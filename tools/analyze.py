#!/usr/bin/env python3
"""
Read variants.json and ask whether the preprocessing should depend on the image.

The variant sweep produced an awkward result. Nearest-neighbour upscaling rescues every
source the official recipe misses — BigGAN 0% -> 100% recall, FLUX 0% -> 75% — and at the
same time sets fire to the real images of the same size: FFHQ-256 goes from 17% false
positives to 92%. Upscaling does not preserve the evidence, it manufactures evidence, and
it manufactures the same evidence for both classes.

Which is a statement about the score *distribution*, not about the ranking. So the
question this script asks is whether the two paths are each fine on their own terms, and
the only mistake is reading both through one threshold. The branch is chosen by image
size, which is known at inference time and has nothing to do with the label.

  python3 analyze.py
"""
import glob, json, os
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 200_000_000
N = 12                                   # variants.py took the first N of each group


def rebuild_index():
    """variants.json is keyed by group; recover which file each score belongs to."""
    rows = []
    for lab in ("real", "ai"):
        for f in sorted(glob.glob(f"data/{lab}/*")):
            src = os.path.basename(f).rsplit("-", 1)[0]
            rows.append((lab, src, f))
    out = {}
    for lab, src, f in rows:
        out.setdefault((lab, src), []).append(f)
    return {k: v[:N] for k, v in out.items()}


def min_side(path):
    with Image.open(path) as im:
        return min(im.size)


def bal_acc(scores, y, t):
    p = scores >= t
    return ((p[y == 1].mean() if (y == 1).any() else 0) +
            ((~p)[y == 0].mean() if (y == 0).any() else 0)) / 2


def best_t(scores, y):
    if len(scores) == 0:
        return 0.5, 0.0
    cands = np.unique(np.round(scores, 4))
    accs = [bal_acc(scores, y, t) for t in cands]
    i = int(np.argmax(accs))
    return float(cands[i]), float(accs[i])


def main():
    V = json.load(open("variants.json"))
    idx = rebuild_index()

    files, y, small = [], [], []
    per = {v: [] for v in V}
    for key in sorted(idx):
        lab, src = key
        fs = idx[key]
        gk = f"{lab}|{src}"
        for i, f in enumerate(fs):
            files.append(f)
            y.append(1 if lab == "ai" else 0)
            small.append(min_side(f) < 384)
            for v in V:
                per[v].append(V[v][gk][i])
    y = np.array(y)
    small = np.array(small)
    for v in per:
        per[v] = np.array(per[v])

    print(f"{len(y)} images · {(y==1).sum()} generated · {small.sum()} smaller than one crop\n")

    for v in per:
        t, a = best_t(per[v], y)
        print(f"  {v:9s} single threshold  best {a*100:5.1f}% @ {t:.4f}")

    print("\n  split by whether the image needed upscaling:")
    for v in per:
        ts, as_ = best_t(per[v][small], y[small])
        tl, al = best_t(per[v][~small], y[~small])
        # combined accuracy of the two-branch rule, scored over everything at once
        pred = np.where(small, per[v] >= ts, per[v] >= tl)
        acc = ((pred[y == 1].mean()) + (~pred[y == 0]).mean()) / 2
        print(f"  {v:9s} small {as_*100:5.1f}% @ {ts:.4f}   large {al*100:5.1f}% @ {tl:.4f}"
              f"   combined {acc*100:5.1f}%")

    # the mixed policy: official for images that already have the pixels, up for the rest
    print("\n  mixed policy (up when the image is smaller than a crop, official otherwise):")
    mixed = np.where(small, per["up"], per["official"])
    ts, as_ = best_t(mixed[small], y[small])
    tl, al = best_t(mixed[~small], y[~small])
    pred = np.where(small, mixed >= ts, mixed >= tl)
    acc = ((pred[y == 1].mean()) + (~pred[y == 0]).mean()) / 2
    print(f"    small {as_*100:5.1f}% @ {ts:.4f}   large {al*100:5.1f}% @ {tl:.4f}"
          f"   combined {acc*100:5.1f}%")
    t1, a1 = best_t(mixed, y)
    print(f"    one threshold instead: {a1*100:5.1f}% @ {t1:.4f}")

    print("\n  combinations (cost = inferences per image):")
    combos = [
        ("official                 1", per["official"]),
        ("mean(official, native)   2", (per["official"] + per["native"]) / 2),
        ("max (official, native)   2", np.maximum(per["official"], per["native"])),
        ("mean(official, up)       2", (per["official"] + per["up"]) / 2),
        ("max (official, up)       2", np.maximum(per["official"], per["up"])),
        ("mean(official, native5)  6", (per["official"] + per["native5"]) / 2),
        ("max (official, native5)  6", np.maximum(per["official"], per["native5"])),
        ("mean of all four         8", sum(per[v] for v in ("official", "native", "native5", "up")) / 4),
    ]
    for name, s in combos:
        t, a = best_t(s, y)
        p = s >= t
        print(f"    {name:26s} best {a*100:5.1f}% @ {t:.4f}"
              f"   recall {p[y==1].mean()*100:5.1f}%  specificity {(~p)[y==0].mean()*100:5.1f}%")

    # The second view only exists when the image actually has 384 real pixels to give.
    # Reflect-padding a smaller one is what variant_native does, and reproducing numpy's
    # reflect padding inside a canvas is a parity risk for a fraction of a point — so
    # measure the version that simply declines to take a second view.
    print("\n  second view only when the image is at least one crop wide:")
    for label, second in (("native", per["native"]), ("native5", per["native5"])):
        s = np.where(small, per["official"], (per["official"] + second) / 2)
        t, a = best_t(s, y)
        p = s >= t
        print(f"    official + {label:8s} when large  best {a*100:5.1f}% @ {t:.4f}"
              f"   recall {p[y==1].mean()*100:5.1f}%  specificity {(~p)[y==0].mean()*100:5.1f}%")

    # A cascade: pay for the second pass only where the first one is not sure. The band is
    # chosen on the first-pass score alone, so the cost is data-dependent but not label-
    # dependent, and the average cost is what a reader actually waits for.
    print("\n  cascade — official first, second pass only inside an uncertain band:")
    o = per["official"]
    for lo, hi in ((0.001, 0.9), (0.0005, 0.99), (0.0001, 0.999)):
        band = (o >= lo) & (o <= hi)
        s = np.where(band, (o + per["native5"]) / 2, o)
        t, a = best_t(s, y)
        print(f"    band [{lo}, {hi}]  {band.mean()*100:4.1f}% of images rescored"
              f"  -> {a*100:5.1f}% @ {t:.4f}"
              f"   mean cost {1 + 5*band.mean():.2f} inferences")


if __name__ == "__main__":
    main()
