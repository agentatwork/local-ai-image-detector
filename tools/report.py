#!/usr/bin/env python3
"""Turn a full-set verify.mjs run into the table the README quotes.

Every headline number in the README has to be a Chrome number, because the Python is what
*chose* the configuration and a measurement taken by the thing that chose it is not a
measurement of the thing that ships. verify.mjs produces per-image probabilities out of the
built extension; this reads them and reports.

Labels and sources are joined by filename from the Python dump, which covers the same 1,020
files -- verify.mjs stages a flat directory and does not carry the label. Sizes are read
from the image headers, not decoded.

One thing to be careful about, because it is easy to report it as if it meant the other
thing: leave-one-generator-out here refits a *clean-set* threshold with a generator held
out. That is not the shipped operating point -- the shipped calibration is fitted by
minimax over delivery conditions (tools/minimax.py), and its held-out figure is
leave-one-*condition*-out. LOGO is a statement about transfer to unseen generators, run the
same way the previous build's LOGO was run so the two can be compared. Both are reported;
neither is allowed to stand in for the other.

  python3 tools/report.py verify_all.json logits.json
"""
import json
import os
import sys

import numpy as np
from PIL import Image

from calibrate import balanced_acc, best_threshold

Image.MAX_IMAGE_PIXELS = 200_000_000
BAR = 0.65


def auroc(s, y):
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), dtype=np.float64)
    r[order] = np.arange(1, len(s) + 1)
    # average ranks within ties, or an AUROC of a coarse score is quietly wrong
    su = np.sort(s)
    i = 0
    while i < len(su):
        j = i
        while j + 1 < len(su) and su[j + 1] == su[i]:
            j += 1
        if j > i:
            r[np.isin(s, su[i])] = (i + j + 2) / 2
        i = j + 1
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main(js_path="verify_all.json", py_path="logits.json", img_root="data"):
    js = json.load(open(js_path))
    py = json.load(open(py_path))
    lab = dict(zip(py["files"], py["labels"]))
    srcs = dict(zip(py["files"], py["sources"]))

    rows = [r for r in js["out"] if not r.get("error") and r["file"] in lab]
    dropped = len(js["out"]) - len(rows)
    files = [r["file"] for r in rows]
    p = np.array([r["probability"] for r in rows], dtype=np.float64)
    raw = np.array([r["raw"] for r in rows], dtype=np.float64)
    y = np.array([lab[f] for f in files], dtype=np.int64)
    src = np.array([srcs[f] for f in files])
    views = js["out"][0].get("viewNames", "unknown")

    acc, tpr, tnr = balanced_acc(p, y, BAR)
    print(f"views {views}   {len(rows)} images ({int((y==1).sum())} AI / {int((y==0).sum())} real)"
          + (f"   [{dropped} rows dropped]" if dropped else ""))
    print(f"\nAT THE BOUNTY'S {BAR}: balanced acc {acc*100:.1f}%   recall(ai) {tpr*100:.1f}%   "
          f"specificity(real) {tnr*100:.1f}%")
    print(f"AUROC {auroc(raw, y):.4f}   (calibration is monotone, so this is view choice only)")

    # What 0.65 costs when the raw score is read as if it were a probability. This is the
    # single biggest number in the whole exercise and it is pure arithmetic.
    acc_raw, _, _ = balanced_acc(raw, y, BAR)
    t_raw, acc_best = best_threshold(raw, y)
    print(f"raw score read at {BAR} without calibrating: {acc_raw*100:.1f}%   "
          f"best raw threshold {t_raw:.6f} -> {acc_best*100:.1f}%")

    # by size, from headers
    side = []
    for f in files:
        for d in ("ai", "real"):
            q = os.path.join(img_root, d, f)
            if os.path.exists(q):
                with Image.open(q) as im:
                    side.append(min(im.size))
                break
        else:
            side.append(-1)
    side = np.array(side)
    if (side > 0).all():
        for name, m in (("min side >= 384px", side >= 384), ("min side <  384px", side < 384)):
            a, _, _ = balanced_acc(p[m], y[m], BAR)
            print(f"  {name}: {a*100:.1f}%  (n={int(m.sum())}, "
                  f"{int((y[m]==1).sum())} AI / {int((y[m]==0).sum())} real)")
    else:
        print(f"  by-size skipped: {int((side<0).sum())} files not found under {img_root}/")

    # leave-one-generator-out, same procedure as the previous build's figure
    z = np.log(np.clip(raw, 1e-12, 1 - 1e-12) / (1 - np.clip(raw, 1e-12, 1 - 1e-12)))
    folds = []
    for s in sorted(set(src[y == 1])):
        held = src == s
        fit = ~held
        t, _ = best_threshold(z[fit], y[fit])
        m = held | (y == 0)
        a, tp, tn = balanced_acc(z[m], y[m], t)
        folds.append((s, a, tp, int(held.sum())))
    logo = float(np.mean([f[1] for f in folds]))
    worst = min(folds, key=lambda f: f[1])
    print(f"\nLEAVE-ONE-GENERATOR-OUT (clean-set refit, NOT the shipped threshold)")
    print(f"  mean {logo*100:.1f}%   worst {worst[1]*100:.1f}% on {worst[0]}   "
          f"above 95%: {sum(f[1] > 0.95 for f in folds)}/{len(folds)}")
    for s, a, tp, n in sorted(folds, key=lambda f: f[1]):
        print(f"    {s:36s} acc {a*100:5.1f}%  recall {tp*100:5.1f}%  n={n}")

    # per-source recall / false-positive rate at the shipped point
    print(f"\nPER SOURCE AT {BAR}")
    for s in sorted(set(src)):
        m = src == s
        is_ai = y[m][0] == 1
        hit = (p[m] >= BAR).mean()
        print(f"  {s:36s} {'recall' if is_ai else 'false-pos'} {hit*100:5.1f}%  n={int(m.sum())}")

    out = dict(views=views, n=len(rows), n_ai=int((y == 1).sum()), n_real=int((y == 0).sum()),
               balanced_accuracy_at_0_65=acc, recall_ai=tpr, specificity_real=tnr,
               auroc=auroc(raw, y), uncalibrated_at_0_65=acc_raw, raw_threshold_equivalent=t_raw,
               leave_one_generator_out=logo,
               logo_worst=dict(source=worst[0], acc=worst[1]),
               by_size=({"min_side_ge_384": balanced_acc(p[side >= 384], y[side >= 384], BAR)[0],
                         "min_side_lt_384": balanced_acc(p[side < 384], y[side < 384], BAR)[0]}
                        if (side > 0).all() else None))
    json.dump(out, open("report.json", "w"), indent=1)
    print("\n-> report.json")


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]) or 0)
