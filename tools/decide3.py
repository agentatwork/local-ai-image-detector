#!/usr/bin/env python3
"""
Same question, in the arithmetic the extension actually uses.

decide2.py averaged views in logit space. `ext/src/detector.js` averages them in
probability space -- `const mean = (p1 + p2) / 2` -- and only then applies the frozen
Platt calibration and the 0.65 cut. Those are different operations, and the difference
showed up as a baseline of 69.9% against a shipped figure of 72.3%. A view change
justified against a baseline computed differently from the shipped path is not justified
at all, so everything below is probability-space, exactly as detector.js does it.

The decision rule, fixed before looking:
  * candidates are scored with ONE threshold fitted on the clean condition only, because
    the browser cannot know which pipeline an image arrived through;
  * a difference under 5.6 points (2 SE at n=320) is not a difference;
  * among subsets that tie within that floor, the one with the fewest forward passes wins,
    since each view is a full ViT pass in a browser.

  python3 decide3.py
"""
import json, itertools
import numpy as np

LOGIT_65 = float(np.log(0.65 / 0.35))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def bacc(s, y, t):
    pred = s >= t
    return ((pred[y == 1].mean() + (~pred)[y == 0].mean()) / 2,
            pred[y == 1].mean(), (~pred)[y == 0].mean())


def oracle(s, y):
    c = np.unique(s)
    mid = np.concatenate([[c[0] - 1], (c[:-1] + c[1:]) / 2, [c[-1] + 1]])
    best, bt = -1.0, 0.0
    for t in mid:
        v = bacc(s, y, t)[0]
        if v > best:
            best, bt = v, t
    return best, bt


P = json.load(open("perview.json"))
old = json.load(open("calibration.json"))
y = np.array(P["_meta"]["labels"])
src = np.array(P["_meta"]["sources"])
views = list(P["_meta"]["views"])
conds = [c for c in P if not c.startswith("_")
         and all(len(P[c].get(v, [])) == len(y) for v in views)]
Q = {c: {v: np.asarray(P[c][v], dtype=np.float64) for v in views} for c in conds}
se = 0.5 * np.sqrt(0.25 / (y == 1).sum() + 0.25 / (y == 0).sum())
floor = 2 * se * 100

# The shipped decision in probability space: calibrate(mean_p) >= 0.65 is exactly
# mean_p >= p_star, with p_star fixed by the frozen a,b.
P_STAR = float(sigmoid((LOGIT_65 - old["b"]) / old["a"]))
print(f"{len(y)} images, {int((y==1).sum())} AI / {int((y==0).sum())} real")
print(f"shipped calibration a={old['a']:.4f} b={old['b']:.4f} -> 0.65 is mean-p >= "
      f"{P_STAR:.5f}")
print(f"noise floor {floor:.1f} points\n")

print("SHIPPED PATH (official+native, probability mean, frozen calibration, 0.65)")
for c in conds:
    s = (Q[c]["official"] + Q[c]["native"]) / 2
    a, r, sp = bacc(s, y, P_STAR)
    print(f"   {c:11s} {a*100:5.1f}%   recall {r*100:5.1f}%  spec {sp*100:5.1f}%")
ship_worst = min(bacc((Q[c]["official"] + Q[c]["native"]) / 2, y, P_STAR)[0]
                 for c in conds)

print("\nCANDIDATES (probability mean, one threshold fitted on 'none' only)")
print(f"   {'combination':30s} {'views':>5s} " + "".join(f"{c:>12s}" for c in conds)
      + f"   {'worst':>7s}")
rows = []
for r in range(1, len(views) + 1):
    for sub in itertools.combinations(views, r):
        _, t = oracle(np.mean([Q["none"][v] for v in sub], axis=0), y)
        accs = {c: bacc(np.mean([Q[c][v] for v in sub], axis=0), y, t)[0] for c in conds}
        rows.append((min(accs.values()), sub, accs, t))
for worst, sub, accs, t in sorted(rows, reverse=True):
    tag = "  <- shipped pair" if set(sub) == {"official", "native"} else ""
    print(f"   {'+'.join(sub):30s} {len(sub):5d} "
          + "".join(f"{accs[c]*100:11.1f}%" for c in conds) + f"   {worst*100:6.1f}%{tag}")

best_worst = max(r[0] for r in rows)
tied = [r for r in rows if r[0] >= best_worst - 2 * se]
pick = min(tied, key=lambda r: (len(r[1]), -r[0]))
print(f"\nbest worst-condition {best_worst*100:.1f}%; "
      f"{len(tied)} subset(s) tie within {floor:.1f}pt")
print(f"CHEAPEST TIED PICK: {'+'.join(pick[1])}  ({len(pick[1])} views)  "
      f"worst {pick[0]*100:.1f}%   threshold on mean-p = {pick[3]:.5f}")
print(f"   vs shipped path today: {ship_worst*100:.1f}%  "
      f"({(pick[0]-ship_worst)*100:+.1f}pt, floor {floor:.1f}pt)")
print(f"   clean condition: {pick[2]['none']*100:.1f}% vs shipped "
      f"{bacc((Q['none']['official']+Q['none']['native'])/2, y, P_STAR)[0]*100:.1f}%")

print(f"\nLEAVE-ONE-GENERATOR-OUT for {'+'.join(pick[1])}")
folds = []
for g in sorted(set(src[y == 1])):
    keep = ~((src == g) & (y == 1))
    _, t = oracle(np.mean([Q["none"][v][keep] for v in pick[1]], axis=0), y[keep])
    m = ((src == g) & (y == 1)) | (y == 0)
    per = {c: bacc(np.mean([Q[c][v][m] for v in pick[1]], axis=0), y[m], t) for c in conds}
    folds.append((min(v[0] for v in per.values()), g, per))
for w, g, per in sorted(folds):
    print(f"   {g:34s} worst {w*100:5.1f}%   "
          + "  ".join(f"{c} rec {per[c][1]*100:5.1f}%" for c in conds))
print(f"   mean {np.mean([f[0] for f in folds])*100:.1f}%   "
      f"median {np.median([f[0] for f in folds])*100:.1f}%")
