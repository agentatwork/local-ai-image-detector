#!/usr/bin/env python3
"""Was the decision rule the right test?

decide3.py compares a candidate against the shipped path using a +-2.8pt standard error
and calls anything under 5.6pt (2 SE) not a difference. That SE is for ONE balanced
accuracy estimated once. But both systems are scored on the SAME 320 images, so their
errors are correlated and the quantity that matters is the standard error of the
DIFFERENCE, which is smaller -- often much smaller. Using an unpaired floor for a paired
comparison is conservative in a way that can discard a real improvement.

This is a post-hoc change of test, which is exactly the move that manufactures results, so
it is reported as one and it does not by itself authorise shipping anything:

  * the pre-registered rule stands as the ship/no-ship gate for today;
  * this script only answers "was that gate measuring the right thing";
  * a paired result that clears here is grounds for a NEW pre-registered validation on
    held-out generators, not for editing detector.js.

Both numbers get published either way.

  python3 tools/paired.py
"""
import json
import itertools
import numpy as np

RNG = np.random.default_rng(0)   # fixed: the bootstrap must not move between runs
B = 20000


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def bacc(s, y, t):
    pred = s >= t
    return (pred[y == 1].mean() + (~pred)[y == 0].mean()) / 2


def oracle(s, y):
    c = np.unique(s)
    mid = np.concatenate([[c[0] - 1], (c[:-1] + c[1:]) / 2, [c[-1] + 1]])
    best, bt = -1.0, 0.0
    for t in mid:
        v = bacc(s, y, t)
        if v > best:
            best, bt = v, t
    return best, bt


P = json.load(open("perview.json"))
old = json.load(open("calibration.json"))
y = np.array(P["_meta"]["labels"])
views = list(P["_meta"]["views"])
conds = [c for c in P if not c.startswith("_")
         and all(len(P[c].get(v, [])) == len(y) for v in views)]
Q = {c: {v: np.asarray(P[c][v], dtype=np.float64) for v in views} for c in conds}

LOGIT_65 = float(np.log(0.65 / 0.35))
P_STAR = float(sigmoid((LOGIT_65 - old["b"]) / old["a"]))

# The challenger, and the threshold it would ship with: fitted on the clean condition only.
CAND = ("native", "squash")
_, T_CAND = oracle(np.mean([Q["none"][v] for v in CAND], axis=0), y)

pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
se_unpaired = 0.5 * np.sqrt(0.25 / len(pos) + 0.25 / len(neg))

print(f"{len(y)} images, {len(pos)} AI / {len(neg)} real, {B} bootstrap resamples")
print(f"unpaired floor used by decide3.py: {2*se_unpaired*100:.1f} points\n")
print(f"{'condition':12s} {'shipped':>8s} {'nat+squash':>11s} {'diff':>7s} "
      f"{'95% CI of diff':>18s} {'P(diff>0)':>10s}")

signs = []
for c in conds:
    ship = (Q[c]["official"] + Q[c]["native"]) / 2
    cand = np.mean([Q[c][v] for v in CAND], axis=0)
    d = bacc(cand, y, T_CAND) - bacc(ship, y, P_STAR)
    signs.append(d > 0)

    # Stratified paired bootstrap: resample AI and real images separately, so each
    # replicate keeps the class balance that balanced accuracy is defined against, and
    # BOTH systems are evaluated on the SAME resampled images. That pairing is the point.
    ip = RNG.integers(0, len(pos), size=(B, len(pos)))
    ineg = RNG.integers(0, len(neg), size=(B, len(neg)))
    sp, sn = pos[ip], neg[ineg]

    def bacc_b(s, t):
        return ((s[sp] >= t).mean(axis=1) + (s[sn] < t).mean(axis=1)) / 2

    db = bacc_b(cand, T_CAND) - bacc_b(ship, P_STAR)
    lo, hi = np.percentile(db, [2.5, 97.5])
    print(f"{c:12s} {bacc(ship, y, P_STAR)*100:7.1f}% {bacc(cand, y, T_CAND)*100:10.1f}% "
          f"{d*100:+6.1f} {lo*100:+8.1f} .. {hi*100:+5.1f} {(db > 0).mean():9.3f}")

print(f"\ncandidate beats shipped on {sum(signs)}/{len(signs)} conditions")
print("Conditions are NOT independent -- same images, same views, different degradations --")
print("so this cannot be turned into a sign test. It is consistency, not significance.")
