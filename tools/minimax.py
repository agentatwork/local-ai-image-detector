#!/usr/bin/env python3
"""Pick the view subset and calibration that maximise the WORST condition, not the clean one.

The shipped extension fits its threshold on undegraded images. That is the wrong objective
for this bounty. The maintainer benchmark is private and described only as "held-out real
and AI images from public datasets and additional web-realistic samples" -- I do not know
which of my eleven delivery pipelines it resembles, and "web-realistic" hints it is not the
clean one. Under that ignorance the defensible choice is minimax: pick the configuration
whose worst condition is highest, because the bar is a floor (75.0%) and a floor is a
worst-case quantity.

The bounty also fixes the decision threshold at 65% *confidence*, so I cannot simply move a
cut point. What I can choose is the calibration that maps the raw score to a confidence, and
the honest way to state the result is: with calibration (a, b), thresholding the reported
confidence at 0.65 is equivalent to thresholding the mean probability at P*. So the search is
over (view subset, P*), and P* is converted back to an (a, b) the extension can ship.

Reported for every candidate: worst condition, mean, and the paired bootstrap CI of the
worst-case difference against the shipped path -- because a minimax winner chosen on the
same data it is evaluated on is a fitted quantity, and its margin needs an error bar.

  python3 tools/minimax.py
"""
import json
import itertools
import numpy as np

RNG = np.random.default_rng(1)
B = 10000
LOGIT_65 = float(np.log(0.65 / 0.35))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def bacc(s, y, t):
    pred = s >= t
    return (pred[y == 1].mean() + (~pred)[y == 0].mean()) / 2


P = json.load(open("perview.json"))
old = json.load(open("calibration.json"))
y = np.array(P["_meta"]["labels"])
views = list(P["_meta"]["views"])
conds = [c for c in P if not c.startswith("_")
         and all(len(P[c].get(v, [])) == len(y) for v in views)]
Q = {c: {v: np.asarray(P[c][v], dtype=np.float64) for v in views} for c in conds}
pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)

P_STAR_SHIPPED = float(sigmoid((LOGIT_65 - old["b"]) / old["a"]))
SHIP = ("official", "native")


def scores(sub, c):
    return np.mean([Q[c][v] for v in sub], axis=0)


def worst_over_conditions(sub, t):
    return min(bacc(scores(sub, c), y, t) for c in conds)


# Candidate thresholds: every midpoint between observed scores, pooled across conditions and
# the subset in question. Searching a grid would quietly miss the optimum between grid points.
def best_threshold(sub):
    pool = np.unique(np.concatenate([scores(sub, c) for c in conds]))
    mids = np.concatenate([[pool[0] - 1e-9], (pool[:-1] + pool[1:]) / 2, [pool[-1] + 1e-9]])
    vals = [worst_over_conditions(sub, t) for t in mids]
    i = int(np.argmax(vals))
    return float(mids[i]), float(vals[i])


subsets = [s for r in (1, 2, 3) for s in itertools.combinations(views, r)]
rows = []
for sub in subsets:
    t, w = best_threshold(sub)
    m = float(np.mean([bacc(scores(sub, c), y, t) for c in conds]))
    rows.append((w, m, t, sub))
rows.sort(reverse=True)

print(f"{len(conds)} conditions, {len(y)} images ({len(pos)} AI / {len(neg)} real)")
print(f"conditions: {', '.join(conds)}\n")
print("SHIPPED, as it stands (threshold fitted on 'none' only)")
sw = worst_over_conditions(SHIP, P_STAR_SHIPPED)
sm = np.mean([bacc(scores(SHIP, c), y, P_STAR_SHIPPED) for c in conds])
print(f"  official+native   worst {sw*100:.1f}%   mean {sm*100:.1f}%\n")

print("MINIMAX over the same conditions (threshold chosen to lift the floor)")
print(f"{'views':28s} {'worst':>7s} {'mean':>7s} {'#views':>7s}  {'P*':>10s}")
for w, m, t, sub in rows[:8]:
    print(f"{'+'.join(sub):28s} {w*100:6.1f}% {m*100:6.1f}% {len(sub):7d}  {t:10.5f}")

# Error bar on the headline improvement. Paired: both configurations scored on the same
# resampled images, and the WORST condition is recomputed inside each replicate rather than
# fixed in advance -- otherwise the minimax selection is not being resampled at all.
w, m, t, sub = rows[0]
ip = RNG.integers(0, len(pos), size=(B, len(pos)))
ineg = RNG.integers(0, len(neg), size=(B, len(neg)))
sp, sn = pos[ip], neg[ineg]


def bacc_b(s, thr):
    return ((s[sp] >= thr).mean(axis=1) + (s[sn] < thr).mean(axis=1)) / 2


cand_w = np.min([bacc_b(scores(sub, c), t) for c in conds], axis=0)
ship_w = np.min([bacc_b(scores(SHIP, c), P_STAR_SHIPPED) for c in conds], axis=0)
d = cand_w - ship_w
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"\nbest = {'+'.join(sub)}  worst-condition {w*100:.1f}% vs shipped {sw*100:.1f}%")
print(f"paired diff of the worst condition: {(w-sw)*100:+.1f}  95% CI {lo*100:+.1f} .. {hi*100:+.1f}"
      f"   P(diff>0) {(d>0).mean():.3f}")
print(f"P(candidate worst >= 75.0%) over resamples: {(cand_w >= 0.75).mean():.3f}"
      f"   (shipped: {(ship_w >= 0.75).mean():.3f})")

# ---------------------------------------------------------------------------------------
# Leave-one-condition-out. The number above is fitted and evaluated on the same conditions,
# so it is an in-sample worst case and it flatters itself. The maintainer benchmark is a
# pipeline I have never seen, so the question that actually matters is: if the threshold is
# chosen without ever seeing a condition, how does it do ON that condition? Hold each one
# out, fit the minimax threshold on the remaining ten, score the held-out one.
print("\nLEAVE-ONE-CONDITION-OUT (threshold never sees the condition it is scored on)")
loco_s = [bacc(scores(SHIP, c), y, P_STAR_SHIPPED) for c in conds]
print(f"shipped as-is: worst {min(loco_s)*100:.1f}%  mean {np.mean(loco_s)*100:.1f}%  "
      f"clearing 75.0%: {sum(v >= 0.75 for v in loco_s)}/{len(conds)}\n")

# Report LOCO for several subsets, not just the minimax winner. Swapping views needs a new
# preprocessing path in the extension and a Python/JS parity check; re-fitting the threshold
# on the views already shipping is one number in config.json. Those are different amounts of
# risk, so they deserve separately measured payoffs rather than one headline.
print(f"{'views':26s} {'LOCO worst':>11s} {'LOCO mean':>10s} {'>=75%':>7s}  {'new view?':>9s}")
loco_table = {}
for _, _, _, s in rows[:5] + [(0, 0, 0, SHIP)]:
    if s in loco_table:
        continue
    got = []
    for held in conds:
        rest = [c for c in conds if c != held]
        pool = np.unique(np.concatenate([scores(s, c) for c in rest]))
        mids = np.concatenate([[pool[0] - 1e-9], (pool[:-1] + pool[1:]) / 2, [pool[-1] + 1e-9]])
        t_l = float(mids[int(np.argmax(
            [min(bacc(scores(s, c), y, x) for c in rest) for x in mids]))])
        got.append(bacc(scores(s, held), y, t_l))
    loco_table[s] = got
    needs = "no" if set(s) <= set(SHIP) else "YES"
    print(f"{'+'.join(s):26s} {min(got)*100:10.1f}% {np.mean(got)*100:9.1f}% "
          f"{sum(v >= 0.75 for v in got):4d}/{len(conds)}  {needs:>9s}")

loco_c = loco_table[sub]

# Convert P* back to a calibration the extension can ship: keep the slope, move the
# intercept so that confidence 0.65 lands exactly on P*.
a = old["a"]
b = LOGIT_65 - a * float(np.log(t / (1 - t)))
print(f"\nship as: views={list(sub)}  a={a:.4f}  b={b:.4f}"
      f"   (checks out: P* = {sigmoid((LOGIT_65 - b)/a):.5f})")
json.dump({"views": list(sub), "p_star": t, "a": a, "b": b,
           "worst": w, "mean": m, "conditions": conds},
          open("minimax.json", "w"), indent=1)
