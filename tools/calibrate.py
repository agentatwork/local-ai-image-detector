#!/usr/bin/env python3
"""
Fit the calibration that maps this model's raw logit onto a probability whose 0.65
point is the decision boundary we actually measured.

Why this is necessary. The bounty scores balanced accuracy at a fixed 65% confidence
threshold. A classifier's raw output is not a probability — it is whatever the training
loss left behind — and this one is badly skewed: it is almost certain about real images
and only mildly confident about generated ones, so at 0.65 it silently becomes a
high-precision, low-recall detector and throws away most of its own accuracy. Platt
scaling fixes the scale without touching the ranking; AUROC is identical before and
after. Nothing here looks at any test image, and no image-specific value is stored.

Why the split is by generator. Holding out random images would let SDXL appear in both
halves, and the number that comes out of that measures memorisation, not detection. The
question a real benchmark asks is "does this work on a generator you have never seen",
so entire sources are held out at a time, and the reported figure is the mean over
leave-one-generator-out folds.

  python3 calibrate.py logits.json
"""
import json, sys
import numpy as np


def load(path):
    d = json.load(open(path))
    z = np.array(d["logits"], dtype=np.float64)
    y = np.array(d["labels"], dtype=np.int64)
    src = np.array(d["sources"])
    return z, y, src


def balanced_acc(z, y, t):
    p = z >= t
    tpr = p[y == 1].mean() if (y == 1).any() else 0.0
    tnr = (~p)[y == 0].mean() if (y == 0).any() else 0.0
    return (tpr + tnr) / 2, tpr, tnr


def best_threshold(z, y):
    cands = np.unique(z)
    if len(cands) > 4000:
        cands = np.quantile(z, np.linspace(0, 1, 4000))
    scores = [balanced_acc(z, y, t)[0] for t in cands]
    i = int(np.argmax(scores))
    return float(cands[i]), float(scores[i])


def _sigmoid(x):
    # branch-free and overflow-free; np.exp(-x) for x = -27 is fine, for x = +800 is not
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def platt(z, y, iters=200, ridge=1e-3):
    """Logistic regression of label on logit, with the two classes weighted equally so
    an unbalanced eval set does not drag the slope around.

    Two details that are not optional here. The targets are Platt's smoothed ones rather
    than hard 0/1 — with hard labels and a nearly separable score, the likelihood keeps
    improving as the slope grows, and the fit runs off to a step function that reports
    every image as exactly 0 or exactly 1. And each Newton step is line-searched against
    the penalised objective, because once the slope is large enough to saturate every
    probability the Hessian goes to zero and an undamped step is unbounded. A degenerate
    slope still scores well on balanced accuracy — it puts the boundary in the same place
    — but it throws away the only thing calibration was for, which is a number between
    the two answers that means something.
    """
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    t = np.where(y == 1, (n1 + 1) / (n1 + 2), 1 / (n0 + 2))
    w = np.where(y == 1, 1.0 / max(n1, 1), 1.0 / max(n0, 1))
    w = w / w.sum() * len(y)
    X = np.stack([z, np.ones_like(z)], 1)

    def obj(v):
        u = X @ v
        # -weighted log-likelihood, computed via log1p(exp(-|u|)) so it never overflows
        ll = w * (t * u - (np.maximum(u, 0) + np.log1p(np.exp(-np.abs(u)))))
        return -ll.sum() + 0.5 * ridge * float(v @ v)

    v = np.array([1.0, 0.0])
    f = obj(v)
    for _ in range(iters):
        p = _sigmoid(X @ v)
        g = X.T @ (w * (p - t)) + ridge * v
        W = w * p * (1 - p)
        H = X.T @ (X * W[:, None]) + (ridge + 1e-9) * np.eye(2)
        step = np.linalg.solve(H, g)
        s = 1.0
        for _ in range(40):                      # backtrack until the objective drops
            v2 = v - s * step
            f2 = obj(v2)
            if f2 <= f:
                break
            s /= 2
        else:
            break                                # no downhill direction left
        moved = np.abs(v2 - v).max()
        v, f = v2, f2
        if moved < 1e-10:
            break
    return float(v[0]), float(v[1])


def leave_one_generator_out(z, y, src):
    """Threshold chosen on every generator but one, scored on the one left out."""
    out = []
    for s in sorted(set(src[y == 1])):
        held = src == s
        # hold out one AI source; keep all real images in the fit, since real is not
        # the axis being generalised over
        fit = ~held
        if not fit.any() or not held.any():
            continue
        t, _ = best_threshold(z[fit], y[fit])
        # score the held-out generator against all real images
        mask = held | (y == 0)
        acc, tpr, tnr = balanced_acc(z[mask], y[mask], t)
        out.append((s, acc, tpr, tnr, t))
    return out


def main(path):
    z, y, src = load(path)
    print(f"{len(z)} images · {(y==1).sum()} generated from {len(set(src[y==1]))} generators"
          f" · {(y==0).sum()} real from {len(set(src[y==0]))} sources\n")

    t_star, acc_in = best_threshold(z, y)
    print(f"in-sample best logit threshold {t_star:+.4f} -> balanced acc {acc_in*100:.1f}%")

    folds = leave_one_generator_out(z, y, src)
    if folds:
        m = np.mean([f[1] for f in folds])
        print(f"leave-one-generator-out mean balanced acc {m*100:.1f}%  "
              f"(worst {min(f[1] for f in folds)*100:.1f}% on {min(folds, key=lambda f: f[1])[0]})\n")
        for s, acc, tpr, tnr, t in sorted(folds, key=lambda f: f[1]):
            print(f"   held out {s:34s} acc {acc*100:5.1f}%  recall {tpr*100:5.1f}%  thr {t:+.3f}")

    a, _ = platt(z, y)
    # Put the measured decision boundary exactly at the threshold the bounty scores at.
    LOGIT_65 = float(np.log(0.65 / 0.35))
    b = LOGIT_65 - a * t_star
    p = _sigmoid(a * z + b)
    acc65, tpr65, tnr65 = balanced_acc(p, y, 0.65)
    print(f"\ncalibration  a={a:.6f}  b={b:.6f}")
    print(f"after calibration, at the bounty's 0.65: balanced acc {acc65*100:.1f}%  "
          f"recall(ai) {tpr65*100:.1f}%  specificity(real) {tnr65*100:.1f}%")
    json.dump({"a": a, "b": b, "logit_threshold": t_star,
               "balanced_acc_at_0.65": acc65,
               "leave_one_generator_out": float(np.mean([f[1] for f in folds])) if folds else None},
              open("calibration.json", "w"), indent=1)
    print("\nwrote calibration.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logits.json")
