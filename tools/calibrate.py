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


def platt(z, y, iters=100):
    """Logistic regression of label on logit, with the two classes weighted equally so
    an unbalanced eval set does not drag the slope around."""
    w = np.where(y == 1, 1.0 / max((y == 1).sum(), 1), 1.0 / max((y == 0).sum(), 1))
    w = w / w.sum() * len(y)
    a, b = 1.0, 0.0
    X = np.stack([z, np.ones_like(z)], 1)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a * z + b)))
        g = X.T @ (w * (p - y))
        W = w * p * (1 - p) + 1e-9
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-9:
            break
    return float(a), float(b)


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
    p = 1 / (1 + np.exp(-(a * z + b)))
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
