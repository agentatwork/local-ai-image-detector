#!/usr/bin/env python3
"""Emit the eleven-condition table AT THE SHIPPED THRESHOLD, as data.

Three files in this directory now hold per-condition balanced accuracies and they are not
the same table, because they answer different questions:

  paired.py   scores both view pairs at a threshold fitted on the CLEAN condition, which is
              what the old build shipped. Right for reading the *difference* between two
              view pairs; wrong as a description of what either scores now.
  minimax.py  fits the threshold by minimax over all eleven and reports only worst and mean.
  this file   the per-condition levels of the configuration that actually ships.

I nearly published paired.py's levels as the shipped build's. They differ by up to 3.6
points. So the shipped numbers get their own script, it reads the threshold out of
minimax.json rather than repeating it, and everything downstream (README, claim text, card)
reads shiptable.json rather than a number I typed twice.

  python3 tools/shiptable.py
"""
import json

import numpy as np

NAMES = {
    "none": "nothing",
    "rescale90": "rescale 90%, no re-encode",
    "jpeg90": "JPEG q90",
    "jpeg75": "JPEG q75",
    "jpeg60": "JPEG q60",
    "cms1600": "CMS resize <=1600px + q85",
    "cms1024": "CMS resize <=1024px + q85",
    "cms640": "CMS resize <=640px + q80",
    "webp80": "WebP q80",
    "sieve_web": "<=768px + JPEG q60",
    "sieve_hard": "<=512px + JPEG q40",
}
BAR = 0.75

P = json.load(open("perview.json"))
M = json.load(open("minimax.json"))
views, t = M["views"], M["p_star"]
y = np.array(P["_meta"]["labels"])
conds = [c for c in P if not c.startswith("_")
         and all(len(P[c].get(v, [])) == len(y) for v in views)]
assert set(conds) == set(M["conditions"]), "perview.json and minimax.json disagree on conditions"

rows = []
for c in conds:
    s = np.mean([np.asarray(P[c][v], dtype=np.float64) for v in views], axis=0)
    pred = s >= t
    rec = float(pred[y == 1].mean())
    spec = float((~pred)[y == 0].mean())
    rows.append(dict(cond=c, label=NAMES.get(c, c), bacc=(rec + spec) / 2,
                     recall=rec, spec=spec))

rows.sort(key=lambda r: -r["bacc"])
clearing = sum(r["bacc"] >= BAR for r in rows)
worst = min(r["bacc"] for r in rows)
mean = float(np.mean([r["bacc"] for r in rows]))

# minimax.py computed worst and mean independently; if this file disagrees, one of them is
# reading a stale perview.json and the published table would be wrong in a way nobody sees.
assert abs(worst - M["worst"]) < 5e-4, f"worst {worst} != minimax {M['worst']}"
assert abs(mean - M["mean"]) < 5e-4, f"mean {mean} != minimax {M['mean']}"

print(f"views {'+'.join(views)}   raw threshold {t:.6f}   "
      f"{len(y)} images ({int((y == 1).sum())} AI / {int((y == 0).sum())} real)\n")
print(f"{'pipeline':28s} {'bacc':>7s} {'recall':>7s} {'spec':>7s}")
for r in rows:
    print(f"{r['label']:28s} {r['bacc']*100:6.1f}% {r['recall']*100:6.1f}% {r['spec']*100:6.1f}%")
print(f"\nworst {worst*100:.1f}%   mean {mean*100:.1f}%   clearing {BAR:.1%}: "
      f"{clearing}/{len(rows)}")

json.dump(dict(views=views, threshold=t, n_ai=int((y == 1).sum()), n_real=int((y == 0).sum()),
               bar=BAR, clearing=clearing, worst=worst, mean=mean, rows=rows),
          open("shiptable.json", "w"), indent=1)
print("-> shiptable.json")
