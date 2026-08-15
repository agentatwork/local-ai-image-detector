#!/usr/bin/env python3
"""Per-view ablation on the clean eval set, from the Chrome run.

Not to be confused with tools/perview.py, which re-runs onnxruntime over the 320-image
degradation sample to ask which view survives a delivery pipeline. This one does no
inference at all: verify.mjs already returns the per-view sigmoids in `views` (in
`viewNames` order), so the ablation on the 1,020 clean files is arithmetic over the same
JSON report.py reads -- and it is a Chrome number, like every other figure in the README.

Each view is scored at *its own* best threshold. Reading a single view at the shipped
0.65 would compare a calibration fitted for the mean against a view it was not fitted
for, which measures the wrong thing and flatters the mean.

  python3 tools/views_clean.py verify_all.json ../logits.json ../data
"""
import json
import os
import sys

import numpy as np
from PIL import Image

from calibrate import balanced_acc, best_threshold

Image.MAX_IMAGE_PIXELS = 200_000_000
BAR = 0.65


def main(js_path="verify_all.json", py_path="logits.json", img_root="data"):
    js = json.load(open(js_path))
    py = json.load(open(py_path))
    lab = dict(zip(py["files"], py["labels"]))

    rows = [r for r in js["out"]
            if not r.get("error") and r["file"] in lab and r.get("views")]
    if not rows:
        sys.exit("no rows with per-view scores -- was this run by an older verify.mjs?")
    files = [r["file"] for r in rows]
    y = np.array([lab[f] for f in files], dtype=np.int64)
    names = rows[0].get("viewNames") or [f"view{i}" for i in range(len(rows[0]["views"]))]
    V = np.array([r["views"] for r in rows], dtype=np.float64)
    mean = np.array([r["raw"] for r in rows], dtype=np.float64)
    p = np.array([r["probability"] for r in rows], dtype=np.float64)

    # Sizes from headers, same convention as report.py, so the two tables split identically.
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
    have_size = bool((side > 0).all())
    big, small = side >= 384, side < 384

    print(f"{len(rows)} images ({int((y==1).sum())} AI / {int((y==0).sum())} real), "
          f"views {names}")
    print(f"{'':32s} {'all':>7s} {'>=384px':>9s} {'<384px':>8s}   threshold")

    def row(label, s, thr):
        a, _, _ = balanced_acc(s, y, thr)
        line = f"{a*100:6.1f}% "
        cells = {"all": a}
        if have_size:
            ab, _, _ = balanced_acc(s[big], y[big], thr)
            asm, _, _ = balanced_acc(s[small], y[small], thr)
            line += f"{ab*100:8.1f}% {asm*100:7.1f}%"
            cells.update(ge_384=ab, lt_384=asm)
        else:
            line += f"{'-':>9s} {'-':>8s}"
        print(f"{label:32s} {line}   {thr:.6g}")
        return dict(cells, threshold=float(thr))

    out = {"views": names, "n": len(rows), "per_view": {}}
    for i, n in enumerate(names):
        t, _ = best_threshold(V[:, i], y)
        out["per_view"][n] = row(f"{n} alone, best threshold", V[:, i], t)
    t, _ = best_threshold(mean, y)
    out["mean_best_threshold"] = row("mean of views, best threshold", mean, t)
    out["shipped_at_0_65"] = row("mean of views, SHIPPED 0.65", p, BAR)

    json.dump(out, open("views_clean.json", "w"), indent=1)
    print("\n-> views_clean.json")


if __name__ == "__main__":
    main(*sys.argv[1:])
