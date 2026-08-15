#!/usr/bin/env python3
"""
Split the shipped score back into its two views, per condition.

The shipped pipeline scores every image twice — the model card's downscale-and-crop
("official") and a crop at the native pixel grid ("native") — and reports the mean. On
pristine files the mean beats both views (86.2% vs 84.2% / 79.7%), which is why it ships.

But the mean is a fixed 50/50 bet, and the two views do not read the same thing. The
official view resamples, so it sees whatever survives a second interpolation. The native
view keeps the pixel grid, so it sees quantisation directly — which is exactly what JPEG
destroys. If the collapse at <=512px + q40 is concentrated in one view, then the fix is a
weight, not a second model: free at inference time, no extra bytes to download, and it
uses evidence the extension already computes.

This writes both view scores per image per condition so that question can be answered
from the numbers instead of argued about.

  python3 tools/perview.py [condition ...]        # writes perview.json
"""
import json, os, sys, time
import numpy as np
import onnxruntime as ort
from PIL import Image

from robust import COND, sample
from variants import VARIANTS

Image.MAX_IMAGE_PIXELS = 200_000_000
OUT = "perview.json"
VIEWS = ("official", "native")


def main(argv):
    global VIEWS
    if "--views" in argv:
        i = argv.index("--views")
        VIEWS = tuple(argv[i + 1].split(","))
        del argv[i:i + 2]
    conds = argv or ["none", "sieve_hard"]
    files, y, srcs = sample(10)

    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession("models/cf2/model.onnx",
                                providers=["CPUExecutionProvider"], sess_options=o)
    inp = sess.get_inputs()[0].name

    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res["_meta"] = {"files": [os.path.basename(f) for f in files],
                    "labels": [int(v) for v in y], "sources": srcs, "views": list(VIEWS)}

    for c in conds:
        if c in res and all(len(res[c].get(v, [])) == len(files) for v in VIEWS):
            print(f"  {c}: cached")
            continue
        fn = COND[c]
        acc = {v: [] for v in VIEWS}
        t0 = time.time()
        for i, f in enumerate(files):
            im = Image.open(f)
            if im.mode != "RGB":
                im = im.convert("RGB")
            im = fn(im)
            for v in VIEWS:
                p = max(float(1 / (1 + np.exp(-sess.run(None, {inp: x[None]})[0][0][0])))
                        for x in VARIANTS[v](im))
                acc[v].append(p)
            if i % 10 == 0:
                el = time.time() - t0
                print(f"  {c} {i}/{len(files)} eta {el/max(i,1)*(len(files)-i)/60:.1f}m",
                      end="\r", file=sys.stderr, flush=True)
        res.setdefault(c, {}).update({v: acc[v] for v in VIEWS})
        json.dump(res, open(OUT, "w"))

        # Oracle-threshold separability per view and for the shipped mean, so the views
        # are compared on ranking alone, independent of any calibration.
        def oracle(p):
            p = np.asarray(p)
            return max((((p >= t)[y == 1].mean() + (p < t)[y == 0].mean()) / 2)
                       for t in np.unique(p))
        parts = "  ".join(f"{v} {oracle(acc[v])*100:5.1f}%" for v in VIEWS)
        allmean = np.mean([acc[v] for v in VIEWS], axis=0)
        print(f"  {c:11s} {parts}   mean-of-all {oracle(allmean)*100:5.1f}%")


if __name__ == "__main__":
    main(sys.argv[1:])
