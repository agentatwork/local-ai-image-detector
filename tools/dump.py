#!/usr/bin/env python3
"""
Score the whole eval set with one or more preprocessing variants and write the results.

Everything downstream — the threshold, the calibration, the number quoted in the README —
is fitted on this file. It stores per-variant probabilities plus the logit of their mean,
because the shipped detector takes two views of every image (the model card's resize, and
a crop at the image's own resolution) and averages them. That pairing is worth 2.4 points
of balanced accuracy over either view alone; see analyze.py.

  python3 dump.py logits.json official native
  python3 dump.py logits_ns.json native squash --every 8    # parity sample, not a fit

Sources are recorded per image because the honest way to fit a threshold is to hold out
whole generators, not random images; see calibrate.py.
"""
import glob, json, os, sys
import numpy as np
import onnxruntime as ort
from PIL import Image

from variants import VARIANTS

Image.MAX_IMAGE_PIXELS = 200_000_000


def main(out="logits.json", *vnames):
    # `--every N` keeps every Nth image. Fitting a threshold wants the whole set, but the
    # JS/Python parity check does not: it compares the two implementations image by image,
    # so a strided sample answers the same question at a fraction of the CPU. The stride is
    # applied after the real/ai concatenation, so it walks both classes and every source in
    # turn rather than taking a prefix.
    every = 1
    vnames = list(vnames)
    for i, a in enumerate(vnames):
        if a.startswith("--every"):
            every = int(a.split("=", 1)[1]) if "=" in a else int(vnames[i + 1])
            vnames = [x for j, x in enumerate(vnames)
                      if j != i and not (("=" not in a) and j == i + 1)]
            break
    vnames = vnames or ["official", "native"]
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    sess = ort.InferenceSession("models/cf2/model.onnx",
                                providers=["CPUExecutionProvider"], sess_options=o)
    inp = sess.get_inputs()[0].name

    files, labels = [], []
    for lab, y in (("real", 0), ("ai", 1)):
        fs = sorted(glob.glob(f"data/{lab}/*"))
        files += fs
        labels += [y] * len(fs)

    if every > 1:
        files, labels = files[::every], labels[::every]

    probs = {v: [] for v in vnames}
    sources = []
    for i, f in enumerate(files):
        im = Image.open(f)
        if im.mode != "RGB":
            im = im.convert("RGB")
        for v in vnames:
            # max over crops where a variant takes several: one region that looks
            # generated is enough to call the image.
            probs[v].append(max(
                float(1 / (1 + np.exp(-sess.run(None, {inp: x[None]})[0][0][0])))
                for x in VARIANTS[v](im)))
        sources.append(os.path.basename(f).rsplit("-", 1)[0])
        if i % 20 == 0:
            print(f"  {i}/{len(files)}", end="\r", file=sys.stderr, flush=True)

    mean = np.mean([probs[v] for v in vnames], axis=0)
    # back to a logit so the calibration has the full dynamic range to work with; a mean
    # probability of 1e-9 and one of 1e-14 are the same number in float32 otherwise.
    z = np.log(np.clip(mean, 1e-12, 1 - 1e-12) / (1 - np.clip(mean, 1e-12, 1 - 1e-12)))

    json.dump({"variants": vnames, "probs": {v: probs[v] for v in vnames},
               "logits": list(map(float, z)), "labels": labels, "sources": sources,
               "files": [os.path.basename(f) for f in files]}, open(out, "w"))
    y = np.array(labels)
    print(f"\nwrote {out}: {len(z)} images over {len(vnames)} views, "
          f"median logit real {np.median(z[y==0]):+.2f} / ai {np.median(z[y==1]):+.2f}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
