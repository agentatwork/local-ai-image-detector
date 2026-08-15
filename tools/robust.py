#!/usr/bin/env python3
"""
Ask the one question the local eval set cannot answer: does this survive the web?

The bounty's benchmark is "a held-out set of real and AI-generated images assembled from
publicly available datasets *and additional web-realistic samples*". Everything in
data/ is a pristine dataset file. A detector that reads resampling and quantisation
artefacts is exactly the kind that can be destroyed by a second round of resampling and
quantisation — which is what every CMS, every social network and every image host does
on the way to a webpage. So the 86.2% figure could be describing a machine that never
meets the images it will be scored on.

This does not retrain anything and does not touch the calibration. It takes the shipped
pipeline as-is (two views, mean, frozen Platt a/b, threshold 0.65) and re-measures
balanced accuracy after putting each image through a plausible delivery pipeline.

  python3 tools/robust.py [--n 10] [condition ...]

Results accumulate in robust.json, one key per condition, so this can be run a condition
at a time on a single core without losing work.
"""
import glob, io, json, os, sys, time
import numpy as np
import onnxruntime as ort
from PIL import Image

from variants import VARIANTS

Image.MAX_IMAGE_PIXELS = 200_000_000
OUT = "robust.json"
CAL = json.load(open("calibration.json"))


# ---------------------------------------------------------------- degradations
# Each takes a PIL RGB image and returns the image as it would arrive in a browser.
# Re-opening from the encoded bytes matters: the point is the decoder's output, not a
# quality parameter written to a file nobody reads.

def _roundtrip(im, fmt, **kw):
    buf = io.BytesIO()
    im.save(buf, fmt, **kw)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert("RGB")


def _fit(im, longest, resample=Image.LANCZOS):
    w, h = im.size
    if max(w, h) <= longest:
        return im
    s = longest / max(w, h)
    return im.resize((max(1, round(w * s)), max(1, round(h * s))), resample)


COND = {
    "none":       lambda im: im,
    "jpeg90":     lambda im: _roundtrip(im, "JPEG", quality=90),
    "jpeg75":     lambda im: _roundtrip(im, "JPEG", quality=75),
    "jpeg60":     lambda im: _roundtrip(im, "JPEG", quality=60),
    # a CMS that caps the long edge and re-encodes: the single most common thing that
    # happens to an image between a generator and a webpage
    "cms1600":    lambda im: _roundtrip(_fit(im, 1600), "JPEG", quality=85),
    "cms1024":    lambda im: _roundtrip(_fit(im, 1024), "JPEG", quality=85),
    "cms640":     lambda im: _roundtrip(_fit(im, 640), "JPEG", quality=80),
    "webp80":     lambda im: _roundtrip(im, "WEBP", quality=80),
    # lossless but resampled — a screenshot, or a retina asset served at 1x
    "rescale90":  lambda im: im.resize((max(1, round(im.size[0] * .9)),
                                        max(1, round(im.size[1] * .9))), Image.BICUBIC),
    # The two degradations the competing "sieve" submission publishes numbers under
    # (github.com/Phineas1500/sieve-ai-image-detector). Reproduced verbatim so that at
    # least two rows of my table and two rows of theirs describe the same thing done to
    # an image — different images and different models, but the same insult.
    "sieve_web":  lambda im: _roundtrip(_fit(im, 768), "JPEG", quality=60),
    "sieve_hard": lambda im: _roundtrip(_fit(im, 512), "JPEG", quality=40),
}


# ---------------------------------------------------------------- scoring
def sample(n):
    """First n files of each source, both classes. Deterministic, and stratified so a
    condition cannot look good by happening to hit the easy generators."""
    groups = {}
    for lab, y in (("real", 0), ("ai", 1)):
        for f in sorted(glob.glob(f"data/{lab}/*")):
            src = os.path.basename(f).rsplit("-", 1)[0]
            groups.setdefault((src, y), []).append(f)
    files, labels, srcs = [], [], []
    for (src, y), fs in sorted(groups.items()):
        for f in fs[:n]:
            files.append(f); labels.append(y); srcs.append(src)
    return files, np.array(labels), srcs


def calibrated(mean_prob):
    m = np.clip(mean_prob, 1e-12, 1 - 1e-12)
    z = np.log(m / (1 - m))
    return 1 / (1 + np.exp(-(CAL["a"] * z + CAL["b"])))


def balanced_acc(p, y, thr=0.65):
    pred = p >= thr
    tpr = pred[y == 1].mean() if (y == 1).any() else 0.0
    tnr = (~pred)[y == 0].mean() if (y == 0).any() else 0.0
    return (tpr + tnr) / 2, tpr, tnr


def score(sess, inp, files, cond):
    fn = COND[cond]
    out = []
    t0 = time.time()
    for i, f in enumerate(files):
        im = Image.open(f)
        if im.mode != "RGB":
            im = im.convert("RGB")
        im = fn(im)
        ps = []
        for v in ("official", "native"):
            ps.append(max(float(1 / (1 + np.exp(-sess.run(None, {inp: x[None]})[0][0][0])))
                          for x in VARIANTS[v](im)))
        out.append(float(np.mean(ps)))
        if i % 10 == 0:
            el = time.time() - t0
            eta = el / max(i, 1) * (len(files) - i)
            print(f"  {cond} {i}/{len(files)}  eta {eta/60:.1f}m",
                  end="\r", file=sys.stderr, flush=True)
    return np.array(out)


def main(argv):
    n = 10
    if "--n" in argv:
        i = argv.index("--n"); n = int(argv[i + 1]); del argv[i:i + 2]
    conds = argv or list(COND)

    files, y, srcs = sample(n)
    print(f"{len(files)} images  ({(y==0).sum()} real, {(y==1).sum()} ai) "
          f"over {len(set(srcs))} sources")

    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession("models/cf2/model.onnx",
                                providers=["CPUExecutionProvider"], sess_options=o)
    inp = sess.get_inputs()[0].name

    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("_meta", {})["n_per_source"] = n
    res["_meta"]["files"] = [os.path.basename(f) for f in files]
    res["_meta"]["labels"] = [int(v) for v in y]
    res["_meta"]["sources"] = srcs

    for c in conds:
        if c.startswith("_"):
            continue
        if c in res and len(res[c].get("probs", [])) == len(files):
            print(f"  {c}: cached")
        else:
            p = score(sess, inp, files, c)
            res[c] = {"probs": [float(v) for v in p]}
            json.dump(res, open(OUT, "w"))
        p = np.array(res[c]["probs"])
        cal = calibrated(p)
        ba, tpr, tnr = balanced_acc(cal, y)
        res[c].update(balanced_acc=ba, recall=tpr, specificity=tnr)
        json.dump(res, open(OUT, "w"))
        print(f"  {c:10s} balanced acc {ba*100:5.1f}%   recall(ai) {tpr*100:5.1f}%"
              f"   specificity(real) {tnr*100:5.1f}%"
              f"   {'PASS' if ba >= 0.75 else 'FAIL'} vs the 75.0% bar")


if __name__ == "__main__":
    main(sys.argv[1:])
