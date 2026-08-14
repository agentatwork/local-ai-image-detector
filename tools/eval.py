#!/usr/bin/env python3
"""
Score every candidate ONNX detector on the local eval set and report balanced
accuracy, AUROC, and a per-source breakdown.

Balanced accuracy is the bounty's metric, and it is the honest one here: an eval set
that is 60% real would let a model that always says "real" score 60% plain accuracy.
The per-source table matters more than the headline, because a detector that is
excellent on GLIDE and useless on Flux averages out to "fine" and is not fine.

  python3 eval.py [model ...]        # default: all in models/
"""
import glob, json, os, sys
import numpy as np
import onnxruntime as ort
from PIL import Image

Image.MAX_IMAGE_PIXELS = 200_000_000
RESAMPLE = {2: Image.BILINEAR, 3: Image.BICUBIC}


def load_spec(mdir):
    pp = json.load(open(os.path.join(mdir, "preprocessor_config.json")))
    cfg = json.load(open(os.path.join(mdir, "config.json")))
    size = pp["size"]
    if isinstance(size, dict) and "shortest_edge" in size:
        resize = ("short", size["shortest_edge"])
    elif isinstance(size, dict):
        resize = (size["width"], size["height"])
    else:
        resize = (size, size)
    crop = pp.get("crop_size")
    if isinstance(crop, dict):
        crop = crop.get("height")
    # a ViT only accepts its trained grid; when the resize target is larger than the
    # model's image_size the extra margin is meant to be centre-cropped away (timm's
    # crop_pct convention, which the HF processor config carries but does not apply).
    img_size = cfg.get("image_size")
    if crop is None and img_size and resize[0] != img_size:
        crop = img_size
    return dict(
        resize=resize, crop=crop,
        mean=np.array(pp["image_mean"], dtype=np.float32).reshape(3, 1, 1),
        std=np.array(pp["image_std"], dtype=np.float32).reshape(3, 1, 1),
        resample=RESAMPLE.get(pp.get("resample", 2), Image.BILINEAR),
        labels=cfg.get("id2label"),
    )


def preprocess(path, spec):
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    if spec["resize"][0] == "short":
        # preserve aspect ratio: scale the shorter side, then centre-crop. Squashing a
        # non-square image to a square is what the upstream config bug used to do, and
        # the resampling it introduces is exactly the signal this model reads.
        s = spec["resize"][1]
        w, h = im.size
        scale = s / min(w, h)
        im = im.resize((max(s, round(w * scale)), max(s, round(h * scale))), spec["resample"])
    else:
        im = im.resize(spec["resize"], spec["resample"])
    if spec["crop"]:
        c = spec["crop"]
        w, h = im.size
        l, t = (w - c) // 2, (h - c) // 2
        im = im.crop((l, t, l + c, t + c))
    x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (x - spec["mean"]) / spec["std"]


def sigmoid_pair(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def auroc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())


def balanced_acc(scores, labels, thr):
    pred = scores >= thr
    tpr = (pred & (labels == 1)).sum() / max((labels == 1).sum(), 1)
    tnr = (~pred & (labels == 0)).sum() / max((labels == 0).sum(), 1)
    return (tpr + tnr) / 2, tpr, tnr


def source_of(path):
    base = os.path.basename(path)
    return base.rsplit("-", 1)[0]


def run(name, weights):
    mdir = os.path.dirname(weights)
    spec = load_spec(mdir)
    sess = ort.InferenceSession(weights, providers=["CPUExecutionProvider"],
                                sess_options=_opts())
    inp = sess.get_inputs()[0].name
    files, labels = [], []
    for lab, y in (("real", 0), ("ai", 1)):
        fs = sorted(glob.glob(f"data/{lab}/*"))
        files += fs
        labels += [y] * len(fs)
    labels = np.array(labels)
    scores = np.zeros(len(files), dtype=np.float64)
    for i, f in enumerate(files):
        try:
            x = preprocess(f, spec)[None]
            out = sess.run(None, {inp: x})[0][0]
            # single-logit heads are sigmoid; two-logit heads are softmax
            scores[i] = 1 / (1 + np.exp(-out[0])) if out.shape[0] == 1 else sigmoid_pair(out)[1]
        except Exception as e:
            scores[i] = 0.5
        if i % 60 == 0:
            print(f"    {i}/{len(files)}", end="\r", file=sys.stderr, flush=True)
    a = auroc(scores, labels)
    flip = a < 0.5
    if flip:
        scores = 1.0 - scores
        a = 1.0 - a
    return dict(name=name, weights=weights, files=files, labels=labels,
                scores=scores, auroc=a, flipped=bool(flip), labelmap=spec["labels"])


def _opts():
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return o


def report(r):
    print(f"\n### {r['name']}   (class-1 = {'index 0' if r['flipped'] else 'index 1'}"
          f"{', labels ' + json.dumps(r['labelmap']) if r['labelmap'] else ''})")
    print(f"  AUROC {r['auroc']:.4f}")
    for thr, tag in ((0.65, "bounty thr 0.65"), (0.5, "thr 0.50")):
        b, tpr, tnr = balanced_acc(r["scores"], r["labels"], thr)
        print(f"  {tag:16s} balanced acc {b*100:5.1f}%   TPR(ai) {tpr*100:5.1f}%  TNR(real) {tnr*100:5.1f}%")
    # best achievable threshold, i.e. what calibration could buy
    best = max(((balanced_acc(r["scores"], r["labels"], t)[0], t)
                for t in np.unique(np.round(r["scores"], 4))), key=lambda x: x[0])
    print(f"  best possible    balanced acc {best[0]*100:5.1f}%  at thr {best[1]:.4f}")
    srcs = {}
    for f, y, s in zip(r["files"], r["labels"], r["scores"]):
        srcs.setdefault((source_of(f), y), []).append(s)
    print("  per source (mean P(ai), and %% called ai at the best threshold):")
    for (src, y), v in sorted(srcs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        v = np.array(v)
        print(f"    {'ai  ' if y else 'real'} {src:32s} n={len(v):3d}  mean {v.mean():.3f}"
              f"  called-ai {100*(v>=best[1]).mean():5.1f}%")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["cf", "dist", "aidet"]
    out = []
    for n in todo:
        w = f"models/{n}/model.onnx"
        if not os.path.exists(w):
            continue
        r = run(n, w)
        report(r)
        np.save(f"models/{n}/scores.npy", r["scores"])
        out.append(r)
    if out:
        json.dump([os.path.basename(f) for f in out[0]["files"]], open("data/files.json", "w"))
        np.save("data/labels.npy", out[0]["labels"])
