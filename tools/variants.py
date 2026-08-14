#!/usr/bin/env python3
"""
Try alternative preprocessing for the Community Forensics ViT.

The model reads resampling and quantisation artefacts, so how you get an image down to
384x384 is not a detail — it is most of the signal. The official recipe (shortest edge
to 440, centre-crop 384) throws away the native pixel grid of anything bigger than that,
and upsamples anything smaller. Both are exactly the operations that erase the evidence.

Variants:
  official   shortest edge -> 440, centre crop 384          (the model card's recipe)
  native     centre crop 384 at native resolution           (no resampling at all)
  native5    five 384 crops at native res, max-pooled       (centre + four corners)
  up         small images upscaled 2x nearest, then crop     (preserve the pixel grid)

  python3 variants.py [n_per_source]
"""
import glob, json, os, sys
import numpy as np
import onnxruntime as ort
from PIL import Image

Image.MAX_IMAGE_PIXELS = 200_000_000
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32).reshape(3, 1, 1)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32).reshape(3, 1, 1)
S = 384


def norm(im):
    x = np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0
    return (x - MEAN) / STD


def crop(im, l, t):
    return im.crop((l, t, l + S, t + S))


def pad_to(im):
    """Images smaller than one crop get the pixel grid kept and the margin reflected."""
    w, h = im.size
    if w >= S and h >= S:
        return im
    a = np.asarray(im)
    ph, pw = max(0, S - h), max(0, S - w)
    a = np.pad(a, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    return Image.fromarray(a)


def variant_official(im):
    w, h = im.size
    sc = 440 / min(w, h)
    im = im.resize((max(440, round(w * sc)), max(440, round(h * sc))), Image.BICUBIC)
    w, h = im.size
    return [norm(crop(im, (w - S) // 2, (h - S) // 2))]


def variant_native(im):
    im = pad_to(im)
    w, h = im.size
    return [norm(crop(im, (w - S) // 2, (h - S) // 2))]


def variant_native5(im):
    im = pad_to(im)
    w, h = im.size
    pts = {((w - S) // 2, (h - S) // 2), (0, 0), (w - S, 0), (0, h - S), (w - S, h - S)}
    return [norm(crop(im, l, t)) for l, t in sorted(pts)]


def variant_up(im):
    w, h = im.size
    if min(w, h) < S:
        # nearest-neighbour keeps each original pixel intact instead of blending it away
        k = -(-S // min(w, h))
        im = im.resize((w * k, h * k), Image.NEAREST)
    im = pad_to(im)
    w, h = im.size
    return [norm(crop(im, (w - S) // 2, (h - S) // 2))]


def variant_ship(im):
    """Exactly what ext/src/detector.js does, so the numbers describe the shipped thing.

    Integer nearest upscale for anything smaller than one crop, then the centre crop,
    plus the four corners when there is enough image for the corners to be different
    pixels rather than the centre again.
    """
    w, h = im.size
    if min(w, h) < S:
        k = -(-S // min(w, h))
        im = im.resize((w * k, h * k), Image.NEAREST)
        w, h = im.size
    pts = [((w - S) // 2, (h - S) // 2)]
    if w >= S * 1.3 and h >= S * 1.3:
        pts += [(0, 0), (w - S, 0), (0, h - S), (w - S, h - S)]
    return [norm(crop(im, l, t)) for l, t in pts]


VARIANTS = dict(official=variant_official, native=variant_native,
                native5=variant_native5, up=variant_up, ship=variant_ship)


def main(n=15):
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    sess = ort.InferenceSession("models/cf2/model.onnx", providers=["CPUExecutionProvider"], sess_options=o)
    inp = sess.get_inputs()[0].name

    groups = {}
    for lab in ("real", "ai"):
        for f in sorted(glob.glob(f"data/{lab}/*")):
            src = os.path.basename(f).rsplit("-", 1)[0]
            groups.setdefault((lab, src), []).append(f)
    for k in groups:
        groups[k] = groups[k][:n]

    res = {v: {} for v in VARIANTS}
    for (lab, src), files in sorted(groups.items()):
        for vname, vf in VARIANTS.items():
            sc = []
            for f in files:
                im = Image.open(f)
                if im.mode != "RGB":
                    im = im.convert("RGB")
                xs = vf(im)
                p = [1 / (1 + np.exp(-sess.run(None, {inp: x[None]})[0][0][0])) for x in xs]
                sc.append(max(p))            # any crop that looks generated counts
            res[vname][(lab, src)] = np.array(sc)
        print(f"  done {lab}/{src}", file=sys.stderr)

    json.dump({v: {f"{l}|{s}": list(map(float, a)) for (l, s), a in d.items()}
               for v, d in res.items()}, open("variants.json", "w"))
    summarise(res)


def summarise(res):
    for vname, d in res.items():
        real = np.concatenate([a for (l, s), a in d.items() if l == "real"])
        ai = np.concatenate([a for (l, s), a in d.items() if l == "ai"])
        sc = np.concatenate([real, ai])
        y = np.concatenate([np.zeros(len(real)), np.ones(len(ai))])
        best = max(((((sc[y == 1] >= t).mean() + (sc[y == 0] < t).mean()) / 2, t)
                    for t in np.unique(np.round(sc, 4))), key=lambda x: x[0])
        b65 = ((ai >= .65).mean() + (real < .65).mean()) / 2
        print(f"\n=== {vname}: bal-acc@0.65 {b65*100:5.1f}%   best {best[0]*100:5.1f}% @ {best[1]:.4f}")
        for (l, s), a in sorted(d.items()):
            hit = (a >= best[1]).mean() if l == "ai" else (a >= best[1]).mean()
            print(f"    {l:4s} {s:32s} mean {a.mean():.3f}  called-ai {hit*100:5.1f}%")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
