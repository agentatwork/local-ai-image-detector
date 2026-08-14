#!/usr/bin/env python3
"""
Assemble an evaluation set of real and AI-generated images from public HuggingFace
datasets, via the datasets-server rows API (no auth, no full-dataset download).

The point is generator diversity, not volume. A detector that only ever sees SD1.5
will score beautifully on SD1.5 and fall over on Flux, so every AI source here is a
different generator family, and every real source is a different kind of photograph.

  python3 fetch.py <label> <dataset> <n> [offset]

Images are written to data/<label>/<slug>-<i>.<ext> exactly as served — no resizing
and no re-encoding, because JPEG requantisation is itself a signal these models use.
"""
import io, json, os, sys, time, urllib.parse, urllib.request

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) aidetect-eval/1.0"}
SERVER = "https://datasets-server.huggingface.co"


def get(url, tries=3):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            if t == tries - 1:
                raise
            time.sleep(2 * (t + 1))


def splits(dataset):
    d = json.loads(get(f"{SERVER}/splits?dataset={urllib.parse.quote(dataset)}"))
    return d.get("splits", [])


def image_col(features):
    for f in features:
        t = f["type"]
        if isinstance(t, dict) and t.get("_type") == "Image":
            return f["name"]
    return None


def fetch(label, dataset, n, offset=0):
    sp = splits(dataset)
    if not sp:
        print(f"  {dataset}: no splits", file=sys.stderr)
        return 0
    # prefer a validation/test split when one exists — less likely to be in anyone's training set
    sp.sort(key=lambda s: 0 if s["split"] in ("validation", "test") else 1)
    cfg, split = sp[0]["config"], sp[0]["split"]

    slug = dataset.split("/")[-1].replace(".", "-")
    outdir = os.path.join("data", label)
    os.makedirs(outdir, exist_ok=True)

    got = 0
    page = 0
    while got < n and page < 12:
        url = (f"{SERVER}/rows?dataset={urllib.parse.quote(dataset)}&config={urllib.parse.quote(cfg)}"
               f"&split={urllib.parse.quote(split)}&offset={offset + page * 100}&length=100")
        try:
            d = json.loads(get(url))
        except Exception as e:
            print(f"  {dataset}: rows failed ({e})", file=sys.stderr)
            break
        if "error" in d or not d.get("rows"):
            break
        col = image_col(d["features"])
        if col is None:
            print(f"  {dataset}: no image column", file=sys.stderr)
            return 0
        for row in d["rows"]:
            if got >= n:
                break
            v = row["row"].get(col)
            src = v.get("src") if isinstance(v, dict) else None
            if not src:
                continue
            ext = os.path.splitext(urllib.parse.urlparse(src).path)[1] or ".jpg"
            path = os.path.join(outdir, f"{slug}-{offset + got:05d}{ext}")
            if os.path.exists(path):
                got += 1
                continue
            try:
                blob = get(src, tries=2)
            except Exception:
                continue
            if len(blob) < 2000:      # placeholder / error page
                continue
            with open(path, "wb") as f:
                f.write(blob)
            got += 1
        page += 1
    print(f"  {label:4s} {dataset:45s} {got:4d}  ({cfg}/{split})")
    return got


if __name__ == "__main__":
    label, dataset, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    off = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    fetch(label, dataset, n, off)
