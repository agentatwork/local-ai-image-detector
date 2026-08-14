#!/usr/bin/env python3
"""
Real images straight off the open web, as a counterweight to the dataset-derived ones.

Curated datasets are clean: one encoder, one resize, one quality setting. Real browsing
is not — it is somebody's phone photo run through three CDNs. A detector tuned only on
tidy dataset reals will call the messy ones fake, so the eval set needs both.

Random Wikimedia Commons files, at a thumbnail width that matches what a page actually
serves, saved as delivered.

  python3 fetch_web.py <n>
"""
import json, os, sys, time, urllib.parse, urllib.request

UA = {"User-Agent": "aidetect-eval/1.0 (https://agentatwork.xyz; agent@agentatwork.xyz)"}
API = "https://commons.wikimedia.org/w/api.php"
WIDTHS = [640, 800, 1024, 1280]      # the widths real pages ask for


def get(url, tries=3):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read()
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(2)


def main(n):
    out = "data/real"
    os.makedirs(out, exist_ok=True)
    got, seen, batch = 0, set(), 0
    while got < n and batch < 60:
        w = WIDTHS[batch % len(WIDTHS)]
        q = urllib.parse.urlencode({
            "action": "query", "generator": "random", "grnnamespace": "6",
            "grnlimit": "20", "prop": "imageinfo",
            "iiprop": "url|mime|size", "iiurlwidth": str(w), "format": "json",
        })
        try:
            d = json.loads(get(f"{API}?{q}"))
        except Exception as e:
            print("api:", e, file=sys.stderr)
            batch += 1
            continue
        for page in d.get("query", {}).get("pages", {}).values():
            if got >= n:
                break
            ii = (page.get("imageinfo") or [{}])[0]
            mime, url = ii.get("mime", ""), ii.get("thumburl") or ii.get("url")
            if mime not in ("image/jpeg", "image/png") or not url or url in seen:
                continue
            seen.add(url)
            try:
                blob = get(url, tries=1)
            except Exception:
                continue
            if len(blob) < 8000:
                continue
            ext = ".jpg" if mime == "image/jpeg" else ".png"
            with open(os.path.join(out, f"commons-{got:05d}{ext}"), "wb") as f:
                f.write(blob)
            got += 1
        batch += 1
    print(f"  real commons{'':38s}{got:4d}  (wikimedia random)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
