#!/usr/bin/env python3
"""
Compare what the shipped JavaScript produces against what the Python produced.

This is the check that decides whether any of the Python measurement transfers. The model
file is identical in both cases; what differs is who decoded the JPEG, who wrote the
bicubic resize, and whether the reflect padding folded the same way. The extension does
its own downscale in explicit code precisely so this comparison is possible — handing the
resize to `ctx.drawImage` would make the answer depend on a Chrome version.

A per-view probability agreeing to ~1e-3 means the two pipelines are the same detector.
A disagreement concentrated on one view tells you which half to go and read.

  python3 compare.py logits.json ext/verify.json
"""
import json, sys
import numpy as np


def main(py_path="logits.json", js_path="ext/verify.json"):
    py = json.load(open(py_path))
    js = json.load(open(js_path))
    by_file = {f: i for i, f in enumerate(py["files"])}

    rows = [(r["file"], by_file[r["file"]], r)
            for r in js["out"] if not r.get("error") and r["file"] in by_file]
    if not rows:
        print("no overlap between the two runs")
        return 1

    # Which views the extension ran is model/config.json's choice, so pair them up by name
    # rather than by position. Comparing views[0] to views[0] was safe while both sides
    # were hard-coded to official+native; with a configurable list it would silently
    # compare two different views and report the mismatch as a parity failure.
    names = js["out"][0].get("viewNames") or py["variants"]
    missing = [k for k in names if k not in py["probs"]]
    if missing:
        print(f"the JS ran views the Python file has no scores for: {missing}\n"
              f"  rerun: python3 tools/dump.py logits.json {' '.join(names)}")
        return 1

    print(f"{len(rows)} images scored by both, views {'+'.join(names)}\n")
    for v, k in enumerate(names):
        a = np.array([py["probs"][k][i] for _, i, _ in rows])
        b = np.array([r["views"][v] for _, _, r in rows])
        d = np.abs(a - b)
        print(f"  view {k:9s} |diff| median {np.median(d):.2e}  p95 {np.quantile(d,.95):.2e}"
              f"  max {d.max():.2e}")

    a = np.array([np.mean([py["probs"][k][i] for k in names]) for _, i, _ in rows])
    b = np.array([r["raw"] for _, _, r in rows])
    d = np.abs(a - b)
    print(f"  mean of views |diff| median {np.median(d):.2e}  max {d.max():.2e}")

    # the only disagreement that can change an answer is one that crosses the boundary
    t = js.get("threshold_raw")
    if t is not None:
        flips = int(((a >= t) != (b >= t)).sum())
        print(f"\n  decisions changed by the disagreement at raw threshold {t:.6f}: {flips}")

    print("\n  worst five by mean-of-views:")
    order = np.argsort(-d)[:5]
    for i in order:
        f = rows[i][0]
        print(f"    {f:44s} py {a[i]:.6f}   js {b[i]:.6f}   d {d[i]:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*(sys.argv[1:] or [])))
