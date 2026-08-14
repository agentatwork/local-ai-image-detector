#!/usr/bin/env node
/*
 * Load the built extension into a real Chrome, score a directory of images through the
 * extension's own code path, and write the results as JSON.
 *
 *   node tools/verify.mjs <image-dir> [out.json] [--offline]
 *
 * This exists because the thing that decides whether this extension works is not the
 * model — it is whether the JavaScript preprocessing produces the same tensor as the
 * Python that measured the model. A half-pixel difference in a crop, or one resize with
 * smoothing left on, moves the score more than swapping the weights would. So the
 * numbers that get quoted anywhere come from here, through Chrome, not from Python.
 *
 * --offline flips the browser to offline after the extension has loaded, which is the
 * state the evaluation runs in: no cloud, no localhost, no second download.
 */
import { readdir, writeFile, mkdir, copyFile } from "node:fs/promises";
import { dirname, join, resolve, extname } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const CHROME = process.env.CHROME_PATH || "/home/agent/opt/chrome-linux64/chrome";
const IMG_EXT = new Set([".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"]);

const [dir, out = "verify.json"] = process.argv.slice(2).filter((a) => !a.startsWith("--"));
const OFFLINE = process.argv.includes("--offline");
if (!dir) { console.error("usage: node tools/verify.mjs <image-dir> [out.json] [--offline]"); process.exit(2); }

// Serve the images from the extension's own origin so the offscreen document can fetch
// them with no network at all — the point is to test our code, not Chrome's networking.
const staging = join(ROOT, "model", "_verify");
await mkdir(staging, { recursive: true });
const files = (await readdir(dir)).filter((f) => IMG_EXT.has(extname(f).toLowerCase())).sort();
for (const f of files) await copyFile(join(dir, f), join(staging, f));
console.log(`staged ${files.length} images`);

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  // one evaluate() call scores the whole directory; the default 180 s protocol timeout
  // would kill a WASM run of any size partway through and report it as a browser crash
  protocolTimeout: 3_600_000,
  args: [
    `--disable-extensions-except=${ROOT}`,
    `--load-extension=${ROOT}`,
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--js-flags=--max-old-space-size=512",
  ],
});

try {
  // find the extension id from the offscreen/service-worker target
  let ext = null;
  for (let i = 0; i < 60 && !ext; i++) {
    const t = browser.targets().find((t) => t.url().startsWith("chrome-extension://"));
    if (t) ext = new URL(t.url()).host;
    else await new Promise((r) => setTimeout(r, 500));
  }
  if (!ext) throw new Error("extension did not start");
  console.log("extension id", ext);

  const page = await browser.newPage();
  await page.goto(`chrome-extension://${ext}/src/offscreen.html`, { waitUntil: "domcontentloaded" });

  if (OFFLINE) {
    await page.setOfflineMode(true);
    console.log("browser set offline");
  }

  const results = await page.evaluate(async (names) => {
    const { load, score, status, config } = await import("./detector.js");
    const t0 = performance.now();
    await load();
    const loadMs = performance.now() - t0;
    // where 0.65 calibrated lands on the raw mean-of-views scale, so compare.py can count
    // the disagreements that would actually change an answer rather than all of them
    const { a, b } = config().calibration;
    const lg = Math.log(0.65 / 0.35);
    const thresholdRaw = 1 / (1 + Math.exp(-(lg - b) / a));
    const out = [];
    for (const n of names) {
      const t = performance.now();
      try {
        const res = await fetch(chrome.runtime.getURL("model/_verify/" + n));
        // same decode settings as offscreen.js, or this measures a different pipeline
        const bmp = await createImageBitmap(await res.blob(),
          { colorSpaceConversion: "none", premultiplyAlpha: "none" });
        const r = await score(bmp);
        bmp.close();
        out.push({ file: n, ...r, ms: performance.now() - t });
      } catch (e) {
        out.push({ file: n, error: String(e && e.message || e) });
      }
    }
    return { status: status(), loadMs, threshold_raw: thresholdRaw, out };
  }, files);

  console.log(`backend ${results.status.backend}, model load ${Math.round(results.loadMs)} ms`);
  const ok = results.out.filter((r) => !r.error);
  const ms = ok.map((r) => r.ms).sort((a, b) => a - b);
  if (ms.length) {
    console.log(`scored ${ok.length}/${files.length}  median ${Math.round(ms[ms.length >> 1])} ms/image`);
  }
  const bad = results.out.filter((r) => r.error);
  if (bad.length) console.log("errors:", bad.slice(0, 5));

  await writeFile(join(ROOT, out), JSON.stringify(results, null, 1));
  console.log("wrote", out);
} finally {
  await browser.close();
}
