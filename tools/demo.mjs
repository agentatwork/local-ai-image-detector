#!/usr/bin/env node
/*
 * Screenshot the extension doing the thing it claims to do: badges appearing on an
 * ordinary webpage, unattended.
 *
 *   node tools/demo.mjs <n-per-class> [out.png]
 *
 * verify.mjs measures the model through the extension's code path but never renders a
 * page -- it calls the scorer directly. This one serves a real page over HTTP, lets
 * content.js find the images with its own IntersectionObserver, and waits for the badges
 * to appear on their own. Nothing on the page tells the extension which images are which;
 * the captions are drawn from the directory each file came from and are there for the
 * reader, after the fact.
 *
 * Images are taken deterministically (every kth file of each class) so re-running does not
 * quietly select a flattering sample.
 *
 * Two things this has to do that are not obvious from the outside:
 *
 *  - `showAll` is off by default, so a real photograph gets no badge at all
 *    (src/content.js:38). A demo that only ever paints the AI half is a worse demo, so
 *    this turns it on through the extension's own storage the way the popup would.
 *  - `.laid-layer` is `position: fixed` (src/content.css), which is correct for not
 *    reflowing the host page and fatal for `fullPage: true` -- Chrome would paint every
 *    badge at its viewport offset against a tall scrolled image. The page is sized to fit
 *    one viewport and the screenshot is a plain one.
 */
import { createServer } from "node:http";
import { readdir, readFile, writeFile, mkdir, copyFile, rm } from "node:fs/promises";
import { dirname, join, resolve, extname, basename } from "node:path";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";
import puppeteer from "puppeteer-core";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const CHROME = process.env.CHROME_PATH || "/home/agent/opt/chrome-linux64/chrome";
const DATA = process.env.DATA_DIR || join(ROOT, "..", "data");
const IMG_EXT = new Set([".jpg", ".jpeg", ".png", ".webp"]);
const PORT = 8765;

const N = Number(process.argv[2] || 4);
const OUT = process.argv[3] || "demo.png";

// Deliberately outside ROOT: Chrome scans the whole extension directory when it loads
// unpacked, and an extension that ships a folder of eval images is a worse thing to hand a
// reviewer than one that does not.
const stage = join(tmpdir(), "laid-demo");
await rm(stage, { recursive: true, force: true });
await mkdir(stage, { recursive: true });

const picked = [];
for (const cls of ["ai", "real"]) {
  const all = (await readdir(join(DATA, cls)))
    .filter((f) => IMG_EXT.has(extname(f).toLowerCase())).sort();
  const step = Math.max(1, Math.floor(all.length / N));
  for (let i = 0; i < N; i++) {
    const f = all[(i * step) % all.length];
    await copyFile(join(DATA, cls, f), join(stage, f));
    picked.push({ cls, file: f, source: basename(f).replace(/[-_]?\d+\.[a-z]+$/i, "") });
  }
}

const cards = picked.map((p) =>
  `<figure><img src="/${encodeURIComponent(p.file)}" alt="">` +
  `<figcaption>${p.source} &mdash; <b>${p.cls === "ai" ? "generated" : "real"}</b></figcaption></figure>`
).join("\n");

const html = `<!doctype html><meta charset="utf-8"><title>a page with pictures on it</title>
<style>
 body{background:#0b0d10;color:#e6edf3;font:16px/1.5 system-ui,sans-serif;margin:0;padding:28px}
 h1{font-size:21px;margin:0 0 4px} p{color:#8b98a5;margin:0 0 20px;font-size:13px}
 .g{display:grid;grid-template-columns:repeat(${N},1fr);gap:18px}
 figure{margin:0} img{width:100%;height:205px;object-fit:cover;border-radius:8px;display:block}
 figcaption{color:#8b98a5;font-size:12px;margin-top:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
</style>
<h1>An ordinary page with pictures on it</h1>
<p>Every badge is the extension, unprompted. Nothing on this page asks for them and nothing here
tells it which is which &mdash; the captions under the images are for you, not for it.</p>
<div class="g">${cards}</div>`;

await writeFile(join(stage, "index.html"), html);

const server = createServer(async (req, res) => {
  const name = decodeURIComponent(req.url.split("?")[0]).replace(/^\//, "") || "index.html";
  try {
    const buf = await readFile(join(stage, name));
    const ext = extname(name).toLowerCase();
    res.writeHead(200, { "content-type": ext === ".html" ? "text/html"
      : ext === ".png" ? "image/png" : ext === ".webp" ? "image/webp" : "image/jpeg" });
    res.end(buf);
  } catch { res.writeHead(404).end(); }
}).listen(PORT, "127.0.0.1");

const browser = await puppeteer.launch({
  executablePath: CHROME, headless: true, protocolTimeout: 1_800_000,
  args: [`--disable-extensions-except=${ROOT}`, `--load-extension=${ROOT}`,
         "--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
         "--js-flags=--max-old-space-size=512"],
});

try {
  // The extension id is not knowable before launch for an unpacked load, so read it off
  // whichever target the extension itself registered.
  let extId = null;
  for (let i = 0; i < 60 && !extId; i++) {
    for (const t of await browser.targets()) {
      const m = /^chrome-extension:\/\/([a-p]{32})\//.exec(t.url() || "");
      if (m) { extId = m[1]; break; }
    }
    if (!extId) await new Promise((r) => setTimeout(r, 500));
  }
  if (!extId) throw new Error("extension never registered a target -- did npm run build run?");

  const opt = await browser.newPage();
  // src/popup.html, not popup.html -- the manifest's default_popup carries the directory and
  // there is nothing at the extension root, so the short path is a plain ERR_FILE_NOT_FOUND.
  await opt.goto(`chrome-extension://${extId}/src/popup.html`, { waitUntil: "domcontentloaded" });
  await opt.evaluate(() => chrome.storage.local.set({ showAll: true, enabled: true }));
  await opt.close();

  const page = await browser.newPage();
  const rows = Math.ceil(picked.length / N);
  await page.setViewport({ width: 1400, height: 150 + rows * 245, deviceScaleFactor: 1 });
  await page.goto(`http://127.0.0.1:${PORT}/`, { waitUntil: "networkidle0" });

  // Wait for every image to carry a badge rather than for a fixed sleep: on one WASM thread
  // this is seconds per image, and a timeout picked in advance either wastes time or
  // screenshots a half-scored page and calls it a demo.
  const total = picked.length;
  const deadline = 25 * 60 * 1000;
  await page.waitForFunction(
    (n) => document.querySelectorAll(".laid-badge").length >= n,
    { timeout: deadline, polling: 2000 }, total,
  ).catch(() => console.warn("  ! not every image got a badge before the timeout"));

  const got = await page.evaluate(() =>
    [...document.querySelectorAll(".laid-badge")].map((n) => n.textContent.trim()));
  console.log(`${got.length}/${total} badged: ${got.join("  ")}`);
  await page.screenshot({ path: OUT });
  console.log("wrote", OUT);
} finally {
  await browser.close();
  server.close();
}
