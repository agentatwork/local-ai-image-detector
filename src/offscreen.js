/*
 * The only page in the extension that holds the model. One copy, shared by every tab.
 *
 * It lives in an offscreen document rather than the service worker because the worker
 * can be killed mid-inference and because WebGPU is not available to it; and rather
 * than in the content script because a model per tab is a model per tab.
 *
 * Images are fetched here, decoded here, scored here, and discarded here. There is no
 * code path in this file that sends anything outward.
 */
import { load, score, status } from "./detector.js";

const queue = [];
let running = false;

async function fetchBitmap(url) {
  // Extension-context fetch: it carries the extension's host permissions, so it works
  // on cross-origin images that a page-context canvas would refuse to read back.
  const res = await fetch(url, { credentials: "omit", cache: "force-cache" });
  if (!res.ok) throw new Error("http " + res.status);
  const blob = await res.blob();
  if (!blob.type.startsWith("image/") && blob.type !== "") throw new Error("not an image");
  if (blob.size < 1024) throw new Error("too small");
  // colorSpaceConversion "none" hands back the decoded pixels as they were stored. The
  // default would apply the file's ICC profile and rewrite every pixel of a Display P3
  // or Adobe RGB photo on the way in — a global, image-wide resample of exactly the
  // values the detector was calibrated on. Decode, don't interpret.
  return createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
}

async function pump() {
  if (running) return;
  running = true;
  try {
    await load();
    while (queue.length) {
      const job = queue.shift();
      try {
        const bmp = await fetchBitmap(job.url);
        // Tiny images are icons, spacers and tracking pixels. Scoring them is noise.
        if (bmp.width < 96 || bmp.height < 96) {
          bmp.close();
          job.done({ url: job.url, skipped: "too small" });
          continue;
        }
        const r = await score(bmp);
        bmp.close();
        job.done({ url: job.url, ...r });
      } catch (e) {
        job.done({ url: job.url, error: String(e.message || e) });
      }
    }
  } catch (e) {
    while (queue.length) queue.shift().done({ error: "model failed to load: " + e.message });
  } finally {
    running = false;
  }
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.target !== "offscreen") return;
  if (msg.type === "status") { sendResponse(status()); return; }
  if (msg.type === "score") {
    queue.push({ url: msg.url, done: sendResponse });
    pump();
    return true;              // keep the channel open for the async reply
  }
});
