/*
 * Router and cache. Holds no model and does no inference — it starts the offscreen
 * document on demand, forwards scoring requests to it, and remembers results so the
 * same image on ten pages is scored once.
 */

const CACHE_MAX = 4000;
const cache = new Map();          // url -> result
let creating = null;

async function ensureOffscreen() {
  const existing = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
  });
  if (existing.length) return;
  if (creating) return creating;
  creating = chrome.offscreen.createDocument({
    url: "src/offscreen.html",
    reasons: ["WORKERS"],
    justification:
      "Runs the image classifier locally. Needs a document context for WebGPU/WASM and " +
      "to keep one model instance alive across tabs.",
  }).catch((e) => {
    // A concurrent call may have created it between the check and the create.
    if (!String(e).includes("Only a single offscreen")) throw e;
  });
  await creating;
  creating = null;
}

function remember(url, result) {
  if (cache.size >= CACHE_MAX) {
    // cheap FIFO eviction — insertion order is Map's iteration order
    for (const k of cache.keys()) { cache.delete(k); if (cache.size < CACHE_MAX * 0.9) break; }
  }
  cache.set(url, result);
}

async function scoreOne(url) {
  if (cache.has(url)) return { ...cache.get(url), cached: true };
  await ensureOffscreen();
  const r = await chrome.runtime.sendMessage({ target: "offscreen", type: "score", url });
  if (r && !r.error) remember(url, r);
  return r;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.target === "offscreen") return;      // not ours; offscreen.js handles it

  if (msg.type === "score") {
    scoreOne(msg.url).then(sendResponse).catch((e) => sendResponse({ error: String(e) }));
    return true;
  }
  if (msg.type === "settings") {
    chrome.storage.local.get(DEFAULTS).then(sendResponse);
    return true;
  }
  if (msg.type === "set-settings") {
    chrome.storage.local.set(msg.values).then(() => sendResponse({ ok: true }));
    return true;
  }
  if (msg.type === "status") {
    ensureOffscreen()
      .then(() => chrome.runtime.sendMessage({ target: "offscreen", type: "status" }))
      .then((s) => sendResponse({ ...s, cached: cache.size }))
      .catch((e) => sendResponse({ ready: false, error: String(e) }));
    return true;
  }
});

export const DEFAULTS = {
  enabled: true,
  threshold: 0.65,        // the operating point the model is calibrated for
  minSize: 128,           // px; below this an image is furniture, not content
  showAll: false,         // badge every image, not just the ones over threshold
};

const MENU_ID = "laid-check-image";

chrome.runtime.onInstalled.addListener(async () => {
  const cur = await chrome.storage.local.get(DEFAULTS);
  await chrome.storage.local.set({ ...DEFAULTS, ...cur });
  // Automatic scoring skips small images, and some readers turn it off entirely. The
  // right-click is the way to ask about one specific image regardless of either.
  chrome.contextMenus.removeAll(() =>
    chrome.contextMenus.create({
      id: MENU_ID,
      title: "Check this image for AI generation",
      contexts: ["image"],
    }));
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== MENU_ID || !info.srcUrl || !tab?.id) return;
  const r = await scoreOne(info.srcUrl).catch((e) => ({ error: String(e) }));
  chrome.tabs.sendMessage(tab.id, { type: "manual-result", url: info.srcUrl, result: r })
    .catch(() => {});   // no content script on this page (a PDF, a chrome:// tab)
});
