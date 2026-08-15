/*
 * Finds images on the page, asks for a score, and puts a badge on the ones that come
 * back generated.
 *
 * Two rules shape this file. Only score what the reader can actually see, because
 * scoring a lazy-loaded feed of four hundred images would heat the laptop for nothing;
 * and never move the page, because an extension that reflows the article it is
 * annotating is worse than no extension.
 */

const MIN_DEFAULT = 128;
// Mirrors DEFAULTS in background.js -- these are what applies for the few ms before storage
// answers, so a mismatch shows up as a flicker rather than as an error.
let settings = { enabled: true, threshold: 0.65, minSize: MIN_DEFAULT, showAll: true };

const seen = new WeakMap();          // img -> {url, state}
let layer = null;
const badges = new Map();            // img -> badge element

/* ---------- overlay ---------- */

function ensureLayer() {
  if (layer && layer.isConnected) return layer;
  layer = document.createElement("div");
  layer.className = "laid-layer";
  document.documentElement.appendChild(layer);
  return layer;
}

function place(img, badge) {
  const r = img.getBoundingClientRect();
  if (r.width < 1 || r.height < 1) { badge.style.display = "none"; return; }
  badge.style.display = "";
  badge.style.transform = `translate(${Math.round(r.left + 6)}px, ${Math.round(r.top + 6)}px)`;
}

function badgeFor(img, result, force = false) {
  const pct = Math.round(result.probability * 100);
  const flagged = result.probability >= settings.threshold;
  if (!flagged && !settings.showAll && !force) return;

  let b = badges.get(img);
  if (!b) {
    b = document.createElement("div");
    b.className = "laid-badge";
    b.addEventListener("click", (e) => { e.stopPropagation(); b.classList.toggle("laid-open"); });
    ensureLayer().appendChild(b);
    badges.set(img, b);
  }
  b.classList.toggle("laid-flag", flagged);
  // The bounty fixes the flag boundary at 0.65, which is above 0.5, so there is a band where
  // the verdict is "not flagged" and P(real) is still under half. Printing "real 37%" there is
  // true and reads as a contradiction, so that band says what it actually is. Every label
  // names the quantity it reports; none of them silently switch scales.
  b.textContent = flagged ? `AI ${pct}%`
    : pct >= 50 ? `unsure ${pct}% AI`
    : `real ${100 - pct}%`;
  b.title = flagged
    ? `Scored ${pct}% likely AI-generated (threshold ${Math.round(settings.threshold * 100)}%). ` +
      `Checked entirely on this device.`
    : `Scored ${pct}% likely AI-generated — below the ${Math.round(settings.threshold * 100)}% threshold.`;
  place(img, b);
}

let toastEl = null, toastTimer = 0;
function toast(text, flagged) {
  if (!toastEl || !toastEl.isConnected) {
    toastEl = document.createElement("div");
    toastEl.className = "laid-toast";
    ensureLayer().appendChild(toastEl);
  }
  toastEl.classList.toggle("laid-flag", !!flagged);
  toastEl.textContent = text;
  toastEl.style.opacity = "1";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { if (toastEl) toastEl.style.opacity = "0"; }, 6000);
}

function reposition() {
  for (const [img, b] of badges) {
    if (!img.isConnected) { b.remove(); badges.delete(img); continue; }
    place(img, b);
  }
}

/* ---------- selection ---------- */

function usableUrl(img) {
  // currentSrc is what the browser actually fetched, which is the variant the reader
  // is looking at; src can point at a different size or at nothing at all.
  const u = img.currentSrc || img.src;
  if (!u || u.startsWith("data:") || u.startsWith("blob:")) return null;
  try {
    const p = new URL(u, location.href);
    return p.protocol === "http:" || p.protocol === "https:" ? p.href : null;
  } catch { return null; }
}

function bigEnough(img) {
  const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
  return w >= settings.minSize && h >= settings.minSize;
}

async function consider(img) {
  if (!settings.enabled) return;
  const url = usableUrl(img);
  if (!url) return;
  const prev = seen.get(img);
  if (prev && prev.url === url) return;
  if (!bigEnough(img)) return;
  seen.set(img, { url, state: "pending" });
  try {
    const r = await chrome.runtime.sendMessage({ type: "score", url });
    if (!r || r.error || r.skipped) { seen.set(img, { url, state: "skip" }); return; }
    seen.set(img, { url, state: "done", result: r });
    badgeFor(img, r);
  } catch {
    // the service worker was asleep and the page is going away; nothing to do
  }
}

/* ---------- wiring ---------- */

const io = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) consider(e.target);
}, { rootMargin: "200px" });

function watch(img) {
  if (img.dataset.laidWatched) return;
  img.dataset.laidWatched = "1";
  if (img.complete) io.observe(img);
  else img.addEventListener("load", () => io.observe(img), { once: true });
}

function scan(root = document) {
  for (const img of root.querySelectorAll("img")) watch(img);
}

const mo = new MutationObserver((muts) => {
  for (const m of muts) {
    for (const n of m.addedNodes) {
      if (n.nodeType !== 1) continue;
      if (n.tagName === "IMG") watch(n);
      else if (n.querySelectorAll) scan(n);
    }
  }
});

let raf = 0;
function onMove() {
  if (raf) return;
  raf = requestAnimationFrame(() => { raf = 0; reposition(); });
}

chrome.runtime.sendMessage({ type: "settings" }).then((s) => {
  if (s) settings = { ...settings, ...s };
  if (!settings.enabled) return;
  scan();
  mo.observe(document.documentElement, { childList: true, subtree: true });
  addEventListener("scroll", onMove, { passive: true, capture: true });
  addEventListener("resize", onMove, { passive: true });
}).catch(() => {});

chrome.storage.onChanged.addListener((ch) => {
  for (const k in ch) settings[k] = ch[k].newValue;
  reposition();
});

chrome.runtime.onMessage.addListener((msg, _s, respond) => {
  if (msg.type === "page-report") {
    const out = [];
    for (const img of document.querySelectorAll("img")) {
      const s = seen.get(img);
      if (s?.state === "done") out.push({ url: s.url, probability: s.result.probability });
    }
    respond({ images: out, threshold: settings.threshold });
  }
  if (msg.type === "manual-result") {
    const r = msg.result || {};
    const text = r.error ? `couldn't read that image (${r.error})`
      : r.skipped ? `too small to judge (${r.skipped})`
      : `${Math.round(r.probability * 100)}% likely AI-generated`;
    toast(text, !r.error && !r.skipped && r.probability >= settings.threshold);
    // if that image is on the page, badge it too, even below the size cutoff
    for (const img of document.querySelectorAll("img")) {
      if (usableUrl(img) === msg.url && !r.error && !r.skipped) {
        seen.set(img, { url: msg.url, state: "done", result: r });
        badgeFor(img, r, true);
      }
    }
    respond({ ok: true });
  }
  if (msg.type === "rescan") {
    for (const [img, b] of badges) b.remove();
    badges.clear();
    for (const img of document.querySelectorAll("img[data-laid-watched]")) {
      delete img.dataset.laidWatched;
      seen.delete(img);
    }
    scan();
    respond({ ok: true });
  }
});
