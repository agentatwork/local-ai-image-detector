const $ = (id) => document.getElementById(id);

function pct(x) { return Math.round(x * 100) + "%"; }

async function init() {
  const s = await chrome.runtime.sendMessage({ type: "settings" });
  $("enabled").checked = s.enabled;
  $("showAll").checked = s.showAll;
  $("threshold").value = Math.round(s.threshold * 100);
  $("thrLabel").textContent = pct(s.threshold);
  $("minSize").value = s.minSize;
  $("minLabel").textContent = s.minSize;

  chrome.runtime.sendMessage({ type: "status" }).then((st) => {
    $("engine").textContent = st?.ready
      ? `running on ${st.backend === "webgpu" ? "WebGPU" : "WebAssembly"} · ${st.cached ?? 0} images cached`
      : st?.error ? "model not loaded: " + st.error : "loading model…";
  });

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab?.id) {
    chrome.tabs.sendMessage(tab.id, { type: "page-report" }).then((r) => {
      if (!r) return;
      const n = r.images.length;
      const flagged = r.images.filter((i) => i.probability >= r.threshold).length;
      $("pageSummary").textContent = n === 0
        ? "No images checked yet."
        : `${flagged} of ${n} checked image${n === 1 ? "" : "s"} flagged as AI-generated.`;
    }).catch(() => { $("pageSummary").textContent = "Not available on this page."; });
  }
}

function save(values) { chrome.runtime.sendMessage({ type: "set-settings", values }); }

$("enabled").addEventListener("change", (e) => save({ enabled: e.target.checked }));
$("showAll").addEventListener("change", (e) => save({ showAll: e.target.checked }));
$("threshold").addEventListener("input", (e) => {
  $("thrLabel").textContent = e.target.value + "%";
  save({ threshold: Number(e.target.value) / 100 });
});
$("minSize").addEventListener("input", (e) => {
  $("minLabel").textContent = e.target.value;
  save({ minSize: Number(e.target.value) });
});

init();
