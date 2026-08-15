/*
 * The detector itself: decode -> N views -> ViT -> calibrated probability.
 *
 * Everything here runs in the offscreen document, which is an ordinary extension page,
 * so it gets WebGPU when the machine has it and WASM everywhere else. Nothing in this
 * file touches the network except the one-time load of the bundled model file off disk.
 *
 * Every image is scored once per view and the probabilities are averaged, because the
 * views disagree in useful ways: a crop at native scale reads quantisation, a squash of
 * the whole frame reads composition, and the conditions that destroy one leave the other
 * intact. Which views to average is a measured choice and lives in model/config.json, so
 * this file iterates whatever it is given rather than naming two. The preprocessing
 * itself lives in preprocess.js and is deliberately hand-written rather than handed to
 * the canvas, because which resize filter ran is part of what this model is reading.
 */
import * as ort from "../vendor/ort.bundle.min.mjs";
import { viewOfficial, viewNative, viewSquash, toCHW, CROP } from "./preprocess.js";

ort.env.wasm.wasmPaths = chrome.runtime.getURL("vendor/");
ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
ort.env.wasm.proxy = false;
ort.env.logLevel = "error";

let session = null;
let cfg = null;
let backend = "unknown";
let loading = null;

export function status() {
  return { ready: !!session, backend, model: cfg?.model_id ?? null };
}

export function config() {
  return cfg;
}

export function load() {
  if (loading) return loading;
  loading = (async () => {
    cfg = await (await fetch(chrome.runtime.getURL("model/config.json"))).json();
    if (cfg.crop_size !== CROP) {
      // preprocess.js is written around one crop size; a config that disagrees would
      // silently feed the model a tensor of the wrong shape or the wrong content.
      throw new Error(`model/config.json wants crop ${cfg.crop_size}, preprocess.js is ${CROP}`);
    }
    if (!Array.isArray(cfg.views) || !cfg.views.every((n) => VIEWS[n])) {
      // A config naming a view this build does not have would otherwise average a
      // shorter list and silently score every image against the wrong calibration.
      throw new Error(`model/config.json names unknown views: ${JSON.stringify(cfg.views)}`);
    }
    const url = chrome.runtime.getURL("model/" + cfg.weights);
    const buf = new Uint8Array(await (await fetch(url)).arrayBuffer());
    // WebGPU when the browser has it, WASM otherwise. Both are local; the only
    // difference is speed, so a failure to get WebGPU is not worth surfacing.
    const tries = [];
    if (navigator.gpu) tries.push("webgpu");
    tries.push("wasm");
    let lastErr;
    for (const ep of tries) {
      try {
        session = await ort.InferenceSession.create(buf, {
          executionProviders: [ep],
          graphOptimizationLevel: "all",
        });
        backend = ep;
        return;
      } catch (e) { lastErr = e; }
    }
    throw lastErr;
  })();
  return loading;
}

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

async function runView(rgb) {
  const f = toCHW(rgb, cfg.image_mean, cfg.image_std);
  const t = new ort.Tensor("float32", f, [1, 3, CROP, CROP]);
  const out = await session.run({ [session.inputNames[0]]: t });
  return sigmoid(out[session.outputNames[0]].data[0]);
}

/**
 * Map the model's own scale onto a probability whose 0.65 point is the decision boundary
 * that was actually measured, rather than the one the training loss happened to leave
 * behind. The raw model is nearly certain about real images and only mildly confident
 * about generated ones, so reading it at a fixed cutoff throws away most of its accuracy.
 * Two-parameter Platt scaling, fitted with whole generators held out; it is monotone, so
 * it changes no ranking and no AUROC, only the number a reader sees. See tools/calibrate.py.
 */
function calibrate(p) {
  const { a, b } = cfg.calibration;
  const q = Math.min(Math.max(p, 1e-12), 1 - 1e-12);
  return sigmoid(a * Math.log(q / (1 - q)) + b);
}

// Which views to average is a measured choice, not a structural one, so it lives in
// model/config.json next to the calibration that was fitted for it. The two travel
// together: a calibration fitted for one view set is meaningless applied to another.
const VIEWS = { official: viewOfficial, native: viewNative, squash: viewSquash };

export async function score(bitmap) {
  if (!session) await load();
  const names = cfg.views;
  const ps = [];
  for (const n of names) ps.push(await runView(VIEWS[n](bitmap)));
  const mean = ps.reduce((s, p) => s + p, 0) / ps.length;
  return { probability: calibrate(mean), raw: mean, views: ps, viewNames: names };
}
