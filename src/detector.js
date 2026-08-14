/*
 * The detector itself: decode -> two views -> ViT -> calibrated probability.
 *
 * Everything here runs in the offscreen document, which is an ordinary extension page,
 * so it gets WebGPU when the machine has it and WASM everywhere else. Nothing in this
 * file touches the network except the one-time load of the bundled model file off disk.
 *
 * Every image is scored twice, because the two views disagree in useful ways: the
 * downscaled view sees the whole frame and the native crop sees pixels that have not
 * been resampled by anyone. Their mean beats either one on the eval set, and beats it on
 * large images and small images separately — see tools/analyze.py. The preprocessing
 * itself lives in preprocess.js and is deliberately hand-written rather than handed to
 * the canvas, because which resize filter ran is part of what this model is reading.
 */
import * as ort from "../vendor/ort.bundle.min.mjs";
import { viewOfficial, viewNative, toCHW, CROP } from "./preprocess.js";

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

export async function score(bitmap) {
  if (!session) await load();
  const p1 = await runView(viewOfficial(bitmap));
  const p2 = await runView(viewNative(bitmap));
  const mean = (p1 + p2) / 2;
  return { probability: calibrate(mean), raw: mean, views: [p1, p2] };
}
