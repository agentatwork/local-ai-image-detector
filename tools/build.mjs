#!/usr/bin/env node
/*
 * Assemble the loadable extension: vendor the ONNX runtime, fetch the model weights,
 * and write the model config the detector reads at startup.
 *
 *   npm install && npm run build
 *
 * This is the only step that touches the network, and it is the "one-time download of
 * publicly available model weights" the extension is allowed. Once it has run, the
 * directory is self-contained: load it unpacked with the network off and it works.
 *
 * The weights are pinned by revision AND verified by SHA-256, so a build either
 * reproduces the exact bytes this was measured against or fails loudly.
 */
import { createHash } from "node:crypto";
import { mkdir, copyFile, writeFile, readFile, readdir, stat } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const MODEL = JSON.parse(await readFile(join(ROOT, "tools", "model.json"), "utf8"));

const ORT_FILES = [
  "ort.bundle.min.mjs",
  "ort-wasm-simd-threaded.jsep.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
];

async function vendorOrt() {
  const src = join(ROOT, "node_modules", "onnxruntime-web", "dist");
  if (!existsSync(src)) throw new Error("run `npm install` first — onnxruntime-web is missing");
  const dst = join(ROOT, "vendor");
  await mkdir(dst, { recursive: true });
  const have = await readdir(src);
  let n = 0;
  for (const f of ORT_FILES) {
    if (!have.includes(f)) { console.warn("  ! not in this onnxruntime-web build:", f); continue; }
    await copyFile(join(src, f), join(dst, f));
    n++;
  }
  console.log(`  vendored ${n} onnxruntime-web files -> vendor/`);
}

async function fetchWeights() {
  const dst = join(ROOT, "model");
  await mkdir(dst, { recursive: true });
  const out = join(dst, MODEL.weights);

  if (existsSync(out) && (await sha256(out)) === MODEL.sha256) {
    console.log("  weights already present and verified");
    return;
  }
  const url = `https://huggingface.co/${MODEL.repo}/resolve/${MODEL.revision}/${MODEL.path}`;
  console.log("  downloading", url);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`weights download failed: HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  await writeFile(out, buf);
  const got = createHash("sha256").update(buf).digest("hex");
  if (got !== MODEL.sha256) {
    throw new Error(`weights hash mismatch\n  expected ${MODEL.sha256}\n  got      ${got}`);
  }
  console.log(`  ${MODEL.weights}: ${(buf.length / 1e6).toFixed(1)} MB, sha256 verified`);
}

async function sha256(path) {
  return createHash("sha256").update(await readFile(path)).digest("hex");
}

async function writeConfig() {
  const cfg = {
    model_id: MODEL.repo + "@" + MODEL.revision.slice(0, 12),
    weights: MODEL.weights,
    crop_size: MODEL.crop_size,
    image_mean: MODEL.image_mean,
    image_std: MODEL.image_std,
    calibration: MODEL.calibration,
    measured: MODEL.measured,
  };
  await writeFile(join(ROOT, "model", "config.json"), JSON.stringify(cfg, null, 2) + "\n");
  console.log("  wrote model/config.json");
}

console.log("building Local AI Image Detector");
await vendorOrt();
await fetchWeights();
await writeConfig();
const size = (await stat(join(ROOT, "model", MODEL.weights))).size;
console.log(`done — load ${ROOT} unpacked at chrome://extensions (model ${(size / 1e6).toFixed(1)} MB)`);
