/*
 * Getting an image into the model, twice, without letting the browser decide how.
 *
 * The detector reads resampling and quantisation artefacts, so the resize filter is not
 * an implementation detail — it is part of the measurement. `ctx.drawImage` into a
 * smaller canvas would use whatever Chrome's compositor feels like today, which is not
 * the filter the thresholds were fitted against and is not even stable across releases.
 * So the downscale is done here, in explicit code, matching Pillow's bicubic exactly:
 * same support scaling, same coefficient normalisation, same rounding between passes.
 * `tools/compare.py` checks that claim against the Python on real images.
 *
 * Two views come out of this file:
 *
 *   official   shortest edge to 440, bicubic, centre crop 384   — sees the whole frame
 *   native     centre crop 384 at the image's own resolution    — sees the actual pixels
 *
 * Averaging them is worth 2.4 points of balanced accuracy over either alone, and neither
 * ordering nor threshold survives dropping one; see tools/analyze.py.
 */

const CROP = 384;
const RESIZE_SHORT = 440;

/* ---------- Pillow-compatible bicubic ---------- */

// Pillow's BICUBIC: Keys' cubic with a = -0.5, support 2.
function cubic(x) {
  x = Math.abs(x);
  const a = -0.5;
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

// Python's round(): half to even. Only differs from Math.round on exact .5, which is
// exactly the case that would silently shift a crop by one pixel.
function pyRound(v) {
  const f = Math.floor(v);
  const d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/**
 * Coefficients for one axis, for output indices [start, start+count).
 * Mirrors Pillow's precompute_coeffs: the filter widens when downscaling, so a 10x
 * reduction averages over 40 input pixels rather than sampling 4 of them.
 */
function coeffs(inSize, outSize, start, count) {
  const scale = inSize / outSize;
  const fscale = Math.max(1, scale);
  const support = 2.0 * fscale;
  const out = [];
  for (let i = 0; i < count; i++) {
    const center = (start + i + 0.5) * scale;
    const xmin = Math.max(0, Math.trunc(center - support + 0.5));
    const xmax = Math.min(inSize, Math.trunc(center + support + 0.5));
    const k = new Float32Array(Math.max(0, xmax - xmin));
    let sum = 0;
    for (let x = xmin; x < xmax; x++) {
      const w = cubic((x - center + 0.5) / fscale);
      k[x - xmin] = w;
      sum += w;
    }
    if (sum !== 0) for (let j = 0; j < k.length; j++) k[j] /= sum;
    out.push({ min: xmin, k });
  }
  return out;
}

const clamp8 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);

/**
 * Resize a source region to the output window [cropX, cropX+CROP) x [cropY, cropY+CROP)
 * of the full (outW x outH) resize, without materialising the rest of it.
 *
 * `readRows(y0, y1)` must return the source pixels for input rows [y0, y1) as RGBA.
 * Only the rows the vertical pass actually touches are ever requested, which is what
 * keeps a 24-megapixel photo from being decoded into half a gigabyte of ImageData.
 */
function resizeWindow(inW, inH, outW, outH, cropX, cropY, readRows) {
  const vert = coeffs(inH, outH, cropY, CROP);
  const horz = coeffs(inW, outW, cropX, CROP);

  let rowMin = inH, rowMax = 0;
  for (const c of vert) {
    if (c.min < rowMin) rowMin = c.min;
    if (c.min + c.k.length > rowMax) rowMax = c.min + c.k.length;
  }
  const src = readRows(rowMin, rowMax);          // Uint8ClampedArray, RGBA, (rowMax-rowMin) rows

  // horizontal pass: full source row range, CROP columns wide, rounded to 8 bit the way
  // Pillow rounds between its two passes
  const nRows = rowMax - rowMin;
  const tmp = new Uint8ClampedArray(nRows * CROP * 3);
  for (let y = 0; y < nRows; y++) {
    const srow = y * inW * 4;
    const trow = y * CROP * 3;
    for (let i = 0; i < CROP; i++) {
      const { min, k } = horz[i];
      let r = 0, g = 0, b = 0;
      for (let j = 0; j < k.length; j++) {
        const p = srow + (min + j) * 4;
        const w = k[j];
        r += w * src[p]; g += w * src[p + 1]; b += w * src[p + 2];
      }
      const o = trow + i * 3;
      tmp[o] = clamp8(Math.round(r));
      tmp[o + 1] = clamp8(Math.round(g));
      tmp[o + 2] = clamp8(Math.round(b));
    }
  }

  // vertical pass straight into the output crop
  const out = new Uint8ClampedArray(CROP * CROP * 3);
  for (let i = 0; i < CROP; i++) {
    const { min, k } = vert[i];
    for (let x = 0; x < CROP; x++) {
      let r = 0, g = 0, b = 0;
      for (let j = 0; j < k.length; j++) {
        const p = ((min + j - rowMin) * CROP + x) * 3;
        const w = k[j];
        r += w * tmp[p]; g += w * tmp[p + 1]; b += w * tmp[p + 2];
      }
      const o = (i * CROP + x) * 3;
      out[o] = clamp8(Math.round(r));
      out[o + 1] = clamp8(Math.round(g));
      out[o + 2] = clamp8(Math.round(b));
    }
  }
  return out;
}

/* ---------- the two views ---------- */

function ctxFor(w, h) {
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = false;
  return ctx;
}

/** shortest edge -> 440 bicubic, centre crop 384. RGB, no alpha. */
export function viewOfficial(bmp) {
  const w = bmp.width, h = bmp.height;
  const sc = RESIZE_SHORT / Math.min(w, h);
  const outW = Math.max(RESIZE_SHORT, pyRound(w * sc));
  const outH = Math.max(RESIZE_SHORT, pyRound(h * sc));
  const cropX = (outW - CROP) >> 1, cropY = (outH - CROP) >> 1;

  const readRows = (y0, y1) => {
    // 1:1 draw with smoothing off is a straight copy of the decoded pixels
    const ctx = ctxFor(w, y1 - y0);
    ctx.drawImage(bmp, 0, y0, w, y1 - y0, 0, 0, w, y1 - y0);
    return ctx.getImageData(0, 0, w, y1 - y0).data;
  };
  return resizeWindow(w, h, outW, outH, cropX, cropY, readRows);
}

// numpy's 'reflect': mirror without repeating the edge pixel, folded as many times as
// the padding needs. Pillow-free, so it has to be spelled out.
function reflectIndex(i, n) {
  if (n === 1) return 0;
  const period = 2 * (n - 1);
  let r = i % period;
  if (r < 0) r += period;
  return r >= n ? period - r : r;
}

/** centre crop 384 at native resolution; anything smaller is reflected out to size. */
export function viewNative(bmp) {
  const w = bmp.width, h = bmp.height;
  const out = new Uint8ClampedArray(CROP * CROP * 3);

  if (w >= CROP && h >= CROP) {
    const ctx = ctxFor(CROP, CROP);
    ctx.drawImage(bmp, (w - CROP) >> 1, (h - CROP) >> 1, CROP, CROP, 0, 0, CROP, CROP);
    const d = ctx.getImageData(0, 0, CROP, CROP).data;
    for (let i = 0, n = CROP * CROP; i < n; i++) {
      out[i * 3] = d[i * 4]; out[i * 3 + 1] = d[i * 4 + 1]; out[i * 3 + 2] = d[i * 4 + 2];
    }
    return out;
  }

  // pad bottom and right, then centre-crop the padded image — the same order as
  // np.pad(..., ((0, ph), (0, pw), (0, 0)), "reflect") followed by a centre crop
  const ctx = ctxFor(w, h);
  ctx.drawImage(bmp, 0, 0);
  const d = ctx.getImageData(0, 0, w, h).data;
  const padW = Math.max(w, CROP), padH = Math.max(h, CROP);
  const l = (padW - CROP) >> 1, t = (padH - CROP) >> 1;
  for (let y = 0; y < CROP; y++) {
    const sy = reflectIndex(t + y, h);
    for (let x = 0; x < CROP; x++) {
      const sx = reflectIndex(l + x, w);
      const p = (sy * w + sx) * 4, o = (y * CROP + x) * 3;
      out[o] = d[p]; out[o + 1] = d[p + 1]; out[o + 2] = d[p + 2];
    }
  }
  return out;
}

/** RGB bytes -> normalised CHW float32, the layout the ViT wants. */
export function toCHW(rgb, mean, std) {
  const n = CROP * CROP;
  const f = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    f[i] = (rgb[i * 3] / 255 - mean[0]) / std[0];
    f[n + i] = (rgb[i * 3 + 1] / 255 - mean[1]) / std[1];
    f[2 * n + i] = (rgb[i * 3 + 2] / 255 - mean[2]) / std[2];
  }
  return f;
}

export { CROP };
