# Local AI Image Detector

A Chrome extension that flags AI-generated images on the pages you visit. Every image is
decoded and classified inside your browser. No cloud inference, no API, no localhost
helper, nothing uploaded — the extension's only network request is for the image the page
had already loaded.

MIT licensed, including the model weights.

---

## Install

Requires Node 18+ and Chrome 116+.

```sh
git clone https://github.com/agentatwork/local-ai-image-detector
cd local-ai-image-detector
npm install        # onnxruntime-web
npm run build      # vendors the runtime, downloads + verifies the model weights
```

Then open `chrome://extensions`, turn on **Developer mode**, choose **Load unpacked**, and
select the repository directory.

`npm run build` is the only step that touches the network. After it finishes the directory
is self-contained: disconnect entirely, restart Chrome, and the extension still works. It
downloads one file — the model weights — from the pinned Hugging Face revision named in
`tools/model.json`, and checks its SHA-256 before writing it. A build either reproduces the
exact bytes these measurements were taken against or fails.

## Use

Browse. Images at least 128px on a side are scored as they scroll into view, and every one
of them gets a badge in its top-left corner carrying what it scored — `AI 91%` in red at or
above the 65% threshold, `real 88%` in grey below it. Click the toolbar icon to badge only
the flagged ones instead, change the threshold, set the minimum size, or turn scoring off.

The threshold is 0.65, which is above one half, so there is a band where an image is not
flagged and is still more likely generated than not. Badging that `real 37%` would be
arithmetically true and would read as a contradiction, so the band reads `unsure 63% AI`
instead. Whatever the badge says, the number is on the scale the word names, and the
tooltip always gives the raw probability of AI.

Scoring every image out loud is the default on purpose. If only the flagged images are
badged, a real photograph and an image the extension never looked at look exactly the same
to you, and a silent detector is indistinguishable from a broken one — which is a failure
mode this model demonstrably has, on small JPEGs, where it degrades by going quiet rather
than by getting things wrong.

To ask about one specific image — including one below the size cutoff, or with automatic
scoring switched off entirely — right-click it and choose **Check this image for AI
generation**.

---

## How it works

```
content.js     finds images in view        (IntersectionObserver, no page reflow)
     |  url
background.js  routes and caches by URL    (holds no model)
     |
offscreen.js   fetches + decodes           (one model instance for all tabs)
     |  ImageBitmap
preprocess.js  two views, 384x384 each     (own bicubic; the canvas is not asked)
     |  tensor x2
detector.js    ViT -> mean -> calibrated probability
```

Inference is a ViT-Small/16 at 384px —
[Community Forensics](https://huggingface.co/papers/2411.04125) (CVPR 2025), trained on
2.7M images from 4,803 different generators, which is why it holds up on generators it has
never seen. It runs under WebGPU where the browser offers it and WebAssembly everywhere
else, via `onnxruntime-web`.

The model lives in an **offscreen document** rather than the service worker (which Chrome
kills mid-inference and which has no WebGPU) or the content script (which would mean one
copy of the model per tab).

### Three things that matter more than the model

**Calibrate the threshold.** Raw model output is not a probability. This one is nearly
certain about real images and only mildly confident about generated ones, so reading it at
a fixed 65% cutoff turns a good detector into a high-precision, low-recall one and throws
away most of its accuracy. `tools/calibrate.py` fits a two-parameter Platt scaling that
puts the measured decision boundary at 0.65. It is a monotone rescaling: it changes no
ranking and no AUROC, only the number a reader sees.

**Score every image twice, and not with two crops.** The obvious intuition is that resizing
destroys the evidence, since this classifier reads resampling artefacts, so you should crop
at native resolution and never scale. Measured, that intuition is half wrong: the native
crop is *worse* on its own than the model card's downscale, and nearest-neighbour upscaling
of small images — which preserves the pixel grid perfectly — is worse still, because it does
not preserve evidence so much as manufacture it, and it manufactures the same evidence for
real photos as for generated ones (FFHQ-256 goes from 17% false positives to 92%).

What works is two views that disagree in useful ways. The pair that ships is `native` (a
384px centre crop at the image's own resolution, which reads quantisation) and `squash` (the
whole frame bicubic-resized to 384×384, aspect abandoned, which reads composition). An
earlier build shipped `native` alongside the model card's downscale-then-crop, `official`;
`squash` displaced it, and the reason is not the one I expected. I assumed `squash` would
win only on heavily downscaled images, where cropping into an already-small frame sees
almost nothing. It wins on **every** condition, undegraded included — 87.5% against
`official`'s 84.1% on pristine files. The distinguishing property is not robustness, it is
that a squash is the only view that never throws pixels away; both crops discard everything
outside the crop window.

The price of that is aspect distortion, and it shows up where you would expect it: on the 49
images in the sweep with an aspect ratio of 1.40 or worse, `squash`'s margin over `official`
narrows from +3.2 to +1.2 points. Forty-nine images cannot establish a two-point
difference-of-differences, so read that as consistent with the explanation rather than as
evidence for it. Which views are averaged lives in `model/config.json`, and a calibration
fitted for one set is meaningless applied to another, so the two travel together.

**Do the resize yourself.** `ctx.drawImage` into a smaller canvas resamples with whatever
filter the compositor picks, which is neither documented nor stable across Chrome
releases — and the choice of filter is part of what the model is reading. `preprocess.js`
implements Pillow's bicubic explicitly: same support scaling, same normalisation, same
rounding between passes. `tools/compare.py` checks that the shipped JavaScript and the
Python that fitted the threshold agree on real images.

---

## Measurements

The eval set is 1,020 images the model has never seen: 540 generated by 18 different
generators, 480 real from 13 public datasets plus 90 random files pulled off Wikimedia
Commons. Balanced accuracy throughout, read at the shipped 0.65 threshold.

Every figure below comes from `tools/verify.mjs` — the built extension, in a real headless
Chrome, with the browser switched offline. All 1,020 images scored, none errored.

| | balanced accuracy | recall (AI) | specificity (real) |
|---|---|---|---|
| **shipped: two views, calibrated** | **86.2%** | 85.9% | 86.5% |
| leave-one-generator-out | 86.4% | | |
| squash alone, best threshold | 85.7% | | |
| native crop alone, best threshold | 79.7% | | |
| two views, *un*calibrated, read at 0.65 | 67.5% | | |

AUROC is 0.9341, and calibration does not change it — it is a monotone rescaling. What it
changes is where 0.65 lands: on the raw scale the shipped boundary sits at **0.003728**, so
reading the model's own output at a fixed 65% cutoff costs 18.7 points.

Split by size, the two-view average wins in both strata, which is why it is not a
composition artefact:

| | all | min side ≥384px | min side <384px |
|---|---|---|---|
| native alone, best threshold | 79.7% | 90.0% | 66.8% |
| squash alone, best threshold | 85.7% | 89.0% | 81.6% |
| mean of the two, best threshold | **87.2%** | 92.4% | 80.7% |
| **mean of the two, shipped 0.65** | **86.2%** | 90.8% | 80.4% |

The last two rows are the price of shipping a fixed operating point rather than an oracle
one: one point of clean accuracy, paid to have a threshold that was chosen without seeing
this set. The shipped threshold is not fitted here at all — it is fitted by minimax over the
eleven delivery conditions in `tools/minimax.py`, which is why it is not the best row.
`tools/views_clean.py` regenerates this table from the same `verify_all.json`.

Nearest-neighbour upscaling of small images was also tried and is worse than either
(78.3% on a balanced subsample, measured against the previous view pair — it was rejected
before the squash view replaced the downscale, and was not re-run). It recovers exactly the
generators the downscale misses —
GenImage BigGAN 0% → 100%, FLUX.1-dev at 256px 0% → 75% — and simultaneously drives FFHQ-256
from 17% false positives to 92%. It does not preserve evidence; it manufactures it, equally
for both classes.

Where it fails, honestly: leave-one-generator-out, the two worst held-out generators are
GenImage BigGAN (50.2%, recall 6.7%) and a 256px FLUX.1-dev subset (53.3%, recall 10.0%) — a
GAN and a small-resolution diffusion crop, both at chance. Nine of the eighteen generators
score above 95%; the full fold list is printed by `tools/report.py`.

Leave-one-generator-out comes out at 86.4%, slightly *above* the 86.2% headline, and that is
not a held-out gain — each fold refits its threshold on clean images, while the headline is
read at the fixed threshold this extension actually ships, which was fitted by minimax over
delivery conditions and never saw this set. The gap is the price of one fixed boundary.

**Python/JavaScript parity.** `tools/verify.mjs` scores images through the built extension
in a real headless Chrome with the browser switched offline; `tools/compare.py` diffs that
against the Python that fitted the calibration. On 128 images — every 8th of the eval set,
so the stride walks both classes and all 31 sources rather than taking a prefix — the two
implementations agree to a median of **3.8e-07** on the mean of views, worst case 4.1e-03,
and **zero decisions change** at the shipped threshold.

The two views disagree by very different amounts, and the reason is worth stating because it
is a check on the implementation rather than a curiosity: `native` agrees to a median of
**4.9e-09** because it does no resampling at all — it is a pure crop, so both sides are
reading the same pixels and only the normalisation arithmetic can differ. `squash` resamples
the entire frame and lands at **7.0e-07**, worst case 8.2e-03. The residual is Pillow's
fixed-point coefficient rounding against JavaScript's float, and it appears exactly where
resampling happens and nowhere else, which is what it should do if the bicubic was
transcribed correctly and not what it would do if the two were running different algorithms.

The headline numbers above come from a separate full-set run through the same harness, so
they are the extension's own scores rather than the Python's. Parity is checked on the
stride because this box has one core and a full-set Python re-scoring costs hours; the
stride answers the question parity actually asks, which is whether the two implementations
agree image by image.

Speed: about 3.3 s per image, two views, on the single WASM thread of a one-core cloud VM. WebGPU and
real hardware are considerably faster; the extension only scores images that scroll into
view, and caches by URL.

---

## Privacy

- No image data leaves the device. There is no code path in this repository that sends
  image bytes, pixel data, features, hashes or scores anywhere.
- The only requests made are `GET`s for image URLs the page has already fetched, without
  credentials.
- Nothing is written to disk except your settings.
- `<all_urls>` host permission exists so the extension can read cross-origin images. A
  page-context canvas cannot read them back, which is why the fetch happens in the
  extension.

## Reproducing the numbers

Every script that produced a number in this README is in [`tools/`](tools/). The eval images
themselves are not redistributed here — the fetchers rebuild the set from public sources.

```sh
pip install onnxruntime pillow numpy
python3 tools/fetch.py real bitmind/MS-COCO 30   # build an eval set from public datasets
python3 tools/fetch_web.py 90                    # plus real images off the open web
python3 tools/variants.py 12                     # which preprocessing keeps the evidence
python3 tools/dump.py logits.json native squash   # score the set the way the extension does
python3 tools/calibrate.py logits.json           # fit the Platt slope on the clean set
python3 tools/perview.py --views official,native,squash  # every view x every delivery pipeline
python3 tools/minimax.py                         # refit the intercept on the WORST pipeline
python3 tools/shiptable.py                       # the eleven-condition table as it ships
npm run build && node tools/verify.mjs data/ai verify.json --offline
python3 tools/compare.py logits.json verify.json # Python vs shipped JavaScript
node tools/demo.mjs 4 demo.png                   # badges on a real page, screenshotted
```

`tools/verify.mjs` scores through the extension's code but never renders a page.
`tools/demo.mjs` is the other half: it serves a grid of eval images over local HTTP, loads
the unpacked extension, and waits for `content.js` to badge them of its own accord. Nothing
in the page tells it which images are which.

`tools/minimax.py` writes the chosen views and calibration to `minimax.json`; those three
values are then set in [`tools/model.json`](tools/model.json), which is the source
`npm run build` writes `model/config.json` from. Editing `model/config.json` directly works
until the next build and then silently reverts — which is a mistake I made and caught only
because `detector.js` refuses to start when `views` is missing.

`tools/verify.mjs` is the one that counts. It loads the built extension into a real Chrome,
switches the browser offline, and scores images through the extension's own code path — so
the reported figures come from the shipped JavaScript, not from the Python that chose the
model.

## Where it broke, and what fixed it

The 86.2% above is measured on pristine dataset files. Images on the web are not pristine, so
the same build was re-scored over 320 stratified images (18 generators x 10, 14 real sources x
10) through eleven delivery pipelines. Nothing changes but the pipeline: same weights, same two
views, same frozen calibration, same 0.65.

| pipeline | before | **now** | recall (AI) | specificity (real) |
|---|---:|---:|---:|---:|
| nothing | 85.7% | **86.2%** | 86.7% | 85.7% |
| rescale 90% | 86.6% | **88.6%** | 92.2% | 85.0% |
| JPEG q90 | 84.7% | **84.2%** | 85.6% | 82.9% |
| JPEG q75 | 83.1% | **85.8%** | 84.4% | 87.1% |
| resize ≤1600px | 82.1% | **81.7%** | 83.3% | 80.0% |
| resize ≤1024px | 82.1% | **81.7%** | 83.3% | 80.0% |
| resize ≤640px | 81.5% | **85.6%** | 86.1% | 85.0% |
| WebP q80 | 79.7% | **81.2%** | 86.7% | 75.7% |
| JPEG q60 | 79.1% | **81.8%** | 75.0% | 88.6% |
| ≤768px + JPEG q60 | 79.4% | **83.3%** | 76.7% | 90.0% |
| **≤512px + JPEG q40** | **72.3%** | **79.0%** | 67.2% | 90.7% |

*before* is the previous build — views `official+native`, threshold fitted on undegraded
images. *now* is what ships. **Eleven of eleven clear the bounty's 75.0% bar**; the pipeline
that used to fail it at 72.3% now scores 79.0%, and its recall — the number that was actually
broken — goes from 51.1% to 67.2%. That failure mode was the point: degraded hard enough, the
old build stopped calling things generated rather than calling them wrongly, which from
outside looks exactly like a detector working correctly on a set of real photographs.

**This was bought, not found, and the price is on the clean row.** Specificity on undegraded
images falls from 91.4% to 85.7% — roughly one extra false positive per seventeen real
photographs, in the condition an extension meets most often while you browse. The threshold
was chosen to maximise the *worst* pipeline rather than the average one, and that is a
deliberate answer to how the bounty is written: 75.0% is a floor, and the images it will be
judged on have been through a delivery path nobody described to me. If you wanted the best
average instead, you would pick a different intercept and get a quieter extension that fails
harder on small recompressed images. Both are defensible; only one of them is the criterion
here, and I have stated which one this is tuned for rather than presenting it as a free win.

Two numbers keep the win in proportion. AUROC at ≤512px + q40 is 0.841 against 0.927 clean,
so a real part of the loss at that pipeline is signal rather than a misplaced boundary, and
no threshold recovers it. And the honest out-of-sample figure is not 79.0% but **76.7%** —
see the next section, which is about how that was validated and why the earlier version of
this section refused to ship a change of exactly this shape.

Two things the average hides:

- **GenImage's ADM subset falls from 90% to 50% between clean and JPEG q75** — four images out
  of ten, from one generator out of eighteen, and that alone is essentially the whole 2.3-point
  recall loss at that step. It survives WebP q80 (90%) and a 90% rescale (100%) untouched, so
  this is JPEG quantisation specifically, not resampling. On the previous build the same subset
  fell to 10%, so this is one of the places the new view pair actually did work.
- **GenImage's BigGAN subset scores 0% in nine of the eleven conditions.** A 2018 GAN against a
  diffusion-era model: a blind spot, not a degradation. The two exceptions are odd enough to
  name — WebP q80 reaches 10% and a 90% rescale reaches 50%, i.e. this build detects a GAN
  *better* after the image has been resampled than before. On ten images that is one and five
  images respectively and I would not build anything on it, but it is the opposite of the
  direction everything else in this table moves, so it is recorded rather than smoothed away.
  Both averages above include a class this build largely does not detect.

Data, per-image probabilities and the script that recomputes all of it:
[agentatwork/c143-survey](https://github.com/agentatwork/c143-survey) · discussion in
[Twenty-two detectors](https://agentatwork.xyz/notes/twenty-two-detectors.html).

### Three attempts to repair the eleventh; the third one shipped

Because the failure is a threshold sitting in the wrong place — the per-condition oracle
reaches 77.3% on the *same scores* that operate at 72.3% — a calibration that knew how
damaged an image was should be able to recover most of the gap. A browser cannot be told
the condition, so it needs to estimate the damage from the decoded bitmap: no re-fetch of
the original bytes, no DQT marker, no network request added to a privacy extension.

**Blockiness works as that estimator.** JPEG quantises each 8×8 block independently, so it
leaves discontinuities on the block grid that are absent between columns inside a block.
The ratio of mean across-boundary to mean interior luma gradient is ~1.0 for an image that
has never been through a block transform, and rises as quality falls. Measured over the
same 320 images (`blockiness.py`, ~20 lines of numpy, no model):

| pipeline | blockiness |
|---|---:|
| rescale 90% | 1.000 |
| nothing | 1.255 |
| JPEG q75 | 1.258 |
| resize ≤1024px | 1.224 |
| ≤512px + JPEG q40 | 1.600 |

It rises from clean to q40 on **99.1% of images individually**, AUC 0.889 as a
clean-vs-degraded test, and a 90% rescale reads exactly 1.000 because resampling destroys
the grid. So the feature is real, cheap and honest.

**Both calibrations built on it are worse than the constant that ships.**

- Refitting the Platt intercept globally on the three mildest pipelines, generators held
  out: worst condition **72.3% → 69.6%**.
- Letting the intercept move with the feature, `b = b₀ + b₁·(blockiness − 1)`, generators
  *and* conditions held out: the fit chooses `t(d) = −2.66 + 2.50·d`, the wrong sign, and
  ≤512px + q40 goes **72.3% → 65.8%** (recall 32.2%). Leave-one-generator-out, worst
  condition averaged over folds: 64.5%, minimum 45.0%.

The diagnosis is worth more than the attempt. The three fit pipelines span blockiness
1.224–1.258 — a range of 0.034 — while the pipeline being repaired sits at 1.600. **The
slope was never identified in-sample; it was extrapolated 2.5× outside the range it was
fitted on.** Nothing in the fit set could see it, and the fitted value's own sign is
therefore noise.

One confound was suspected and ruled out rather than assumed away: if real photographs
arrive as JPEG and generated images as PNG, blockiness would be a class label in disguise.
It is not — AUC of blockiness against the AI label is 0.496 / 0.509 / 0.474 / 0.479 across
four pipelines (0.5 is chance), and the file extensions are balanced (170 of 180 AI and
140 of 140 real are JPEG).

A third repair was tried, and it produced the most useful negative result of the three.

The two views shipping at the time were both *crops*, so each showed the model a fraction of the frame at
close to native scale — good for reading quantisation, which is exactly what heavy JPEG
destroys. So a fourth view was added: `squash`, the whole frame bicubic-resized to 384×384,
keeping no pixel grid at all and keeping composition instead. It costs no extra download,
because it is the same model asked a different question.

Read on ranking alone it is the best single view by a clear margin, and it is the most
compression-robust:

| view | clean | ≤512px + q40 |
|---|---:|---:|
| `official` (shipped then) | 84.1% | 74.0% |
| `native` (shipped, then and now) | 81.7% | 76.7% |
| `up` | 78.3% | 69.8% |
| **`squash`** (shipped now) | **87.5%** | **79.2%** |

**It is still not shipped, and the reason is a trap worth more than the view.**

Views have to be combined, and there are two obvious ways: average the probabilities, or
average the logits. `detector.js` averages probabilities — `const mean = (p1 + p2) / 2`.
Averaging logits instead is the kind of choice that looks like a matter of taste. It is
not. Scoring every subset of the four views both ways, with one threshold fitted on the
clean condition only:

| combination | worst condition, logit mean | worst condition, probability mean |
|---|---:|---:|
| `official+native+up+squash` | 77.4% | 69.8% |
| `official+squash` | 77.1% | 72.3% |
| `native+squash` | 74.8% | 75.4% |
| `official+native` (shipped then) | 71.9% | 72.7% |

Same scores, same images, same threshold procedure — **a 7.6-point swing on the four-view
combination purely from which space the mean is taken in**, and it reverses the ranking:
the best combination in logit space is the worst in probability space. The check that
proves it is arithmetic rather than a bug is that all four *single*-view numbers are
identical in both tables (73.5%, 70.8%, 73.5%, 68.5%) — with one view there is no average
to take.

In the space the extension actually uses, the best combination reached 75.4% against the
shipped path's 72.3%. At the time I wrote: *+3.1 points, inside the ±2.8-point standard
error, so it is not a result.* **That sentence was wrong, and the third repair is now what
ships.** What follows is why, kept in this order because the reasoning matters more than
the outcome.

**The error bar was the wrong one.** ±2.8 is the standard error of *one* balanced accuracy
estimated once — the right bar for an absolute claim like "this clears 75%". But comparing
two view pairs on the *same images* is a paired design: both see the same photographs and
make correlated errors on them, so what governs the comparison is the standard error of the
difference, which is smaller. Measured with a stratified paired bootstrap (20,000
resamples, AI and real resampled separately to hold the class balance): +3.0 on the worst
pipeline, 95% CI **+0.8 .. +5.4**. The interval excludes zero. Using the unpaired figure as
a floor for a difference is not conservative in a harmless direction — it throws away real
improvements, and it threw away this one. Across all eleven pipelines the candidate wins on
ten and eight of the eleven intervals exclude zero.

**The fit objective was also wrong,** and this mattered more than the error bar. The old
threshold was fitted on undegraded images, which optimises the condition the extension is
least likely to be in. Refitting it to maximise the *worst* of the eleven pipelines — the
minimax choice, since the bounty's 75.0% is a floor — moves the worst pipeline to 79.0%
rather than 75.4%.

**Which raises the obvious objection: a threshold fitted on eleven conditions and then
reported on those same eleven conditions is a memorisation score.** It is, so that is not
the number to trust. The honest one is leave-one-condition-out: fit the threshold on ten
pipelines, score the eleventh, rotate. Held out that way the worst condition is **76.7%**
and all eleven still clear 75.0%. The previous configuration under the same procedure gets
71.5% and clears nine of eleven. 76.7% against a 75.0% bar is a thin margin and I am not
going to dress it up as more; it is, however, the number that answers the question, and it
is the one this build is chosen on.

**And the trap above still stands.** Nothing here rehabilitates averaging in logit space —
`detector.js` averages probabilities, and every number in this section is measured in that
space, which is the only space the claim is about. The 7.6-point swing was real and the
lesson survives the reversal of the conclusion.

Two things I want stated plainly, because each is a way this could be misread. **I changed
the test after seeing the data.** That is the move that manufactures results, and naming it
does not neutralise it; what makes this a fix rather than a rationalisation is that the
paired test and the minimax objective are both more appropriate *a priori*, for reasons
that have nothing to do with which answer they give. And **"the improvement is real" is not
"the result clears the bar."** Those are two different questions taking two different error
bars: the gain is established by the paired interval, while 76.7% against 75.0% is an
absolute claim carrying the full unpaired ±2.8. The gain is established. The clearance is
not — it is the best estimate I have, on a benchmark I have never seen.

The full argument, in a form someone else can run against their own detector rather than
mine, is in [PROTOCOL.md](https://github.com/agentatwork/c143-survey/blob/main/PROTOCOL.md).

## Limits

Worth saying plainly:

- A confident score is not proof. Heavily edited photographs, AI-upscaled real photos and
  screenshots of generated images all sit in genuine grey area.
- Small images carry less evidence. Below about 200px the score should be read as a hint.
- Every detector degrades on generators released after its training data. This one
  degrades more slowly than most, which is the whole reason it was chosen, but it still
  degrades.

## Licence

MIT — see [LICENSE](LICENSE). The model weights are MIT from
`buildborderless/CommunityForensics-DeepfakeDet-ViT`. `onnxruntime-web` is MIT.
