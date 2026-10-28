# Real handwritten kana bank (`manga_ocr_dev/real_kana/`)

Optional pipeline that replaces the kana in a rendered line with **real
handwritten kana** cut from AnimeText-OCR crops. Everything else — kanji,
Latin letters, digits and punctuation — still comes from the selected
handwriting font, so the ground truth and the drawn pixels stay in agreement.

The bank is a local directory (default `assets/handwriting/`, override with
`--root` or `REAL_KANA_ROOT`). It is empty by default; when it is empty the
renderer silently falls back to font rendering.

```
AnimeText-OCR ──► harvest ──► segment ──► clean ──► upscale ──► verify ──► upright ──► build ──► renderer
   (stream)       candidates   charspot     RGBA       ×2        (Gemini)   rotate      manifest   kana_bank=
                                boxes +                            optional
                                labels
```

Detection and recognition are local (`manga-ocr-charspot-small`). The vision
model is only a **final verifier** of the already-cleaned glyphs, and it is
optional.

## Install

The core package has no extra dependency. The local detector and the optional
vision/scoring stages need a little more:

```bash
# character detection + recognition (recommended; the checkout in this repo
# already contains the model files)
pip install -e 'manga-ocr-charspot-small[cpu]'   # adds onnxruntime on CPU
# or clone muscgab/manga-ocr-charspot-small and pass --local-model /path

# vision verifier (pick one or both)
pip install google-genai        # Gemini / Vertex AI (default)
pip install openai              # any OpenAI-compatible endpoint

# Mercury handwriting classifier (optional; already in [kana])
pip install requests
export OPENROUTER_API_KEY=...

# trained handwriting head (optional; heuristic fallback works without it)
pip install torch torchvision transformers scikit-learn

# or use the extras
pip install -e '.[kana]'             # google-genai + openai + requests
pip install -e '.[kana,scoring]'     # + torch/torchvision/transformers/scikit-learn
```

The local model is found automatically in `manga-ocr-charspot-small/` next to
the project root, from an installed package, or via
`MANGA_OCR_CHARSPOT_SMALL_DIR` / `--local-model`. CPU ONNX is the default;
`--local-backend torch` uses CUDA when available.

Real-ESRGAN is **not** installed by an extra because its `basicsr` dependency
is fragile on modern Python. Install `realesrgan`/`basicsr` manually and pass
`--realesrgan-model /path/to/RealESRGAN_x4plus.pth`; without it the pipeline
uses Lanczos.

## Credentials

The verifier defaults to `gemini-3.5-flash-lite` (override with
`REAL_KANA_VISION_MODEL` or `--model`). Provide exactly one of:

```bash
export GEMINI_API_KEY=...          # API key (also accepts GOOGLE_API_KEY)
# or Vertex AI:
export GOOGLE_CLOUD_PROJECT=my-project
export GOOGLE_CLOUD_LOCATION=us-central1
```

For a local/OpenAI-compatible verifier:

```bash
export OPENAI_API_KEY=...          # may be any non-empty value for vLLM
python -m manga_ocr_dev.real_kana verify \
  --provider local --model Qwen/Qwen2.5-VL-7B-Instruct \
  --base-url http://localhost:8000/v1
```

Verification is the only paid stage. By default it checks only doubtful crops:
low local-model confidence, clean/upscale flags, or crops already marked
`watch`. Pass `--all` to audit everything. Vision responses are cached by
`(model, prompt version, image hashes)` under `<root>/vision_cache/`, so
retrying a stage does not re-bill completed crops. Use `--no-cache` to bypass
it.

Mercury uses a separate OpenRouter key:

```bash
export OPENROUTER_API_KEY=...
# or pass --openrouter-api-key before the subcommand
```

Mercury is free-tier but rate-limited; see below. Its responses are cached
under `<root>/vision_cache/mercury/` keyed by the prepared PNG, model, question
version and endpoint. `--no-cache` bypasses that cache too.

## Commands

```bash
# 1. Harvest candidates (streams the pinned AnimeText-OCR revision)
python -m manga_ocr_dev.real_kana harvest --limit 5000 --max-scan 250000

# ... or classify handwriting with Mercury instead of the local heuristic
python -m manga_ocr_dev.real_kana harvest \
  --handwriting-backend mercury --limit 1000

# 2. Local detection + recognition (no network, no Gemini calls)
python -m manga_ocr_dev.real_kana segment --limit 500

# 3. Local pixel stages
python -m manga_ocr_dev.real_kana clean
python -m manga_ocr_dev.real_kana upscale --scale 2

# 4. Optional final Gemini verification over the upscaled glyphs
python -m manga_ocr_dev.real_kana verify --limit 100
python -m manga_ocr_dev.real_kana verify --all      # audit every glyph

# 5. Rotate and build
python -m manga_ocr_dev.real_kana upright
python -m manga_ocr_dev.real_kana build
python -m manga_ocr_dev.real_kana audit

# Inspect
python -m manga_ocr_dev.real_kana stats
python -m manga_ocr_dev.real_kana sheets --per-sheet 64

# Render a sample with the bank
python -m manga_ocr_dev.real_kana render-demo --text おはよう --output demo.png
```

`run` chains harvest → segment → clean → upscale → verify → upright → build
for a small trial; pass `--handwriting-backend mercury` to use Mercury for the
harvest filter, and `--no-verify` to skip the paid verifier. Every stage
is resumable (`meta.json`, `verify_<stage>.json`); pass `--no-resume` to redo.
Stage limits are for smoke tests, not production size.

### Harvest filters

`harvest` applies the article's metadata filters before decoding pixels:

- Japanese (`language == "ja"`) and 3–40 characters with at least 2 kana;
- `seq_confidence` and `min_token_confidence` ≥ 0.90;
- shortest image side ≥ 18 px;
- likely SFX (short all-katakana or repeated syllables) excluded;
- handwriting score ≥ 0.95, after de-duplication by crop id and image hash.
  With `--handwriting-backend mercury` the default cut is 0.5 instead.

The dataset revision is pinned in `config.py`
(`9c70c6b5609c4378d46ac0df4927c4119ebfd83d`) and stored in every record.

### Detection, recognition and verification

`segment` runs `manga-ocr-charspot-small` on each candidate crop. The model
returns every character in reading order with a tight box and a confidence
score. Boxes are already in source pixels, so no coordinate mapping is needed.
The pipeline keeps only kana, groups them into lines/columns, and uses the
model's scores with a deliberately low threshold (`local_min_confidence`,
default `0.20`) because genuine handwritten kana can score around `0.3`.

`verify` is a separate, optional pass over `upscaled/` records. It builds a
numbered contact sheet of the glyphs and asks the vision model to correct
labels, remove neighbour fragments/background, and confirm that each crop is a
single readable kana. Verification never changes the stage status: records
remain `status="upscaled"` so `upright` still sees them. By default only
doubtful glyphs are sent; `--all` audits everything.

Gemini is never used to detect or recognise characters. The old
`review`/`segment --review` paths remain only as compatibility aliases for the
raw stage.

Verification must run before `upright`/`build`. Re-running `upright` removes
stale final glyphs left by an earlier build (for example a glyph the verifier
later deleted or relabelled), so a resumed bank does not keep orphans.

### Handwriting scoring

`HandwritingScorer` uses a frozen DINOv2 CLS embedding plus a logistic head
(`train-head`) when one is supplied. It only decides **handwritten vs printed**
during harvest; it is not the kana detector/recognizer.

A head needs both classes, but normal harvest keeps only crops above the
handwriting threshold. Build a mixed pool first:

```bash
# keep every metadata-passing candidate, including printed ones
python -m manga_ocr_dev.real_kana --root assets/handwriting harvest \
  --limit 2000 --max-scan 250000 --score-threshold 0

# 1. write a starting CSV (heuristic guesses, then fix labels by hand)
python -m manga_ocr_dev.real_kana --root assets/handwriting \
  heuristic-labels --output assets/handwriting/labels.csv

# or weak-label a small sample with the vision model (one call per crop)
python -m manga_ocr_dev.real_kana --root assets/handwriting \
  label-handwriting --output assets/handwriting/labels.csv --limit 300

# or weak-label with Mercury (one cached OpenRouter request per crop)
python -m manga_ocr_dev.real_kana --root assets/handwriting \
  label-handwriting --backend mercury \
  --output assets/handwriting/labels.csv --limit 1000

# 2. correct the CSV (`path,label`; 1 = handwritten, 0 = printed), then train
python -m manga_ocr_dev.real_kana train-head \
  --labels assets/handwriting/labels.csv \
  --output assets/handwriting/handwriting_head.npz \
  --threshold 0.95

# 3. harvest with the head (its stored threshold is used automatically)
python -m manga_ocr_dev.real_kana --root assets/handwriting harvest \
  --head assets/handwriting/handwriting_head.npz --limit 5000
```

`train-head` requires at least 20 labelled crops with both classes; use several
hundred to a few thousand balanced examples for a useful head. It reports a
5-fold cross-validated AUC, stores the chosen threshold in the `.npz`, and
caches DINO embeddings next to it (`handwriting_head.embeddings.npz`). Delete
that cache to force re-embedding after changing the labelled files.

Without a head (or without torch/transformers) a deterministic heuristic is
used. It is deliberately conservative and must not be treated as the article's
classifier.

### Mercury handwriting classifier

`inception/mercury-decide:free` is a non-generative decision model: it answers
one typed `noul` question (“is this text handwritten?”) with a probability.
It is used only for the handwritten-vs-printed decision, never for detection,
recognition or glyph verification.

```bash
# label a mixed pool (resumable; the CSV label is preserved on rerun)
python -m manga_ocr_dev.real_kana --root assets/handwriting \
  label-handwriting --backend mercury \
  --output assets/handwriting/labels.csv --limit 1000

# or skip the CSV and filter during harvest
python -m manga_ocr_dev.real_kana --root assets/handwriting harvest \
  --handwriting-backend mercury --score-threshold 0.5 --limit 2000
```

The default decision threshold is `0.5` for both `label-handwriting --backend
mercury` and `harvest --handwriting-backend mercury` (override with
`--threshold` and `--score-threshold` respectively). Mercury is not confident
on tiny or low-contrast crops: it works on a compact PNG (small crops are
enlarged to 96 px, large ones downscaled to `--mercury-max-side`, default 768).

Useful options (global, before the subcommand, or after `harvest` /
`label-handwriting`):

- `--openrouter-api-key` — defaults to `$OPENROUTER_API_KEY`;
- `--mercury-model` — defaults to `inception/mercury-decide:free`;
- `--mercury-url` — defaults to `https://openrouter.ai/api/alpha/decisions`;
- `--mercury-timeout` — seconds per request (default 120);
- `--mercury-retries` — retries for 429/5xx/network errors (default 4);
- `--mercury-max-side` — downscale bound (default 768).

Responses are cached as `<root>/vision_cache/mercury/<sha256>.json`, so an
interrupted labelling or harvest run does not spend the same request twice.
The free tier is rate-limited; use `--limit` for a first pass and let resume
finish the rest rather than raising concurrency. Cached files store the raw
`noul`, so changing only `--threshold` does not trigger new requests.

### Mercury-distilled CNN filter

Mercury is accurate but too slow to label hundreds of thousands of crops.
Label a few thousand with Mercury, train the small MobileNetV3 filter, and use
it on the rest:

```bash
# 1. build a mixed pool (metadata filters only)
python -m manga_ocr_dev.real_kana --root /mnt/real-kana-pool harvest \
  --limit 400000 --max-scan 3000000 --score-threshold 0

# 2. label a random sample with Mercury (path,label CSV)
python -m manga_ocr_dev.real_kana --root /mnt/real-kana-pool \
  label-handwriting --backend mercury --output /mnt/real-kana-pool/models/weak.csv \
  --limit 2500

# 3. train and validate by image_id (GroupShuffleSplit)
python -m manga_ocr_dev.real_kana train-cnn \
  --labels /mnt/real-kana-pool/models/weak.csv \
  --output /mnt/real-kana-pool/models/handwriting_cnn.pt

# 4. score the pool and hard-link the top crops into a clean bank
python -m manga_ocr_dev.real_kana --root /mnt/real-kana-pool \
  filter-cnn --checkpoint /mnt/real-kana-pool/models/handwriting_cnn.pt \
  --output-root /mnt/real-kana-bank --limit 100000 --threshold 0.0 \
  --mercury-uncertain --mercury-band 0.1 --mercury-max-calls 3000

# 5. continue with the normal segment/clean/upscale/verify/upright/build chain
python -m manga_ocr_dev.real_kana --root /mnt/real-kana-bank segment
```

`filter-cnn` writes per-crop scores to `<pool>/candidates/cnn_scores.jsonl`,
reports the validation AUC/precision/recall, and records
`selection_method="mercury_distilled_cnn"` in the new bank. The `image_id`
column is used for the grouped split; without it, the crop filename prefix is
used. The CNN is only a handwriting/printed filter — detection and recognition
still come from `manga-ocr-charspot-small`.

Scores are appended to `<pool>/candidates/cnn_score_cache.jsonl`, so a re-run
(or a run interrupted by a network drop) only scores crops added since the last
pass. `--mercury-uncertain` spends a bounded number of Mercury calls on the
band around the selection cutoff, where the CNN is least sure; those crops are
re-ranked by Mercury `noul`. The band defaults to `cutoff ± --mercury-band`
(`--mercury-low/--mercury-high` override it explicitly) and only the
`--mercury-max-calls` crops closest to the cutoff are sent. With
`--threshold 0.0` the selection is simply the top `--limit` crops by score.

Re-running `filter-cnn` into an existing bank keeps the hard links it already
created and, with `--prune-stale`, deletes `raw/`, `cleaned/` and `upscaled/`
directories for crops that dropped out of the new selection, so stale glyphs
cannot leak into the manifest.

Segmentation can be split across processes safely; each shard takes every
`n`-th candidate and writes into the same bank:

```bash
for i in 0 1 2 3; do
  python -m manga_ocr_dev.real_kana --root /mnt/real-kana-bank segment \
    --shard "$i" --shards 4 &
done
wait
```

`--limit` is per shard, and the existing `meta.json` resume marker still skips
crops another shard already finished.

## Bank layout

```
assets/handwriting/
  candidates/candidates.jsonl        # harvested metadata + score
  candidates/images/<crop_id>.png
  raw/<crop_id>/source.png           # original candidate
                  overlay.png        # charspot boxes drawn on it
                  crops.png          # numbered contact sheet
                  records.json       # one row per kana box
                  <glyph_id>.png
  cleaned/<crop_id>/<glyph_id>.png   # RGBA, colour-separated alpha
  upscaled/<crop_id>/<glyph_id>.png
  upscaled/<crop_id>/verify_upscaled.json       # verifier report (if run)
  upscaled/<crop_id>/verify_upscaled_sheet.png  # what the verifier saw
  glyphs/<kana>/<glyph_id>.png       # final upright glyphs
  glyphs/<kana>/<glyph_id>.json      # sidecar: stroke width, angle, hash…
  quarantine/                        # rejected crops, kept for inspection
  review/                            # per-label human review sheets
  vision_cache/                      # cached verifier responses
  manifest.jsonl                     # one row per final glyph
  manifest.meta.json                 # config + counts
  audit.json
```

The manifest is the only file the renderer needs. `KanaBank.load()` rebuilds
it from `glyphs/` if it is missing.

## Glyph matching rules

`KanaBank.pick(label, target_stroke, target_height)` enforces the article's
constraints:

- **Stroke width**: the glyph's stroke, scaled to the requested height, must be
  within ×1.35 of the line's target stroke.
- **Per-image consistency**: all real glyphs used in one rendered image must
  stay within ×1.65 of each other (`StrokeBudget`).
- **No over-enlargement**: a glyph is never scaled beyond ×4 of its native
  resolution; because `upscale` is recorded, the cap is applied as
  `4 / upscale` for upscaled glyphs.
- **Small kana**: if no native `っ`/`ゃ`/`ぁ`… exists, a full-size counterpart
  can be derived at 50% (`small_kana_scale = 0.50`). Derived glyphs are marked
  in the sidecar and slightly penalised during selection.
- Reviewed glyphs are preferred; `watch` glyphs (border ink, likely
  neighbour) are slightly penalised.

The target stroke is sampled per image from 7–10% of the character cell, with
pen pressure 0.75–1.35.

## Renderer integration

Real kana is an opt-in flag, like the `sfx` split: the default pipeline keeps
rendering everything with fonts.

```python
from manga_ocr_dev.real_kana import KanaBank
from manga_ocr_dev.synthetic_data_generator.renderer import Renderer

bank = KanaBank.load("/mnt/real-kana-final")
renderer = Renderer(kana_bank=bank, real_kana=True)

img, params = renderer.render(
    ["おはよう、元気？"],
    {
        "vertical": True,
        "font_size": 48,
        "real_kana_prob": 0.7,    # optional: share of eligible kana replaced
        "real_kana_seed": 1234,   # optional: reproducible attachment
    },
)
print(params["real_kana_stats"])  # {"attached": 3, "skipped": 2, ...}
```

`real_kana` can also be set per sample (`{"real_kana": True}`) and
`{"kana_bank_root": path}` overrides the bank for one render. Bank discovery
order is `REAL_KANA_ROOT`, `assets/handwriting`, `/mnt/real-kana-final`,
`/mnt/real-kana-bank`; `Renderer(kana_bank=path)` is explicit. A missing or
broken bank is ignored rather than raising, and `params["real_kana_stats"]`
reports what happened. SFX keeps its decorative fonts unless
`real_kana_sfx` is set; even then only flat SFX ops (no perspective/aspect
warp) use the bank, because a flat glyph raster cannot be matched to a warp.

The showcase includes four real-kana samples (`synth_samples/29`–`32`) and
skips them when no bank is installed; see [`showcase.md`](showcase.md).

Only kana ops go through the bank; kanji, Latin, digits and punctuation stay
font-rendered, so labels and pixels remain aligned. Ink colour, stroke
outlines, hollow lettering, shadow and extrusion are all supported: the glyph
alpha is re-tinted per op, and the outline is a dilation of the same alpha.
Kana ops that are warped (`persp`/`aspect`) or smaller than
`min_font_size_for_bank` are left to the font. Small kana are rendered at
`small_kana_scale` (50%) of the cell height; their strokes are proportionally
thicker, so stroke matching still uses the full line target.

`KanaBank.load()` only reads the manifest (glyph size/stroke are recorded
there) and opens glyph pixels lazily on first use, so a 61k-glyph bank loads
in a couple of seconds per worker.

### CLI

```bash
python -m manga_ocr_dev.synthetic_data_generator.run_generate \
  --package 0 --n_random 2000 --real_kana --kana_bank /mnt/real-kana-final
```

`--real_kana` is False by default. `--real_kana_prob` mixes real and font kana
in the same image (default 1.0 when the flag is on).

## How this differs from the article

| Article | This implementation |
|---|---|
| Several independent GPT “Sol” sessions for detection and review | Local `manga-ocr-charspot-small` for detection/recognition; one configurable vision model (`gemini-3.5-flash-lite` by default) as a final verifier |
| Real-ESRGAN ×4 on every glyph | ×2 Lanczos by default; glyphs ≥48 px are left alone; Real-ESRGAN optional |
| Manual review of every upscaled glyph | Optional verifier on doubtful crops; automatic artefact flags (`white_cores`, `speckles`) |
| Trained DINO classifier | Same DINO + logistic design, plus a deterministic heuristic fallback when no labelled set exists |
| Separate scripts per stage | One CLI with resumable stages, disk-cached verifier calls, pinned dataset revision |
| Browser-based rendering | Existing Pillow renderer; font is only replaced for kana |

## Limitations

- The repository ships an **empty** bank. Nothing is downloaded at render
  time; run the pipeline to create one.
- The local model is an 11M-parameter OCR model. It occasionally merges two
  kana or drops a faint one; the verifier catches some of these, but a reviewed
  bank is better than an unreviewed one. Keep an eye on `sequence_match` in
  each `raw/<crop_id>/meta.json`.
- Verification is budgeted. Use `--limit` and the default doubtful-only mode
  before running `--all` over a large bank.
- Very small or low-resolution source crops produce noisy alpha masks. The
  cleaner quarantines obvious failures, but a reviewed bank is still better
  than an unreviewed one.
- The heuristic scorer is a bootstrap tool, not a replacement for the trained
  head.
- Kana only. Kanji, Latin, digits, punctuation, `ー`, `・` and iteration marks
  remain font-rendered.

## Tests

```bash
pytest tests/test_real_kana.py -v
```

The tests cover kana helpers, stroke estimation, colour-alpha cleaning,
segmentation validation, local charspot response mapping and grouping,
verification bookkeeping (including `status="upscaled"` preservation), tilt
estimation, bank matching and derived small kana, manifest/audit, the full
offline pipeline and the renderer hook. They need no network or API key (fake
local and vision models are used).
