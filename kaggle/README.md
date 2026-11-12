# Kaggle: append synthetic crops to `hayai-synth-dataset`

`append_hayai_synth.ipynb` generates new synthetic Japanese OCR crops and appends
them to `JustANormalTinkerer/hayai-synth-dataset` as **new parquet shards**. It
never rewrites, renames, or deletes existing shards.

New files are named:

```
data/<split>-append-<run-id>-00000.parquet
```

They still match the card globs (`data/japanese-*.parquet`, `data/sfx-*.parquet`,
`data/handwritten-*.parquet`), and keep the existing schema:

| Column | Type | Notes |
|---|---|---|
| `image` | Image (raw JPEG bytes + `<image_id>.jpg` path) | same layout as existing shards |
| `transcription` | string | cleaned line text |
| `language` | string | always `ja` |
| `source` | string | e.g. `dialogue_anime_tv`, `novel_Narou_NSFW`, local file stem |
| `image_id` | string | unique per crop, encodes preset/run/index |
| `vertical` | bool | writing direction |

## Required Kaggle inputs

Attach these as Kaggle datasets (the notebook auto-detects them under
`/kaggle/input`, `/mnt`, and `~/data`), or set the paths in the config cell:

| Input | Contents |
|---|---|
| backgrounds | `manifest.csv` plus the `*.png` crops produced by `generate_backgrounds_animetext` |
| fonts | font tree whose relative paths match `assets/fonts.csv` |
| real kana (optional) | bank folder with `manifest.jsonl` and `glyphs/<kana>/<id>.png` |
| lines (optional) | `.csv` / `.parquet` / `.jsonl` / `.txt` with a `line`, `text`, or `dialogue` column |

The notebook's font check samples `assets/fonts.csv` across the whole file and
picks the font root with the best match, so a partial font pack is detected
immediately instead of mid-run.

Add a Kaggle secret named **`HF_TOKEN`** (write access to the dataset repo)
before enabling upload.

## What to choose in the config cell

- **`SPLIT`** — `japanese`, `sfx`, or `handwritten`.
- **`CROP_PRESET`** — `auto`, `dialogue`, `dialogue_bubble`, `dialogue_flat`,
  `dialogue_tight`, `handwritten`, `sfx`, `mixed`, or `custom`.
- **`LINES_MODE`** — `hf_dialogue`, `hf_novel`, `local`, `random`, or `sfx`.
- **`N_IMAGES`** / **`ROWS_PER_SHARD`** — total crops and shard size.
- **Crop controls** — `VERTICAL`, `BUBBLE`, `BUBBLE_PROB`, `TIGHT`, `IRREGULAR`,
  `USE_REAL_KANA`, `REAL_KANA_PROB`, `COLOR`, `FONT_SIZE_MIN/MAX`,
  `STROKE_SIZE`, `SLANT_DEG`, `DOUBLE_OUTLINE`, `HARD_INK`, `OVERRIDE_JSON`.
- **Upload controls** — `PUSH_TO_HUB` (off by default), `UPDATE_CARD`,
  `DRY_RUN_CARD` (on by default), `VERIFY_REMOTE`.

Typical pairings:

| Goal | `SPLIT` | `CROP_PRESET` | `LINES_MODE` |
|---|---|---|---|
| more dialogue bubbles | `japanese` | `dialogue_bubble` | `hf_novel` |
| tight text-on-art crops | `japanese` | `dialogue_tight` | `hf_novel` |
| handwritten dialogue | `handwritten` | `handwritten` | `hf_dialogue` |
| onomatopoeia | `sfx` | `sfx` | `sfx` |
| balanced batch | any | `mixed` | any |

## Notebook flow

1. Install dependencies.
2. Set configuration (the only cell you normally edit).
3. Locate the repo, or clone it with `REPO_URL`.
4. Auto-discover backgrounds, fonts, kana bank, assets, and line files.
5. Sanity-check the font root.
6. Build the shared CLI command and write `run_config-<run-id>.json`.
7. Preview a few crops (no shards written).
8. Generate shards.
9. Validate schema, JPEG magic, transcription, labels, and unique ids.
10. Upload only shards that do not already exist on the Hub.
11. Add the uploaded row counts to the dataset card (dry-run first by default).
12. Optionally load one remote shard back as a spot-check.

## CLI equivalent

The notebook is a thin wrapper around `tools/append_hayai_synth.py`:

```bash
# generate
python tools/append_hayai_synth.py generate \
  --split japanese --preset dialogue_bubble --lines hf:novel:Narou_NSFW \
  --n-images 5000 --rows-per-shard 5000 --workers 4 \
  --out-dir /kaggle/working/hayai_append --run-id my-run

# validate
python tools/append_hayai_synth.py validate \
  --out-dir /kaggle/working/hayai_append --split japanese --run-id my-run

# upload (append-only)
python tools/append_hayai_synth.py upload \
  --out-dir /kaggle/working/hayai_append --split japanese --run-id my-run \
  --repo-id JustANormalTinkerer/hayai-synth-dataset

# update card counts
python tools/append_hayai_synth.py update-card \
  --repo-id JustANormalTinkerer/hayai-synth-dataset \
  --add japanese=5000 --dry-run
```

Environment variables are set from the flags before the generator is imported,
so they override the defaults in `manga_ocr_dev/env.py`:
`FONTS_ROOT`, `MANGA_OCR_ASSETS_PATH`, `BACKGROUND_ANIMETEXT_DIR`, and
`REAL_KANA_ROOT`.

## Safety notes

- Upload is append-only: `upload` never deletes or replaces remote files.
- `PUSH_TO_HUB` defaults to `False`.
- Card updates are additive (`--add split=N`) and default to `DRY_RUN_CARD=True`.
- If a run is interrupted, rerun with `RESUME=True` **and the same `RUN_ID`**;
  generation skips already-written source lines and the upload step skips
  shards already present on the Hub.
- Use a fresh `RUN_ID` for a new batch so shard names cannot collide.

---

# Kaggle / Colab: Multilingual Synthetic OCR (`multilingual_synth.ipynb`)

`multilingual_synth.ipynb` extends synthetic generation to **Chinese (`zh`)**,
**Korean (`ko`)**, **English (`en`)**, and **Japanese (`ja`)**.

It works seamlessly on **Google Colab**, **Kaggle**, and **local workstations** (`/mnt/data`):

1. **Fonts**:
   - Priority 1: `/mnt/data/jp_fonts` (local dev machine)
   - Priority 2: `/kaggle/input/...` (attached Kaggle dataset)
   - Priority 3: Streaming/selective download from private Hugging Face repo [`JustANormalTinkerer/manga-ocr-multilingual-fonts`](https://huggingface.co/datasets/JustANormalTinkerer/manga-ocr-multilingual-fonts) (downloads only the language folder needed for the active run: `Chinese_Additions`, `Korean`, `Latin_Comic`, or `精简包`).
2. **Backgrounds**:
   - Priority 1: `/mnt/data/manga/AnimeText_background` (local)
   - Priority 2: Streaming from [`deepghs/AnimeText`](https://huggingface.co/datasets/deepghs/AnimeText) on the fly (masks text boxes and generates clean crops + `manifest.csv` automatically).
3. **Upload & Splits**:
   - Configurable split names (e.g. `chinese`, `chinese_sfx`, `korean`, `korean_sfx`, `english`, `english_sfx`).
   - Generates Parquet shards with schema `(image, transcription, language, source, image_id, vertical)`.
   - Append-only upload to any Hugging Face dataset (e.g. `JustANormalTinkerer/hayai-synth-multilingual`).

