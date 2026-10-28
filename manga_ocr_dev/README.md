# manga_ocr_dev

Development toolkit for generating high-fidelity synthetic Japanese manga OCR training datasets.

## Project Structure

```
assets/                       # Asset files (fonts.csv, len_to_p.csv, vocab.csv, etc.)
manga_ocr_dev/                # Core development package
   env.py                     # Environment variables and path configurations
   data/                      # Text corpus & background image preprocessing
   synthetic_data_generator/  # Pillow-based high-throughput synthetic text generator
tests/                        # Test suite for rendering engine and text processing
```

## Assets

### `fonts.csv`
CSV mapping font paths to character coverage and frequency weights:
- `font_path`: Path to font file, relative to `FONTS_ROOT`.
- `supported_chars`: String of characters verified to render without missing glyph boxes.
- `num_chars`: Number of supported characters.
- `label`: `common`, `regular`, or `special` (used to weight font sampling).

To generate `fonts.csv` for your own font collection, run `manga_ocr_dev/synthetic_data_generator/scan_fonts.py`.

### `lines_example.csv`
Sample text line package with columns: `source`, `id`, `line`.

### `len_to_p.csv`
Probability distribution over line lengths derived from empirical manga corpora.

### `vocab.csv`
List of supported characters recognized by the tokenizer.

---

## Configuration (`env.py`)

Paths are configured via environment variables with sensible defaults:

| Variable | Description | Default |
|---|---|---|
| `MANGA_OCR_ASSETS_PATH` | Path to asset files directory | `./assets` |
| `FONTS_ROOT` | Directory containing font collections | `~/data/jp_fonts` |
| `DATA_SYNTHETIC_ROOT` | Destination directory for output packages | `~/data/synthetic` (or `/mnt/Synth_Images` if present) |
| `BACKGROUND_DIR` | Manga109-s background crops directory | `~/data/manga/Manga109s/background` |
| `BACKGROUND_ANIMETEXT_DIR` | AnimeText background crops directory | `~/data/manga/AnimeText_background` |
| `MANGA109_ROOT` | Manga109-s dataset root | `~/data/manga/Manga109s` |

---

## Data Pipeline

### 1. Extract Text Lines

Generate line packages from streaming Hugging Face datasets or local files:

```bash
# Mixed dialogue + novels (streams from HuggingFace on the fly)
python -m manga_ocr_dev.data.process_text --num_packages=10 --num_lines_in_each_package=100000

# Dialogue-only (subtitles from anime/drama)
python -m manga_ocr_dev.data.process_dialogue --num_packages=10

# Web novels only (Narou, Kakuyomu, Hameln, etc.)
python -m manga_ocr_dev.data.process_text --source_type=novel --num_packages=10
```

Supported online datasets:
- [Japanese Dialogue Dataset](https://huggingface.co/datasets/JustANormalTinkerer/Japanese-Dialogue-Dataset) (`anime_tv`, `anime_movie`, `drama_tv`, `drama_movie`, `unsorted`)
- [Japanese Novels 23M](https://huggingface.co/datasets/OmniAICreator/Japanese-Novels-23M) (`Narou_NSFW`, `Kakuyomu`, `Narou_SFW`, `Hameln`, `aozorabunko`, `novecomi`, `pixiv`)

### 2. Extract Backgrounds

Extract clean, text-free background patches:

```bash
# From AnimeText dataset (recommended)
python -m manga_ocr_dev.data.generate_backgrounds_animetext --n_images 10000

# Or from Manga109-s (requires Manga109-s dataset)
python -m manga_ocr_dev.data.generate_backgrounds
```

### 3. Scan Font Coverage

Index and verify character support across your local font library:

```bash
python -m manga_ocr_dev.synthetic_data_generator.scan_fonts
```

### 4. Run Synthetic Generation

Render synthetic image-text pairs:

```bash
# Generate dialogue package 0000
python -m manga_ocr_dev.synthetic_data_generator.run_generate --package 0 --max_workers 16

# Generate SFX (sound effect) split
python -m manga_ocr_dev.synthetic_data_generator.run_generate --package 0 --split sfx --n_random 50000 --max_workers 16
```

---

## Directory Layout for Synthetic Output

```
<DATA_SYNTHETIC_ROOT>/
   img/           # Generated dialogue images
      0000/
      ...
   img_sfx/       # Generated SFX images
      0000/
      ...
   lines/         # Text line packages (input to generation)
      0000.csv
      ...
   meta/          # Dialogue image-text metadata
      0000.csv
      ...
   meta_sfx/      # SFX image-text metadata
      0000.csv
      ...
```
