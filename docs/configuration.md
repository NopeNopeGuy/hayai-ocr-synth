# Configuration (`manga_ocr_dev/env.py`)

Single source of truth for filesystem locations. All 6 values are `pathlib.Path` objects overridden by environment variables. See `manga_ocr_dev/env.py:1-37`.

| Constant | Env var | Default |
|---|---|---|
| `ASSETS_PATH` | `MANGA_OCR_ASSETS_PATH` | `<package>/assets` if present else `<repo>/assets` (`env.py:9-14`) |
| `FONTS_ROOT` | `FONTS_ROOT` | `~/data/jp_fonts` (`env.py:17`) |
| `DATA_SYNTHETIC_ROOT` | `DATA_SYNTHETIC_ROOT` | `/mnt/Synth_Images` if mounted else `~/data/synthetic` (`env.py:20-25`) |
| `BACKGROUND_DIR` | `BACKGROUND_DIR` | `~/data/manga/Manga109s/background` (`env.py:28-30`) |
| `BACKGROUND_ANIMETEXT_DIR` | `BACKGROUND_ANIMETEXT_DIR` | `~/data/manga/AnimeText_background` (`env.py:32-34`) |
| `MANGA109_ROOT` | `MANGA109_ROOT` | `~/data/manga/Manga109s` (`env.py:37`) |

Usage:

```bash
export FONTS_ROOT="/path/to/japanese/fonts"
export DATA_SYNTHETIC_ROOT="/path/to/output/synthetic_data"
export MANGA_OCR_ASSETS_PATH="./assets"
export BACKGROUND_ANIMETEXT_DIR=~/data/manga/AnimeText_background
```

Consumers:

- `ASSETS_PATH` → `utils.py:get_charsets/get_font_meta/get_font_profiles` load `vocab.csv`, `fonts.csv`, `font_profiles.json`.
- `FONTS_ROOT` → `scan_fonts.py`, `scan_font_profiles.py`, `generator_legacy.py` font selection; `run_generate.py` relativizes `font_path` to it for portability.
- `DATA_SYNTHETIC_ROOT` → `process_text.py:export_lines` writes `lines[/_ko/_zh/_en]`, `run_generate.py:_package_paths` writes `img[/_lang]/meta[/_lang]`, `process_cc100.py` reads `<lang>.txt`.
- `BACKGROUND_*_DIR` → `renderer.py:get_default_background_df` (AnimeText preferred, Manga109-s fallback), `generate_backgrounds*.py` output.
- `MANGA109_ROOT` → `process_manga109s.py`, `generate_backgrounds.py` input.

`manga_ocr_dev/README.md:38-50` documents the same table.
