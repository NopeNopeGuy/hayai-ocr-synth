# Manga OCR Better Synth

A fast, flexible, and robust synthetic Japanese manga OCR data generation toolkit.

This repository provides an end-to-end pipeline to generate realistic synthetic training datasets imitating Japanese manga speech bubbles, handwritten lettering, and sound effects (SFX), with zero browser dependencies.

---

## Key Features

- **Pure Pillow Rendering Engine**: Replaces legacy headless Chrome (`html2image`) browser screenshots with direct Pillow rendering. Runs in ~10–25 ms per image (~30x faster) without external browser binaries.
- **Multiprocess Parallelism**: True multi-core generation using `process_map` for maximum hardware utilization.
- **Rich Japanese Layouts**:
  - Vertical text (`vertical-rl`) with right-to-left column progression and horizontal text.
  - Furigana / ruby text (`<ruby>漢字<rt>かんじ</rt></ruby>`).
  - Tate-chū-yoko (`text-combine-upright`) for mixed vertical Latin digits/words.
  - Irregular hand-lettered speech bubble variations: staggered line starts, per-line font sizes, baseline wobble, slight tilt, and shrunken small kana.
  - Whole-block slant / shear (mild italic leans to dramatic 30° skews).
- **Dedicated SFX (Onomatopoeia) Mode**:
  - Over 200 base Japanese sound effects combined procedurally (stretched vowels, repeats, stutters, compound SFX).
  - 7 baseline curve paths (`arc`, `wave`, `diagonal`, `straight`, `zigzag`, `bounce`, `hill`).
  - True per-glyph perspective warping, per-glyph size/tilt, thick stroke outlines, and double (black-outer / white-inner) edges.
  - Radiating speedlines and burst balloons.
  - Accurate ground truth: only font-rendered glyphs are included in labels (never background decor).
- **Chromatic & Multi-Color Text**:
  - Supports vibrant color pages, covers, and SFX lettering (hot pink, red, cyan, lime, yellow, purple, etc.).
  - `per_line` color alternation (emphasis lettering) and `per_char` rainbow fills.
  - Automatic WCAG contrast enforcement (light ink automatically receives dark outlines and vice-versa).
- **Corpus Processing & Streaming**:
  - Stream dialogue subtitles from [Japanese Dialogue Dataset](https://huggingface.co/datasets/JustANormalTinkerer/Japanese-Dialogue-Dataset).
  - Stream web novel chapters from [Japanese Novels 23M](https://huggingface.co/datasets/OmniAICreator/Japanese-Novels-23M).
  - Automated subtitle ASS markup stripping, Aozora Bunko formatting cleanup, and rolling de-duplication.
- **Tofu Detection & Font Auditing**:
  - Pre-scans and verifies font character coverage against FreeType glyph rasters to prevent missing character boxes in synthetic output.
- **Optional Real Handwritten Kana**:
  - Harvest annotated crops from AnimeText-OCR, cut kana locally with `manga-ocr-charspot-small`, optionally verify with Gemini, then clean/upscale/upright them into a local kana bank. Enable it per run with `--real_kana` (off by default, like the SFX split) to swap only eligible kana for real handwriting; kanji, Latin, digits and punctuation stay font-rendered, so labels and pixels remain aligned.

---

## Installation

```bash
git clone https://github.com/your-username/manga-ocr-better-synth.git
cd manga-ocr-better-synth

# Install required dependencies
pip install -r manga_ocr_dev/requirements.txt
```

Optional real handwritten-kana pipeline (vision model + optional DINO or
Mercury handwriting scoring):

```bash
pip install -e '.[kana,scoring]'
```

See [`docs/real-kana.md`](docs/real-kana.md) for credentials, commands and the
bank format. The bank is empty by default; without one, rendering is unchanged.

---

## Quick Start

### 1. Configure Paths

Set optional environment variables in your shell or edit `manga_ocr_dev/env.py`:

```bash
export FONTS_ROOT="/path/to/japanese/fonts"
export DATA_SYNTHETIC_ROOT="/path/to/output/synthetic_data"
```

### 2. Prepare Dialogue & Novel Text

Download or stream Japanese text lines into package CSVs:

```bash
# Export 10 packages (100,000 lines each) of mixed dialogue and novel text
python -m manga_ocr_dev.data.process_text --num_packages=10 --num_lines_in_each_package=100000

# Export dialogue only (anime & drama subtitles)
python -m manga_ocr_dev.data.process_dialogue --num_packages=10
```

### 3. Generate Background Images

Extract text-free background crops from manga/anime illustrations:

```bash
python -m manga_ocr_dev.data.generate_backgrounds_animetext --n_images 5000
```

### 4. Scan Font Coverage

Index character support across your local font directory:

```bash
python -m manga_ocr_dev.synthetic_data_generator.scan_fonts
```

### 5. Generate Synthetic Images

Run batch generation for dialogue and sound effects:

```bash
# Standard dialogue package
python -m manga_ocr_dev.synthetic_data_generator.run_generate --package 0 --max_workers 16

# Sound effects (SFX) split
python -m manga_ocr_dev.synthetic_data_generator.run_generate --package 0 --split sfx --n_random 50000 --max_workers 16
```

---

## Python API

You can also use the renderer directly in your own scripts:

```python
from manga_ocr_dev.synthetic_data_generator.renderer import Renderer

renderer = Renderer()

# Render custom multi-line Japanese text with stroke outline
img, params = renderer.render(
    lines=["やはり俺の青春ラブコメは", "まちがっている。"],
    override_css_params={
        "vertical": True,
        "font_size": 48,
        "stroke_size": 4,
        "text_color": "black",
        "stroke_color": "white",
    },
)
```

---

## Testing

Run the test suite with pytest:

```bash
pytest tests/ -v
```

---

## Showcase Samples

Sample images generated using the renderer are located in `synth_samples/`. They use fonts sampled
from the installed manga-font library and are composited over the real background dataset. To
regenerate them:

```bash
python synth_samples/generate.py
```

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
