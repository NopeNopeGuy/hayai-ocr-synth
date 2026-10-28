# Installation

## Requirements

- Python `>=3.9` (`pyproject.toml:6`)
- OS: Linux (tested). No browser, no Rust toolchain, no temp-file pipeline.

## Core dependencies (`manga_ocr_dev/requirements.txt`, `pyproject.toml:14-25`)

| Package | Use |
|---|---|
| `pillow>=9.0.0` | glyph rasterization, text layer compositing |
| `opencv-python>=4.5.0` | BGR I/O, bubble warps, degradations |
| `numpy>=1.20.0` | image arrays, RNG |
| `albumentations>=1.1.0` | `LongestMaxSize`, blur/noise degradations |
| `fonttools>=4.30.0` | cmap coverage without rasterizing |
| `datasets>=2.0.0` | streaming dialogue/novel corpora |
| `pandas>=1.3.0` | `lines/*.csv`, `meta/*.csv`, manifests |
| `budoux>=0.1.0` | Japanese/Chinese word-break for line wrapping |
| `fire>=0.4.0` | CLI for all `process_*`, `run_generate`, `synth_samples/generate.py` |
| `tqdm>=4.60.0` | progress + `process_map` multiprocessing |

Dev: `pytest>=7.0.0`, `ruff>=0.4.0` (`pyproject.toml:27-31`).

Optional — real handwritten kana bank (`docs/real-kana.md`):

| Extra | Packages | Use |
|---|---|---|
| `kana` | `google-genai`, `openai`, `requests` | optional Gemini/OpenAI verifier and Mercury handwriting classifier (detection/recognition is local) |
| `scoring` | `torch`, `torchvision`, `transformers`, `scikit-learn` | frozen DINOv2 + logistic head, or the faster Mercury-distilled CNN |

The local detector/recognizer is `manga-ocr-charspot-small`; install it from
the checkout with `pip install -e manga-ocr-charspot-small` (CPU ONNX) or point
`--local-model` / `MANGA_OCR_CHARSPOT_SMALL_DIR` at its files.

```bash
pip install -e '.[kana,scoring]'
```

Real-ESRGAN (`realesrgan` + `basicsr`) is optional and installed manually; the
upscaler falls back to Lanczos without it.

## Install

```bash
git clone https://github.com/your-username/manga-ocr-better-synth.git
cd manga-ocr-better-synth
pip install -r manga_ocr_dev/requirements.txt
# or: pip install -e .   # installs manga-ocr-better-synth 0.2.0
pip install pytest  # for tests
```

System Japanese font for tests (tests skip without it):

```bash
# Debian/Ubuntu example
apt install fonts-noto-cjk
ls /usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc
```

`synth_samples/generate.py` uses the font library (`FONTS_ROOT`/`assets/fonts.csv`) and
the background dataset (`BACKGROUND_ANIMETEXT_DIR` or `BACKGROUND_DIR`); without
backgrounds it falls back to transparent text layers.

## Verify

```bash
pytest tests/ -v
python synth_samples/generate.py  # regenerates 32 PNGs (seeded; 29-32 need a real-kana bank)
```
