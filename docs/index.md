# Manga OCR Better Synth — Documentation

Fast, flexible synthetic Japanese manga OCR data generation toolkit (`manga_ocr_dev`, v0.2.0).

End-to-end pipeline to generate realistic synthetic training data imitating manga speech bubbles, hand lettering, and sound effects (SFX), with zero browser dependencies (pure Pillow rendering, ~10–25 ms/image).

## Where to start

- New install: [`installation.md`](installation.md)
- Paths / env vars: [`configuration.md`](configuration.md)
- End-to-end flow: [`pipeline.md`](pipeline.md)
- Input text packages: [`data-corpus.md`](data-corpus.md)
- Background crops: [`backgrounds.md`](backgrounds.md)
- Rendering engine: [`renderer.md`](renderer.md)
- Dialogue/SFX sampling + batch driver: [`generator.md`](generator.md)
- SFX text composer: [`sfx.md`](sfx.md)
- Multilingual support (`ja/ko/zh/en`): [`multilingual.md`](multilingual.md)
- Fonts (coverage scan, profiles, `FontMap`): [`fonts.md`](fonts.md)
- Real handwritten kana bank (optional): [`real-kana.md`](real-kana.md)
- Shipped CSV/JSON assets: [`assets.md`](assets.md)
- Tests: [`testing.md`](testing.md)
- Showcase images: [`showcase.md`](showcase.md)
- Public Python API: [`api.md`](api.md)

## Repository layout

```
assets/                        # fonts.csv, font_profiles.{json,csv}, vocab.csv, len_to_p.csv, lines_example.csv, examples/
manga_ocr_dev/
  __init__.py                  # version 0.2.0
  env.py                       # global paths (FONTS_ROOT, DATA_SYNTHETIC_ROOT, …)
  data/                        # corpus + background preprocessing
    process_text.py            # mixed dialogue+novel exporter (canonical)
    process_dialogue.py        # dialogue-only wrapper
    process_cc100.py           # CC-100 slicer (ko/zh/en)
    process_manga109s.py       # Manga109-s frames/crops exporter
    generate_backgrounds.py    # Manga109-s text-free crops
    generate_backgrounds_animetext.py  # AnimeText text-free crops (recommended)
  synthetic_data_generator/
    renderer.py                # Pillow renderer (3006 lines): uniform/irregular/SFX layouts
    generator_legacy.py        # base multilingual generator (988 lines)
    generator.py               # hard-dialogue wrapper (furigana 0.70)
    run_generate.py            # multiprocess batch driver (fire CLI)
    sfx_text.py                # procedural SFX composer (ja/ko/zh/en)
    lang.py                    # per-language typographic config
    utils.py                   # FontMap, tofu raster check, background/vocab loaders
    scan_fonts.py              # fonts.csv coverage scanner
    scan_font_profiles.py      # archetype profiler → font_profiles.json
    test_generator_accuracy.py # label==pixels accuracy suite
  real_kana/                   # optional real handwritten-kana bank pipeline
    dataset.py                 # AnimeText-OCR harvesting
    segment.py                 # vision kana boxes + review
    clean.py                   # colour-separated alpha + component cleanup
    upscale.py                 # ×2 Lanczos / optional Real-ESRGAN
    upright.py                 # per-line rotation
    bank.py                    # manifest, matching rules, renderer hook
    cli.py                     # python -m manga_ocr_dev.real_kana …
tests/
  test_process_text.py         # corpus cleaning/packaging
  test_pil_renderer.py         # renderer (~30 tests)
  test_multilingual.py         # lang + per-lang generation
  test_real_kana.py            # offline real-kana pipeline + renderer hook
synth_samples/generate.py      # 32 seeded showcase PNGs
docs/                          # this documentation
```

## Quick start

```bash
pip install -r manga_ocr_dev/requirements.txt
export FONTS_ROOT="/path/to/japanese/fonts"
export DATA_SYNTHETIC_ROOT="/path/to/output/synthetic_data"

# 1. Text lines
python -m manga_ocr_dev.data.process_text --num_packages=10 --num_lines_in_each_package=100000
# 2. Backgrounds
python -m manga_ocr_dev.data.generate_backgrounds_animetext --n_images 5000
# 3. Font coverage
python -m manga_ocr_dev.synthetic_data_generator.scan_fonts
# 4. Generate
python -m manga_ocr_dev.synthetic_data_generator.run_generate --package 0 --max_workers 16
python -m manga_ocr_dev.synthetic_data_generator.run_generate --package 0 --split sfx --n_random 50000 --max_workers 16
```

Python API:

```python
from manga_ocr_dev.synthetic_data_generator.renderer import Renderer
renderer = Renderer()
img, params = renderer.render(
    lines=["やはり俺の青春ラブコメは", "まちがっている。"],
    override_css_params={"vertical": True, "font_size": 48,
                         "stroke_size": 4, "text_color": "black",
                         "stroke_color": "white"},
)
```

## Key invariants

- **GT == drawn glyphs.** Ground truth is captured before furigana markup is added, and only font-rendered glyphs (including `♡♥★` when in the string) are labeled. Decor (speedlines, bursts) is never labeled. The renderer raises `MissingGlyphError` instead of emitting tofu.
- **Three tofu layers.** `fonts.csv` bitmap (`FontMap`), live `fontTools` cmap, and FreeType raster comparison (`find_bad_glyphs` / `is_drawable`).
- **Reproducibility.** Batch workers seed `numpy`+`random` from `crc32(id)`; showcase layouts are seeded.
