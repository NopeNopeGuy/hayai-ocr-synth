# Public Python API

## `manga_ocr_dev` (`manga_ocr_dev/__init__.py:1-3`)

`__version__ = "0.2.0"`.

## `manga_ocr_dev.synthetic_data_generator` (`synthetic_data_generator/__init__.py:1-23`)

```python
from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
from manga_ocr_dev.synthetic_data_generator.lang import SUPPORTED_LANGS, detect_lang, normalize_lang
from manga_ocr_dev.synthetic_data_generator.renderer import Renderer
from manga_ocr_dev.synthetic_data_generator.sfx_text import (
    ChineseSFXComposer, EnglishSFXComposer, KoreanSFXComposer, SFXComposer)
```

`__all__` exports the 8 names above.

Minimal use:

```python
from manga_ocr_dev.synthetic_data_generator import SyntheticDataGenerator, Renderer

gen = SyntheticDataGenerator()
img, text_gt, params = gen.process("おはよう！", sfx=False, lang="ja")
img, text_gt, params = gen.process(None, sfx=True, lang="ja")  # random SFX

r = Renderer()
img, params = r.render(["約束だよ", "必ず帰る"],
    override_css_params={"vertical": True, "font_size": 48, "stroke_size": 2})
img_bgra, params = r.render_text(["水平線の向こう側"],
    override_css_params={"vertical": False, "shadow_size": 3})
```

`Renderer.render` returns BGR (color) or grayscale + `params`; `render_text` returns BGRA + `params`. `Generator.process` returns `(img, text_gt, params)` with GT captured before furigana.

## `manga_ocr_dev.data` (`manga_ocr_dev/data/__init__.py:1-25`)

```python
from manga_ocr_dev.data import (export_dialogue, export_lines, clean_dialogue_line,
    clean_novel_text, detect_line_lang, iter_dialogue_stream, iter_mixed_stream,
    iter_novel_stream, split_novel_chapter)
```

## CLIs (all `fire`)

```bash
python -m manga_ocr_dev.data.process_text --help
python -m manga_ocr_dev.data.process_dialogue --help
python -m manga_ocr_dev.synthetic_data_generator.scan_fonts --help
python -m manga_ocr_dev.synthetic_data_generator.run_generate --help
python synth_samples/generate.py -- --help
```
`process_manga109s.py` and `generate_backgrounds.py` have no CLI (edit constants / import functions). `scan_font_profiles.main()` takes no args.
