# Testing

Run: `pytest tests/ -v` (`pyproject.toml:44-45` sets `testpaths=["tests","manga_ocr_dev"]`).

## `tests/test_process_text.py` (132 lines)

- `test_clean_dialogue_line_subtitles`, `test_clean_novel_text_aozora`, `test_split_novel_chapter`, `test_rolling_dedup`, `test_iter_mixed_stream`, `test_export_lines_local_mock` (local `dummy_dialogue.txt` → `0000.csv`, 3 rows, checks `source,id,line` schema).
- Verifies ASS/HTML/speaker/music/quote handling and packaging. No network (local mock). 6 tests pass.

## `tests/test_pil_renderer.py` (602 lines, ~30 tests)

Covers `_parse_line` (ruby/tcy), BGRA output, vertical-rl vs horizontal ordering, stroke/shadow/ruby/tcy ink, `full_render` grayscale vs color BGR, irregular/slant/SFX (`persp/curve/speedlines/burst/double_outline`), color sampling (`solid/per_line/per_char`, contrast/outline legibility), `get_background_df` with/without manifest.
Requires `/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc` else skipped.

## `tests/test_multilingual.py` (311 lines)

- `test_supported_langs`, `test_normalize_lang_aliases`, `test_detect_lang_scripts`, `test_vertical_probs_ordering`, `test_*_sfx_composer_vocab_only[ko|zh|en]`, `test_detect_line_lang_wrapper`, `test_export_lines_lang_filter` (mixed ja/ko/zh/en txt → `lines_ko` keeps 1 Korean line), `test_renderer_renders_each_language`, `test_renderer_vertical_distribution_per_lang`, `test_sfx_renders_each_language`, `test_generator_*_per_lang/auto_detects`.
- Validates `lang.py` + `process_text` lang filter + per-lang rendering.

## `manga_ocr_dev/synthetic_data_generator/test_generator_accuracy.py` (110 lines)

Label==pixels suite; needs real `fonts.csv`/`vocab.csv`/backgrounds. Collected by default `testpaths` (`pytest -q test_generator_accuracy.py`).

- Fixture `gen()` (module-scoped `SyntheticDataGenerator`); helper `_layout(vertical, lines)` via `Renderer._layout_sfx` (`font_size=100, line_height=0.3, margin=20, pad=4, curve=straight`).
- `test_sfx_vertical_first_line_is_rightmost`, `test_sfx_horizontal_first_line_is_top` — reading order.
- `test_gt_only_contains_glyphs_of_the_font[sfx True/False]` (60 seeds) — GT ⊆ `font_map[font_path]` + spaces.
- `test_sfx_fonts_cover_kana` — every SFX font contains `ドンバガッー`.
- `test_sfx_is_diverse` (200 seeds, >150 uniques), `test_fonts_csv_has_no_tofu` (25 random fonts, `find_bad_glyphs`), `test_generated_text_is_drawable[sfx]` (60 seeds offset 1000, `is_drawable`), `test_renderer_refuses_tofu` (probe `"U+10FFFD U+E0FF"`, expects `MissingGlyphError`).
- Failures point to stale `fonts.csv` → re-run `scan_fonts.py`.

## Lint

`ruff` (`line-length=120`, `indent-width=4`, `pyproject.toml:40-42`).
