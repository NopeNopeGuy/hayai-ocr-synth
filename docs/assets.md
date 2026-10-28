# Shipped assets (`assets/`)

Resolved via `ASSETS_PATH` (`configuration.md`).

| File | Schema / content | Producer / consumer |
|---|---|---|
| `fonts.csv` | `font_path,supported_chars,num_chars,label` (`label`: `common`/`regular`/`special` = sampling weight). `font_path` relative to `FONTS_ROOT`. | `scan_fonts.py` + `scan_font_profiles.py`; consumed by `utils.get_font_meta()` → `FontMap` |
| `font_profiles.json` | `{rel_path: {font_path, font_type, priority, is_manga, omit_reason, dialogue_weight, sfx_weight, dialogue_size_min/max, sfx_size_min/max, stroke_min/max, stroke_ratio, stroke_prob, double_outline_prob, lang_hint}}` | `scan_font_profiles.py`; consumed by `utils.get_font_profiles()` at generation time |
| `font_profiles.csv` | same as JSON, tabular report | `scan_font_profiles.py` |
| `vocab.csv` | `id,char` — supported chars for the tokenizer | `utils.get_charsets()` (with `_EXCLUDE_HIRA/KATA` denylists) |
| `len_to_p.csv` | `len,p` — line-length distribution from empirical manga corpora | `generator_legacy` line-length sampling |
| `lines_example.csv` | `source,id,line` — sample line package | example input to `run_generate.py` (dialogue split) |
| `examples/` | `00.jpg…11.jpg, cc-100.jpg, random.jpg` — reference outputs | visual sanity checks |

`manga_ocr_dev/README.md:16-35` documents `fonts.csv`/`lines_example.csv`/`len_to_p.csv`/`vocab.csv`.
Regenerate coverage for your own collection: `python -m manga_ocr_dev.synthetic_data_generator.scan_fonts`.
