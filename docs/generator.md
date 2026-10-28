# Generator + batch driver (`synthetic_data_generator/`)

## `generator_legacy.py` (988 lines) — base multilingual generator

Owns font selection, random text, line wrapping, furigana, and SFX dispatch. No CLI (library).

`__init__` loads `get_charsets()`, `len_to_p.csv`, budoux parsers, `get_font_meta()` (label probs `special 0.50 / common 0.35 / regular 0.15`), `Renderer()`, and per-lang `SFXComposer`s.

Key methods (`generator_legacy.py`):

- `process(text=None, override_css_params=None, sfx=False, lang=None)` — normalizes `sfx` + `lang` (`override["lang"]` > `detect_lang(text)` > `normalize_lang`, default `ja`); SFX → `process_sfx()`, else 5× retry `_prepare_dialogue_lines()`.
- `_prepare_dialogue_lines(text, override, lang)` — if `text is None` picks `get_random_font(lang)` then `get_random_words(vocab)`; else normalizes `　/…`, `split_into_words()` (budoux `ja/zh_hans/zh_hant`, regex `ko/en`) → `words_to_lines()` (poisson(6) clip 1–20 for `ja/zh`, poisson(14) clip 8–28 for `ko/en`). Applies `font_profiles.json` sizing (`dialogue_size_min/max`, `stroke_prob/ratio/min/max`, `double_outline_prob`). Strips `<>&`, filters `is_drawable()`.
- `process_sfx(text=None, override=None, lang)` — merges `get_random_sfx_params(lang)`, forces `sfx=True, irregular=False`; 10× retry `get_random_sfx_font()` + `get_random_sfx_text()` (20 tries, lexicon fallback, kana-burst fallback, full `is_drawable` gate); splits to 1–3 lines (`>6:45%`, `>9:25%` 3-way, `>4:30%`; `en` with spaces stays single).
- `get_random_words(vocab, font_path, lang)` + `_get_random_words_{ja,ko,zh,en}` — `ja` kana/kanji/kata/punct/num (0.40/0.28/0.10/0.12/0.05); `ko` Hangul 0.80 spaced + frequent-syllable weighting; `zh` Han 0.72 spaceless; `en` `EN_WORDS` 0.82 with per-sample upper 0.3–0.7. `_punct_for_lang`, `_punct_word`, `_pick`, `_uniform_fallback`, `_pools`, `_vocab_probe`.
- `get_random_sfx_text(vocab, font_path, lang)` — delegates to per-lang `SFXComposer.sample`.
- `get_random_font(text, lang)` / `_dialogue_font_candidates` / `_find_font_by_cmap(text, is_label, lang, limit=60)` / `get_font_labels_prob` / `is_font_supporting_text` (+ alias `is_font_supporting_text_any`) / `_fonts_covering(need, thresh=0.98)` / `_cmap_hit` / `get_sfx_fonts(lang)` + `_get_sfx_fonts_{ja,en,cjk}` / `get_random_sfx_font(lang)` / `_sfx_weights` — three-layer defense: `fonts.csv` bitmap `FontMap`, live `fontTools` cmap, raster `is_drawable()`, with `_pool_cache` (32) + `_lang_font_cache/_sfx_font_cache`.
- `split_into_words(text, lang)`, `words_to_lines(words, lang)` (+ `_words_to_lines_spaced`), `add_random_furigana(line, word_prob=1.0, vocab=None, font_path=None)` — 50% `ja` adds `<ruby><rt>` + `<span text-combine-upright>` for ≤3 ASCII. GT is joined **before** furigana.

GT invariant: filtering, never truncating; `MissingGlyphError` on tofu.

## `generator.py` (81 lines) — hard-dialogue wrapper

`from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator` is the default import (`synthetic_data_generator/__init__.py:3`).

- `FURIGANA_PROB_JA = 0.70` (legacy 0.50).
- `sample_hard_ink() -> dict` (`generator.py:26-41`): `distress` 30% (strength 0.3–0.7), `faint` 20% (alpha 0.4–0.7), `extrude` 12%, `hollow` 10%, `irregular` 12%.
- `SyntheticDataGenerator.process(text, override, sfx, lang)` — SFX short-circuits to legacy; else detect/normalize lang, `setdefault` hard inks (caller wins), `_process_dialogue` (5 retries, GT-before-furigana, `params["lang"]=lang`).

Random-text sampling unchanged from legacy.

## `run_generate.py` (155 lines) — parallel batch driver

`fire.Fire(run)` CLI:

| Flag | Default | Meaning |
|---|---|---|
| `--package` | `0` | `0000` dir/file stem |
| `--n_random` | `1000` | random dialogue rows (dialogue split) or SFX rows (sfx split) |
| `--n_limit` | `None` | subsample total rows |
| `--max_workers` | `16` | `process_map` workers |
| `--split` | `dialogue` | `dialogue` (`img/meta`) vs `sfx` (`img_sfx/meta_sfx`) |
| `--lang` | `ja` | `ja` keeps legacy paths, else `img_{lang}/meta_{lang}/lines_{lang}` |

Functions:

- `get_generator()` — per-process lazy singleton.
- `_random_frame(prefix, lang, package_str, n)` — generated-only rows (`line=None`).
- `_package_paths(split, lang, package_str)` (`run_generate.py:42-51`) — output dir + meta path.
- `_worker_process_sample(args)` — `crc32(id)` seeds `numpy`+`random`, `text=None` if NaN, `sample_lang=lang if text is None else None` (corpus rows auto-detect), `generator.process(text,sfx,lang)`, `cv2.imwrite(id.jpg)`, relativize `font_path` to `FONTS_ROOT`.
- `run(...)` — builds workload (dialogue: `lines{suffix}/{package}.csv` + random frame; sfx: all-random), `process_map`, collects `(source,id,text_gt,vertical,font_path)`, traceback-prints + skips failures, writes `meta{suffix}/{package}.csv`.

```bash
python -m manga_ocr_dev.synthetic_data_generator.run_generate --package 0 --max_workers 16
python -m manga_ocr_dev.synthetic_data_generator.run_generate --package 0 --split sfx --n_random 50000 --max_workers 16
```
