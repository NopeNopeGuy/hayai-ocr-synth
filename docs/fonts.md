# Fonts (`scan_fonts.py`, `scan_font_profiles.py`, `utils.py`)

Three-layer tofu defense: `fonts.csv` bitmap → live `fontTools` cmap → FreeType raster check.

## `utils.py` (460 lines) — runtime font/background layer

- `class FontMap(paths, bits, extra, lru_size=64)` — compact read-only `font_path→chars`: 8 KB BMP bitmap per font + non-BMP `extra` string. `pack(s)`, `__len__/__contains__/__iter__/__getitem__/get/items`, `_has(ch)`, `support_mask(text)`, `coverage(chars)`.
- `class _FontProbe(font_path, size=48)` + `MissingGlyphError(ValueError)` — probe cache; raised instead of tofu.
- `get_background_df(background_dir) -> DataFrame[path,h,w,ratio]` — `manifest.csv` if present else `*_ymin_ymax_xmin_xmax.ext` filename parse.
- `is_kanji/is_hiragana/is_katakana/is_hangul/is_ascii(ch)` — `unicodedata.name` / `ord<128`.
- `get_charsets(vocab_path=None) -> (vocab, hiragana, katakana)` — filters `vocab.csv` with `_EXCLUDE_HIRA/KATA` denylists.
- `get_font_meta() -> (fonts_df, FontMap)` — streams `fonts.csv` (chunksize 500), drops empty `supported_chars`, packs via `FontMap.pack`.
- `get_font_profiles() -> dict`, `get_font_profile(font_path) -> dict`.
- `find_bad_glyphs(font_path, chars, size=48) -> set` — rejects blank, `.notdef`-identical (probes `NOTDEF_PROBES`), or shared-raster placeholders in horizontal + `ttb` modes. Compares actual glyph rasters (`_glyph_key/_notdef_keys`), not cmap presence.
- `is_drawable(font_path, ch) -> bool` — cached `_probe_for` runtime check; space always True.
- `get_font_cmap_chars(font_path) -> frozenset` (fontTools `TTCollection/TTFont.getBestCmap`, 64-entry cache), `font_supports_text_cmap(font_path, text) -> bool` — multilingual (ko/zh) selection beyond `vocab.csv`.

All cached for forked workers.

## `scan_fonts.py` (272 lines) — coverage scanner → `assets/fonts.csv`

`fire.Fire(main)`; `main(chunk_size=250, max_workers=None, audit_lang=None)`. If `audit_lang` set (`ja/ko/zh/en/all`), runs `audit_langs(langs, limit=2000)` instead of scanning.

- `get_vocab()`, `open_font(path) -> TTFont`, `candidates_of(font) -> list[str]` (best-cmap + all unicode tables, excl. `.notdef`), `process(font_path) -> str` (verified string).
- `_hash_file`, `_regroup`, `_regroup_by_size`, `dedupe_files(items, threads=32)` — inode → size → edge blake2b (64 KiB ends) → full blake2b; largest-first, raster-check one rep per group (~19k fonts).
- `load_checkpoint(path) -> dict` — append-only `fonts.checkpoint.csv` with torn-write truncation + `flush+fsync` every `chunk_size`.
- Pipeline: walk `FONTS_ROOT` (`.ttf/.otf/.ttc/.otc`, face 0), dedupe, `ProcessPoolExecutor.map(process, chunksize=2)` with `find_bad_glyphs` raster gate, replicate to duplicates, atomic `assets/fonts.csv` write (`font_path,supported_chars,num_chars,label=regular` via `.tmp+os.replace`), delete checkpoint, then `from scan_font_profiles import run_profiling; run_profiling()`.

```bash
python -m manga_ocr_dev.synthetic_data_generator.scan_fonts
```

## `scan_font_profiles.py` (495 lines) — archetype profiler → `font_profiles.json`

No CLI args (`main()` takes none; invoked programmatically after `scan_fonts`).

- `ARCHETYPE_CONFIGS` — 12 keys (`dialogue_gothic/dialogue_round/narration_serif/sfx_impact/sfx_brush/sfx_cute/sfx_horror/handwritten/display_headline/comic_latin/regular_cjk/omit`), each with `priority/dialogue_weight/sfx_weight/dialogue_size/sfx_size/stroke_range/stroke_ratio/stroke_prob/double_outline_prob`. E.g. `sfx_impact` 70–195 px, stroke 4–14 prob 0.85.
- `BANNED_OFFICE_FONTS`, `COMIC_MAYO_WHITELIST`, `check_omit(rel_path) -> (bool, reason)` — omits vendor latin/symbols, `yahei/arial/roboto/...`, coding, arabic/hebrew/thai/…, `mayo/` non-comic (whitelist `comic/manga/bb/blambot/...` or CJK filename).
- `classify_font(rel_path) -> str` — keyword-path matching, precedence horror→brush→cute→impact→handwritten→display→comic_latin→round→serif→gothic→`regular_cjk`.
- `detect_lang_hint(rel_path) -> str` (`ko/zh/ja/en/cjk` from path prefixes), `profile_single_font(rel_path) -> dict`, `update_fonts_csv_with_profiles(profiles, out_csv)`, `main()` / alias `run_profiling = main`.
- Pipeline: enumerate `FONTS_ROOT` rel paths sorted → profile each → print archetype/priority breakdowns → write `assets/font_profiles.json` (keyed by rel path, runtime) + `assets/font_profiles.csv` (report) → `ProcessPoolExecutor(min(cpu,16))` coverage for disk fonts missing from `fonts.csv` via `scan_fonts.process`, append rows, overwrite `label` with `priority`, atomic replace.

`get_font_profile()` is consumed at generation time for sizing/stroke/outline draws.
