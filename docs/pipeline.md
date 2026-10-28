# End-to-end pipeline (`docs/pipeline.md`)

```
1. Corpus ──► 2. Backgrounds ──► 3. Font scan ──► 4. Batch generation
process_text/    generate_          scan_fonts +      run_generate
process_dialogue backgrounds_*      scan_font_        + generator/
process_cc100    (AnimeText /      profiles          renderer
 Manga109s XML   Manga109s)
```

## 1. Extract text lines → `DATA_SYNTHETIC_ROOT/lines/*.csv`

- Mixed dialogue+novels (streaming HF): `python -m manga_ocr_dev.data.process_text --num_packages=10 --num_lines_in_each_package=100000`
- Dialogue only: `python -m manga_ocr_dev.data.process_dialogue --num_packages=10`
- Novels only: `python -m manga_ocr_dev.data.process_text --source_type=novel --num_packages=10`
- CC-100 per language: `process_cc100.export_lines(lang="ko|zh|en")`

Output schema: `source,id,line` (see `assets.md`). Rolling dedup window 50k. Details: `data-corpus.md`.

## 2. Extract backgrounds → `BACKGROUND_*_DIR/*.png`

- Recommended: `python -m manga_ocr_dev.data.generate_backgrounds_animetext --n_images 10000`
- Legacy Manga109-s (needs `MANGA109_ROOT` + `process_manga109s.py` first): `python -m manga_ocr_dev.data.generate_backgrounds`

Both grow random text-free rectangles from an occupancy mask (`find_rectangle`). Details: `backgrounds.md`.

## 3. Scan font coverage → `assets/fonts.csv` + `assets/font_profiles.json`

```bash
python -m manga_ocr_dev.synthetic_data_generator.scan_fonts
# optional audit without rescan:
# python -m manga_ocr_dev.synthetic_data_generator.scan_fonts --audit_lang all
```

- `scan_fonts.main(chunk_size=250)` dedupes by content hash, raster-checks tofu, writes `fonts.csv`, then calls `scan_font_profiles.run_profiling()`.
- `scan_font_profiles` classifies 12 archetypes (dialogue_gothic, sfx_impact, …) and relabels `fonts.csv`. Details: `fonts.md`.

## 4. Render packages → `img*/meta*`

```bash
python -m manga_ocr_dev.synthetic_data_generator.run_generate --package 0 --max_workers 16
python -m manga_ocr_dev.synthetic_data_generator.run_generate --package 0 --split sfx --n_random 50000 --max_workers 16
```

- Dialogue: reads `lines/{package:04d}.csv` + appends `n_random` random-dialogue rows; auto-detects language per row.
- SFX: fully synthetic from `SFXComposer` (no corpus needed).
- Output tree (Japanese keeps legacy names, other langs get suffix):

```
<DATA_SYNTHETIC_ROOT>/
  img/0000/*.jpg  img_sfx/0000/*.jpg        # ja
  img_ko/… meta_ko/…  lines_ko/…            # ko/zh/en
  lines/0000.csv  meta/0000.csv  meta_sfx/0000.csv
```

`meta` schema: `source,id,text,vertical,font_path` (font path relative to `FONTS_ROOT`). Each worker seeds RNGs from `crc32(id)`. Details: `generator.md`.

## Python one-liner path

`Renderer().render(lines, override)` → BGRA/BGR `uint8` array; `SyntheticDataGenerator().process(text, override, sfx, lang)` → `(img, text_gt, params)`. Details: `renderer.md`, `generator.md`, `api.md`.
