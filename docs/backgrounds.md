# Backgrounds (`manga_ocr_dev/data/generate_backgrounds*.py`)

Both scripts extract text-free crops for compositing behind synthetic text. Filenames encode the crop box: `<src>_<ymin>_<ymax>_<xmin>_<xmax>.png`, or with `manifest.csv` for AnimeText (consumed by `utils.get_background_df`).

Shared algorithm: pick a random unmasked seed pixel inside a frame, greedily expand a rectangle while it stays text-free and within `aspect_ratio_range=(0.33, 3.0)` and `>= min_size` px.

## `generate_backgrounds_animetext.py` (118 lines) — recommended

- `yolo_boxes_to_mask(h, w, bboxes, dilation=3) -> bool mask` — converts normalized `xc,yc,w,h` boxes to a dilated (`3×3`) occupancy mask. Both category 0 (text) and 1 (hard_negative) are masked.
- `generate_backgrounds_animetext(n_images=500, crops_per_image=5, min_size=40, out_dir=None, split="train", seed=0, dilation=3) -> {images_seen, kept, keep_rate}` — streams HF `deepghs/AnimeText:<split>` (`image` PIL + `objects.bbox`), reuses `find_rectangle` from `generate_backgrounds.py` for expansion, `RandomState(seed)` sampling.
- `fire.Fire(generate_backgrounds_animetext)`; `os._exit(0)` after run.

Inputs: streaming AnimeText (no full download). Outputs: `BACKGROUND_ANIMETEXT_DIR/<image_id>_<ymin>_<ymax>_<xmin>_<xmax>.png` + `manifest.csv[source,image_id,ymin/ymax/xmin/xmax,h,w,n_boxes,path]`.

```bash
python -m manga_ocr_dev.data.generate_backgrounds_animetext --n_images 10000
python -m manga_ocr_dev.data.generate_backgrounds_animetext --n_images 500 --out_dir ~/data/manga/AnimeText_background --split train
```

## `generate_backgrounds.py` (107 lines) — Manga109-s variant

No CLI (edit constants or call programmatically).

- `find_rectangle(mask, y, x, aspect_ratio_range=(0.33,3.0))` — greedy expansion from seed `(y,x)`; also imported by the AnimeText script.
- `generate_backgrounds(crops_per_page=5, min_size=40)` — needs `MANGA109_ROOT/data.csv` (text mask) + `frames.csv` (panel mask) + page JPEGs. Mask = `text_mask | ~frames_mask`; skips fully-masked pages; up to `crops_per_page` rectangles per page.

Inputs: `process_manga109s.py` outputs under `MANGA109_ROOT`. Outputs: `BACKGROUND_DIR/<book>_<page>_<ymin>_<ymax>_<xmin>_<xmax>.png`.

## Consumption (`renderer.py:get_default_background_df`, `utils.py:get_background_df`)

`get_default_background_df()` (`renderer.py:89-97`) prefers `BACKGROUND_ANIMETEXT_DIR`, falls back to `BACKGROUND_DIR`. `get_background_df(dir)` reads `manifest.csv` if present else parses `*_ymin_ymax_xmin_xmax.ext` filenames → `DataFrame[path,h,w,ratio]`.
