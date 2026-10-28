# Text corpus (`manga_ocr_dev/data/`)

Package init re-exports the public surface (`manga_ocr_dev/data/__init__.py:1-25`): `export_dialogue, export_lines, clean_dialogue_line, clean_novel_text, detect_line_lang, iter_dialogue/novel/mixed_stream, split_novel_chapter`.

## `process_text.py` (623 lines) — canonical exporter

`fire.Fire(export_lines)`; all flags map 1:1 to `export_lines(...)` params.

Key API (`process_text.py`):

- `clean_dialogue_line(text) -> str` — strips ASS override blocks `{...}`, HTML tags, speaker prefixes `Name:`, music notes `♪♫`, leading/trailing quotes/brackets; normalizes whitespace.
- `clean_novel_text(text) -> str` — strips Aozora Bunko ruby `《...》`/`［...］` notes, HTML, headers/footers.
- `split_novel_chapter(text, min_len=2, max_len=150, lang=None) -> list[str]` — quote-aware splitter: CJK on `。！？…` vs English on `.!?`+space; clamps length.
- `class RollingDedup(window_size=50000): is_duplicate(text)` — rolling hash window.
- `iter_dialogue_split(dataset, split, streaming=True, min_len=2, max_len=200)` — yields `{source,id,line}` from one HF split.
- `iter_novel_config(dataset, config, ...)` — yields from one novel config (`main_text`/`text` + preface/afterword).
- `iter_dialogue_stream(...)`, `iter_novel_stream(...)`, `iter_mixed_stream(dialogue_iter, novel_iter, dialogue_ratio=0.5, seed=42)` — streaming combinators.
- `export_lines(num_lines_in_each_package=100000, num_packages=10, start_package=0, source_type="both", dialogue_ratio=0.5, ..., out_dir=None, min_len=2, max_len=150, dedup_window=50000, seed=42, lang=None) -> list[Path]`

Inputs:

- Dialogue default `JustANormalTinkerer/Japanese-Dialogue-Dataset`, splits `anime_tv, anime_movie, drama_tv, drama_movie, unsorted` (`dialogue` field).
- Novels default `OmniAICreator/Japanese-Novels-23M`, configs `Narou_NSFW, Kakuyomu, Narou_SFW, Hameln, aozorabunko, novecomi, pixiv`.
- Or local files via `dialogue_data_dir/novel_data_dir` glob/dir (`parquet/csv/jsonl/txt`) resolved by `_resolve_data_files()`.

Outputs: `out_dir` (default `DATA_SYNTHETIC_ROOT/lines`) `0000.csv…` with columns `source,id,line`. Optional `lang` filter via `detect_line_lang()` wrapper around `lang.detect_lang`.

CLI:

```bash
python -m manga_ocr_dev.data.process_text --num_packages=10 --num_lines_in_each_package=100000
python -m manga_ocr_dev.data.process_text --source_type=novel --num_packages=10
python -m manga_ocr_dev.data.process_text --source_type=dialogue --dialogue_ratio 0.5
```

## `process_dialogue.py` (82 lines) — dialogue convenience wrapper

- `export_dialogue(...)` — identical signature to `export_lines` but defaults `source_type="dialogue", max_len=200`; forwards everything unchanged. `fire.Fire(export_dialogue)`.
- Exists so `python -m manga_ocr_dev.data.process_dialogue --num_packages=10` needs no `source_type` flag.

## `process_cc100.py` (85 lines) — CC-100 slicer for ko/zh/en

- `CC100_FILES = {"ja": ["ja.txt"], "ko": ["ko.txt"], "zh": ["zh-Hans.txt", ...], "en": ["en.txt"]}`
- `export_lines(num_lines_in_each_package=100000, num_packages=10, lang="ja", out_dir=None)` — reads first existing `DATA_SYNTHETIC_ROOT/<candidate>`, skips `len<=2`, sequential `id_count` across packages, writes `lines/` for `ja` else `lines_<lang>/` (matches `run_generate --lang` convention) with `source=cc-100-<lang>`. No cleaning/dedup/lang-detect. `fire.Fire(export_lines)`.

## `process_manga109s.py` (110 lines) — Manga109-s annotation export

No CLI (`if __name__ == "__main__": export_frames(); export_crops()`).

- `get_books() -> DataFrame[book,annotations,images]` — parses `Manga109s_released_2021_02_28/books.txt`.
- `export_frames()` → `MANGA109_ROOT/frames.csv` (`book,page_index,page_path,page_width/height,id,xmin/ymin/xmax/ymax`) from `<frame>` XML nodes.
- `export_crops()` → `MANGA109_ROOT/data.csv` (same + `text,split,crop_path`; random 10% `test`) + `MANGA109_ROOT/crops/<id>.png` (10px margin, `cv2.imwrite`) from `<text>` nodes. Paths repo-relative. Foundation for eval crops and `generate_backgrounds.py`.
