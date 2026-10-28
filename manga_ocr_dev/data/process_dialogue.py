"""Process Japanese dialogue datasets into line packages for synthetic manga OCR data generation.

Extracts cleaned dialogue lines from:
- Nan-Do/OpenSubtitlesJapanese (OpenSubtitles.parquet: one row per subtitle block)
- Optional novel sources or mixed dialogue/novel streams via `--source_type`.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import fire

# Ensure manga_ocr_dev root is on sys.path if run as a standalone script
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manga_ocr_dev.data.process_text import (
    DEFAULT_DIALOGUE_DATA_FILE,
    DEFAULT_DIALOGUE_DATASET,
    DEFAULT_NOVEL_CONFIGS,
    DEFAULT_NOVEL_DATASET,
    export_lines,
)


def export_dialogue(
    num_lines_in_each_package: int = 100_000,
    num_packages: int = 10,
    start_package: int = 0,
    source_type: str = "dialogue",
    dialogue_data_file: str = DEFAULT_DIALOGUE_DATA_FILE,
    novel_configs: Sequence[str] = DEFAULT_NOVEL_CONFIGS,
    dialogue_ratio: float = 0.5,
    streaming: bool = True,
    dialogue_dataset_name: str = DEFAULT_DIALOGUE_DATASET,
    novel_dataset_name: str = DEFAULT_NOVEL_DATASET,
    dialogue_data_dir: str | None = None,
    novel_data_dir: str | None = None,
    out_dir: str | Path | None = None,
    min_len: int = 2,
    max_len: int = 200,
    dedup_window: int = 50_000,
    seed: int = 42,
    lang: str | None = None,
    strip_leading_quotes: bool = True,
) -> list[Path]:
    """Export dialogue lines into package CSV files for synthetic data generation.

    :param lang: Keep only lines of this language ('ja', 'ko', 'zh', 'en').
        None (default) keeps every line.
    """
    return export_lines(
        num_lines_in_each_package=num_lines_in_each_package,
        num_packages=num_packages,
        start_package=start_package,
        source_type=source_type,
        dialogue_ratio=dialogue_ratio,
        dialogue_data_file=dialogue_data_file,
        novel_configs=novel_configs,
        streaming=streaming,
        dialogue_dataset_name=dialogue_dataset_name,
        novel_dataset_name=novel_dataset_name,
        dialogue_data_dir=dialogue_data_dir,
        novel_data_dir=novel_data_dir,
        out_dir=out_dir,
        min_len=min_len,
        max_len=max_len,
        dedup_window=dedup_window,
        seed=seed,
        lang=lang,
        strip_leading_quotes=strip_leading_quotes,
    )


def main():
    fire.Fire(export_dialogue)


if __name__ == "__main__":
    main()
