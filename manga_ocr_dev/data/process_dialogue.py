"""
Process dialogue dataset (https://huggingface.co/datasets/JustANormalTinkerer/Japanese-Dialogue-Dataset)
into line packages for synthetic manga OCR data generation.

Splits: anime_tv, anime_movie, drama_tv, drama_movie, unsorted.
Can also process novel sources or mix with novels via --source_type=both or novel options.
"""

import sys
from collections.abc import Sequence
from pathlib import Path

import fire

# Ensure manga_ocr_dev can be imported
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manga_ocr_dev.data.process_text import (
    DEFAULT_DIALOGUE_DATASET,
    DEFAULT_DIALOGUE_SPLITS,
    DEFAULT_NOVEL_CONFIGS,
    DEFAULT_NOVEL_DATASET,
    export_lines,
)


def export_dialogue(
    num_lines_in_each_package: int = 100_000,
    num_packages: int = 10,
    start_package: int = 0,
    source_type: str = "dialogue",
    dialogue_splits: Sequence[str] = DEFAULT_DIALOGUE_SPLITS,
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
):
    """Export dialogue lines from JustANormalTinkerer/Japanese-Dialogue-Dataset

    into package CSV files for the synthetic generator.
    """
    return export_lines(
        num_lines_in_each_package=num_lines_in_each_package,
        num_packages=num_packages,
        start_package=start_package,
        source_type=source_type,
        dialogue_ratio=dialogue_ratio,
        dialogue_splits=dialogue_splits,
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
    )


def main():
    fire.Fire(export_dialogue)


if __name__ == "__main__":
    main()
