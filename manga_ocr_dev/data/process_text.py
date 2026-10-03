"""
Process dialogue and novel datasets into line packages for synthetic manga OCR data generation.

Supported datasets:
1. Japanese Dialogue Dataset (https://huggingface.co/datasets/JustANormalTinkerer/Japanese-Dialogue-Dataset)
   - Splits: anime_tv, anime_movie, drama_tv, drama_movie, unsorted
   - Feature: dialogue (subtitles/dialogues from anime and drama; includes Japanese, English, and mixed lines)
2. Japanese Novels 23M (https://huggingface.co/datasets/OmniAICreator/Japanese-Novels-23M)
   - Configs: Hameln, Kakuyomu, Narou_NSFW, Narou_SFW, aozorabunko, novecomi, pixiv
   - Features: main_text (Narou_NSFW, Narou_SFW) or text (Hameln, Kakuyomu, aozorabunko, novecomi, pixiv)
   - Explicitly supports and includes the NSFW subset (Narou_NSFW).
3. Local file sources (parquet, csv, jsonl, txt).
"""

import gc
import glob
import random
import re
import sys
from collections import deque
from collections.abc import Generator, Iterator, Sequence
from pathlib import Path

import fire
import pandas as pd
from tqdm import tqdm

# Ensure manga_ocr_dev can be imported even when run as a standalone script
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manga_ocr_dev.env import DATA_SYNTHETIC_ROOT

DEFAULT_DIALOGUE_DATASET = "JustANormalTinkerer/Japanese-Dialogue-Dataset"
DEFAULT_DIALOGUE_SPLITS = ("anime_tv", "anime_movie", "drama_tv", "drama_movie", "unsorted")

DEFAULT_NOVEL_DATASET = "OmniAICreator/Japanese-Novels-23M"
DEFAULT_NOVEL_CONFIGS = (
    "Narou_NSFW",
    "Kakuyomu",
    "Narou_SFW",
    "Hameln",
    "aozorabunko",
    "novecomi",
    "pixiv",
)

# Regex patterns for subtitle and novel cleaning
RE_ASS_NEWLINE = re.compile(r"\\+[Nnh]")
RE_ASS_TAGS = re.compile(r"\{[^\}]*\}")
RE_HTML_TAGS = re.compile(r"<[^>]+>")
RE_SPEAKER_PREFIX = re.compile(r"^[（(\[][^）)\]]+[）)\]]\s*(?=\S)")
RE_MUSIC_NOTES = re.compile(r"[♪♫♬♩]+")
RE_ASS_CODE = re.compile(r"^!(?:retime|maxloop|template)|math\.(?:floor|ceil|random)")
RE_AOZORA_RUBY = re.compile(r"｜?([^《\r\n]+)《[^》\r\n]+》")
RE_AOZORA_NOTES = re.compile(r"［＃[^］\r\n]+］")
RE_SEPARATOR_LINE = re.compile(r"^[=\-*_~#・]{3,}$")
RE_SENTENCE_DELIM = re.compile(r"(?<=[。！？!?])")
RE_VALID_CHAR = re.compile(r"[\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")
RE_WHITESPACE = re.compile(r"\s+")


def clean_dialogue_line(text: str | None) -> str:
    """Clean a subtitle/dialogue line by removing ASS styling, HTML tags, speaker prefixes,

    and outer quotation marks while preserving valid text (Japanese, English, punctuation).
    """
    if not text or not isinstance(text, str):
        return ""

    # Filter out ASS karaoke/script code artifacts
    if RE_ASS_CODE.search(text):
        return ""

    # Replace ASS line break markup with space
    text = RE_ASS_NEWLINE.sub(" ", text)
    # Remove ASS override tags
    text = RE_ASS_TAGS.sub("", text)
    # Remove HTML formatting
    text = RE_HTML_TAGS.sub("", text)
    # Remove leading speaker tag if followed by actual text, e.g. （ジェイ）私から見ても -> 私から見ても
    text = RE_SPEAKER_PREFIX.sub("", text)
    # Remove music notes
    text = RE_MUSIC_NOTES.sub("", text)

    # Strip matching outer quotes (straight or curly)
    text = text.strip()
    if len(text) >= 2 and (text[0] in ('"', "'", "“", "‘") and text[-1] in ('"', "'", "”", "’")):
        text = text[1:-1].strip()

    # Normalize full-width spaces and collapse redundant spaces
    text = text.replace("\u3000", " ")
    text = RE_WHITESPACE.sub(" ", text).strip()
    return text


def clean_novel_text(text: str | None) -> str:
    """Clean novel chapter text by stripping Aozora Bunko ruby text, editorial notes,

    and formatting tags while preserving narrative and dialogue content.
    """
    if not text or not isinstance(text, str):
        return ""

    # Strip Aozora Bunko ruby text: 漢字《かんじ》 -> 漢字
    text = RE_AOZORA_RUBY.sub(r"\1", text)
    # Remove Aozora Bunko editorial notes: ［＃...］
    text = RE_AOZORA_NOTES.sub("", text)
    # Remove HTML tags if present
    text = RE_HTML_TAGS.sub("", text)
    return text


def split_novel_chapter(
    text: str | None,
    min_len: int = 2,
    max_len: int = 150,
) -> list[str]:
    """Split a full novel chapter into individual dialogue lines and sentences suitable

    for speech bubble and text block rendering in synthetic manga data.
    """
    text = clean_novel_text(text)
    if not text:
        return []

    lines: list[str] = []
    for raw_para in text.splitlines():
        para = raw_para.strip().strip("\u3000")
        if not para:
            continue
        # Skip web novel horizontal rule separators
        if RE_SEPARATOR_LINE.match(para):
            continue

        # If line contains dialogue quotes 「...」 or 『...』, extract quotes & narrative
        if "「" in para or "『" in para:
            tokens = re.findall(r"「[^」]*」|『[^』]*』|[^「『]+", para)
            for tok in tokens:
                tok = tok.strip()
                if len(tok) >= min_len:
                    if len(tok) > max_len:
                        for s in RE_SENTENCE_DELIM.split(tok):
                            s = s.strip()
                            if len(s) >= min_len:
                                lines.append(s)
                    else:
                        lines.append(tok)
        elif len(para) > max_len:
            # Long narrative paragraph: split on sentence boundaries
            for s in RE_SENTENCE_DELIM.split(para):
                s = s.strip()
                if len(s) >= min_len:
                    lines.append(s)
        else:
            if len(para) >= min_len:
                lines.append(para)

    # Filter lines that have at least one meaningful character (letters, kana, kanji, digits)
    valid_lines: list[str] = []
    for l in lines:
        l = RE_WHITESPACE.sub(" ", l).strip()
        if len(l) >= min_len and RE_VALID_CHAR.search(l):
            valid_lines.append(l)

    return valid_lines


def _resolve_data_files(path_or_glob: str) -> list[str]:
    """Resolve file paths from a path or glob expression."""
    p = Path(path_or_glob)
    if p.is_dir():
        files = sorted(str(f) for f in p.glob("**/*") if f.is_file() and f.suffix.lower() in {".parquet", ".csv", ".jsonl", ".txt"})
        if not files:
            files = sorted(str(f) for f in p.iterdir() if f.is_file())
        return files
    expanded = sorted(glob.glob(path_or_glob, recursive=True))
    if expanded:
        return expanded
    if p.is_file():
        return [str(p)]
    return []


class RollingDedup:
    """Rolling window de-duplicator to prevent repeating identical lines within a package."""

    def __init__(self, window_size: int = 50_000):
        self.window_size = window_size
        self.seen: set[str] = set()
        self.queue: deque = deque()

    def is_duplicate(self, text: str) -> bool:
        if self.window_size <= 0:
            return False
        if text in self.seen:
            return True
        self.seen.add(text)
        self.queue.append(text)
        if len(self.queue) > self.window_size:
            oldest = self.queue.popleft()
            self.seen.discard(oldest)
        return False


def iter_dialogue_split(
    dataset_name: str,
    split: str,
    streaming: bool = True,
    min_len: int = 2,
    max_len: int = 200,
) -> Generator[dict[str, str], None, None]:
    """Stream lines from a single dialogue split with minimal memory footprint."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    id_cnt = 0
    for row in ds:
        raw_dialogue = row.get("dialogue") or row.get("text") or row.get("line")
        if not raw_dialogue or not isinstance(raw_dialogue, str):
            continue

        for line in RE_ASS_NEWLINE.sub("\n", raw_dialogue).splitlines():
            cleaned = clean_dialogue_line(line)
            if len(cleaned) < min_len or len(cleaned) > max_len:
                continue
            if not RE_VALID_CHAR.search(cleaned):
                continue

            id_cnt += 1
            yield {
                "source": f"dialogue_{split}",
                "id": f"dialogue_{split}_{id_cnt}",
                "line": cleaned,
            }


def iter_novel_config(
    dataset_name: str,
    config: str,
    streaming: bool = True,
    min_len: int = 2,
    max_len: int = 150,
) -> Generator[dict[str, str], None, None]:
    """Stream lines from a single novel config with minimal memory footprint."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, config, split="train", streaming=streaming)
    id_cnt = 0
    for row in ds:
        primary_text = row.get("main_text") or row.get("text")
        extra_texts = []
        for extra_field in ("preface", "afterword", "maegaki", "atogaki", "synopsis"):
            val = row.get(extra_field)
            if val and isinstance(val, str) and len(val) >= min_len:
                extra_texts.append(val)

        all_texts = ([primary_text] if primary_text and isinstance(primary_text, str) else []) + extra_texts

        for t in all_texts:
            for line in split_novel_chapter(t, min_len=min_len, max_len=max_len):
                id_cnt += 1
                yield {
                    "source": f"novel_{config}",
                    "id": f"novel_{config}_{id_cnt}",
                    "line": line,
                }


def iter_dialogue_stream(
    dataset_name: str = DEFAULT_DIALOGUE_DATASET,
    splits: Sequence[str] = DEFAULT_DIALOGUE_SPLITS,
    streaming: bool = True,
    data_files: str | Sequence[str] | None = None,
    min_len: int = 2,
    max_len: int = 200,
) -> Generator[dict[str, str], None, None]:
    """Stream dialogue lines across splits or local files with low memory usage."""
    from datasets import load_dataset

    if isinstance(splits, str):
        splits = [s.strip() for s in splits.split(",") if s.strip()]

    if data_files:
        if isinstance(data_files, str):
            resolved = _resolve_data_files(data_files)
        else:
            resolved = []
            for df in data_files:
                resolved.extend(_resolve_data_files(df))

        if not resolved:
            raise FileNotFoundError(f"No dialogue data files found for {data_files}")

        ext = Path(resolved[0]).suffix.lower()
        if ext == ".parquet":
            ds = load_dataset("parquet", data_files=resolved, split="train", streaming=streaming)
        elif ext in {".csv", ".tsv"}:
            ds = load_dataset("csv", data_files=resolved, split="train", streaming=streaming)
        elif ext in {".json", ".jsonl"}:
            ds = load_dataset("json", data_files=resolved, split="train", streaming=streaming)
        elif ext in {".txt", ".text"}:
            ds = load_dataset("text", data_files=resolved, split="train", streaming=streaming)
        else:
            ds = load_dataset("parquet", data_files=resolved, split="train", streaming=streaming)

        id_cnt = 0
        for row in ds:
            raw_dialogue = row.get("dialogue") or row.get("text") or row.get("line")
            if not raw_dialogue or not isinstance(raw_dialogue, str):
                continue
            for line in RE_ASS_NEWLINE.sub("\n", raw_dialogue).splitlines():
                cleaned = clean_dialogue_line(line)
                if len(cleaned) < min_len or len(cleaned) > max_len:
                    continue
                if not RE_VALID_CHAR.search(cleaned):
                    continue
                id_cnt += 1
                yield {"source": "dialogue_local", "id": f"dialogue_local_{id_cnt}", "line": cleaned}
        return

    # Cycle sequentially across splits to keep at most one split stream open at a time
    for split in splits:
        try:
            yield from iter_dialogue_split(
                dataset_name=dataset_name,
                split=split,
                streaming=streaming,
                min_len=min_len,
                max_len=max_len,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Warning: split '{split}' encountered error: {e}", file=sys.stderr)
        gc.collect()


def iter_novel_stream(
    dataset_name: str = DEFAULT_NOVEL_DATASET,
    configs: Sequence[str] = DEFAULT_NOVEL_CONFIGS,
    streaming: bool = True,
    data_files: str | Sequence[str] | None = None,
    min_len: int = 2,
    max_len: int = 150,
) -> Generator[dict[str, str], None, None]:
    """Stream novel lines across configs or local files with low memory usage."""
    from datasets import load_dataset

    if isinstance(configs, str):
        configs = [c.strip() for c in configs.split(",") if c.strip()]

    if data_files:
        if isinstance(data_files, str):
            resolved = _resolve_data_files(data_files)
        else:
            resolved = []
            for df in data_files:
                resolved.extend(_resolve_data_files(df))

        if not resolved:
            raise FileNotFoundError(f"No novel data files found for {data_files}")

        ext = Path(resolved[0]).suffix.lower()
        if ext == ".parquet":
            ds = load_dataset("parquet", data_files=resolved, split="train", streaming=streaming)
        elif ext in {".json", ".jsonl"}:
            ds = load_dataset("json", data_files=resolved, split="train", streaming=streaming)
        elif ext in {".txt", ".text"}:
            ds = load_dataset("text", data_files=resolved, split="train", streaming=streaming)
        else:
            ds = load_dataset("parquet", data_files=resolved, split="train", streaming=streaming)

        id_cnt = 0
        for row in ds:
            primary_text = row.get("main_text") or row.get("text")
            if primary_text and isinstance(primary_text, str):
                for line in split_novel_chapter(primary_text, min_len=min_len, max_len=max_len):
                    id_cnt += 1
                    yield {"source": "novel_local", "id": f"novel_local_{id_cnt}", "line": line}
        return

    # Cycle sequentially across configs (Hameln, Kakuyomu, Narou_NSFW, etc.)
    for cfg in configs:
        try:
            yield from iter_novel_config(
                dataset_name=dataset_name,
                config=cfg,
                streaming=streaming,
                min_len=min_len,
                max_len=max_len,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Warning: config '{cfg}' encountered error: {e}", file=sys.stderr)
        gc.collect()


def iter_mixed_stream(
    dialogue_iter: Iterator[dict[str, str]],
    novel_iter: Iterator[dict[str, str]],
    dialogue_ratio: float = 0.5,
    seed: int = 42,
) -> Generator[dict[str, str], None, None]:
    """Interleave lines from dialogue stream and novel stream according to dialogue_ratio."""
    rng = random.Random(seed)
    dialogue_alive = True
    novel_alive = True

    while dialogue_alive or novel_alive:
        choose_dialogue = rng.random() < dialogue_ratio

        if choose_dialogue and dialogue_alive:
            try:
                yield next(dialogue_iter)
            except StopIteration:
                dialogue_alive = False
        elif novel_alive:
            try:
                yield next(novel_iter)
            except StopIteration:
                novel_alive = False
        elif dialogue_alive:
            try:
                yield next(dialogue_iter)
            except StopIteration:
                dialogue_alive = False


def export_lines(
    num_lines_in_each_package: int = 100_000,
    num_packages: int = 10,
    start_package: int = 0,
    source_type: str = "both",
    dialogue_ratio: float = 0.5,
    dialogue_splits: Sequence[str] = DEFAULT_DIALOGUE_SPLITS,
    novel_configs: Sequence[str] = DEFAULT_NOVEL_CONFIGS,
    streaming: bool = True,
    dialogue_dataset_name: str = DEFAULT_DIALOGUE_DATASET,
    novel_dataset_name: str = DEFAULT_NOVEL_DATASET,
    dialogue_data_dir: str | None = None,
    novel_data_dir: str | None = None,
    out_dir: str | Path | None = None,
    min_len: int = 2,
    max_len: int = 150,
    dedup_window: int = 50_000,
    seed: int = 42,
) -> list[Path]:
    """Export lines into packages for the synthetic data generator.

    :param num_lines_in_each_package: Number of text lines per package CSV. Default 100,000.
    :param num_packages: Total number of packages to generate. Default 10.
    :param start_package: Index of first package (e.g. 0 creates 0000.csv). Default 0.
    :param source_type: Data source to use: 'both' (mixed), 'dialogue', or 'novel' / 'novels'.
    :param dialogue_ratio: Proportion of dialogue lines when source_type='both'. Default 0.5.
    :param dialogue_splits: Splits to use from Japanese-Dialogue-Dataset.
        Default: ('anime_tv', 'anime_movie', 'drama_tv', 'drama_movie', 'unsorted').
    :param novel_configs: Configs to use from Japanese-Novels-23M.
        Default: ('Narou_NSFW', 'Kakuyomu', 'Narou_SFW', 'Hameln', 'aozorabunko', 'novecomi', 'pixiv').
        (Includes Narou_NSFW as requested).
    :param streaming: Whether to stream data from HuggingFace without downloading the whole corpus.
    :param dialogue_dataset_name: HuggingFace dataset name for dialogues.
    :param novel_dataset_name: HuggingFace dataset name for novels.
    :param dialogue_data_dir: Optional local path/glob to dialogue files (parquet/csv/txt).
    :param novel_data_dir: Optional local path/glob to novel files (parquet/csv/txt).
    :param out_dir: Output directory for packages. Defaults to DATA_SYNTHETIC_ROOT / 'lines'.
    :param min_len: Minimum line length in characters. Default 2.
    :param max_len: Maximum line length in characters. Default 150.
    :param dedup_window: Size of rolling window for skipping exact duplicate lines. Default 50,000.
    :param seed: Random seed for stream mixing.
    :return: List of created CSV file paths.
    """
    if out_dir is None:
        out_dir = DATA_SYNTHETIC_ROOT / "lines"
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_type = source_type.lower()
    if source_type in {"dialogue", "subtitles"}:
        stream = iter_dialogue_stream(
            dataset_name=dialogue_dataset_name,
            splits=dialogue_splits,
            streaming=streaming,
            data_files=dialogue_data_dir,
            min_len=min_len,
            max_len=max_len,
        )
    elif source_type in {"novel", "novels"}:
        stream = iter_novel_stream(
            dataset_name=novel_dataset_name,
            configs=novel_configs,
            streaming=streaming,
            data_files=novel_data_dir,
            min_len=min_len,
            max_len=max_len,
        )
    elif source_type in {"both", "mixed", "all"}:
        d_stream = iter_dialogue_stream(
            dataset_name=dialogue_dataset_name,
            splits=dialogue_splits,
            streaming=streaming,
            data_files=dialogue_data_dir,
            min_len=min_len,
            max_len=max_len,
        )
        n_stream = iter_novel_stream(
            dataset_name=novel_dataset_name,
            configs=novel_configs,
            streaming=streaming,
            data_files=novel_data_dir,
            min_len=min_len,
            max_len=max_len,
        )
        stream = iter_mixed_stream(
            dialogue_iter=d_stream,
            novel_iter=n_stream,
            dialogue_ratio=dialogue_ratio,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown source_type '{source_type}'. Choose from 'both', 'dialogue', 'novel'.")

    dedup = RollingDedup(window_size=dedup_window)
    created_files: list[Path] = []

    print(
        f"Exporting {num_packages} package(s) ({num_lines_in_each_package} lines each) "
        f"starting from package {start_package:04d} -> {out_dir}"
    )

    for package_idx in range(start_package, start_package + num_packages):
        package_str = f"{package_idx:04d}"
        package_file = out_dir / f"{package_str}.csv"

        data: list[dict[str, str]] = []
        pbar = tqdm(
            total=num_lines_in_each_package,
            desc=f"Package {package_str} ({package_idx - start_package + 1}/{num_packages})",
        )

        while len(data) < num_lines_in_each_package:
            try:
                item = next(stream)
            except StopIteration:
                print(f"Stream exhausted at package {package_str} ({len(data)} lines collected).")
                break

            line_text = item["line"]
            if dedup.is_duplicate(line_text):
                continue

            data.append(item)
            pbar.update(1)

        pbar.close()

        if not data:
            print(f"No data available for package {package_str}. Stopping.")
            break

        df = pd.DataFrame(data, columns=["source", "id", "line"])
        df.to_csv(package_file, index=False)
        print(f"Saved {len(df)} lines to {package_file}")
        created_files.append(package_file)
        gc.collect()

    return created_files


def main():
    fire.Fire(export_lines)


if __name__ == "__main__":
    main()
