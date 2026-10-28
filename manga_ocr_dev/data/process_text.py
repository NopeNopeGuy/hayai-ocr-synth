"""
Process dialogue and novel datasets into line packages for synthetic manga OCR data generation.

Supported datasets:
1. OpenSubtitles Japanese (https://huggingface.co/datasets/Nan-Do/OpenSubtitlesJapanese)
   - File: OpenSubtitles.parquet (the metadata/OA parquet files are not used)
   - Columns: ID, BlockNum, InitTime, EndTime, Sentences (list of subtitle lines per block)
   - Each element of ``Sentences`` is one subtitle line; a leading ``- `` speaker dash is removed.
2. Japanese Novels 23M (https://huggingface.co/datasets/OmniAICreator/Japanese-Novels-23M)
   - Configs: Hameln, Kakuyomu, Narou_NSFW, Narou_SFW, aozorabunko, novecomi, pixiv
   - Features: main_text (Narou_NSFW, Narou_SFW) or text (Hameln, Kakuyomu, aozorabunko, novecomi, pixiv)
   - Supports web novel subsets including Narou_NSFW.
3. Local file sources (parquet, csv, jsonl, txt).
"""

from __future__ import annotations

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

DEFAULT_DIALOGUE_DATASET = "Nan-Do/OpenSubtitlesJapanese"
DEFAULT_DIALOGUE_DATA_FILE = "OpenSubtitles-2023-04-19.parquet"
# Column names that may hold the subtitle lines. "Sentences" is the OpenSubtitlesJapanese
# column; the rest are fallbacks for local files exported from older dialogue datasets.
SUBTITLE_TEXT_FIELDS = ("Sentences", "sentences", "dialogue", "text", "line")

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
RE_SENTENCE_DELIM_EN = re.compile(r"(?<=[.!?…])\s+")
# Explicit Hangul coverage (syllables, Jamo, compat letters); \w already
# matches most of these, but multilingual filtering should not depend on
# regex Unicode-word semantics.
RE_VALID_CHAR = re.compile(r"[\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]")
RE_WHITESPACE = re.compile(r"\s+")
# Leading speaker-change dash used in subtitles ("- 試して"). Not stripped when the dash is
# part of a number or word, e.g. "-5度" keeps its minus sign.
RE_SUBTITLE_DASH = re.compile(r"^[-－–—]+(?:\s+|(?=[^\d\s]))")
# Japanese quotations. re.split with this capturing pattern yields narrative and quote
# parts alternately, so quote parts are always at odd indexes.
RE_QUOTE_SPLIT = re.compile(r"(「[^「」]*」|『[^『』]*』)")
_OPEN_TO_CLOSE_BRACKET = {"「": "」", "『": "』"}
_CLOSE_TO_OPEN_BRACKET = {v: k for k, v in _OPEN_TO_CLOSE_BRACKET.items()}


def detect_line_lang(text: str) -> str:
    """Script-based language guess (``ja``/``ko``/``zh``/``en``) for one line."""
    from manga_ocr_dev.synthetic_data_generator.lang import detect_lang

    return detect_lang(text)


def balance_quote_brackets(text: str) -> str:
    """Remove 「」 / 『』 brackets that have no partner inside the same line.

    Subtitle blocks and novel paragraphs often split one quotation across two lines, so a
    line can start with a lone 「 or end with a lone 」. Matched pairs are kept as they are;
    unmatched brackets are dropped so no training line has a dangling bracket.
    """
    if not any(ch in text for ch in "「」『』"):
        return text

    open_stack: list[int] = []  # indexes of brackets still waiting for a closer
    unmatched: set[int] = set()
    for i, ch in enumerate(text):
        if ch in _OPEN_TO_CLOSE_BRACKET:
            open_stack.append(i)
        elif ch in _CLOSE_TO_OPEN_BRACKET:
            if open_stack and text[open_stack[-1]] == _CLOSE_TO_OPEN_BRACKET[ch]:
                open_stack.pop()
            else:
                unmatched.add(i)
    unmatched.update(open_stack)
    if not unmatched:
        return text
    return "".join(ch for i, ch in enumerate(text) if i not in unmatched)


def _find_partner_closer(text: str, start: int) -> int | None:
    """Index of the bracket that closes the opener at ``start``, or None if it never closes."""
    stack: list[str] = []
    for i in range(start, len(text)):
        ch = text[i]
        if ch in _OPEN_TO_CLOSE_BRACKET:
            stack.append(ch)
        elif ch in _CLOSE_TO_OPEN_BRACKET and stack and stack[-1] == _CLOSE_TO_OPEN_BRACKET[ch]:
            stack.pop()
            if not stack:
                return i
    return None


def strip_leading_quote(text: str) -> str:
    """Remove 「」/『』 that open a line, together with the bracket that closes them.

    Manga speech bubbles show the spoken text without outer quotation brackets, so
    ``「やあ」と彼は言った`` becomes ``やあと彼は言った``. Quotes inside a sentence,
    such as ``彼は「やあ」と言った``, are not at the start of the line and are kept.
    """
    while text and text[0] in _OPEN_TO_CLOSE_BRACKET:
        partner = _find_partner_closer(text, 0)
        drop = {0} if partner is None else {0, partner}
        text = "".join(ch for i, ch in enumerate(text) if i not in drop).strip()
    return balance_quote_brackets(text).strip()


def _split_sentences_outside_quotes(para: str) -> list[str]:
    """Split a paragraph after 。！？ only where it is not inside a balanced 「」/『』 quote.

    A stray unclosed bracket does not block splitting, because only balanced quotations
    (as matched by RE_QUOTE_SPLIT) are protected.
    """
    protected = [m.span() for m in RE_QUOTE_SPLIT.finditer(para)]

    def inside_quote(i: int) -> bool:
        return any(start <= i < end for start, end in protected)

    pieces: list[str] = []
    start = 0
    for i, ch in enumerate(para):
        if ch in "。！？!?" and not inside_quote(i):
            pieces.append(para[start : i + 1])
            start = i + 1
    if start < len(para):
        pieces.append(para[start:])
    return pieces


def _pack_quote_segments(text: str, max_len: int, min_len: int, sentence_split) -> list[str]:
    """Cut an over-long sentence only between quotation and narration segments.

    Segments are packed greedily into lines of at most ``max_len`` characters. A quotation
    that is itself too long is split by sentence, keeping its brackets balanced.
    """
    lines: list[str] = []
    current = ""
    for idx, seg in enumerate(RE_QUOTE_SPLIT.split(text)):
        if not seg:
            continue
        if idx % 2 == 1 and len(seg) > max_len:
            if current:
                lines.append(current)
                current = ""
            lines.extend(_split_long_quote(seg, min_len, sentence_split))
            continue
        if current and len(current) + len(seg) > max_len:
            lines.append(current)
            current = ""
        current += seg
    if current:
        lines.append(current)
    return [l.strip() for l in lines if len(l.strip()) >= min_len]


def _split_long_quote(quote: str, min_len: int, sentence_split) -> list[str]:
    """Split an over-long 「…」 / 『…』 quotation into sentence-sized quotations.

    Each piece is wrapped in the original brackets, so every output line is a complete
    quotation: 「今日は晴れだ。明日は雨だ。」 -> 「今日は晴れだ。」 and 「明日は雨だ。」
    """
    open_bracket, close_bracket = quote[0], quote[-1]
    pieces: list[str] = []
    for sentence in sentence_split(quote[1:-1]):
        sentence = sentence.strip()
        if len(sentence) >= min_len:
            pieces.append(f"{open_bracket}{sentence}{close_bracket}")
    return pieces


def _as_text_list(value) -> list[str]:
    """Return the non-empty strings held by a dataset cell.

    A cell can be a plain string, a list of strings (OpenSubtitles ``Sentences``), or a
    numpy array of strings when data comes from pandas/arrow.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str) and v]
    return []


def _subtitle_entries(row: dict) -> list[str]:
    """Return the raw subtitle entries of one dataset row (one entry per subtitle line)."""
    for field in SUBTITLE_TEXT_FIELDS:
        entries = _as_text_list(row.get(field))
        if entries:
            return entries
    return []


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
    # Remove leading subtitle speaker dash, e.g. "- 試して" -> "試して"
    text = RE_SUBTITLE_DASH.sub("", text.strip())
    # Remove leading speaker tag if followed by actual text, e.g. （ジェイ）私から見ても -> 私から見ても
    text = RE_SPEAKER_PREFIX.sub("", text)
    # Remove music notes
    text = RE_MUSIC_NOTES.sub("", text)

    # Strip matching outer quotes (straight or curly)
    text = text.strip()
    if len(text) >= 2 and (text[0] in ('"', "'", "“", "‘") and text[-1] in ('"', "'", "”", "’")):
        text = text[1:-1].strip()

    # Subtitles often cut a quotation across blocks; drop the bracket that has no partner
    text = balance_quote_brackets(text)

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
    lang: str | None = None,
) -> list[str]:
    """Split a full novel chapter into individual dialogue lines and sentences suitable

    for speech bubble and text block rendering in synthetic manga data.
    ``lang="en"`` splits on ASCII sentence boundaries (``.``/``!``/``?`` +
    space); CJK scripts split on ``。！？!?``.
    """
    text = clean_novel_text(text)
    if not text:
        return []

    # English sentence splitting needs the trailing space kept as the boundary;
    # CJK scripts split on full-width stops.
    if lang == "en":
        sentence_split = lambda p: RE_SENTENCE_DELIM_EN.split(p)  # noqa: E731
    else:
        sentence_split = lambda p: RE_SENTENCE_DELIM.split(p)  # noqa: E731

    lines: list[str] = []
    for raw_para in text.splitlines():
        para = raw_para.strip().strip("\u3000")
        if not para:
            continue
        # Skip web novel horizontal rule separators
        if RE_SEPARATOR_LINE.match(para):
            continue

        # If line contains dialogue quotes 「...」/『...』("..." for en), extract quotes & narrative
        if "「" in para or "『" in para:
            # Quotations stay inside their sentence, so a short paragraph is one line, as a
            # paragraph without quotes is. Long paragraphs are cut only between sentences
            # and never inside a quotation.
            chunks = [para] if len(para) <= max_len else _split_sentences_outside_quotes(para)
            for chunk in chunks:
                chunk = chunk.strip()
                if len(chunk) < min_len:
                    continue
                if len(chunk) <= max_len:
                    lines.append(chunk)
                elif RE_QUOTE_SPLIT.fullmatch(chunk):
                    # A single over-long quotation: split it, keeping each piece wrapped
                    lines.extend(_split_long_quote(chunk, min_len, sentence_split))
                else:
                    lines.extend(_pack_quote_segments(chunk, max_len, min_len, sentence_split))
        elif lang == "en" and ('"' in para or '"' in para or "'" in para):
            tokens = re.findall(r'"[^"]*"|\'[^\']*\'|[^"\']+', para)
            for tok in tokens:
                tok = tok.strip()
                if len(tok) >= min_len:
                    if len(tok) > max_len:
                        for s in sentence_split(tok):
                            s = s.strip()
                            if len(s) >= min_len:
                                lines.append(s)
                    else:
                        lines.append(tok)
        elif len(para) > max_len:
            # Long narrative paragraph: split on sentence boundaries
            for s in sentence_split(para):
                s = s.strip()
                if len(s) >= min_len:
                    lines.append(s)
        else:
            if len(para) >= min_len:
                lines.append(para)

    # Filter lines that have at least one meaningful character (letters, kana, kanji, digits)
    valid_lines: list[str] = []
    for l in lines:
        # Narrative fragments can still carry a bracket whose partner is in another paragraph
        l = balance_quote_brackets(l)
        l = RE_WHITESPACE.sub(" ", l).strip()
        if len(l) >= min_len and RE_VALID_CHAR.search(l):
            valid_lines.append(l)

    return valid_lines


def _resolve_data_files(path_or_glob: str) -> list[str]:
    """Resolve file paths from a path or glob expression."""
    p = Path(path_or_glob)
    if p.is_dir():
        files = sorted(
            str(f) for f in p.glob("**/*") if f.is_file() and f.suffix.lower() in {".parquet", ".csv", ".jsonl", ".txt"}
        )
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


def iter_subtitle_lines(
    ds,
    source: str,
    min_len: int = 2,
    max_len: int = 200,
) -> Generator[dict[str, str], None, None]:
    """Yield one cleaned line per subtitle entry from an opened subtitle dataset.

    Each dataset row is one subtitle block. Its subtitle list (``Sentences``) holds one
    entry per on-screen line; an entry can still contain several lines joined by ASS
    ``\\N`` markup or real newlines, which are split here.
    """
    fallback_cnt = 0
    for row in ds:
        block_key = None
        if row.get("ID") is not None and row.get("BlockNum") is not None:
            block_key = f"{row['ID']}_{row['BlockNum']}"

        entry_cnt = 0
        for entry in _subtitle_entries(row):
            for raw_line in RE_ASS_NEWLINE.sub("\n", entry).splitlines():
                cleaned = clean_dialogue_line(raw_line)
                if len(cleaned) < min_len or len(cleaned) > max_len:
                    continue
                if not RE_VALID_CHAR.search(cleaned):
                    continue

                entry_cnt += 1
                if block_key is not None:
                    line_id = f"{source}_{block_key}_{entry_cnt}"
                else:
                    fallback_cnt += 1
                    line_id = f"{source}_{fallback_cnt}"
                yield {"source": source, "id": line_id, "line": cleaned}


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
    data_file: str = DEFAULT_DIALOGUE_DATA_FILE,
    streaming: bool = True,
    data_files: str | Sequence[str] | None = None,
    min_len: int = 2,
    max_len: int = 200,
) -> Generator[dict[str, str], None, None]:
    """Stream subtitle lines from OpenSubtitlesJapanese or local files with low memory usage."""
    from datasets import load_dataset

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

        yield from iter_subtitle_lines(ds, source="dialogue_local", min_len=min_len, max_len=max_len)
        return

    # Only the subtitle table is read; the metadata and Open Assistant parquet files are skipped.
    ds = load_dataset(dataset_name, data_files=data_file, split="train", streaming=streaming)
    yield from iter_subtitle_lines(ds, source="dialogue_opensubtitles", min_len=min_len, max_len=max_len)


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
    dialogue_data_file: str = DEFAULT_DIALOGUE_DATA_FILE,
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
    lang: str | None = None,
    strip_leading_quotes: bool = True,
) -> list[Path]:
    """Export lines into packages for the synthetic data generator.

    :param num_lines_in_each_package: Number of text lines per package CSV. Default 100,000.
    :param num_packages: Total number of packages to generate. Default 10.
    :param start_package: Index of first package (e.g. 0 creates 0000.csv). Default 0.
    :param source_type: Data source to use: 'both' (mixed), 'dialogue', or 'novel' / 'novels'.
    :param dialogue_ratio: Proportion of dialogue lines when source_type='both'. Default 0.5.
    :param dialogue_data_file: Parquet file inside the OpenSubtitles dataset to read.
        Default: 'OpenSubtitles.parquet'.
    :param novel_configs: Configs to use from Japanese-Novels-23M.
        Default: ('Narou_NSFW', 'Kakuyomu', 'Narou_SFW', 'Hameln', 'aozorabunko', 'novecomi', 'pixiv').
        Includes web novel sources (such as Narou_NSFW, Kakuyomu, Hameln).
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
    :param lang: Keep only lines of this language ('ja', 'ko', 'zh', 'en';
        script-detected per line). None (default) keeps every line. Useful to
        extract e.g. English lines from mixed corpora, or to build per-language
        ``lines_ko`` packages consumed by ``run_generate --lang ko``.
    :param strip_leading_quotes: Remove 「」/『』 that open a line (with their closing bracket),
        e.g. 「やあ」と言った -> やあと言った. Quotes inside a sentence are kept. Default True,
        since manga speech bubbles usually omit the outer quotation brackets.
    :return: List of created CSV file paths.
    """
    if out_dir is None:
        out_dir = DATA_SYNTHETIC_ROOT / "lines"
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_lang = None
    if lang is not None:
        from manga_ocr_dev.synthetic_data_generator.lang import normalize_lang

        keep_lang = normalize_lang(lang)

    source_type = source_type.lower()
    if source_type in {"dialogue", "subtitles"}:
        stream = iter_dialogue_stream(
            dataset_name=dialogue_dataset_name,
            data_file=dialogue_data_file,
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
            data_file=dialogue_data_file,
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
            if strip_leading_quotes:
                line_text = strip_leading_quote(line_text)
                if len(line_text) < min_len or not RE_VALID_CHAR.search(line_text):
                    continue
            if keep_lang is not None and detect_line_lang(line_text) != keep_lang:
                continue
            if dedup.is_duplicate(line_text):
                continue

            data.append({"source": item["source"], "id": item["id"], "line": line_text})
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
