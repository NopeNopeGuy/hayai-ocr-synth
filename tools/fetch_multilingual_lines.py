#!/usr/bin/env python3
"""Fetch and filter Chinese, English, and Korean text lines for synthetic manga OCR.

Uses Wikipedia (wikimedia/wikipedia) and OpenSubtitles
(sentence-transformers/parallel-sentences-opensubtitles), packing 1-3 short
consecutive sentences/turns into realistic speech-bubble lengths.
"""

from __future__ import annotations

import argparse
import itertools
import random
import re
import unicodedata
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_WIKI_STOP_HEADERS = {
    "see also",
    "references",
    "external links",
    "further reading",
    "notes",
    "bibliography",
    "sources",
    "参见",
    "參見",
    "參考資料",
    "参考资料",
    "外部連結",
    "外部链接",
    "延伸閱讀",
    "延伸阅读",
    "같이 보기",
    "각주",
    "참고 문헌",
    "외부 링크",
    "둘러보기",
}

_RE_WS = re.compile(r"\s+")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_SDH = re.compile(r"\[[^\]]*\]|\([^)]*(?:laugh|sigh|music|groan|scream|whisper|applause)[^)]*\)", re.I)
_RE_MUSIC = re.compile(r"[♪♫♬♩]+")
_RE_SUB_DASH = re.compile(r"^[-－–—]+(?:\s+|(?=[^\d\s]))")
_RE_WIKI_HEADER = re.compile(r"^=+.*=+$")
_RE_BULLET = re.compile(r"^[•·*\-–—\d+[.)\]]\s+")

_RE_EN_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")
_RE_ZH_SENT = re.compile(r"(?<=[。！？!?])")
_RE_KO_SENT = re.compile(r"(?<=[.!?。！？])\s+")

_RE_HAS_HAN = re.compile(r"[\u4e00-\u9fff]")
_RE_PINYIN_TONE = re.compile(r"[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]")
_RE_ZH_SPACE = re.compile(r"(?<=[\u4e00-\u9fff，。！？；：、（）「」『』《》])\s+|\s+(?=[\u4e00-\u9fff，。！？；：、（）「」『』《》])")
_RE_HAS_HANGUL = re.compile(r"[\uac00-\ud7af]")
_RE_HAS_LATIN = re.compile(r"[A-Za-z]")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _RE_URL.sub("", text)
    return _RE_WS.sub(" ", text).strip()


def _clean_sub_line(text: str) -> str:
    text = _RE_SDH.sub("", text)
    text = _RE_MUSIC.sub("", text)
    text = _RE_SUB_DASH.sub("", text.strip())
    return _norm(text)


def _valid_for_lang(text: str, lang: str) -> bool:
    if not text:
        return False
    if lang == "en":
        alpha = sum(c.isalpha() and ord(c) < 128 for c in text)
        return _RE_HAS_LATIN.search(text) is not None and alpha >= max(8, len(text) * 0.55)
    if lang == "zh":
        if _RE_PINYIN_TONE.search(text) or "读作" in text or "讀作" in text:
            return False
        if text[-1] in ",，、:：":
            return False
        han = len(_RE_HAS_HAN.findall(text))
        return han >= max(4, int(len(text) * 0.50))
    if lang == "ko":
        ko = len(_RE_HAS_HANGUL.findall(text))
        return ko >= max(4, int(len(text) * 0.40))
    return True


def _pack_sentences(
    sentences: Iterator[str],
    lang: str,
    min_chars: int,
    max_chars: int,
    seed: int = 42,
) -> Iterator[str]:
    """Combine 1-3 short consecutive sentences into realistic bubble lengths."""
    rng = random.Random(seed)
    joiner = "" if lang == "zh" else " "
    buf = ""
    lo_target = min(max_chars, min_chars + max(1, (max_chars - min_chars) // 4))
    target_chars = rng.randint(lo_target, max(lo_target, max_chars))

    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) > max_chars:
            continue
        if not buf:
            buf = sent
        elif len(buf) + len(joiner) + len(sent) <= target_chars:
            buf = f"{buf}{joiner}{sent}"
        else:
            if len(buf) >= min_chars and _valid_for_lang(buf, lang):
                yield buf
            buf = sent
            target_chars = rng.randint(lo_target, max(lo_target, max_chars))

        if len(buf) >= target_chars:
            if len(buf) >= min_chars and _valid_for_lang(buf, lang):
                yield buf
            buf = ""
            target_chars = rng.randint(lo_target, max(lo_target, max_chars))

    if buf and len(buf) >= min_chars and _valid_for_lang(buf, lang):
        yield buf


def _iter_wiki_sentences(config: str, lang: str, seed: int = 42) -> Iterator[str]:
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    splitter = {"en": _RE_EN_SENT, "zh": _RE_ZH_SENT, "ko": _RE_KO_SENT}[lang]

    for row in ds:
        text = row.get("text") or ""
        if not text:
            continue
        for raw_para in text.splitlines():
            para = raw_para.strip()
            if not para:
                continue
            if para.lower() in _WIKI_STOP_HEADERS:
                break
            if _RE_WIKI_HEADER.match(para) or _RE_BULLET.match(para):
                continue
            if len(para) < 35 and para[-1] not in ".!?。！？":
                continue
            para = _norm(para)
            if not para:
                continue
            for sent in splitter.split(para):
                sent = sent.strip()
                if len(sent) < 8 or "|" in sent or "ISBN" in sent:
                    continue
                if _valid_for_lang(sent, lang):
                    yield sent


def _load_opensubtitles_stream(configs: tuple[str, ...], seed: int):
    from datasets import load_dataset

    last_err = None
    for cfg in configs:
        try:
            ds = load_dataset(
                "sentence-transformers/parallel-sentences-opensubtitles",
                cfg,
                split="train",
                streaming=True,
            )
            return ds.shuffle(seed=seed, buffer_size=20_000)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"Could not load OpenSubtitles configs {configs}: {last_err}")


def _iter_opensubtitles_lines(lang: str, seed: int = 42) -> Iterator[str]:
    if lang == "ko":
        configs, col = ("en-ko",), "non_english"
    elif lang == "zh":
        configs, col = ("en-zh_cn", "en-zh", "en-zh_tw"), "non_english"
    elif lang == "en":
        configs, col = ("en-ko", "en-zh_cn", "en-zh"), "english"
    else:
        raise ValueError(f"Unsupported lang for opensubtitles: {lang!r}")

    ds = _load_opensubtitles_stream(configs, seed=seed)
    for row in ds:
        line = _clean_sub_line(str(row.get(col) or ""))
        if lang == "zh":
            line = _RE_ZH_SPACE.sub("", line)
        if not line or len(line) < 4:
            continue
        if _valid_for_lang(line, lang):
            yield line


def _interleave(streams: list[Iterator[str]], weights: list[float], seed: int = 42) -> Iterator[str]:
    rng = random.Random(seed)
    active = list(zip(streams, weights))
    while active:
        _, w_list = zip(*active)
        chosen = rng.choices(range(len(active)), weights=w_list, k=1)[0]
        stream, _ = active[chosen]
        try:
            yield next(stream)
        except StopIteration:
            active.pop(chosen)


def stream_wikipedia_lines(
    lang: str,
    limit: int | None = None,
    min_chars: int | None = None,
    max_chars: int | None = None,
    seed: int = 42,
) -> Iterator[str]:
    cfg_map = {"en": "20231101.en", "zh": "20231101.zh", "ko": "20231101.ko"}
    defaults = {"en": (28, 125), "zh": (10, 42), "ko": (12, 65)}
    lo, hi = defaults[lang]
    min_chars = lo if min_chars is None else min_chars
    max_chars = hi if max_chars is None else max_chars
    gen = _pack_sentences(_iter_wiki_sentences(cfg_map[lang], lang, seed=seed), lang, min_chars, max_chars, seed=seed)
    return itertools.islice(gen, limit) if limit is not None else gen


def stream_opensubtitles_lines(
    lang: str,
    limit: int | None = None,
    min_chars: int | None = None,
    max_chars: int | None = None,
    seed: int = 42,
) -> Iterator[str]:
    defaults = {"en": (22, 115), "zh": (6, 40), "ko": (10, 60)}
    lo, hi = defaults[lang]
    min_chars = lo if min_chars is None else min_chars
    max_chars = hi if max_chars is None else max_chars
    gen = _pack_sentences(_iter_opensubtitles_lines(lang, seed=seed), lang, min_chars, max_chars, seed=seed + 1)
    return itertools.islice(gen, limit) if limit is not None else gen


def stream_english_lines(limit: int | None = None, seed: int = 42) -> Iterator[str]:
    """English comic/manga lines: 60% OpenSubtitles dialogue + 40% Wikipedia."""
    gen = _interleave(
        [stream_opensubtitles_lines("en", seed=seed), stream_wikipedia_lines("en", seed=seed + 10)],
        weights=[0.60, 0.40],
        seed=seed,
    )
    return itertools.islice(gen, limit) if limit is not None else gen


def stream_korean_lines(limit: int | None = None, seed: int = 42) -> Iterator[str]:
    """Korean manhwa lines: 60% OpenSubtitles dialogue + 40% Wikipedia."""
    gen = _interleave(
        [stream_opensubtitles_lines("ko", seed=seed), stream_wikipedia_lines("ko", seed=seed + 10)],
        weights=[0.60, 0.40],
        seed=seed,
    )
    return itertools.islice(gen, limit) if limit is not None else gen


def stream_chinese_lines(limit: int | None = None, seed: int = 42) -> Iterator[str]:
    """Chinese manhua lines: 50% OpenSubtitles dialogue + 50% Wikipedia."""
    gen = _interleave(
        [stream_opensubtitles_lines("zh", seed=seed), stream_wikipedia_lines("zh", seed=seed + 10)],
        weights=[0.50, 0.50],
        seed=seed,
    )
    return itertools.islice(gen, limit) if limit is not None else gen


def save_lines_parquet(lines: Iterator[str], out_path: Path, language: str, source: str) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        pa.field("line", pa.string()),
        pa.field("source", pa.string()),
        pa.field("language", pa.string()),
    ])
    batch_size = 5000
    rows, total, writer = [], 0, None
    for line in lines:
        rows.append({"line": line, "source": source, "language": language})
        if len(rows) >= batch_size:
            table = pa.Table.from_pylist(rows, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), schema=schema, compression="ZSTD")
            writer.write_table(table)
            total += len(rows)
            rows.clear()
    if rows:
        table = pa.Table.from_pylist(rows, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), schema=schema, compression="ZSTD")
        writer.write_table(table)
        total += len(rows)
    if writer is not None:
        writer.close()
    print(f"Saved {total:,} lines ({language}) -> {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = Path("/mnt/data/lines") if Path("/mnt/data").is_dir() else Path.home() / "data" / "lines"
    parser.add_argument("--out-dir", default=str(default_out))
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--lang", default="all", choices=["all", "zh", "en", "ko"])
    parser.add_argument("--wiki-only", action="store_true", help="Fetch only wikimedia/wikipedia")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    langs = ["zh", "en", "ko"] if args.lang == "all" else [args.lang]

    for lg in langs:
        if args.wiki_only:
            stream = stream_wikipedia_lines(lg, limit=args.limit)
            src, fname = f"wikipedia_{lg}", f"{lg}_wikipedia.parquet"
        else:
            fn = {"zh": stream_chinese_lines, "en": stream_english_lines, "ko": stream_korean_lines}[lg]
            stream = fn(limit=args.limit)
            src, fname = f"{lg}_opensubtitles_wiki", f"{lg}_opensubtitles_wiki.parquet"
        save_lines_parquet(stream, out_dir / fname, language=lg, source=src)


if __name__ == "__main__":
    main()
