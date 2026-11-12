#!/usr/bin/env python3
"""Fetch and filter Chinese, English, and Korean text lines for synthetic manga OCR.

Datasets used:
  - Chinese: wdndev/webnovel-chinese (webnovel chapters -> cleaned narrative & dialogue lines)
  - English: mookiezi/Discord-Dialogues (natural conversation -> clean chat dialogue)
  - Korean:  heegyu/kowiki-sentences + jojo0217/korean_safe_conversation (wiki sentences + QA dialogue)

Outputs are saved to /mnt/data/lines/ (or --out-dir) as compressed Parquet files
with columns: ['line', 'source', 'language'].
"""

from __future__ import annotations

import argparse
import html
import os
import re
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def clean_chinese_text(raw_text: str) -> Iterator[str]:
    """Extract and filter clean Chinese dialogue and narrative lines from webnovel text."""
    # Split paragraphs by newline or explicit dialogue quotes
    for para in raw_text.split("\n"):
        para = html.unescape(para).strip()
        if not para:
            continue
        # Split long paragraphs into sentence chunks
        chunks = re.split(r"(?<=[。！？!?…])\s*", para)
        for line in chunks:
            line = line.strip()
            # Strip outer quotes and brackets
            line = re.sub(r'^[“"\'「『]+|[”"\'」』]+$', '', line).strip()
            # Skip chapter headings, URLs, author notes
            if re.match(r'^(第[0-9一二三四五六七八九十百千万]+[章节回卷]|PS|作者有话说|内容简介)', line):
                continue
            if re.search(r'https?://|www\.|<.*?>', line):
                continue
            # Length filter: manga OCR crops can be single characters (>= 1)
            if len(line) < 1 or len(line) > 75:
                continue
            # Hanzi density filter (require >= 60% CJK characters)
            han_count = sum('\u4e00' <= c <= '\u9fff' for c in line)
            if han_count / len(line) < 0.60:
                continue
            yield line


def clean_english_discord_text(raw_text: str) -> Iterator[str]:
    """Extract and filter clean English dialogue from Discord chat transcripts."""
    # Strip ChatML and special tokens
    clean = re.sub(r'<\|im_start\|>\s*(?:user|assistant|system)?|<\|im_end\|>|<\|[^>]+?\|>', ' ', raw_text)
    clean = html.unescape(clean)

    for line in clean.split("\n"):
        line = line.strip()
        if line.lower() in {"user", "assistant", "system", "bot", "discord", "human"}:
            continue
        # Remove discord formatting: custom emojis <:name:123>, user mentions <@123>, channel tags <#123>
        line = re.sub(r'<a?:\w+:\d+>', '', line)
        line = re.sub(r'<@[!&]?\d+>', '', line)
        line = re.sub(r'<#\d+>', '', line)
        line = re.sub(r'https?://\S+', '', line)
        line = re.sub(r'```.*?```', '', line)
        line = re.sub(r'`.*?`', '', line)
        # Remove bot command prefixes like !play, /help, .kick
        if re.match(r'^[!/.$~#^]\w+', line):
            continue
        # Strip outer quotes and normalize spaces
        line = re.sub(r'^\s*["\']+|["\']+\s*$', '', line)
        line = re.sub(r'\s+', ' ', line).strip()
        if line.lower() in {"user", "assistant", "system", "bot", "discord", "human"}:
            continue
        # Length filter: 1 to 90 characters
        if len(line) < 1 or len(line) > 90:
            continue
        # English alphabet ratio
        alpha_count = sum(c.isalpha() for c in line)
        if alpha_count / max(1, len(line)) < 0.50:
            continue
        yield line


def clean_korean_text(raw_text: str) -> Iterator[str]:
    """Extract and filter clean Korean sentences and dialogue lines."""
    raw = html.unescape(raw_text).strip()
    # Remove wiki references [1], [2], parenthetical explanations
    clean = re.sub(r'\[\d+\]', '', raw)
    clean = re.sub(r'\([^\)]*?\)', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', clean)
    for line in sentences:
        line = line.strip()
        # Strip outer quotes
        line = re.sub(r'^[“"\'「『]+|[”"\'」』]+$', '', line).strip()
        if re.search(r'https?://|www\.|<.*?>', line):
            continue
        # Length filter: 1 to 75 characters
        if len(line) < 1 or len(line) > 75:
            continue
        # Hangul density filter (require >= 50% Hangul characters)
        hangul_count = sum('\uac00' <= c <= '\ud7af' for c in line)
        if hangul_count / max(1, len(line)) < 0.50:
            continue
        yield line


def stream_chinese_lines(limit: int = 50000) -> Iterator[str]:
    """Stream clean Chinese lines from wdndev/webnovel-chinese."""
    from datasets import load_dataset
    print("Streaming Chinese lines from wdndev/webnovel-chinese...")
    ds = load_dataset("wdndev/webnovel-chinese", data_files="data/webnovel_*.jsonl", split="train", streaming=True)
    seen = set()
    count = 0
    for row in ds:
        text = row.get("text") or ""
        for line in clean_chinese_text(text):
            if line not in seen:
                seen.add(line)
                yield line
                count += 1
                if count >= limit:
                    return


def stream_english_lines(limit: int = 50000) -> Iterator[str]:
    """Stream clean English dialogue lines from mookiezi/Discord-Dialogues."""
    from datasets import load_dataset
    print("Streaming English dialogue from mookiezi/Discord-Dialogues...")
    ds = load_dataset("mookiezi/Discord-Dialogues", split="train", streaming=True)
    seen = set()
    count = 0
    for row in ds:
        text = row.get("text") or ""
        for line in clean_english_discord_text(text):
            if line not in seen:
                seen.add(line)
                yield line
                count += 1
                if count >= limit:
                    return


def stream_korean_lines(limit: int = 50000) -> Iterator[str]:
    """Stream clean Korean sentences and dialogue from heegyu/kowiki-sentences & jojo0217."""
    from datasets import load_dataset
    print("Streaming Korean lines from heegyu/kowiki-sentences and conversation datasets...")
    seen = set()
    count = 0

    # 1. First fetch Wikipedia sentences
    try:
        ds_wiki = load_dataset("heegyu/kowiki-sentences", split="train", streaming=True)
        for row in ds_wiki:
            sentence = row.get("sentence") or ""
            for line in clean_korean_text(sentence):
                if line not in seen:
                    seen.add(line)
                    yield line
                    count += 1
                    if count >= limit // 2:
                        break
            if count >= limit // 2:
                break
    except Exception as exc:
        print(f"Notice during Korean wiki streaming: {exc}")

    # 2. Next fetch conversational dialogue
    try:
        ds_conv = load_dataset("jojo0217/korean_safe_conversation", split="train", streaming=True)
        for row in ds_conv:
            for key in ["instruction", "output"]:
                text = row.get(key) or ""
                for line in clean_korean_text(text):
                    if line not in seen:
                        seen.add(line)
                        yield line
                        count += 1
                        if count >= limit:
                            return
            if count >= limit:
                return
    except Exception as exc:
        print(f"Notice during Korean conversation streaming: {exc}")


def save_lines_parquet(lines: Iterator[str], out_path: Path, language: str, source: str) -> int:
    """Save lines into a Parquet file with ['line', 'source', 'language']."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        pa.field("line", pa.string()),
        pa.field("source", pa.string()),
        pa.field("language", pa.string()),
    ])

    batch_size = 5000
    rows = []
    total = 0
    writer = None

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = Path("/mnt/data/lines") if Path("/mnt/data").is_dir() else Path.home() / "data" / "lines"
    parser.add_argument("--out-dir", default=str(default_out), help="Directory to save parquet files")
    parser.add_argument("--limit", type=int, default=50000, help="Number of lines per language")
    parser.add_argument("--lang", default="all", choices=["all", "zh", "en", "ko"], help="Languages to fetch")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving fetched lines to: {out_dir}")

    langs = ["zh", "en", "ko"] if args.lang == "all" else [args.lang]

    if "zh" in langs:
        zh_path = out_dir / "chinese_webnovel.parquet"
        save_lines_parquet(stream_chinese_lines(args.limit), zh_path, language="zh", source="wdndev_webnovel_chinese")

    if "en" in langs:
        en_path = out_dir / "english_discord.parquet"
        save_lines_parquet(stream_english_lines(args.limit), en_path, language="en", source="mookiezi_discord_dialogues")

    if "ko" in langs:
        ko_path = out_dir / "korean_wiki_dialogue.parquet"
        save_lines_parquet(stream_korean_lines(args.limit), ko_path, language="ko", source="kowiki_and_safe_conversation")

    print("\nAll requested lines successfully fetched and saved!")


if __name__ == "__main__":
    main()

