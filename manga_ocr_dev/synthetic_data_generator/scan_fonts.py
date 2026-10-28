"""Scan and extract character coverage metadata for Japanese fonts.

Performs deduplication and glyph-level rendering checks to detect blank or missing
glyphs (tofu) and outputs fonts.csv for the synthetic generator.
"""

from __future__ import annotations

import csv
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import fire
import pandas as pd
from fontTools.ttLib import TTFont
from tqdm import tqdm

from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT
from manga_ocr_dev.synthetic_data_generator.utils import find_bad_glyphs, get_font_meta

_vocab = None


def get_vocab():
    """Lazily load the vocabulary character list."""
    global _vocab
    if _vocab is None:
        _vocab = pd.read_csv(ASSETS_PATH / "vocab.csv").char.values
    return _vocab


def open_font(font_path: str | Path) -> TTFont:
    """Open TrueType/OpenType font; .ttc/.otc collections map to face index 0 matching Pillow."""
    font_path_str = str(font_path)
    if font_path_str.lower().endswith((".ttc", ".otc")):
        return TTFont(font_path_str, fontNumber=0, lazy=True)
    return TTFont(font_path_str, lazy=True)


def candidates_of(font: TTFont) -> list[str]:
    """Filter vocabulary characters present in Unicode cmap whose glyph is not .notdef."""
    vocab = get_vocab()
    mapping = dict(font.getBestCmap() or {})
    for table in font["cmap"].tables:
        if table.isUnicode():
            for cp, name in table.cmap.items():
                mapping.setdefault(cp, name)
    notdef = font.getGlyphOrder()[0]
    return [c for c in vocab if mapping.get(ord(c), notdef) != notdef]


def process(font_path: str | Path) -> str:
    """Extract string of verified supported characters for a given font file."""
    try:
        font_path = str(font_path)
        candidates = candidates_of(open_font(font_path))
        if not candidates:
            return ""
        bad = find_bad_glyphs(font_path, candidates)
        return "".join(c for c in candidates if c not in bad)
    except Exception as e:  # noqa: BLE001
        print(f"Error while processing {font_path}: {e}")
        return ""


_EDGE = 1 << 16  # 64 KiB read from each end for rapid pre-hashing


def _hash_file(path: Path, edges_only: bool) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        if edges_only:
            size = os.fstat(f.fileno()).st_size
            h.update(f.read(_EDGE))
            if size > _EDGE:
                f.seek(max(size - _EDGE, _EDGE))
                h.update(f.read(_EDGE))
        else:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
    return h.digest()


def _regroup(groups, keyfn, desc, threads):
    """Split each group by keyfn(first path); singletons pass through unread."""
    out, pending = [], []
    for g in groups:
        (pending if len(g) > 1 else out).append(g)
    if pending:
        flat = [m for g in pending for m in g]
        with ThreadPoolExecutor(threads) as ex:
            keys = list(tqdm(ex.map(lambda m: keyfn(m[0]), flat), total=len(flat), desc=desc, leave=False))
        it = iter(keys)
        for g in pending:
            sub = {}
            for m in g:
                sub.setdefault(next(it), []).append(m)
            out.extend(sub.values())
    return out


def _regroup_by_size(groups):
    by_size = {}
    for g in groups:
        by_size.setdefault(g[0][0].stat().st_size, []).append(g)
    return [[m for g in same for m in g] for same in by_size.values()]


def dedupe_files(items, threads: int = 32):
    """Group byte-identical font files with minimal I/O overhead."""
    by_inode = {}
    for p, r in items:
        st = p.stat()
        by_inode.setdefault((st.st_dev, st.st_ino), []).append((p, r))
    groups = list(by_inode.values())
    groups = _regroup_by_size(groups)
    groups = _regroup(groups, lambda p: _hash_file(p, True), "Pre-hash", threads)
    groups = _regroup(groups, lambda p: _hash_file(p, False), "Full hash", threads)
    return groups


def load_checkpoint(path: Path) -> dict[str, str]:
    """Load existing scan checkpoint, safely handling torn lines."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    text = text[: text.rfind("\n") + 1]
    done = {}
    for row in csv.reader(text.splitlines()):
        if len(row) == 2 and row[0] != "font_path":
            done[row[0]] = row[1]
    return done


def main(chunk_size: int = 250, max_workers: int | None = None, audit_lang: str | None = None):
    """Scan all fonts under FONTS_ROOT and output verified supported characters to fonts.csv.

    :param audit_lang: Optional language ('ja', 'ko', 'zh', 'en' or 'all') to
        audit instead of scanning: prints how many installed fonts can draw
        dialogue text and the SFX lexicon of each language (cmap check, so
        Hangul/SFX Han outside vocab.csv are counted correctly). Useful to
        verify multilingual readiness without rescanning.
    """
    if audit_lang is not None:
        audit_langs(audit_lang)
        return
    max_workers = max_workers or os.cpu_count() or 4
    out_path = ASSETS_PATH / "fonts.csv"
    ckpt_path = ASSETS_PATH / "fonts.checkpoint.csv"

    suffixes = {".ttf", ".otf", ".ttc", ".otc"}
    font_paths = sorted(
        Path(root) / f
        for root, _, files in os.walk(FONTS_ROOT, followlinks=True)
        for f in files
        if Path(f).suffix.lower() in suffixes
    )
    rel_paths = [str(p.relative_to(FONTS_ROOT)) for p in font_paths]

    done = load_checkpoint(ckpt_path)
    if done:
        print(f"Resuming from checkpoint: {len(done)}/{len(rel_paths)} done")

    todo = [(p, r) for p, r in zip(font_paths, rel_paths) if r not in done]
    groups = dedupe_files(todo)
    groups = [sorted(g, key=lambda m: m[1]) for g in groups]
    unique = sorted(
        (g[0][0] for g in groups),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    print(f"Scanning {len(unique)} unique fonts ({len(todo)} remaining, {len(done)} already done)")
    group_of = {g[0][0]: g for g in groups}

    new_file = not ckpt_path.exists()
    with open(ckpt_path, "a", newline="", encoding="utf-8") as ckpt, ProcessPoolExecutor(max_workers) as pool:
        w = csv.writer(ckpt, lineterminator="\n")
        if new_file:
            w.writerow(["font_path", "supported_chars"])
        results = pool.map(process, [str(p) for p in unique], chunksize=2)
        for n, (p, chars) in enumerate(tqdm(zip(unique, results), total=len(unique), desc="Scanning fonts"), 1):
            for _, rel in group_of[p]:
                done[rel] = chars
                w.writerow([rel, chars])
            if n % chunk_size == 0:
                ckpt.flush()
                os.fsync(ckpt.fileno())

    data = pd.DataFrame({"font_path": rel_paths, "supported_chars": [done[r] for r in rel_paths]})
    data["num_chars"] = data.supported_chars.str.len()
    data["label"] = "regular"
    tmp_out = out_path.with_suffix(".tmp.csv")
    data.to_csv(tmp_out, index=False)
    os.replace(tmp_out, out_path)
    ckpt_path.unlink(missing_ok=True)
    print(f"Wrote {len(data)} fonts to {out_path}")

    # Seamlessly update font profiles, archetypes, and priorities
    print("\nUpdating font profiles, archetypes, and priority labels...")
    from manga_ocr_dev.synthetic_data_generator.scan_font_profiles import run_profiling
    run_profiling()


if __name__ == "__main__":
    fire.Fire(main)


def audit_langs(langs: str = "all", limit: int = 2000):
    """Print per-language font readiness (dialogue + SFX coverage).

    Dialogue capability is measured on fonts.csv coverage (fast); SFX lexicon
    coverage is measured on the real font cmap (bounded to ``limit`` fonts per
    language, best-first), so Korean Hangul and Chinese SFX Han outside
    vocab.csv are counted correctly.
    """
    import numpy as np

    from manga_ocr_dev.synthetic_data_generator.lang import (
        LANG_CORE_CHARS,
        SFX_BASES as LANG_SFX,
        normalize_lang,
    )
    from manga_ocr_dev.synthetic_data_generator.utils import (
        get_font_cmap_chars,
        is_hangul,
        is_kanji,
    )

    want = ["ja", "ko", "zh", "en"] if str(langs).lower() == "all" else [normalize_lang(langs)]
    vocab = get_vocab()
    fonts_df, font_map = get_font_meta()
    paths = font_map.paths
    print(f"fonts.csv: {len(paths)} usable fonts")
    for lang in want:
        import unicodedata

        if lang == "ja":
            probe = [c for c in vocab if "HIRAGANA" in unicodedata.name(c, "")]
            need = set("ドンバガッー")
            thresh = 0.50
        elif lang == "ko":
            probe = [c for c in vocab if is_hangul(c) and "\uac00" <= c <= "\ud7a3"]
            need = set("".join(LANG_SFX[lang])) | set(LANG_CORE_CHARS[lang])
            thresh = 0.30
        elif lang == "zh":
            probe = [c for c in vocab if is_kanji(c)]
            need = set("".join(LANG_SFX[lang])) | set(LANG_CORE_CHARS[lang])
            thresh = 0.50
        else:
            probe = [c for c in vocab if ord(c) < 128 and c.isalpha()]
            need = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
            thresh = 0.50
        cover = font_map.coverage(probe) if probe else np.zeros(len(paths))
        order = np.argsort(-cover)[:limit]
        n_dialogue = int((cover >= thresh).sum())
        # SFX: cmap check on the best candidates (bounded).
        n_sfx = 0
        checked = 0
        for idx in order:
            p = paths[int(idx)]
            try:
                cmap = get_font_cmap_chars(p)
            except Exception:  # noqa: BLE001
                continue
            checked += 1
            if sum(ord(c) in cmap for c in need) / max(1, len(need)) >= 0.85:
                n_sfx += 1
            if checked >= limit:
                break
        print(f"[{lang}] dialogue-capable: {n_dialogue} | SFX-capable (cmap, top-{checked}): {n_sfx}")
