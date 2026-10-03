import csv
import hashlib
import os
from pathlib import Path

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from fontTools.ttLib import TTFont
from tqdm import tqdm

from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT
from manga_ocr_dev.synthetic_data_generator.utils import find_bad_glyphs

vocab = pd.read_csv(ASSETS_PATH / "vocab.csv").char.values


def open_font(font_path):
    # .ttc/.otc hold several faces; face 0 matches what PIL's truetype() renders.
    if font_path.lower().endswith((".ttc", ".otc")):
        return TTFont(font_path, fontNumber=0, lazy=True)
    return TTFont(font_path, lazy=True)


def candidates_of(font):
    """Vocab chars present in a Unicode cmap whose glyph is not .notdef.

    Both filters are exact: an absent or .notdef-mapped char is rendered as the
    .notdef box, which find_bad_glyphs would reject anyway - we just skip the
    (expensive) render.
    """
    mapping = dict(font.getBestCmap() or {})
    for table in font["cmap"].tables:
        if table.isUnicode():
            for cp, name in table.cmap.items():
                mapping.setdefault(cp, name)
    notdef = font.getGlyphOrder()[0]
    return [c for c in vocab if mapping.get(ord(c), notdef) != notdef]


def process(font_path):
    """
    Get supported characters list for a given font.

    The cmap only narrows the candidates; the verdict comes from actually
    rendering each one (horizontal + vertical) via find_bad_glyphs, which rejects
    blank glyphs, .notdef, and placeholder boxes shared by several characters.
    """
    try:
        font_path = str(font_path)
        candidates = candidates_of(open_font(font_path))
        if not candidates:
            return ""
        bad = find_bad_glyphs(font_path, candidates)
        return "".join(c for c in candidates if c not in bad)
    except Exception as e:
        print(f"Error while processing {font_path}: {e}")
        return ""


_EDGE = 1 << 16  # bytes read from each end for the cheap pre-hash


def _hash_file(path, edges_only):
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


def dedupe_files(items, threads=32):
    """Group byte-identical font files, reading as little as possible.

    Stages (each only touches files still ambiguous): same inode (symlinks and
    hard links, no read) -> same size (stat only) -> same first+last 64 KiB ->
    full hash. Only a full-hash match merges files, so grouping is exact.
    """
    by_inode = {}
    for p, r in items:
        st = p.stat()
        by_inode.setdefault((st.st_dev, st.st_ino), []).append((p, r))
    groups = list(by_inode.values())
    groups = _regroup_by_size(groups)
    groups = _regroup(groups, lambda p: _hash_file(p, True), "Pre-hash", threads)
    groups = _regroup(groups, lambda p: _hash_file(p, False), "Full hash", threads)
    return groups


def _regroup_by_size(groups):
    by_size = {}
    for g in groups:
        by_size.setdefault(g[0][0].stat().st_size, []).append(g)
    # merge groups that share a size into candidate sets; identical files are
    # confirmed later, so just concatenate them for the next stage.
    return [[m for g in same for m in g] for same in by_size.values()]


def load_checkpoint(path):
    """Append-only checkpoint; a torn last line from a kill is discarded."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    text = text[: text.rfind("\n") + 1]
    done = {}
    for row in csv.reader(text.splitlines()):
        if len(row) == 2 and row[0] != "font_path":
            done[row[0]] = row[1]
    return done


def main(chunk_size=250, max_workers=None):
    max_workers = max_workers or os.cpu_count()
    out_path = ASSETS_PATH / "fonts.csv"
    ckpt_path = ASSETS_PATH / "fonts.checkpoint.csv"

    suffixes = {".ttf", ".otf", ".ttc", ".otc"}
    # os.walk with followlinks: FONTS_ROOT may contain symlinks to vendor dirs.
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

    # Identical files (symlinks, copies in vendor dirs) are scanned once.
    todo = [(p, r) for p, r in zip(font_paths, rel_paths) if r not in done]
    groups = dedupe_files(todo)
    groups = [sorted(g, key=lambda m: m[1]) for g in groups]
    unique = sorted(
        (g[0][0] for g in groups),
        key=lambda p: p.stat().st_size,
        reverse=True,  # biggest first keeps workers balanced to the end
    )
    print(
        f"Scanning {len(unique)} unique fonts "
        f"({len(todo)} remaining, {len(done)} already done)"
    )
    group_of = {g[0][0]: g for g in groups}

    new_file = not ckpt_path.exists()
    with open(ckpt_path, "a", newline="", encoding="utf-8") as ckpt, ProcessPoolExecutor(
        max_workers
    ) as pool:
        w = csv.writer(ckpt, lineterminator="\n")
        if new_file:
            w.writerow(["font_path", "supported_chars"])
        results = pool.map(process, [str(p) for p in unique], chunksize=2)
        for n, (p, chars) in enumerate(
            tqdm(zip(unique, results), total=len(unique), desc="Scanning"), 1
        ):
            for _, rel in group_of[p]:
                done[rel] = chars
                w.writerow([rel, chars])
            if n % chunk_size == 0:
                ckpt.flush()
                os.fsync(ckpt.fileno())

    data = pd.DataFrame(
        {"font_path": rel_paths, "supported_chars": [done[r] for r in rel_paths]}
    )
    data["num_chars"] = data.supported_chars.str.len()
    data["label"] = "regular"
    tmp_out = out_path.with_suffix(".tmp.csv")
    data.to_csv(tmp_out, index=False)
    os.replace(tmp_out, out_path)
    ckpt_path.unlink(missing_ok=True)
    print(f"Wrote {len(data)} fonts to {out_path}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
