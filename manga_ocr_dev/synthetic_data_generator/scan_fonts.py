import os
from pathlib import Path

import pandas as pd
from fontTools.ttLib import TTFont
from tqdm.contrib.concurrent import process_map

from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT
from manga_ocr_dev.synthetic_data_generator.utils import find_bad_glyphs

vocab = pd.read_csv(ASSETS_PATH / "vocab.csv").char.values


def cmap_of(font):
    """Union of all Unicode cmap tables of a font."""
    chars = set()
    for table in font["cmap"].tables:
        if table.isUnicode():
            chars.update(table.cmap.keys())
    return chars


def process(font_path):
    """
    Get supported characters list for a given font.

    Font metadata is not reliable, so the cmap is only a first filter. Every
    candidate is then rendered (horizontal and vertical mode) and rejected when
    it is blank, equal to the font's ``.notdef`` glyph, or has the same raster
    as other characters - fonts that map unsupported code points to a
    placeholder box (not ``.notdef``) are caught by that last rule.
    """

    try:
        font_path = str(font_path)
        # .ttc/.otc collections hold several fonts; TTFont() alone raises on them,
        # which used to silently discard every collection font. Use face 0.
        if font_path.lower().endswith((".ttc", ".otc")):
            ttfont = TTFont(font_path, fontNumber=0, lazy=True)
        else:
            ttfont = TTFont(font_path, lazy=True)
        in_cmap = cmap_of(ttfont)

        candidates = [c for c in vocab if ord(c) in in_cmap]
        bad = find_bad_glyphs(font_path, candidates)
        supported_chars = "".join(c for c in candidates if c not in bad)
    except Exception as e:
        print(f"Error while processing {font_path}: {e}")
        supported_chars = ""

    return supported_chars


def main(chunk_size=250, max_workers=4):
    path_in = FONTS_ROOT
    out_path = ASSETS_PATH / "fonts.csv"
    ckpt_path = ASSETS_PATH / "fonts.checkpoint.csv"

    suffixes = {".ttf", ".otf", ".ttc", ".otc"}
    # os.walk with followlinks: FONTS_ROOT may contain symlinks to vendor dirs
    # (pathlib glob/rglob do not descend into symlinked dirs by default).
    font_paths = sorted(
        Path(root) / f
        for root, _, files in os.walk(path_in, followlinks=True)
        for f in files
        if Path(f).suffix.lower() in suffixes
    )

    rel_paths = [str(p.relative_to(FONTS_ROOT)) for p in font_paths]

    # Resume from checkpoint if present. Checkpoint holds font_path ->
    # supported_chars for completed fonts, written after every chunk so a kill
    # only loses the in-flight chunk.
    done = {}
    if ckpt_path.exists():
        try:
            ckpt_df = pd.read_csv(ckpt_path, keep_default_na=False)
            done = dict(zip(ckpt_df["font_path"], ckpt_df["supported_chars"]))
            print(f"Resuming from checkpoint: {len(done)}/{len(rel_paths)} done")
        except Exception as e:
            print(f"Ignoring unreadable checkpoint {ckpt_path}: {e}")

    todo_idx = [i for i, r in enumerate(rel_paths) if r not in done]
    print(f"Scanning {len(todo_idx)} remaining fonts ({len(done)} already done)")

    for start in range(0, len(todo_idx), chunk_size):
        chunk_idx = todo_idx[start : start + chunk_size]
        chunk_paths = [font_paths[i] for i in chunk_idx]
        chunk_data = process_map(
            process,
            chunk_paths,
            max_workers=max_workers,
            desc=f"Scanning {start + len(chunk_idx)}/{len(todo_idx)}",
        )
        for i, s in zip(chunk_idx, chunk_data):
            done[rel_paths[i]] = s if s is not None else ""

        # Atomic checkpoint write: tmp + rename so a kill mid-write can't corrupt it.
        ckpt_df = pd.DataFrame(
            {"font_path": list(done.keys()), "supported_chars": list(done.values())}
        )
        tmp_path = ckpt_path.with_suffix(".tmp.csv")
        ckpt_df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, ckpt_path)

    data = [done[r] for r in rel_paths]
    data = pd.DataFrame({"font_path": rel_paths, "supported_chars": data})
    data["num_chars"] = data.supported_chars.str.len()
    data["label"] = "regular"
    tmp_out = out_path.with_suffix(".tmp.csv")
    data.to_csv(tmp_out, index=False)
    os.replace(tmp_out, out_path)
    try:
        ckpt_path.unlink()
    except FileNotFoundError:
        pass
    print(f"Wrote {len(data)} fonts to {out_path}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
