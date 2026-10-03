from pathlib import Path
import unicodedata
from collections import OrderedDict
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image, ImageFont, features

from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT


def get_background_df(background_dir):
    background_dir = Path(background_dir).expanduser()
    manifest_path = background_dir / "manifest.csv"
    if manifest_path.is_file():
        try:
            df = pd.read_csv(manifest_path)
            if not df.empty and "path" in df.columns:
                df["path"] = df["path"].apply(
                    lambda p: str(background_dir / p) if not Path(p).is_absolute() else str(p)
                )
                if "h" not in df.columns or "w" not in df.columns:
                    if all(col in df.columns for col in ["ymin", "ymax", "xmin", "xmax"]):
                        df["h"] = df["ymax"] - df["ymin"]
                        df["w"] = df["xmax"] - df["xmin"]
                if "ratio" not in df.columns and "h" in df.columns and "w" in df.columns:
                    df["ratio"] = df["w"] / df["h"]
                if "h" in df.columns and "w" in df.columns:
                    df = df[(df["h"] > 0) & (df["w"] > 0)].reset_index(drop=True)
                return df
        except Exception:
            pass

    background_df = []
    for path in background_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            continue
        parts = path.stem.split("_")
        if len(parts) < 4:
            continue
        try:
            ymin, ymax, xmin, xmax = [int(v) for v in parts[-4:]]
        except ValueError:
            continue
        h = ymax - ymin
        w = xmax - xmin
        if h <= 0 or w <= 0:
            continue
        ratio = w / h

        background_df.append(
            {
                "path": str(path),
                "h": h,
                "w": w,
                "ratio": ratio,
            }
        )
    background_df = pd.DataFrame(background_df)
    return background_df


def is_kanji(ch):
    return "CJK UNIFIED IDEOGRAPH" in unicodedata.name(ch, "")


def is_hiragana(ch):
    return "HIRAGANA" in unicodedata.name(ch, "")


def is_katakana(ch):
    return "KATAKANA" in unicodedata.name(ch, "")


def is_hangul(ch):
    return "HANGUL" in unicodedata.name(ch, "")


def is_ascii(ch):
    return ord(ch) < 128


_EXCLUDE_HIRA = set("゙゚゛゜ゝゞ゠ー")
_EXCLUDE_KATA = set("゙゚゛゜゠")


def get_charsets(vocab_path=None):
    if vocab_path is None:
        vocab_path = ASSETS_PATH / "vocab.csv"
    vocab = pd.read_csv(vocab_path).char.values
    # Explicit denylist, not positional slicing: vocab.csv ordering depends on
    # tokenizer ids (BPE frequency order), so [:-6]/[3:] would drop arbitrary
    # kana. These are combining/iteration marks and the prolonged mark (the
    # latter is sampled via the katakana pool).
    hiragana = np.array([c for c in vocab if is_hiragana(c) and c not in _EXCLUDE_HIRA])
    katakana = np.array([c for c in vocab if is_katakana(c) and c not in _EXCLUDE_KATA])
    return vocab, hiragana, katakana


_BMP = 0x10000


class FontMap:
    """Memory-compact, read-only ``font_path -> supported characters`` mapping.

    The old ``{path: set(chars)}`` needs one Python set (tens of KB to MBs each,
    with a fresh str object per CJK character) per font. With ~20k fonts that is
    tens of GB of RAM before the first sample is drawn, and forked workers make
    it worse by touching refcounts (copy-on-write). Here every font is a 8 KB
    bitmap over the BMP (~160 MB for 20k fonts) plus a tiny string for the rare
    non-BMP characters. Per-font ``frozenset``s are materialised on demand and
    kept in a small LRU. Bulk questions ("which fonts can draw this text?")
    are answered vectorised over all fonts at once.
    """

    def __init__(self, paths, bits, extra, lru_size=64):
        self.paths = list(paths)
        self.index = {p: i for i, p in enumerate(self.paths)}
        self.bits = bits  # (n_fonts, 8192) uint8, little-endian bit order
        self.extra = extra  # row -> str of non-BMP chars (rare)
        self._lru = OrderedDict()
        self._lru_size = lru_size

    @staticmethod
    def pack(s):
        """supported-chars string -> (8 KB bitmap row, non-BMP chars str)."""
        cp = np.frombuffer(s.encode("utf-32-le", "surrogatepass"), dtype=np.uint32)
        row = np.zeros(_BMP, dtype=bool)
        row[cp[cp < _BMP]] = True
        hi = cp[cp >= _BMP]
        extra = "".join(map(chr, np.unique(hi).tolist())) if hi.size else ""
        return np.packbits(row, bitorder="little"), extra

    def __len__(self):
        return len(self.paths)

    def __contains__(self, path):
        return path in self.index

    def __iter__(self):
        return iter(self.paths)

    def _build(self, i):
        row = np.unpackbits(self.bits[i], bitorder="little")
        chars = "".join(map(chr, np.flatnonzero(row).tolist())) + self.extra.get(i, "")
        return frozenset(chars)

    def __getitem__(self, path):
        i = self.index[path]  # KeyError for unknown fonts, like a dict
        s = self._lru.get(i)
        if s is None:
            s = self._build(i)
            self._lru[i] = s
            while len(self._lru) > self._lru_size:
                self._lru.popitem(last=False)
        else:
            self._lru.move_to_end(i)
        return s

    def get(self, path, default=None):
        return self[path] if path in self.index else default

    def items(self):
        for p in self.paths:
            yield p, self[p]

    def _has(self, ch):
        """bool array over all fonts: does the font contain ``ch``?"""
        cp = ord(ch)
        if cp < _BMP:
            return (self.bits[:, cp >> 3] >> (cp & 7)) & 1 == 1
        out = np.zeros(len(self.paths), dtype=bool)
        for i, s in self.extra.items():
            out[i] = ch in s
        return out

    def support_mask(self, text):
        """bool array over all fonts: font can draw every non-space char of ``text``."""
        mask = np.ones(len(self.paths), dtype=bool)
        for ch in set(text):
            if not ch.isspace():
                mask &= self._has(ch)
        return mask

    def coverage(self, chars):
        """float array over all fonts: fraction of ``chars`` each font contains."""
        chars = set(chars)
        total = np.zeros(len(self.paths), dtype=np.float64)
        for ch in chars:
            total += self._has(ch)
        return total / max(1, len(chars))


def get_font_meta():
    """Return ``(fonts_df, FontMap)``; rows of ``fonts_df`` match ``FontMap.paths``.

    ``fonts.csv`` is streamed in chunks so the huge ``supported_chars`` strings
    (one per font) are never all in memory at once, and the column is dropped
    from the returned frame.
    """
    frames, paths, rows, extra = [], [], [], {}
    for chunk in pd.read_csv(ASSETS_PATH / "fonts.csv", chunksize=500):
        # Fonts that failed scanning have empty supported_chars (read back as NaN).
        # Drop them: otherwise fonts_df and the map disagree.
        chunk = chunk.dropna(subset=["supported_chars"])
        chunk = chunk[chunk.supported_chars.str.len() > 0].copy()
        chunk["font_path"] = chunk.font_path.apply(lambda x: str(FONTS_ROOT / x))
        for s in chunk.supported_chars:  # pack now; the strings die with the chunk
            row, hi = FontMap.pack(s)
            if hi:
                extra[len(rows)] = hi
            rows.append(row)
        frames.append(chunk.drop(columns=["supported_chars"]))
        paths.extend(chunk.font_path.tolist())
    if not rows:
        raise RuntimeError("fonts.csv has no usable fonts: run scan_fonts.py first")
    df = pd.concat(frames, ignore_index=True)
    return df, FontMap(paths, np.stack(rows), extra)


# --------------------------------------------------------------------------
# Tofu ("missing glyph" box) detection.
#
# A cmap lookup is not proof that a font can draw a character: many fonts map
# unsupported code points to a placeholder box glyph that is *not* ``.notdef``.
# The only reliable test is to look at what FreeType actually draws, so we
# render every character and reject it when it is blank, equal to the font's
# ``.notdef`` glyph, or identical to other unrelated characters (a placeholder
# is by definition shared by many code points).
#
# Rendering goes through FreeTypeFont.getmask2 directly (the same call
# ImageDraw.text makes) instead of drawing onto a canvas, converting to numpy
# and hashing: same glyph bitmap + same placement, ~1.5x less work per char.
# --------------------------------------------------------------------------
NOTDEF_PROBES = ["\uFFFF", "\uFFFE", "\U0010FFFD", "\uE0FF"]
_PROBE_SIZE = 48
# Identical bitmaps are legitimate only for a few punctuation variants; for
# letter-like characters (kana, kanji, latin, digits) any sharing is a red flag.
_MAX_SHARED_PUNCT = 3


@lru_cache(maxsize=1)
def _has_raqm():
    try:
        return bool(features.check("raqm"))
    except Exception:
        return False


_HORIZ_KW = {"anchor": "la"}
_VERT_KW = {"direction": "ttb", "language": "ja", "anchor": "lt"}


def _glyph_key(pil_font, char, ttb=False):
    """Hashable identity of what ``char`` looks like (bitmap + placement).

    Returns ``None`` for a blank glyph. Raises if FreeType cannot render it.
    """
    mask, offset = pil_font.getmask2(char, "L", **(_VERT_KW if ttb else _HORIZ_KW))
    bbox = mask.getbbox()
    if bbox is None:
        return None
    # Crop to the inked area: the mask is padded to the text advance, so two
    # identical outlines with different side bearings must still compare equal.
    ink = mask.crop(bbox)
    return (ink.size, (offset[0] + bbox[0], offset[1] + bbox[1]), Image.Image()._new(ink).tobytes())


def _notdef_keys(pil_font, ttb):
    keys = set()
    for p in NOTDEF_PROBES:
        try:
            k = _glyph_key(pil_font, p, ttb)
        except Exception:
            continue
        if k is not None:
            keys.add(k)
    return keys


def _is_letterlike(ch):
    return unicodedata.category(ch)[0] in "LNM"


def find_bad_glyphs(font_path, chars, size=_PROBE_SIZE):
    """Return the subset of ``chars`` that ``font_path`` cannot really draw.

    Rejected: blank glyphs (non-space), glyphs identical to ``.notdef``,
    and glyphs whose raster is shared with other characters (placeholder
    boxes). Checked in horizontal mode and, for non-ASCII characters, in the
    vertical (``ttb``) mode that the renderer uses for upright CJK text.
    """
    font_path = str(font_path)
    pil_font = ImageFont.truetype(font_path, size)
    chars = list(dict.fromkeys(chars))
    bad = set()
    modes = [False] + ([True] if _has_raqm() else [])
    for ttb in modes:
        notdef = _notdef_keys(pil_font, ttb)
        groups = {}
        for ch in chars:
            if ch in bad or ch.isspace() or (ttb and is_ascii(ch)):
                continue
            try:
                k = _glyph_key(pil_font, ch, ttb)
            except Exception:
                bad.add(ch)
                continue
            if k is None or k in notdef:
                bad.add(ch)
                continue
            groups.setdefault(k, []).append(ch)
        for members in groups.values():
            if len(members) < 2:
                continue
            if any(_is_letterlike(c) for c in members) or len(members) > _MAX_SHARED_PUNCT:
                bad.update(members)
    return bad


class _FontProbe:
    """Per-font cache for the cheap runtime check (no whole-font scan)."""

    def __init__(self, font_path, size=_PROBE_SIZE):
        self.font = ImageFont.truetype(str(font_path), size)
        self.raqm = _has_raqm()
        self.notdef = {t: _notdef_keys(self.font, t) for t in ((False, True) if self.raqm else (False,))}
        self.cache = {}

    def drawable(self, ch):
        if ch.isspace():
            return True
        ok = self.cache.get(ch)
        if ok is None:
            ok = True
            for ttb in (False, True) if (self.raqm and not is_ascii(ch)) else (False,):
                try:
                    k = _glyph_key(self.font, ch, ttb)
                except Exception:
                    ok = False
                    break
                if k is None or k in self.notdef[ttb]:
                    ok = False
                    break
            self.cache[ch] = ok
        return ok


# Each probe holds an open FreeType face; keep the cache small.
@lru_cache(maxsize=32)
def _probe_for(font_path):
    return _FontProbe(font_path)


def is_drawable(font_path, ch):
    """True if ``font_path`` really draws ``ch`` (not blank, not a tofu box).

    Runtime safety net that works even when ``fonts.csv`` is stale. Cheap:
    one small render per (font, char), cached for the process lifetime.
    """
    try:
        return _probe_for(str(font_path)).drawable(ch)
    except Exception:
        return False


class MissingGlyphError(ValueError):
    """Raised instead of ever rendering a tofu box."""
