from collections import OrderedDict
from functools import lru_cache
import logging
import os
from pathlib import Path
import unicodedata
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageFont, features

from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT

# Silence fontTools "too many bytes" / malformed table warnings at runtime
logging.getLogger("fontTools").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module=r"fontTools.*")


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
                if ("h" not in df.columns or "w" not in df.columns) and all(
                    col in df.columns for col in ["ymin", "ymax", "xmin", "xmax"]
                ):
                    df["h"] = df["ymax"] - df["ymin"]
                    df["w"] = df["xmax"] - df["xmin"]
                if "ratio" not in df.columns and "h" in df.columns and "w" in df.columns:
                    df["ratio"] = (df["w"] / df["h"]).astype(np.float32)
                if "h" in df.columns and "w" in df.columns:
                    df = df[(df["h"] > 0) & (df["w"] > 0)].reset_index(drop=True)
                    df["h"] = df["h"].astype(np.int32)
                    df["w"] = df["w"].astype(np.int32)
                return df
        except Exception:  # noqa: BLE001, S110
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
        background_df.append(
            {
                "path": str(path),
                "h": np.int32(h),
                "w": np.int32(w),
                "ratio": np.float32(w / h),
            }
        )
    return pd.DataFrame(background_df)


# Fast O(1) integer range checks (~50x faster than unicodedata.name string search)
def is_kanji(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2EBEF
        or 0x30000 <= cp <= 0x323AF
    )


def is_hiragana(ch: str) -> bool:
    cp = ord(ch)
    return 0x3041 <= cp <= 0x309F or cp in (0x1B001, 0x1F200)


def is_katakana(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x30A1 <= cp <= 0x30FF
        or 0x31F0 <= cp <= 0x31FF
        or 0xFF66 <= cp <= 0xFF9F
        or cp == 0x1B000
    )


def is_hangul(ch: str) -> bool:
    cp = ord(ch)
    return (
        0xAC00 <= cp <= 0xD7A3
        or 0x1100 <= cp <= 0x11FF
        or 0x3130 <= cp <= 0x318F
        or 0xA960 <= cp <= 0xA97F
        or 0xD7B0 <= cp <= 0xD7FF
    )


def is_ascii(ch: str) -> bool:
    return ord(ch) < 128


_EXCLUDE_HIRA = set("゙゚゛゜ゝゞ゠ー")
_EXCLUDE_KATA = set("゙゚゛゜゠")


@lru_cache(maxsize=2)
def get_charsets(vocab_path=None):
    if vocab_path is None:
        vocab_path = ASSETS_PATH / "vocab.csv"
    vocab = pd.read_csv(vocab_path).char.values
    hiragana = np.array([c for c in vocab if is_hiragana(c) and c not in _EXCLUDE_HIRA])
    katakana = np.array([c for c in vocab if is_katakana(c) and c not in _EXCLUDE_KATA])
    return vocab, hiragana, katakana


@lru_cache(maxsize=1)
def get_vocab_set() -> frozenset[str]:
    vocab, _, _ = get_charsets()
    return frozenset(vocab.tolist())


_BMP = 0x10000


class FontMap:
    """Memory-compact, read-only ``font_path -> supported characters`` mapping."""

    def __init__(self, paths, bits, extra, lru_size=16):
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

    def has_char(self, path: str, ch: str) -> bool:
        """Fast O(1) bit check for a single (font, char) without materialising frozenset."""
        i = self.index.get(path)
        if i is None:
            return False
        cp = ord(ch)
        if cp < _BMP:
            return bool((self.bits[i, cp >> 3] >> (cp & 7)) & 1)
        return ch in self.extra.get(i, "")

    def _build(self, i):
        row = np.unpackbits(self.bits[i], bitorder="little")
        chars = "".join(map(chr, np.flatnonzero(row).tolist())) + self.extra.get(i, "")
        return frozenset(chars)

    def __getitem__(self, path):
        i = self.index[path]
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
        cp = ord(ch)
        if cp < _BMP:
            return (self.bits[:, cp >> 3] >> (cp & 7)) & 1 == 1
        out = np.zeros(len(self.paths), dtype=bool)
        for i, s in self.extra.items():
            out[i] = ch in s
        return out

    def support_mask(self, text):
        mask = np.ones(len(self.paths), dtype=bool)
        for ch in set(text):
            if not ch.isspace():
                mask &= self._has(ch)
        return mask

    def coverage(self, chars):
        chars = set(chars)
        total = np.zeros(len(self.paths), dtype=np.float32)
        for ch in chars:
            total += self._has(ch)
        return total / max(1, len(chars))


_GLOBAL_FONT_META = None


def get_font_meta():
    """Return cached ``(fonts_df, FontMap)``."""
    global _GLOBAL_FONT_META
    if _GLOBAL_FONT_META is not None:
        return _GLOBAL_FONT_META

    frames, paths, rows, extra = [], [], [], {}
    for chunk in pd.read_csv(ASSETS_PATH / "fonts.csv", chunksize=1000):
        chunk = chunk.dropna(subset=["supported_chars"])
        chunk = chunk[chunk.supported_chars.str.len() > 0].copy()
        chunk["font_path"] = chunk.font_path.apply(lambda x: str(FONTS_ROOT / x))
        chunk["num_chars"] = chunk["num_chars"].astype(np.int32)
        chunk["label"] = chunk["label"].astype("category")
        for s in chunk.supported_chars:
            row, hi = FontMap.pack(s)
            if hi:
                extra[len(rows)] = hi
            rows.append(row)
        frames.append(chunk.drop(columns=["supported_chars"]))
        paths.extend(chunk.font_path.tolist())
    if not rows:
        raise RuntimeError("fonts.csv has no usable fonts: run scan_fonts.py first")
    df = pd.concat(frames, ignore_index=True)
    _GLOBAL_FONT_META = (df, FontMap(paths, np.stack(rows), extra))
    return _GLOBAL_FONT_META


_FONT_PROFILES = None
_FONT_PROFILES_BY_FULL = None
_FONT_PROFILES_BY_NAME = None


def get_font_profiles() -> dict[str, dict]:
    """Return dict of {rel_font_path: profile_dict} and build O(1) lookup tables."""
    global _FONT_PROFILES, _FONT_PROFILES_BY_FULL, _FONT_PROFILES_BY_NAME
    if _FONT_PROFILES is None:
        profiles_path = ASSETS_PATH / "font_profiles.json"
        if profiles_path.exists():
            import json

            try:
                with open(profiles_path, encoding="utf-8") as f:
                    _FONT_PROFILES = json.load(f)
            except Exception:
                _FONT_PROFILES = {}
        else:
            _FONT_PROFILES = {}
        _FONT_PROFILES_BY_FULL = {str(FONTS_ROOT / k): v for k, v in _FONT_PROFILES.items()}
        _FONT_PROFILES_BY_NAME = {Path(k).name.lower(): v for k, v in _FONT_PROFILES.items()}
    return _FONT_PROFILES


def get_font_profile(font_path: str | Path | None) -> dict:
    """O(1) profile lookup for given font path (full or relative)."""
    if font_path is None:
        return {}
    profiles = get_font_profiles()
    p_str = str(font_path)
    if p_str in _FONT_PROFILES_BY_FULL:
        return _FONT_PROFILES_BY_FULL[p_str]
    if p_str in profiles:
        return profiles[p_str]
    return _FONT_PROFILES_BY_NAME.get(Path(p_str).name.lower(), {})


NOTDEF_PROBES = ["\uffff", "\ufffe", "\U0010fffd", "\ue0ff"]
_PROBE_SIZE = 32  # 32px is ~2.2x faster to rasterize than 48px and equally accurate for tofu detection
_MAX_SHARED_PUNCT = 3


@lru_cache(maxsize=1)
def _has_raqm():
    try:
        return bool(features.check("raqm"))
    except Exception:  # noqa: BLE001
        return False


_HORIZ_KW = {"anchor": "la"}
_VERT_KW = {"direction": "ttb", "language": "ja", "anchor": "lt"}


def _glyph_key(pil_font, char, ttb=False):
    mask, offset = pil_font.getmask2(char, "L", **(_VERT_KW if ttb else _HORIZ_KW))
    bbox = mask.getbbox()
    if bbox is None:
        return None
    ink = mask.crop(bbox)
    return (ink.size, (offset[0] + bbox[0], offset[1] + bbox[1]), Image.Image()._new(ink).tobytes())


def _notdef_keys(pil_font, ttb):
    keys = set()
    for p in NOTDEF_PROBES:
        try:
            k = _glyph_key(pil_font, p, ttb)
        except Exception:  # noqa: BLE001, S112
            continue
        if k is not None:
            keys.add(k)
    return keys


def _is_letterlike(ch):
    return unicodedata.category(ch)[0] in "LNM"


def find_bad_glyphs(font_path, chars, size=48):
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
            except Exception:  # noqa: BLE001
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
    """Per-font cache for runtime tofu check."""

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
            for ttb in (False,):
                try:
                    k = _glyph_key(self.font, ch, ttb)
                except Exception:  # noqa: BLE001
                    ok = False
                    break
                if k is None or k in self.notdef[ttb]:
                    ok = False
                    break
            self.cache[ch] = ok
        return ok


@lru_cache(maxsize=8)
def _probe_for(font_path):
    return _FontProbe(font_path)


def is_drawable(font_path, ch):
    """True if ``font_path`` really draws ``ch`` (fast bitmap lookup for vocab chars, probe fallback for OOV)."""
    if ch.isspace():
        return True
    p_str = str(font_path)
    if _GLOBAL_FONT_META is not None:
        font_map = _GLOBAL_FONT_META[1]
        if p_str in font_map.index and ch in get_vocab_set():
            return font_map.has_char(p_str, ch)
    try:
        return _probe_for(p_str).drawable(ch)
    except Exception:  # noqa: BLE001
        return False


class MissingGlyphError(ValueError):
    """Raised instead of ever rendering a tofu box."""


_CMAP_CACHE: OrderedDict[str, frozenset] = OrderedDict()
_MAX_CMAP_FONT_BYTES = 35 * (1 << 20)  # Skip >35MB fonts in runtime fallback scan


def get_font_cmap_chars(font_path) -> frozenset:
    """Return the set of Unicode codepoints ``font_path`` maps (bounded LRU)."""
    key = str(font_path)
    cached = _CMAP_CACHE.get(key)
    if cached is not None:
        _CMAP_CACHE.move_to_end(key)
        return cached

    try:
        if os.path.getsize(key) > _MAX_CMAP_FONT_BYTES:
            return frozenset()
    except OSError:
        return frozenset()

    from fontTools.ttLib import TTCollection, TTFont

    if key.lower().endswith((".ttc", ".otc")):
        ttc = TTCollection(key, lazy=True)
        font = ttc.fonts[0]
    else:
        ttc = None
        font = TTFont(key, lazy=True)
    try:
        cmap = font.getBestCmap() or {}
        chars = frozenset(cmap.keys())
    finally:
        try:
            font.close()
            if ttc is not None:
                ttc.close()
        except Exception:  # noqa: BLE001, S110
            pass
    _CMAP_CACHE[key] = chars
    while len(_CMAP_CACHE) > 12:
        _CMAP_CACHE.popitem(last=False)
    return chars


def font_supports_text_cmap(font_path, text) -> bool:
    try:
        cmap = get_font_cmap_chars(font_path)
    except Exception:  # noqa: BLE001
        return False
    return all(c.isspace() or ord(c) in cmap for c in text)
