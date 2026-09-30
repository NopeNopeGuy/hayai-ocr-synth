import hashlib
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, features

from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT


def get_background_df(background_dir):
    background_df = []
    for path in background_dir.iterdir():
        ymin, ymax, xmin, xmax = [int(v) for v in path.stem.split("_")[-4:]]
        h = ymax - ymin
        w = xmax - xmin
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


def is_ascii(ch):
    return ord(ch) < 128


def get_charsets(vocab_path=None):
    if vocab_path is None:
        vocab_path = ASSETS_PATH / "vocab.csv"
    vocab = pd.read_csv(vocab_path).char.values
    hiragana = vocab[[is_hiragana(c) for c in vocab]][:-6]
    katakana = vocab[[is_katakana(c) for c in vocab]][3:]
    return vocab, hiragana, katakana


def get_font_meta():
    df = pd.read_csv(ASSETS_PATH / "fonts.csv")
    # Fonts that failed scanning have empty supported_chars (read back as NaN).
    # Drop them from the frame itself: otherwise fonts_df and font_map disagree
    # and get_random_font() sampled a dead path -> KeyError in font_map.
    df = df.dropna(subset=["supported_chars"])
    df = df[df.supported_chars.str.len() > 0].reset_index(drop=True)
    df.font_path = df.font_path.apply(lambda x: str(FONTS_ROOT / x))
    font_map = {row.font_path: set(row.supported_chars) for row in df.itertuples()}
    return df, font_map


# --------------------------------------------------------------------------
# Tofu ("missing glyph" box) detection.
#
# A cmap lookup is not proof that a font can draw a character: many fonts map
# unsupported code points to a placeholder box glyph that is *not* ``.notdef``.
# The only reliable test is to look at what FreeType actually draws, so we
# render every character and reject it when it is blank, equal to the font's
# ``.notdef`` glyph, or identical to other unrelated characters (a placeholder
# is by definition shared by many code points).
# --------------------------------------------------------------------------
NOTDEF_PROBES = ["\uFFFF", "\uFFFE", "\U0010FFFD", "\uE0FF"]
_PROBE_SIZE = 48
# Identical bitmaps are legitimate only for a few punctuation variants; for
# letter-like characters (kana, kanji, latin, digits) any sharing is a red flag.
_MAX_SHARED_PUNCT = 3


def _has_raqm():
    try:
        return bool(features.check("raqm"))
    except Exception:
        return False


def _render_probe(pil_font, char, ttb=False, size=_PROBE_SIZE):
    img = Image.new("L", (size * 2, size * 2), 255)
    kw = {"direction": "ttb", "language": "ja"} if ttb else {}
    ImageDraw.Draw(img).text((size // 2, 0), char, 0, font=pil_font, **kw)
    return np.array(img)


def _key(bitmap):
    return hashlib.blake2b(bitmap.tobytes(), digest_size=8).digest()


def _notdef_keys(pil_font, ttb):
    return {_key(_render_probe(pil_font, p, ttb)) for p in NOTDEF_PROBES}


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
                bmp = _render_probe(pil_font, ch, ttb, size)
            except Exception:
                bad.add(ch)
                continue
            if (bmp != 255).sum() == 0:
                bad.add(ch)
                continue
            k = _key(bmp)
            if k in notdef:
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
        self.size = size
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
                    bmp = _render_probe(self.font, ch, ttb, self.size)
                except Exception:
                    ok = False
                    break
                if (bmp != 255).sum() == 0 or _key(bmp) in self.notdef[ttb]:
                    ok = False
                    break
            self.cache[ch] = ok
        return ok


@lru_cache(maxsize=1024)
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
