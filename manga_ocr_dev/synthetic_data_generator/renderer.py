"""Fast Pillow-based text renderer for synthetic manga OCR data.

Renders text layouts (vertical/horizontal text, furigana/ruby, tate-chu-yoko,
stroke/shadow styles) directly with Pillow, with no browser, no temp files
and no subprocesses.

Public API::

    Renderer().render(lines, override_css_params) -> (img_bgra, params)
"""

from collections import OrderedDict
import ctypes
import gc
import logging
import math
import os
import re
import threading

import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from manga_ocr_dev.env import ASSETS_PATH, BACKGROUND_ANIMETEXT_DIR, BACKGROUND_DIR
from manga_ocr_dev.synthetic_data_generator.utils import (
    MissingGlyphError,
    get_background_df,
    is_ascii,
    is_drawable,
)

try:
    _LIBC = ctypes.CDLL("libc.so.6")
except Exception:
    _LIBC = None


def _trim_native_memory():
    gc.collect()
    if _LIBC is not None:
        try:
            _LIBC.malloc_trim(0)
        except Exception:
            pass


def get_default_background_df():
    """AnimeText backgrounds replace Manga109s; fall back to the old dir if empty/missing."""
    try:
        df = get_background_df(BACKGROUND_ANIMETEXT_DIR)
        if len(df):
            return df
    except (FileNotFoundError, NotADirectoryError):
        pass
    return get_background_df(BACKGROUND_DIR)


BODY_MARGIN = 20  # px, matches the old CSS ``body { margin: 20px; }``
RUBY_SCALE = 0.45  # furigana size relative to the main font size
RUBY_GAP = 6  # px gap between base text and furigana
MIN_PAD = 4  # extra padding so stroke/blur never clips at canvas edges

# Real handwritten-kana bank hook.  It is opt-in (``real_kana=True``), just
# like the SFX split: the default font rendering path is untouched.
REAL_KANA_PROB = 0.60
_BANK_ROOT_ENV = "REAL_KANA_ROOT"
_REAL_KANA_BANK_CACHE = {}
_REAL_KANA_BANK_LOCK = threading.Lock()
_REAL_KANA_MISSING_WARNED = False
log = logging.getLogger(__name__)


def _bank_root_candidates(default=None):
    """Yield bank roots in priority order (explicit root wins)."""

    from pathlib import Path

    if default is not None:
        yield Path(default).expanduser()
        return
    env_root = os.environ.get(_BANK_ROOT_ENV)
    if env_root:
        yield Path(env_root).expanduser()
    yield ASSETS_PATH / "handwriting"
    # Conventional dataset mounts used by the bank pipeline.
    yield Path("/mnt/real-kana-final")
    yield Path("/mnt/real-kana-bank")


def _is_bank_root(root) -> bool:
    return (root / "manifest.jsonl").is_file() or any((root / "glyphs").glob("*/*.png"))


def _get_real_kana_bank(root=None, cfg=None):
    """Load and cache a bank, or return ``None`` when none is installed."""

    from pathlib import Path

    roots = list(_bank_root_candidates(root))
    chosen = None
    for candidate in roots:
        if _is_bank_root(candidate):
            chosen = candidate
            break
    if chosen is None:
        global _REAL_KANA_MISSING_WARNED
        if root is not None and not _REAL_KANA_MISSING_WARNED:
            log.warning("real-kana bank not found at %s; falling back to fonts", root)
            _REAL_KANA_MISSING_WARNED = True
        return None
    key = str(chosen.resolve())
    with _REAL_KANA_BANK_LOCK:
        bank = _REAL_KANA_BANK_CACHE.get(key)
        if bank is not None:
            return bank
        try:
            from manga_ocr_dev.real_kana.bank import KanaBank

            bank = KanaBank.load(chosen, cfg=cfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("cannot load real-kana bank at %s: %s", chosen, exc)
            return None
        _REAL_KANA_BANK_CACHE[key] = bank
        return bank


IRREGULAR_PROB = 0.55
LINE_SCALE_RANGE = (0.6, 1.55)
LINE_SHIFT_EM = 1.5
CHAR_SCALE_RANGE = (0.65, 1.6)
CHAR_JITTER_PROB = 0.55
WOBBLE_EM = 0.09
ROTATE_DEG = 7.0
SMALL_KANA = frozenset("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ")
SMALL_KANA_SCALE = (0.55, 0.8)
MIN_CHAR_SIZE = 8

SLANT_PROB = 0.22
SLANT_MILD_RANGE = (8.0, 15.0)
SLANT_STRONG_RANGE = (20.0, 38.0)
SLANT_STRONG_PROB = 0.4

DIALOGUE_FONT_SIZE_RANGE = (26, 56)
DIALOGUE_LINE_HEIGHT_RANGE = (0.22, 0.62)

SFX_FONT_SIZE_RANGE = (38, 190)
SFX_CHAR_SCALE_RANGE = (0.75, 2.1)
SFX_LINE_SCALE_RANGE = (0.8, 1.3)
SFX_PERSP_RANGE = (0.10, 0.45)
SFX_ROTATE_DEG = 35.0
SFX_BLOCK_ROTATE_RANGE = (-35.0, 35.0)
SFX_BLOCK_ROTATE_90_PROB = 0.05
SFX_CURVE_MODES = ("arc", "wave", "diagonal", "straight", "zigzag", "bounce", "hill", "steep", "circle")
SFX_CURVE_P = (0.18, 0.15, 0.13, 0.12, 0.09, 0.09, 0.09, 0.08, 0.07)
SFX_ADJACENT_SCALE_MAX = 1.8
SFX_ASPECT_MAX = 0.7
SFX_SPACING_RANGE = (-0.18, 0.22)
SFX_CURVE_AMP_RANGE = (0.15, 0.6)
SFX_SPEEDLINE_PROB = 0.6
SFX_SPEEDLINE_N_RANGE = (24, 80)
SFX_BURST_PROB = 0.35
SFX_DOUBLE_OUTLINE_PROB = 0.3
SFX_VERTICAL_PROB = 0.4
SFX_CROP_MARGIN_FRAC = (0.02, 0.06)
SFX_CROP_MARGIN_MIN = 2
SFX_INK_ALPHA_THRESHOLD = 8
TRUNC_PROB = 0.2
TRUNC_FRAC_RANGE = (0.10, 0.30)
TRUNC_INK_BITE_EM = 0.10
SHADOW_GLOW_GAIN = 2.0

COLOR_PROB = 0.18
SFX_COLOR_PROB = 0.28
MULTICOLOR_PROB = 0.35
SFX_MULTICOLOR_PROB = 0.4
MUTED_COLOR_PROB = 0.5
MULTICOLOR_MODES = ("per_line", "per_char")
TEXT_COLOR_NAMES = (
    "red",
    "pink",
    "lightpink",
    "orange",
    "yellow",
    "green",
    "lime",
    "cyan",
    "blue",
    "purple",
    "magenta",
    "brown",
)
TEXT_COLOR_P = (0.20, 0.18, 0.08, 0.08, 0.07, 0.07, 0.04, 0.05, 0.09, 0.05, 0.05, 0.04)

STROKE_PROB = 0.40
SHADOW_PROB = 0.08
STROKE_SIZES = (1, 2, 3, 4, 6)
STROKE_SIZE_P = (0.35, 0.30, 0.20, 0.10, 0.05)
SFX_STROKE_FRAC = 0.05
SFX_OUTER_RING = 2

BG_BLUR_PROB = 0.08
BG_TONE_MODES = ("natural", "midtone", "ink_match")
BG_TONE_P = (0.2, 0.35, 0.45)
BG_MIN_STD = (30.0, 75.0)
BG_INK_OFFSET = (35.0, 95.0)
BG_MIN_EDGE_DIFF = 75.0
BG_MAX_LOW_CONTRAST = 0.20
BG_MAX_TRIES = 6
BG_CAMO_PROB = 0.15
BG_SCREENTONE_PROB = 0.32
BG_GRADIENT_PROB = 0.32
BG_NOISE_PROB = 0.35
BG_HATCH_PROB = 0.22
BG_COLOR_ART_PROB = 0.65
BUBBLE_FILL_MODES = ("white", "gray", "tint", "tone", "dark")
BUBBLE_FILL_P = (0.45, 0.20, 0.10, 0.15, 0.10)

HOLLOW_PROB = 0.12
GRADIENT_FILL_PROB = 0.12
STRIPE_FILL_PROB = 0.10
EXTRUDE_PROB = 0.15
DISTRESS_PROB = 0.22
FAINT_PROB = 0.10
FG_OCCLUDE_PROB = 0.35
DISTRACTOR_PROB = 0.0
DEGRADE_JPEG_PROB = 0.5
DEGRADE_SCALE_PROB = 0.3
DEGRADE_BLUR_PROB = 0.25
DEGRADE_SHARPEN_PROB = 0.25
DEGRADE_PHOTO_PROB = 0.2
DEGRADE_COLOR_JITTER_PROB = 0.6

INK_DENSITY_PROB = 0.5
INK_DENSITY_DEPTH = (0.05, 0.25)
INK_EDGE_PROB = 0.5
INK_EDGE_SIGMA = (8.0, 28.0)
SCAN_ILLUM_PROB = 0.4
SCAN_ILLUM_DEPTH = (0.03, 0.18)
SCAN_SPREAD_PROB = 0.2
SCAN_DEFOCUS_PROB = 0.3
SCAN_DEFOCUS_SIGMA = (0.35, 0.9)
SCAN_NOISE_PROB = 0.35
SCAN_NOISE_SIGMA = (2.0, 7.0)

RUBY_RE = re.compile(r"<ruby>(.*?)<rt>(.*?)</rt></ruby>")
TCY_RE = re.compile(r"<span[^>]*?text-combine-upright[^>]*?>(.*?)</span>")
TAG_RE = re.compile(r"<[^>]+>")

_NAMED_COLORS = {
    "black": (0, 0, 0, 255),
    "white": (255, 255, 255, 255),
    "transparent": (0, 0, 0, 0),
    "red": (255, 0, 0, 255),
    "pink": (255, 80, 160, 255),
    "lightpink": (255, 170, 200, 255),
    "orange": (255, 130, 0, 255),
    "yellow": (255, 235, 0, 255),
    "gold": (255, 200, 0, 255),
    "green": (0, 180, 0, 255),
    "lime": (150, 220, 0, 255),
    "cyan": (0, 220, 220, 255),
    "teal": (0, 170, 170, 255),
    "blue": (0, 80, 255, 255),
    "purple": (150, 0, 220, 255),
    "magenta": (255, 0, 150, 255),
    "brown": (150, 75, 0, 255),
}

_FONT_CACHE_MAX = 16
_LARGE_FONT_BYTES = 25 * (1 << 20)  # 25 MB: evict giant fonts as soon as sample switches
_BG_IMAGE_CACHE = OrderedDict()
_BG_IMAGE_CACHE_MAX = 8


def _read_bg_cached(path: str):
    img = _BG_IMAGE_CACHE.get(path)
    if img is not None:
        _BG_IMAGE_CACHE.move_to_end(path)
        return img
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        h, w = img.shape[:2]
        if max(h, w) > 1200:
            scale = 1200.0 / max(h, w)
            img = cv2.resize(img, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA)
        _BG_IMAGE_CACHE[path] = img
        while len(_BG_IMAGE_CACHE) > _BG_IMAGE_CACHE_MAX:
            _BG_IMAGE_CACHE.popitem(last=False)
    return img


_FONT_CACHE = OrderedDict()
_FONT_CACHE_LOCK = threading.Lock()
_RENDER_COUNT = 0
_LAST_HUGE_FONT: str | None = None


def _normalize_color(value, default="black"):
    """Map a CSS color name, hex code, or color tuple to an RGBA tuple."""
    if value is None:
        value = default
    if isinstance(value, (tuple, list)):
        value = value[0] if len(value) > 0 else default
    if not isinstance(value, str):
        value = default
    key = value.strip().lower()
    if key in _NAMED_COLORS:
        return _NAMED_COLORS[key]
    if key.startswith("#"):
        hexpart = key[1:]
        try:
            if len(hexpart) == 3:
                r, g, b = (int(c * 2, 16) for c in hexpart)
                return (r, g, b, 255)
            if len(hexpart) == 6:
                r = int(hexpart[0:2], 16)
                g = int(hexpart[2:4], 16)
                b = int(hexpart[4:6], 16)
                return (r, g, b, 255)
        except ValueError:
            pass
    return _NAMED_COLORS[default]


def _sample_text_color(sfx=False):
    prob = SFX_COLOR_PROB if sfx else COLOR_PROB
    if np.random.rand() >= prob:
        return "black"
    return str(np.random.choice(TEXT_COLOR_NAMES, p=TEXT_COLOR_P))


def _sample_text_colors(n, sfx=False):
    n = max(2, int(n))
    chosen = []
    for _ in range(20):
        c = str(np.random.choice(TEXT_COLOR_NAMES, p=TEXT_COLOR_P))
        if c not in chosen:
            chosen.append(c)
        if len(chosen) >= min(n, len(TEXT_COLOR_NAMES)):
            break
    while len(chosen) < 2:
        c = str(np.random.choice(TEXT_COLOR_NAMES, p=TEXT_COLOR_P))
        if c not in chosen:
            chosen.append(c)
    return chosen[: max(2, min(n, len(chosen)))]


def _sample_color_mode(sfx=False):
    if sfx:
        return str(np.random.choice(MULTICOLOR_MODES, p=[0.4, 0.6]))
    return str(np.random.choice(MULTICOLOR_MODES, p=[0.7, 0.3]))


def _maybe_multicolor(params, text_color, sfx=False):
    prob = SFX_MULTICOLOR_PROB if sfx else MULTICOLOR_PROB
    if text_color == "black" or np.random.rand() >= prob:
        return params
    params["color_mode"] = _sample_color_mode(sfx=sfx)
    params["text_colors"] = _sample_text_colors(int(np.random.randint(2, 5)), sfx=sfx)
    if text_color not in params["text_colors"]:
        params["text_colors"][0] = text_color
    return params


def _params_are_color(params):
    mode = str(params.get("color_mode", "solid"))
    if mode in ("per_line", "per_char"):
        return True
    colors = params.get("text_colors", None)
    if isinstance(colors, (list, tuple)) and len(colors):
        return any(_is_chromatic_color(c) for c in colors)
    return _is_chromatic_color(params.get("text_color", "black"))


def _assign_op_colors(ops, params):
    shadow_fallback = _normalize_color(params.get("shadow_color", "white"), "white")
    black = (0, 0, 0, 255)
    white = (255, 255, 255, 255)

    def contrasting_edge(fill):
        return black if _needs_dark_outline(fill) else white

    mode = str(params.get("color_mode", "solid"))
    if mode not in ("per_line", "per_char"):
        fill = _maybe_mute_fill(_normalize_color(params.get("text_color", "black"), "black"))
        for op in ops:
            op["fill"] = fill
            op["stroke"] = contrasting_edge(fill)
            op["shadow"] = black if _needs_dark_outline(fill) else shadow_fallback
        return ops
    colors = params.get("text_colors", None)
    if not isinstance(colors, (list, tuple)) or not len(colors):
        colors = _sample_text_colors(3)
    fills = [_maybe_mute_fill(_normalize_color(c, "black")) for c in colors]
    if mode == "per_line":
        line_to_fill = {}
        for op in ops:
            li = int(op.get("line", 0))
            if li not in line_to_fill:
                line_to_fill[li] = fills[int(np.random.randint(len(fills)))]
            op["fill"] = line_to_fill[li]
            op["stroke"] = contrasting_edge(op["fill"])
            op["shadow"] = black if _needs_dark_outline(op["fill"]) else shadow_fallback
    else:  # per_char
        for op in ops:
            op["fill"] = fills[int(np.random.randint(len(fills)))]
            op["stroke"] = contrasting_edge(op["fill"])
            op["shadow"] = black if _needs_dark_outline(op["fill"]) else shadow_fallback
    return ops


def _is_chromatic_color(value):
    try:
        r, g, b, a = _normalize_color(value, "black")
    except (TypeError, ValueError):
        return False
    if a == 0:
        return False
    return not (r == g == b)


def _relative_luminance(rgba):
    r, g, b = (c / 255.0 for c in rgba[:3])

    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _contrast_ratio(a, b):
    la = _relative_luminance(_normalize_color(a, "black"))
    lb = _relative_luminance(_normalize_color(b, "black"))
    hi, lo = (la, lb) if la >= lb else (lb, la)
    return (hi + 0.05) / (lo + 0.05)


MIN_FILL_CONTRAST = 3.0


def _needs_dark_outline(fill_rgba):
    white = (255, 255, 255, 255)
    la = _relative_luminance(fill_rgba)
    lw = _relative_luminance(white)
    hi, lo = (lw, la) if lw >= la else (la, lw)
    return (hi + 0.05) / (lo + 0.05) < MIN_FILL_CONTRAST


def _desaturate_rgba(fill_rgba, amount=0.4):
    r, g, b, a = (int(v) for v in fill_rgba[:4])
    lum = int(round(0.299 * r + 0.587 * g + 0.114 * b))
    k = float(np.clip(amount, 0.0, 1.0))
    return (int(round(r + (lum - r) * k)), int(round(g + (lum - g) * k)), int(round(b + (lum - b) * k)), a)


def _maybe_mute_fill(fill_rgba):
    try:
        r, g, b, a = fill_rgba[:4]
    except (TypeError, ValueError):
        return fill_rgba
    if r == g == b or a == 0:
        return fill_rgba
    if np.random.rand() < MUTED_COLOR_PROB:
        return _desaturate_rgba(fill_rgba, float(np.random.uniform(0.25, 0.65)))
    return fill_rgba


def _ensure_legible_outline(params):
    mode = str(params.get("color_mode", "solid"))
    if mode in ("per_line", "per_char"):
        if int(params.get("stroke_size", 0) or 0) <= 0 and int(params.get("shadow_size", 0) or 0) <= 0:
            params["stroke_size"] = 2
            params["stroke_color"] = "white"
        elif int(params.get("stroke_size", 0) or 0) <= 0:
            params["shadow_color"] = "black"
            params["shadow_size"] = max(int(params.get("shadow_size", 0) or 0), 3)
        return params
    fill = _normalize_color(params.get("text_color", "black"), "black")
    if not _needs_dark_outline(fill):
        return params
    if int(params.get("stroke_size", 0) or 0) > 0:
        params["stroke_color"] = "black"
        params["stroke_size"] = max(int(params["stroke_size"]), 2)
    elif int(params.get("shadow_size", 0) or 0) > 0:
        params["shadow_color"] = "black"
        params["shadow_size"] = max(int(params["shadow_size"]), 3)
    else:
        params["stroke_size"] = 2
        params["stroke_color"] = "black"
    if params.get("sfx") and int(params.get("stroke_size", 0) or 0) >= 5:
        params["sfx_double_outline"] = True
    return params


def _get_font(font_path, size):
    """Load (and cache) a TrueType/OpenType font. Thread-safe."""
    global _LAST_HUGE_FONT
    p_str = str(font_path)
    sz = int(size)
    key = (p_str, sz)
    with _FONT_CACHE_LOCK:
        if _LAST_HUGE_FONT is not None and _LAST_HUGE_FONT != p_str:
            _FONT_CACHE.clear()
            _LAST_HUGE_FONT = None
        font = _FONT_CACHE.get(key)
        if font is None:
            font = ImageFont.truetype(p_str, sz)
            try:
                is_huge = os.path.getsize(p_str) > _LARGE_FONT_BYTES
            except OSError:
                is_huge = False
            if is_huge:
                _LAST_HUGE_FONT = p_str
            max_entries = 4 if is_huge else _FONT_CACHE_MAX
            _FONT_CACHE[key] = font
            while len(_FONT_CACHE) > max_entries:
                _FONT_CACHE.popitem(last=False)
        else:
            _FONT_CACHE.move_to_end(key)
        return font


def _default_font_path():
    """Fallback Japanese font for direct Renderer use without font_path."""
    import glob

    candidates = [
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        try:
            with open(path, "rb"):
                return path
        except OSError:
            continue
    for pattern in (
        "/usr/share/fonts/**/NotoSansJP*.otf",
        "/usr/share/fonts/**/*JP*.ttf",
        "/usr/share/fonts/noto-cjk/*.ttc",
    ):
        found = glob.glob(pattern, recursive=True)
        if found:
            return min(found)
    raise OSError("No Japanese font found; pass font_path explicitly.")


def _assert_no_tofu(parsed, font_path):
    """Hard guarantee: never rasterize a character the font cannot draw."""
    for tokens in parsed:
        for tok in tokens:
            text = "".join(_strip_tags(part) for part in tok[1:] if isinstance(part, str))
            for ch in text:
                if not is_drawable(font_path, ch):
                    raise MissingGlyphError(f"{font_path!r} cannot draw {ch!r} (U+{ord(ch):04X})")


def _strip_tags(s):
    return TAG_RE.sub("", s)


def _parse_inline(s):
    tokens = []
    pos = 0
    for m in TCY_RE.finditer(s):
        if m.start() > pos:
            tokens.append(("text", s[pos : m.start()]))
        tokens.append(("tcy", m.group(1)))
        pos = m.end()
    if pos < len(s):
        tokens.append(("text", s[pos:]))
    return tokens


def _parse_line(line):
    tokens = []
    pos = 0
    for m in RUBY_RE.finditer(line):
        if m.start() > pos:
            tokens.extend(_parse_inline(line[pos : m.start()]))
        tokens.append(("ruby", m.group(1), m.group(2)))
        pos = m.end()
    if pos < len(line):
        tokens.extend(_parse_inline(line[pos:]))
    return tokens


def _hatch_overlay(h, w, strength=0.35):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = float(np.random.uniform(0, np.pi))
    period = float(np.random.uniform(4.0, 9.0))
    u = (xx * np.cos(ang) + yy * np.sin(ang)) * (2 * np.pi / period)
    lines = (np.sin(u) > float(np.random.uniform(0.2, 0.7))).astype(np.float32)
    mask = (np.random.rand(h, w).astype(np.float32) < float(np.random.uniform(0.3, 0.7))).astype(np.float32)
    return (lines * mask * float(strength)).astype(np.float32)


def _colorize_background(bg_bgr, ref_bgr, strength=0.5):
    try:
        ref = np.asarray(ref_bgr, dtype=np.float32).reshape(1, 1, 3)
        ref_hsv = cv2.cvtColor(ref.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    except Exception:  # noqa: BLE001
        return bg_bgr
    bg_hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    dh = (ref_hsv[0, 0, 0] - bg_hsv[:, :, 0] + 90.0) % 180.0 - 90.0
    bg_hsv[:, :, 0] = (bg_hsv[:, :, 0] + dh * float(strength)) % 180.0
    bg_hsv[:, :, 1] = np.clip(
        bg_hsv[:, :, 1] * float(np.random.uniform(0.6, 1.2)) + ref_hsv[0, 0, 1] * 0.25 * float(strength),
        0,
        255,
    )
    out = cv2.cvtColor(np.clip(bg_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    h, w = out.shape[:2]
    wash = np.zeros_like(out, dtype=np.float32)
    c = np.asarray(ref_bgr, dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = float(np.random.uniform(0, 2 * np.pi))
    g = ((xx - w / 2) * np.cos(ang) + (yy - h / 2) * np.sin(ang)) / max(1.0, float(np.abs((w + h) / 2)))
    wash += (c[None, None, :] * (0.5 + 0.5 * g)[..., None] * 0.25 * float(strength)).astype(np.float32)
    return np.clip(out.astype(np.float32) * 0.8 + wash * 0.2, 0, 255).astype(np.uint8)


def _apply_ink_modulation(main_layer, mode):
    if mode not in ("gradient", "stripe"):
        return main_layer
    arr = np.array(main_layer)
    alpha = arr[:, :, 3]
    m = alpha > 8
    if not m.any():
        return main_layer
    h = arr.shape[0]
    if mode == "gradient":
        t = np.linspace(0.55, 1.15, h, dtype=np.float32)[:, None]
        t = np.broadcast_to(t, (h, arr.shape[1]))
        for c in range(3):
            ch = arr[:, :, c].astype(np.float32)
            ch[m] = np.clip(ch[m] * t[m], 0, 255)
            arr[:, :, c] = ch.astype(np.uint8)
    else:
        yy, xx = np.mgrid[0 : arr.shape[0], 0 : arr.shape[1]].astype(np.float32)
        if np.random.rand() < 0.5:
            pat = (np.sin((xx + yy * 0.4) * 2 * np.pi / float(np.random.uniform(4, 9))) > 0).astype(np.float32)
        else:
            pat = ((_screentone(arr.shape[0], arr.shape[1]) > 0.35)).astype(np.float32)
        dim = (1.0 - float(np.random.uniform(0.35, 0.6)) * pat).astype(np.float32)
        for c in range(3):
            ch = arr[:, :, c].astype(np.float32)
            ch[m] *= dim[m]
            arr[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _distress_rgba(main_layer, strength=0.5):
    arr = np.array(main_layer)
    h, w = arr.shape[:2]
    a = arr[:, :, 3].astype(np.float32)
    if (a > 8).sum() < 20:
        return main_layer
    n = max(1, int(float(strength) * float(np.random.randint(3, 9))))
    for _ in range(n):
        if np.random.rand() < 0.6:
            x0, y0 = int(np.random.randint(0, w)), int(np.random.randint(0, h))
            ang = float(np.random.uniform(0, 2 * np.pi))
            ln = float(np.random.uniform(min(h, w) * 0.2, min(h, w) * 0.9))
            x1, y1 = int(x0 + ln * np.cos(ang)), int(y0 + ln * np.sin(ang))
            cv2.line(a, (x0, y0), (x1, y1), 0, max(1, int(np.random.randint(1, 3))))
        else:
            cx, cy = int(np.random.randint(0, w)), int(np.random.randint(0, h))
            cv2.circle(a, (cx, cy), max(1, int(np.random.randint(2, 6))), 0, -1)
    arr[:, :, 3] = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _draw_foreground_occluders_bgr(img_bgr, density=1.0):
    h, w = img_bgr.shape[:2]
    out = img_bgr
    if np.random.rand() < 0.45 * density:
        side = int(np.random.randint(0, 4))
        th = int(np.random.randint(2, 7))
        col = (0, 0, 0) if np.random.rand() < 0.8 else (255, 255, 255)
        if side == 0:
            cv2.line(out, (0, int(np.random.randint(0, h))), (w, int(np.random.randint(0, h))), col, th)
        elif side == 1:
            cv2.line(out, (int(np.random.randint(0, w)), 0), (int(np.random.randint(0, w)), h), col, th)
        elif side == 2:
            cv2.line(out, (0, 0), (w, h), col, th)
        else:
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), col, th)
    n = int(np.random.randint(2, 7) * density)
    for _ in range(n):
        x0, y0 = int(np.random.randint(0, w)), int(np.random.randint(0, h))
        ang = float(np.random.uniform(0, 2 * np.pi))
        ln = float(np.random.uniform(min(h, w) * 0.4, max(h, w) * 1.1))
        x1, y1 = int(x0 + ln * np.cos(ang)), int(y0 + ln * np.sin(ang))
        gray = int(np.random.randint(0, 255))
        col = (gray, gray, gray) if out.ndim == 3 and out.shape[2] == 3 else gray
        cv2.line(out, (x0, y0), (x1, y1), col, int(np.random.choice([1, 1, 2, 3])))
    if np.random.rand() < 0.4 * density:
        dots = _screentone(h, w)
        f = out.astype(np.float32)
        f *= (1.0 - float(np.random.uniform(0.15, 0.4)) * dots)[..., None] if out.ndim == 3 else (1.0 - 0.3 * dots)
        out = np.clip(f, 0, 255).astype(np.uint8)
    return out


_DISTRACTOR_CHARS = {
    "ja": list("アカサタナハマヤラワガザダバパドンビキクケコ0123456789"),
    "ko": list("가나다라마바사아자차카타파하거너더러머버서어0123456789"),
    "zh": list("的一是不了我有个人好这中大来上0123456789"),
    "en": list("ABCDEFGHKMRWZabcdegkmwxyz0123456789"),
}


def _draw_distractor_fragments_bgr(img_bgr, font_path, cell=48, lang="ja"):
    return img_bgr


def _truncate_bgr(img, ink_box=None, bite_px=0, frac_range=TRUNC_FRAC_RANGE):
    h, w = img.shape[:2]
    f = float(np.random.uniform(*frac_range))
    side = int(np.random.randint(0, 4))
    want = int(h * f) if side in (0, 1) else int(w * f)
    if ink_box is not None:
        x0, y0, x1, y1 = ink_box
        room = {0: y0, 1: h - y1, 2: x0, 3: w - x1}[side]
        want = min(want, max(0, int(room)) + int(bite_px))
    if want < 2:
        return img
    if side == 0:
        out = img[want:, :, ...] if h - want >= 8 else img
    elif side == 1:
        out = img[: h - want, :, ...] if h - want >= 8 else img
    elif side == 2:
        out = img[:, want:, ...] if w - want >= 8 else img
    else:
        out = img[:, : w - want, ...] if w - want >= 8 else img
    return np.ascontiguousarray(out)


def _blur_rgba_premultiplied(layer, sigma, gain=1.0):
    arr = np.asarray(layer, dtype=np.float32)
    a = arr[:, :, 3:4] / 255.0
    pm = np.concatenate([arr[:, :, :3] * a, a], axis=2)
    pm = cv2.GaussianBlur(pm, (0, 0), float(sigma))
    a2 = pm[:, :, 3:4]
    rgb = np.where(a2 > 1e-6, pm[:, :, :3] / np.maximum(a2, 1e-6), 0.0)
    a2 = np.clip(a2 * float(gain), 0.0, 1.0)
    out = np.concatenate([rgb, a2 * 255.0], axis=2)
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


def _sweep_layer(layer, dx, dy):
    arr = np.asarray(layer)
    h, w = arr.shape[:2]
    n = max(abs(int(dx)), abs(int(dy)), 1)
    out = np.zeros_like(arr)
    for k in range(0, n + 1):
        sx, sy = int(round(dx * k / n)), int(round(dy * k / n))
        ys0, ys1 = max(0, sy), min(h, h + sy)
        xs0, xs1 = max(0, sx), min(w, w + sx)
        if ys1 <= ys0 or xs1 <= xs0:
            continue
        dst = out[ys0:ys1, xs0:xs1]
        np.maximum(dst, arr[ys0 - sy : ys1 - sy, xs0 - sx : xs1 - sx], out=dst)
    return Image.fromarray(out)


def _scale_layer_alpha(layer, factor):
    arr = np.array(layer)
    arr[:, :, 3] = np.clip(arr[:, :, 3].astype(np.float32) * float(factor), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _smooth_field(h, w, grid=4):
    coarse = np.random.uniform(-1.0, 1.0, (grid, grid)).astype(np.float32)
    return np.clip(cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC), -1.0, 1.0)


def _ink_realism(layer):
    arr = np.array(layer)
    alpha = arr[:, :, 3].astype(np.float32)
    ink = alpha > 8
    if not ink.any():
        return layer
    h, w = alpha.shape
    if np.random.rand() < INK_DENSITY_PROB:
        depth = float(np.random.uniform(*INK_DENSITY_DEPTH))
        field = (_smooth_field(h, w, grid=int(np.random.randint(3, 7))) + 1.0) * 0.5
        alpha = np.where(ink, alpha * (1.0 - depth * field), alpha)
    if np.random.rand() < INK_EDGE_PROB:
        edge = ink & (alpha < 247)
        sigma = float(np.random.uniform(*INK_EDGE_SIGMA))
        alpha = np.where(edge, alpha + np.random.normal(0.0, sigma, (h, w)).astype(np.float32), alpha)
    arr[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _scan_model_bgr(img):
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    if np.random.rand() < SCAN_ILLUM_PROB:
        depth = float(np.random.uniform(*SCAN_ILLUM_DEPTH))
        field = (_smooth_field(h, w, grid=3) + 1.0) * 0.5
        out *= (1.0 - depth * field)[..., None]
    if np.random.rand() < SCAN_SPREAD_PROB:
        out = cv2.erode(out, np.ones((3, 3), np.uint8))
    if np.random.rand() < SCAN_DEFOCUS_PROB:
        out = cv2.GaussianBlur(out, (0, 0), float(np.random.uniform(*SCAN_DEFOCUS_SIGMA)))
    if np.random.rand() < SCAN_NOISE_PROB:
        sigma = float(np.random.uniform(*SCAN_NOISE_SIGMA))
        out = out + np.random.normal(0.0, sigma, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _degrade_bgr(img, is_color):
    out = img
    if np.random.rand() < DEGRADE_SCALE_PROB:
        h, w = out.shape[:2]
        s = float(np.random.uniform(0.4, 0.8))
        nh, nw = max(8, int(h * s)), max(8, int(w * s))
        small = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    r = np.random.rand()
    if r < DEGRADE_BLUR_PROB * 0.6:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    elif r < DEGRADE_BLUR_PROB:
        k = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        amt = float(np.random.uniform(0.2, 0.6))
        sharp = cv2.filter2D(out, -1, k)
        out = np.clip(out.astype(np.float32) * (1 - amt) + sharp.astype(np.float32) * amt, 0, 255).astype(np.uint8)
    elif np.random.rand() < DEGRADE_SHARPEN_PROB:
        pass
    if is_color and np.random.rand() < DEGRADE_COLOR_JITTER_PROB:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= float(np.random.uniform(0.4, 1.6))
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * float(np.random.uniform(0.85, 1.15)), 0, 255)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    elif not is_color and np.random.rand() < 0.4:
        gain = float(np.random.uniform(0.85, 1.15))
        bias = float(np.random.uniform(-12, 12))
        out = np.clip(out.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    if np.random.rand() < DEGRADE_JPEG_PROB:
        q = int(np.random.randint(25, 71))
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    if np.random.rand() < DEGRADE_PHOTO_PROB:
        h, w = out.shape[:2]
        ang = float(np.random.uniform(-3.0, 3.0))
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


class Renderer:
    """Pillow-based renderer. Drop-in replacement for the old browser one."""

    def __init__(self, background_df=None, kana_bank=None, real_kana=False, real_kana_prob=None):
        self.background_df = background_df if background_df is not None else get_default_background_df()
        self.kana_bank = kana_bank
        self.real_kana = bool(real_kana)
        self.real_kana_prob = real_kana_prob
        self.max_size = 600

    def render(self, lines, override_css_params=None):
        global _RENDER_COUNT
        _RENDER_COUNT += 1
        if _RENDER_COUNT % 250 == 0:
            with _FONT_CACHE_LOCK:
                _FONT_CACHE.clear()
            _trim_native_memory()

        img, params = self.render_text(lines, override_css_params)
        draw_bubble = None if not params.get("sfx") else False
        img, ink_box = self.render_background(
            img, draw_bubble=draw_bubble, tight=bool(params.get("sfx")), return_ink_box=True
        )
        if np.random.rand() < TRUNC_PROB and min(img.shape[:2]) > 24:
            bite = TRUNC_INK_BITE_EM * float(params.get("font_size", 48))
            img = _truncate_bgr(img, ink_box=ink_box, bite_px=bite)
        if np.random.rand() < FG_OCCLUDE_PROB:
            img = _draw_foreground_occluders_bgr(img, density=float(np.random.uniform(0.7, 1.4)))
        if max(img.shape[:2]) > self.max_size:
            img = A.LongestMaxSize(self.max_size)(image=img)["image"]
        is_color = _params_are_color(params)
        img = _scan_model_bgr(img)
        img = _degrade_bgr(img, is_color)
        if is_color:
            return img, params
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, params

    def render_text(self, lines, override_css_params=None):
        """Render text on a transparent background and return as BGRA image."""
        sfx_hint = bool((override_css_params or {}).get("sfx", False))
        lang_hint = str((override_css_params or {}).get("lang", "ja"))
        params = self.get_random_css_params(sfx=sfx_hint, lang=lang_hint)
        if override_css_params:
            params.update(override_css_params)
            params["lang"] = str(params.get("lang", lang_hint))
            if (
                "text_color" in override_css_params
                and "color_mode" not in override_css_params
                and "text_colors" not in override_css_params
            ):
                params["color_mode"] = "solid"
            if params.get("color_mode", "solid") == "solid":
                params.pop("text_colors", None)
        _ensure_legible_outline(params)

        sfx = bool(params.get("sfx", False))
        font_size = int(params.get("font_size", 48))
        font_path = params.get("font_path") or _default_font_path()
        vertical = bool(params.get("vertical", True))
        line_height = float(params.get("line_height", 0.5))
        upright = params.get("text_orientation", None) == "upright"
        stroke_size = int(params.get("stroke_size", 0) or 0)
        shadow_size = int(params.get("shadow_size", 0) or 0)
        letter_spacing = params.get("letter_spacing", None)
        letter_spacing_px = round(float(letter_spacing) * font_size) if letter_spacing else 0

        text_rgba = _normalize_color(params.get("text_color", "black"), "black")
        stroke_rgba = _normalize_color(params.get("stroke_color", "black"), "black")
        shadow_rgba = _normalize_color(params.get("shadow_color", "black"), "black")
        bg_rgba = _normalize_color(params.get("background_color", "transparent"), "transparent")

        irregular = bool(params.get("irregular", False)) and not sfx
        pad = max(stroke_size, 3 * shadow_size, 2 if params.get("hollow") else 0, 0) + MIN_PAD

        parsed = [_parse_line(line) for line in lines]
        _assert_no_tofu(parsed, font_path)

        if sfx:
            width, height, ops = self._layout_sfx(
                parsed,
                vertical=vertical,
                font_path=font_path,
                font_size=font_size,
                line_height=line_height,
                letter_spacing=letter_spacing,
                upright=upright,
                margin=BODY_MARGIN,
                pad=pad,
                char_scale_min=float(params.get("sfx_char_scale_min", SFX_CHAR_SCALE_RANGE[0])),
                char_scale_max=float(params.get("sfx_char_scale_max", SFX_CHAR_SCALE_RANGE[1])),
                curve=str(params.get("sfx_curve", "arc")),
                curve_amp=float(params.get("sfx_curve_amp", 0.3)),
                curve_sign=float(params.get("sfx_curve_sign", 1.0)),
                persp=float(params.get("sfx_persp", 0.2)),
                rotate=float(params.get("sfx_rotate", 14.0)),
                wobble=float(params.get("sfx_wobble", 0.05)),
                small_kana=bool(params.get("small_kana", True)),
                aspect=float(params.get("sfx_aspect", 0.0)),
            )
        elif irregular:
            width, height, ops = self._layout_irregular(
                parsed,
                vertical=vertical,
                font_path=font_path,
                font_size=font_size,
                line_height=line_height,
                letter_spacing=letter_spacing,
                upright=upright,
                margin=BODY_MARGIN,
                pad=pad,
                line_scale_min=float(params.get("line_scale_min", LINE_SCALE_RANGE[0])),
                line_scale_max=float(params.get("line_scale_max", LINE_SCALE_RANGE[1])),
                line_shift=float(params.get("line_shift", LINE_SHIFT_EM)),
                char_scale_min=float(params.get("char_scale_min", CHAR_SCALE_RANGE[0])),
                char_scale_max=float(params.get("char_scale_max", CHAR_SCALE_RANGE[1])),
                char_jitter_prob=float(params.get("char_jitter_prob", CHAR_JITTER_PROB)),
                wobble=float(params.get("wobble", WOBBLE_EM)),
                rotate=float(params.get("rotate", ROTATE_DEG)),
                small_kana=bool(params.get("small_kana", True)),
            )
        else:
            ruby_size = max(8, round(font_size * RUBY_SCALE))
            font = _get_font(font_path, font_size)
            ruby_font = _get_font(font_path, ruby_size)
            width, height, ops = self._layout_uniform(
                parsed,
                vertical=vertical,
                font=font,
                ruby_font=ruby_font,
                font_size=font_size,
                ruby_size=ruby_size,
                line_height=line_height,
                letter_spacing_px=letter_spacing_px,
                upright=upright,
                margin=BODY_MARGIN,
                pad=pad,
            )

        base = Image.new("RGBA", (width, height), bg_rgba)
        if sfx:
            decor = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            if bool(params.get("sfx_speedlines", False)):
                if str(params.get("sfx_speedline_style", "radial")) == "parallel":
                    lines_layer = _draw_parallel_speedlines_layer(
                        width, height, n_lines=int(params.get("sfx_speedline_n", 40)), color=(0, 0, 0, 255)
                    )
                else:
                    lines_layer = _draw_speedlines_layer(
                        width, height, n_lines=int(params.get("sfx_speedline_n", 40)), color=(0, 0, 0, 255)
                    )
                decor = Image.alpha_composite(decor, lines_layer)
            if bool(params.get("sfx_burst", False)):
                decor = Image.alpha_composite(decor, _draw_burst_layer(width, height, ops))
            base = Image.alpha_composite(base, decor)

        main_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        _assign_op_colors(ops, params)
        kana_bank = self._attach_real_kana(ops, params, sfx=sfx, irregular=irregular)
        faint_k = float(params.get("faint_alpha", 0.65)) if bool(params.get("faint", False)) else 1.0
        if shadow_size > 0:
            shadow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            self._execute_ops(
                shadow_layer,
                ops,
                vertical=vertical,
                fill=shadow_rgba,
                stroke_width=stroke_size,
                stroke_fill=None,
                font_path=font_path,
                font_size=font_size,
                use_op_fill=False,
                lang=str(params.get("lang", "ja")),
                kana_bank=kana_bank,
            )
            shadow_layer = _blur_rgba_premultiplied(shadow_layer, shadow_size, gain=SHADOW_GLOW_GAIN)
            if faint_k < 1.0:
                shadow_layer = _scale_layer_alpha(shadow_layer, faint_k)
            base = Image.alpha_composite(base, shadow_layer)

        hollow = bool(params.get("hollow", False))
        if hollow and stroke_size <= 0:
            stroke_size = 2
        ink_texture = str(params.get("ink_texture", "solid"))

        extrude_layer = None
        if bool(params.get("extrude", False)):
            ref_fill = ops[0].get("fill", text_rgba) if ops else text_rgba
            ext_rgb = (0, 0, 0) if _relative_luminance(ref_fill) > 0.12 else (110, 110, 110)
            ext_fill = (*ext_rgb, 255)
            extrude_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            extrude_ops = [{**op, "fill": ext_fill, "stroke": ext_fill} for op in ops]
            self._execute_ops(
                extrude_layer,
                extrude_ops,
                vertical=vertical,
                fill=ext_fill,
                stroke_width=stroke_size,
                stroke_fill=ext_fill,
                font_path=font_path,
                font_size=font_size,
                use_op_fill=True,
                lang=str(params.get("lang", "ja")),
                kana_bank=kana_bank,
            )
            mag = int(np.clip(round(font_size * float(np.random.uniform(0.05, 0.11))), 2, 9))
            dx = mag * int(np.random.choice([-1, 1]))
            dy = mag * int(np.random.choice([-1, 1]))
            extrude_layer = _sweep_layer(extrude_layer, dx, dy)
            if faint_k < 1.0:
                extrude_layer = _scale_layer_alpha(extrude_layer, faint_k)
            base = Image.alpha_composite(base, extrude_layer)

        outer_layer = None
        if sfx and bool(params.get("sfx_double_outline", False)) and stroke_size > 0:
            outer_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            self._execute_ops(
                outer_layer,
                ops,
                vertical=vertical,
                fill=text_rgba,
                stroke_width=stroke_size + SFX_OUTER_RING,
                stroke_fill=(0, 0, 0, 255),
                font_path=font_path,
                font_size=font_size,
                use_op_fill=True,
                edge_override=(0, 0, 0, 255),
                lang=str(params.get("lang", "ja")),
                kana_bank=kana_bank,
            )
            if faint_k < 1.0:
                outer_layer = _scale_layer_alpha(outer_layer, faint_k)
            base = Image.alpha_composite(base, outer_layer)

        if hollow:
            hollow_ops = [{**op, "fill": (0, 0, 0, 0)} for op in ops]
            self._execute_ops(
                main_layer,
                hollow_ops,
                vertical=vertical,
                fill=(0, 0, 0, 0),
                stroke_width=max(2, stroke_size),
                stroke_fill=stroke_rgba,
                font_path=font_path,
                font_size=font_size,
                use_op_fill=True,
                lang=str(params.get("lang", "ja")),
                kana_bank=kana_bank,
            )
        else:
            self._execute_ops(
                main_layer,
                ops,
                vertical=vertical,
                fill=text_rgba,
                stroke_width=stroke_size,
                stroke_fill=stroke_rgba if stroke_size > 0 else None,
                font_path=font_path,
                font_size=font_size,
                use_op_fill=True,
                lang=str(params.get("lang", "ja")),
                kana_bank=kana_bank,
            )
        if ink_texture in ("gradient", "stripe"):
            main_layer = _apply_ink_modulation(main_layer, ink_texture)
        if bool(params.get("distress", False)):
            main_layer = _distress_rgba(main_layer, strength=float(params.get("distress_strength", 0.5)))
        main_layer = _ink_realism(main_layer)
        if bool(params.get("faint", False)):
            arr = np.array(main_layer)
            arr[:, :, 3] = (arr[:, :, 3].astype(np.float32) * faint_k).astype(np.uint8)
            main_layer = Image.fromarray(arr)
        base = Image.alpha_composite(base, main_layer)

        ink = None
        if sfx:
            ink_alpha = main_layer.getchannel("A")
            if outer_layer is not None:
                ink_alpha = ImageChops.lighter(ink_alpha, outer_layer.getchannel("A"))
            if extrude_layer is not None:
                ink_alpha = ImageChops.lighter(ink_alpha, extrude_layer.getchannel("A"))
            ink = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            ink.putalpha(ink_alpha)

        if sfx:
            block_rotate = float(params.get("sfx_block_rotate", 0.0))
            if block_rotate:
                base = base.rotate(block_rotate, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0))
                ink = ink.rotate(block_rotate, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0))

        slant = float(params.get("slant", 0.0))
        if slant:
            base = _shear_layer(base, slant)
            if ink is not None:
                ink = _shear_layer(ink, slant)

        if ink is not None:
            a = ink.getchannel("A").point(lambda v: 255 if v > SFX_INK_ALPHA_THRESHOLD else 0)
            bbox = a.getbbox()
            if bbox is not None:
                base = base.crop(bbox)

        img = np.array(base)
        img = img[:, :, [2, 1, 0, 3]]  # RGBA -> BGRA
        return np.ascontiguousarray(img), params

    def _attach_real_kana(self, ops, params, *, sfx=False, irregular=False):
        """Attach real-kana glyphs to eligible ops; return the bank or None.

        Opt-in only: enabled with ``real_kana=True`` on the renderer or in the
        per-sample params.  Disabled for non-Japanese text.  SFX keeps the
        decorative fonts unless ``real_kana_sfx`` is set explicitly.
        """

        if sfx and not bool(params.get("real_kana_sfx", False)):
            return None
        if str(params.get("lang", "ja")).lower() not in {"ja", "japanese", "jp"}:
            return None
        enabled = params.get("real_kana", None)
        if enabled is None:
            enabled = getattr(self, "real_kana", False)
        if not enabled:
            return None
        bank = getattr(self, "kana_bank", None)
        if isinstance(bank, (str, os.PathLike)):
            bank = _get_real_kana_bank(bank)
        if bank is None:
            bank = _get_real_kana_bank(params.get("kana_bank_root"))
        if bank is None or len(bank) == 0:
            return None
        cfg = getattr(bank, "cfg", None)
        try:
            from manga_ocr_dev.real_kana.bank import attach_real_kana

            default_prob = getattr(self, "real_kana_prob", None)
            if default_prob is None:
                default_prob = getattr(cfg, "real_kana_prob", REAL_KANA_PROB)
            prob = params.get("real_kana_prob", default_prob)
            seed = params.get("real_kana_seed", None)
            rng = np.random.default_rng(int(seed) if seed is not None else None)
            stats = attach_real_kana(ops, bank, cfg=cfg, rng=rng, probability=float(prob), lang="ja")
            params["real_kana_stats"] = {
                "attached": stats["attached"],
                "skipped": stats["skipped"],
                "probability": float(prob),
            }
        except Exception as exc:  # noqa: BLE001
            # A missing/broken bank must never break normal rendering.
            log.warning("real-kana rendering failed, using fonts: %s", exc)
            return None
        return bank

    @staticmethod
    def _layout_uniform(
        parsed,
        vertical,
        font,
        ruby_font,
        font_size,
        ruby_size,
        line_height,
        letter_spacing_px,
        upright,
        margin,
        pad,
    ):
        pitch_base = round(font_size * (1.0 + line_height))
        ops = []

        if vertical:
            counts = []
            has_ruby = []
            for tokens in parsed:
                n = 0
                ruby = False
                for tok in tokens:
                    if tok[0] == "text":
                        n += len(_strip_tags(tok[1]))
                    elif tok[0] == "ruby":
                        n += len(_strip_tags(tok[1]))
                        ruby = True
                    elif tok[0] == "tcy":
                        n += 1
                counts.append(n)
                has_ruby.append(ruby)
            inner_h = max(counts, default=0) * font_size
            height = max(1, inner_h + 2 * margin + 2 * pad)
            pitches = [max(pitch_base, font_size + (ruby_size + RUBY_GAP if r else 0) + RUBY_GAP) for r in has_ruby]
            width = max(1, sum(pitches) + 2 * margin + 2 * pad)

            slot_right = width - margin - pad
            y_top = margin + pad
            for li, (tokens, pitch, ruby) in enumerate(zip(parsed, pitches, has_ruby)):
                op_start = len(ops)
                slot_left = slot_right - pitch
                base_cx = slot_right - (ruby_size + RUBY_GAP if ruby else 0) - font_size / 2
                ruby_cx = slot_right - ruby_size / 2
                y = y_top
                for tok in tokens:
                    if tok[0] == "text":
                        for ch in _strip_tags(tok[1]):
                            cy = y + font_size / 2
                            ops.append(_char_op(ch, base_cx, cy, font, vertical, upright))
                            y += font_size
                    elif tok[0] == "ruby":
                        base = _strip_tags(tok[1])
                        rt = _strip_tags(tok[2])
                        block_h = len(base) * font_size
                        total_rt_h = len(rt) * ruby_size
                        ry = y + (block_h - total_rt_h) / 2
                        for ch in base:
                            cy = y + font_size / 2
                            ops.append(_char_op(ch, base_cx, cy, font, vertical, upright))
                            y += font_size
                        for ch in rt:
                            ops.append(
                                {
                                    "kind": "char",
                                    "ch": ch,
                                    "cx": ruby_cx,
                                    "cy": ry + ruby_size / 2,
                                    "font": ruby_font,
                                    "rotate": False,
                                    "ttb": vertical and not is_ascii(ch),
                                }
                            )
                            ry += ruby_size
                    elif tok[0] == "tcy":
                        cy = y + font_size / 2
                        ops.append({"kind": "tcy", "text": _strip_tags(tok[1]), "cx": base_cx, "cy": cy})
                        y += font_size
                for op in ops[op_start:]:
                    op["line"] = li
                slot_right = slot_left
        else:
            widths = []
            has_ruby = []
            for tokens in parsed:
                w = 0
                ruby = False
                for tok in tokens:
                    if tok[0] == "text":
                        for ch in _strip_tags(tok[1]):
                            w += _advance(font, ch, letter_spacing_px)
                    elif tok[0] == "ruby":
                        for ch in _strip_tags(tok[1]):
                            w += _advance(font, ch, letter_spacing_px)
                        ruby = True
                    elif tok[0] == "tcy":
                        for ch in _strip_tags(tok[1]):
                            w += _advance(font, ch, letter_spacing_px)
                widths.append(w)
                has_ruby.append(ruby)
            inner_w = max(widths, default=0)
            width = max(1, int(inner_w) + 2 * margin + 2 * pad)
            pitches = [max(pitch_base, font_size + (ruby_size + RUBY_GAP if r else 0) + RUBY_GAP) for r in has_ruby]
            height = max(1, sum(pitches) + 2 * margin + 2 * pad)

            x_left = margin + pad
            slot_top = margin + pad
            for li, (tokens, pitch, ruby) in enumerate(zip(parsed, pitches, has_ruby)):
                op_start = len(ops)
                base_cy = slot_top + (ruby_size + RUBY_GAP if ruby else 0) + font_size / 2
                ruby_cy = slot_top + ruby_size / 2
                x = x_left
                for tok in tokens:
                    if tok[0] == "text":
                        for ch in _strip_tags(tok[1]):
                            adv = _advance(font, ch, letter_spacing_px)
                            ops.append(
                                {
                                    "kind": "char",
                                    "ch": ch,
                                    "cx": x + adv / 2,
                                    "cy": base_cy,
                                    "font": font,
                                    "rotate": False,
                                    "ttb": False,
                                }
                            )
                            x += adv
                    elif tok[0] == "ruby":
                        base = _strip_tags(tok[1])
                        rt = _strip_tags(tok[2])
                        block_w = sum(_advance(font, ch, letter_spacing_px) for ch in base)
                        total_rt_w = sum(ruby_font.getlength(ch) for ch in rt)
                        rx = x + (block_w - total_rt_w) / 2
                        for ch in base:
                            adv = _advance(font, ch, letter_spacing_px)
                            ops.append(
                                {
                                    "kind": "char",
                                    "ch": ch,
                                    "cx": x + adv / 2,
                                    "cy": base_cy,
                                    "font": font,
                                    "rotate": False,
                                    "ttb": False,
                                }
                            )
                            x += adv
                        for ch in rt:
                            adv = ruby_font.getlength(ch)
                            ops.append(
                                {
                                    "kind": "char",
                                    "ch": ch,
                                    "cx": rx + adv / 2,
                                    "cy": ruby_cy,
                                    "font": ruby_font,
                                    "rotate": False,
                                    "ttb": False,
                                }
                            )
                            rx += adv
                    elif tok[0] == "tcy":
                        for ch in _strip_tags(tok[1]):
                            adv = _advance(font, ch, letter_spacing_px)
                            ops.append(
                                {
                                    "kind": "char",
                                    "ch": ch,
                                    "cx": x + adv / 2,
                                    "cy": base_cy,
                                    "font": font,
                                    "rotate": False,
                                    "ttb": False,
                                }
                            )
                            x += adv
                for op in ops[op_start:]:
                    op["line"] = li
                slot_top += pitch

        return width, height, ops

    @staticmethod
    def _layout_irregular(
        parsed,
        vertical,
        font_path,
        font_size,
        line_height,
        letter_spacing,
        upright,
        margin,
        pad,
        line_scale_min=LINE_SCALE_RANGE[0],
        line_scale_max=LINE_SCALE_RANGE[1],
        line_shift=LINE_SHIFT_EM,
        char_scale_min=CHAR_SCALE_RANGE[0],
        char_scale_max=CHAR_SCALE_RANGE[1],
        char_jitter_prob=CHAR_JITTER_PROB,
        wobble=WOBBLE_EM,
        rotate=ROTATE_DEG,
        small_kana=True,
    ):
        n_lines = len(parsed)
        line_scales = [float(np.random.uniform(line_scale_min, line_scale_max)) for _ in range(n_lines)]
        line_fs = [max(MIN_CHAR_SIZE, round(font_size * s)) for s in line_scales]
        line_jitter = [bool(np.random.rand() < char_jitter_prob) for _ in range(n_lines)]
        raw_starts = [float(np.random.uniform(-line_shift, line_shift)) * font_size for _ in range(n_lines)]

        spacing_em = float(letter_spacing) if letter_spacing else 0.0

        def scaled_font(size):
            return _get_font(font_path, max(MIN_CHAR_SIZE, int(size)))

        def char_spec(ch, fs, jitter_on):
            scale = 1.0
            if small_kana and ch in SMALL_KANA:
                scale *= float(np.random.uniform(*SMALL_KANA_SCALE))
            if jitter_on:
                scale *= float(np.random.uniform(char_scale_min, char_scale_max))
            size = max(MIN_CHAR_SIZE, round(fs * scale))
            angle = float(np.random.uniform(-rotate, rotate)) if rotate else 0.0
            cross = float(np.random.uniform(-wobble, wobble)) * fs if wobble else 0.0
            return scaled_font(size), size, angle, cross

        pitches = []
        ruby_sizes = []
        segments_per_line = []
        lengths = []
        for li, tokens in enumerate(parsed):
            fs = line_fs[li]
            ruby_size = max(MIN_CHAR_SIZE, round(fs * RUBY_SCALE))
            ruby_sizes.append(ruby_size)
            sp = round(spacing_em * fs)
            has_ruby = any(tok[0] == "ruby" for tok in tokens)
            pitch = round(fs * (1.0 + line_height))
            if has_ruby:
                pitch = max(pitch, fs + ruby_size + 2 * RUBY_GAP)
            else:
                pitch = max(pitch, fs + RUBY_GAP)
            pitches.append(pitch)
            base_font = scaled_font(fs)
            rt_font = scaled_font(ruby_size)

            segments = []
            total = 0
            for tok in tokens:
                if tok[0] == "text":
                    specs = []
                    seg_len = 0
                    for ch in _strip_tags(tok[1]):
                        font, cell, angle, cross = char_spec(ch, fs, line_jitter[li])
                        adv = (cell + sp) if vertical else _advance(font, ch, sp)
                        specs.append((ch, font, cell, adv, angle, cross))
                        seg_len += adv
                    segments.append(("chars", specs, seg_len))
                    total += seg_len
                elif tok[0] == "ruby":
                    base = _strip_tags(tok[1])
                    rt = _strip_tags(tok[2])
                    if vertical:
                        base_advs = [fs + sp for _ in base]
                        rt_len = len(rt) * ruby_size
                    else:
                        base_advs = [_advance(base_font, ch, sp) for ch in base]
                        rt_len = sum(rt_font.getlength(ch) for ch in rt)
                    seg_len = sum(base_advs)
                    segments.append(("ruby", base, rt, base_advs, seg_len, rt_len))
                    total += seg_len
                elif tok[0] == "tcy":
                    text = _strip_tags(tok[1])
                    if vertical:
                        adv = fs + sp
                    else:
                        adv = sum(_advance(base_font, ch, 0) for ch in text) + sp
                    angle = float(np.random.uniform(-rotate, rotate)) if rotate else 0.0
                    cross = float(np.random.uniform(-wobble, wobble)) * fs if wobble else 0.0
                    segments.append(("tcy", text, adv, angle, cross))
                    total += adv
            segments_per_line.append(segments)
            lengths.append(total)

        min_start = min(raw_starts, default=0.0)
        starts = [s - min_start for s in raw_starts]
        ends = [s + ln for s, ln in zip(starts, lengths)]
        inner = max(ends, default=0)

        max_fs = max(line_fs, default=font_size)
        wob_px = abs(wobble) * max_fs
        rot_px = math.sin(math.radians(abs(rotate))) * max_fs if rotate else 0.0
        pad_irr = pad + math.ceil(wob_px + rot_px)

        ops = []
        if vertical:
            width = max(1, sum(pitches) + 2 * margin + 2 * pad_irr)
            height = max(1, math.ceil(inner) + 2 * margin + 2 * pad_irr)
            slot_right = width - margin - pad_irr
            y_top = margin + pad_irr
            for li, (pitch, fs, ruby_size, start) in enumerate(zip(pitches, line_fs, ruby_sizes, starts)):
                op_start = len(ops)
                has_ruby = any(tok[0] == "ruby" for tok in parsed[li])
                slot_left = slot_right - pitch
                base_cx = slot_right - (ruby_size + RUBY_GAP if has_ruby else 0) - fs / 2
                ruby_cx = slot_right - ruby_size / 2
                base_font = scaled_font(fs)
                rt_font = scaled_font(ruby_size)
                y = y_top + start
                for seg in segments_per_line[li]:
                    if seg[0] == "chars":
                        for ch, font, cell, adv, angle, cross in seg[1]:
                            cy = y + adv / 2
                            ops.append(_char_op_ex(ch, base_cx + cross, cy, font, cell, angle, vertical, upright))
                            y += adv
                    elif seg[0] == "ruby":
                        _, base, rt, base_advs, seg_len, rt_len = seg
                        ry = y + (seg_len - rt_len) / 2
                        for ch, adv in zip(base, base_advs):
                            cy = y + adv / 2
                            ops.append(_char_op_ex(ch, base_cx, cy, base_font, fs, 0.0, vertical, upright))
                            y += adv
                        for ch in rt:
                            ops.append(
                                {
                                    "kind": "char",
                                    "ch": ch,
                                    "cx": ruby_cx,
                                    "cy": ry + ruby_size / 2,
                                    "font": rt_font,
                                    "cell": ruby_size,
                                    "angle": 0.0,
                                    "rotate": False,
                                    "ttb": vertical and not is_ascii(ch),
                                }
                            )
                            ry += ruby_size
                    elif seg[0] == "tcy":
                        _, text, adv, angle, cross = seg
                        cy = y + adv / 2
                        ops.append(
                            {
                                "kind": "tcy",
                                "text": text,
                                "cx": base_cx + cross,
                                "cy": cy,
                                "cell": fs,
                                "angle": angle,
                            }
                        )
                        y += adv
                for op in ops[op_start:]:
                    op["line"] = li
                slot_right = slot_left
        else:
            width = max(1, math.ceil(inner) + 2 * margin + 2 * pad_irr)
            height = max(1, sum(pitches) + 2 * margin + 2 * pad_irr)
            x_left = margin + pad_irr
            slot_top = margin + pad_irr
            for li, (pitch, fs, ruby_size, start) in enumerate(zip(pitches, line_fs, ruby_sizes, starts)):
                op_start = len(ops)
                has_ruby = any(tok[0] == "ruby" for tok in parsed[li])
                base_cy = slot_top + (ruby_size + RUBY_GAP if has_ruby else 0) + fs / 2
                ruby_cy = slot_top + ruby_size / 2
                base_font = scaled_font(fs)
                rt_font = scaled_font(ruby_size)
                sp = round(spacing_em * fs)
                x = x_left + start
                for seg in segments_per_line[li]:
                    if seg[0] == "chars":
                        for ch, font, cell, adv, angle, cross in seg[1]:
                            ops.append(
                                {
                                    "kind": "char",
                                    "ch": ch,
                                    "cx": x + adv / 2,
                                    "cy": base_cy + cross,
                                    "font": font,
                                    "cell": cell,
                                    "angle": angle,
                                    "rotate": False,
                                    "ttb": False,
                                }
                            )
                            x += adv
                    elif seg[0] == "ruby":
                        _, base, rt, base_advs, seg_len, rt_len = seg
                        rx = x + (seg_len - rt_len) / 2
                        for ch, adv in zip(base, base_advs):
                            ops.append(
                                {
                                    "kind": "char",
                                    "ch": ch,
                                    "cx": x + adv / 2,
                                    "cy": base_cy,
                                    "font": base_font,
                                    "cell": fs,
                                    "angle": 0.0,
                                    "rotate": False,
                                    "ttb": False,
                                }
                            )
                            x += adv
                        for ch in rt:
                            adv = rt_font.getlength(ch)
                            ops.append(
                                {
                                    "kind": "char",
                                    "ch": ch,
                                    "cx": rx + adv / 2,
                                    "cy": ruby_cy,
                                    "font": rt_font,
                                    "cell": ruby_size,
                                    "angle": 0.0,
                                    "rotate": False,
                                    "ttb": False,
                                }
                            )
                            rx += adv
                    elif seg[0] == "tcy":
                        _, text, _adv, angle, cross = seg
                        for ch in text:
                            adv = _advance(base_font, ch, 0)
                            ops.append(
                                {
                                    "kind": "char",
                                    "ch": ch,
                                    "cx": x + adv / 2,
                                    "cy": base_cy + cross,
                                    "font": base_font,
                                    "cell": fs,
                                    "angle": angle,
                                    "rotate": False,
                                    "ttb": False,
                                }
                            )
                            x += adv
                        x += sp
                for op in ops[op_start:]:
                    op["line"] = li
                slot_top += pitch

        return width, height, ops

    @staticmethod
    def _layout_sfx(
        parsed,
        vertical,
        font_path,
        font_size,
        line_height,
        letter_spacing,
        upright,
        margin,
        pad,
        char_scale_min=SFX_CHAR_SCALE_RANGE[0],
        char_scale_max=SFX_CHAR_SCALE_RANGE[1],
        curve="arc",
        curve_amp=0.3,
        curve_sign=1.0,
        persp=0.2,
        rotate=SFX_ROTATE_DEG / 1.5,
        wobble=0.05,
        small_kana=True,
        aspect=0.0,
    ):
        n_lines = max(1, len(parsed))
        line_scales = [float(np.random.uniform(*SFX_LINE_SCALE_RANGE)) for _ in range(n_lines)]
        line_fs = [max(MIN_CHAR_SIZE, round(font_size * s)) for s in line_scales]
        spacing_em = float(letter_spacing) if letter_spacing else 0.0
        sign = 1.0 if float(curve_sign) >= 0 else -1.0
        amp_em = abs(float(curve_amp))
        persp_max = abs(float(persp))
        aspect_max = abs(float(aspect))
        rot_max = abs(float(rotate))
        wob_em = abs(float(wobble))
        mode = str(curve) if str(curve) in SFX_CURVE_MODES else "arc"
        freq = float(np.random.uniform(1.0, 1.75)) if mode == "wave" else 1.0
        freq_zz = float(np.random.choice([1.5, 2.0, 2.5, 3.0]))

        def scaled_font(size):
            return _get_font(font_path, max(MIN_CHAR_SIZE, int(size)))

        def char_spec(ch, fs, prev_scale, cap_ref, cap_mult):
            scale = float(np.random.uniform(char_scale_min, char_scale_max))
            if prev_scale is not None:
                scale = float(np.clip(scale, prev_scale / SFX_ADJACENT_SCALE_MAX, prev_scale * SFX_ADJACENT_SCALE_MAX))
            base_scale = scale
            is_small = bool(small_kana and ch in SMALL_KANA)
            if is_small:
                scale *= float(np.random.uniform(*SMALL_KANA_SCALE))
                if cap_ref is not None:
                    scale = min(scale, cap_ref * cap_mult)
            else:
                np.random.uniform(*SMALL_KANA_SCALE)
            size = max(MIN_CHAR_SIZE, round(fs * scale))
            angle = float(np.random.uniform(-1.0, 1.0)) * rot_max
            cross = float(np.random.uniform(-1.0, 1.0)) * wob_em * fs
            p = float(np.random.uniform(0.0, 1.0)) * persp_max
            ax = 1.0 + float(np.random.uniform(-1.0, 1.0)) * aspect_max
            ay = 1.0 + float(np.random.uniform(-1.0, 1.0)) * aspect_max
            return scaled_font(size), size, angle, cross, p, (ax, ay), base_scale, scale, is_small

        def flat_chars(tokens):
            out = []
            for tok in tokens:
                if tok[0] in ("text", "tcy") or tok[0] == "ruby":
                    out.extend(list(_strip_tags(tok[1])))
            return out

        def curve_offset(t):
            if mode == "arc":
                return sign * amp_em * 4.0 * t * (1.0 - t)
            if mode == "wave":
                return sign * amp_em * math.sin(2.0 * math.pi * freq * t)
            if mode == "diagonal":
                return sign * amp_em * 2.0 * (t - 0.5)
            if mode == "zigzag":
                tri = 2.0 * abs(2.0 * ((t * freq_zz) % 1.0) - 1.0) - 1.0
                return sign * amp_em * tri * 0.6
            if mode == "bounce":
                return sign * amp_em * abs(math.sin(math.pi * freq_zz * t)) * 1.2 - sign * amp_em * 0.6
            if mode == "hill":
                return sign * amp_em * (math.sin(math.pi * min(1.0, t**0.6)) * 1.1 - 0.4)
            if mode == "steep":
                return sign * amp_em * 2.2 * (t - 0.5)
            if mode == "circle":
                return sign * amp_em * (math.cos(math.pi * (t - 0.5) * 1.6) * 1.4 - 0.5)
            return 0.0

        def curve_slope(t):
            e = 1e-3
            return (curve_offset(min(1.0, t + e)) - curve_offset(max(0.0, t - e))) / (2.0 * e)

        specs_per_line = []
        totals = []
        pitches = []
        for li, tokens in enumerate(parsed):
            fs = line_fs[li]
            sp = round(spacing_em * fs)
            chars = flat_chars(tokens)
            specs = []
            total = 0
            prev_scale = None
            cap_ref, cap_mult = None, SMALL_KANA_SCALE[1]
            for ch in chars:
                font, cell, angle, cross, p, asp, prev_scale, final_scale, was_small = char_spec(
                    ch, fs, prev_scale, cap_ref, cap_mult
                )
                cap_ref, cap_mult = (final_scale, 1.0) if was_small else (final_scale, SMALL_KANA_SCALE[1])
                adv = (cell + sp) if vertical else _advance(font, ch, sp)
                rad = math.radians(angle)
                adv = adv * (1.0 + 0.8 * (abs(math.cos(rad)) + abs(math.sin(rad)) - 1.0) + 0.6 * p)
                adv = adv * (asp[1] if vertical else asp[0]) if aspect_max else adv
                specs.append([ch, font, cell, adv, angle, cross, p, asp])
                total += adv
            specs_per_line.append((fs, sp, specs))
            totals.append(total)
            pitches.append(round(fs * (1.0 + line_height)) + RUBY_GAP)

        ops = []
        line_origins = []
        acc = 0
        for pitch in pitches:
            line_origins.append(acc)
            acc += pitch
        if vertical:
            line_origins = [acc - o - p for o, p in zip(line_origins, pitches)]
        for li, (fs, sp, specs) in enumerate(specs_per_line):
            total = totals[li] or 1
            pos = 0
            for ch, font, cell, adv, angle, cross, p, asp in specs:
                t = (pos + adv / 2) / total
                off_px = curve_offset(t) * fs
                slope = curve_slope(t)
                tangent_deg = math.degrees(math.atan(slope * fs / max(total, 1)))
                if vertical:
                    cx = line_origins[li] + fs / 2 + off_px + cross
                    cy = pos + adv / 2
                    final_angle = angle + tangent_deg
                else:
                    cx = pos + adv / 2
                    cy = line_origins[li] + fs / 2 + off_px + cross
                    final_angle = angle + tangent_deg
                if vertical and is_ascii(ch) and not upright and not ch.isspace():
                    rot_flag = True
                else:
                    rot_flag = False
                ops.append(
                    {
                        "kind": "char",
                        "ch": ch,
                        "cx": cx,
                        "cy": cy,
                        "font": font,
                        "cell": cell,
                        "angle": final_angle,
                        "persp": p,
                        "pcoef": tuple(float(v) for v in np.random.uniform(-1.0, 1.0, 8)),
                        "aspect": asp,
                        "rotate": rot_flag,
                        "ttb": vertical and not is_ascii(ch),
                        "line": li,
                    }
                )
                pos += adv

        if not ops:
            width = max(1, 2 * margin + 2 * pad)
            height = max(1, 2 * margin + 2 * pad)
            return width, height, ops

        max_cell = max(op.get("cell", font_size) for op in ops)
        extra = math.ceil(max_cell * 0.9 + amp_em * max(line_fs))
        xs = [op["cx"] for op in ops]
        ys = [op["cy"] for op in ops]
        xmin, xmax = min(xs) - max_cell, max(xs) + max_cell
        ymin, ymax = min(ys) - max_cell, max(ys) + max_cell
        width = max(1, math.ceil(xmax - xmin) + 2 * margin + 2 * pad + 2 * extra)
        height = max(1, math.ceil(ymax - ymin) + 2 * margin + 2 * pad + 2 * extra)
        dx = margin + pad + extra - xmin
        dy = margin + pad + extra - ymin
        for op in ops:
            op["cx"] += dx
            op["cy"] += dy
        return width, height, ops

    @staticmethod
    def _execute_ops(
        layer,
        ops,
        vertical,
        fill,
        stroke_width,
        stroke_fill,
        font_path,
        font_size,
        use_op_fill=True,
        edge_override=None,
        lang="ja",
        kana_bank=None,
    ):
        from manga_ocr_dev.synthetic_data_generator.lang import TTB_LANGUAGE

        ttb_lang = TTB_LANGUAGE.get(str(lang), "ja")
        draw = ImageDraw.Draw(layer)
        if any("persp" in op for op in ops):
            ops = sorted(ops, key=lambda o: -float(o.get("cell") or 0))
        for op in ops:
            angle = float(op.get("angle", 0) or 0)
            persp = float(op.get("persp", 0) or 0)
            if use_op_fill:
                op_fill = op.get("fill", fill)
                op_edge = edge_override if edge_override is not None else op.get("stroke", stroke_fill)
            else:
                op_fill = op.get("shadow", fill)
                op_edge = op_fill
            if kana_bank is not None and op.get("kana"):
                from manga_ocr_dev.real_kana.bank import draw_bank_op

                if draw_bank_op(
                    layer,
                    op,
                    kana_bank,
                    fill=op_fill,
                    stroke_width=stroke_width,
                    stroke_fill=op_edge,
                    angle=angle,
                ):
                    continue
            if op["kind"] == "tcy":
                _draw_tcy_tile(
                    layer,
                    op["text"],
                    op["cx"],
                    op["cy"],
                    font_path=font_path,
                    font_size=op.get("cell") or font_size,
                    fill=op_fill,
                    stroke_width=stroke_width,
                    stroke_fill=op_edge,
                    angle=angle,
                )
            elif persp or op.get("aspect"):
                _draw_perspective_char(
                    layer,
                    op["ch"],
                    op["cx"],
                    op["cy"],
                    font=op["font"],
                    cell=op.get("cell") or font_size,
                    angle=angle,
                    persp=persp,
                    fill=op_fill,
                    stroke_width=stroke_width,
                    stroke_fill=op_edge,
                    rotate_mixed=bool(op.get("rotate", False)),
                    ttb=bool(op.get("ttb", False)),
                    lang=ttb_lang,
                    aspect=op.get("aspect"),
                    pcoef=op.get("pcoef"),
                )
            elif op.get("rotate"):
                _draw_rotated_char(
                    layer,
                    op["ch"],
                    op["cx"],
                    op["cy"],
                    font=op["font"],
                    cell=op.get("cell") or font_size,
                    fill=op_fill,
                    stroke_width=stroke_width,
                    stroke_fill=op_edge,
                    angle=angle,
                )
            elif angle:
                _draw_tilted_char(
                    layer,
                    op["ch"],
                    op["cx"],
                    op["cy"],
                    font=op["font"],
                    cell=op.get("cell") or font_size,
                    angle=angle,
                    fill=op_fill,
                    stroke_width=stroke_width,
                    stroke_fill=op_edge,
                )
            else:
                kw = {"font": op["font"], "fill": op_fill, "anchor": "mm"}
                if op.get("ttb"):
                    kw.update(direction="ttb", language=ttb_lang)
                if stroke_width:
                    kw.update(stroke_width=stroke_width, stroke_fill=op_edge)
                draw.text((op["cx"], op["cy"]), op["ch"], **kw)

    @staticmethod
    def get_random_sfx_params(lang="ja"):
        from manga_ocr_dev.synthetic_data_generator.lang import SFX_VERTICAL_PROB, normalize_lang

        lang = normalize_lang(lang)
        font_size = int(np.random.uniform(*SFX_FONT_SIZE_RANGE))
        text_color = _sample_text_color(sfx=True)
        if np.random.rand() < 0.2:
            stroke_choice = int(np.random.choice([0, 1, 2]))
        else:
            stroke_choice = int(np.random.choice([2, 3, 4, 5, 6, 7]))
        stroke_size = min(stroke_choice, max(2 if stroke_choice >= 2 else 0, round(SFX_STROKE_FRAC * font_size) + 2))
        if font_size < 55:
            stroke_size = min(stroke_size, 3)
        params = {
            "sfx": True,
            "lang": lang,
            "font_size": font_size,
            "vertical": bool(np.random.rand() < SFX_VERTICAL_PROB.get(lang, 0.4)),
            "line_height": float(np.random.uniform(0.18, 0.42)),
            "background_color": "transparent",
            "text_color": text_color,
            "color_mode": "solid",
            "irregular": False,
            "stroke_size": stroke_size,
            "stroke_color": "white",
            "sfx_curve": str(np.random.choice(SFX_CURVE_MODES, p=SFX_CURVE_P)),
            "sfx_aspect": float(np.random.uniform(0.0, SFX_ASPECT_MAX)) if np.random.rand() < 0.65 else 0.0,
            "sfx_speedline_style": str(np.random.choice(["radial", "parallel"], p=[0.6, 0.4])),
            "letter_spacing": float(np.random.uniform(*SFX_SPACING_RANGE)),
            "sfx_curve_amp": float(np.random.uniform(*SFX_CURVE_AMP_RANGE)),
            "sfx_curve_sign": float(np.random.choice([-1.0, 1.0])),
            "sfx_persp": float(np.random.uniform(*SFX_PERSP_RANGE)),
            "sfx_rotate": float(np.random.uniform(6.0, SFX_ROTATE_DEG)),
            "sfx_char_scale_min": float(SFX_CHAR_SCALE_RANGE[0]),
            "sfx_char_scale_max": float(np.random.uniform(1.5, SFX_CHAR_SCALE_RANGE[1])),
            "sfx_wobble": float(np.random.uniform(0.03, 0.12)),
            "sfx_speedlines": bool(np.random.rand() < SFX_SPEEDLINE_PROB),
            "sfx_speedline_n": int(np.random.randint(*SFX_SPEEDLINE_N_RANGE)),
            "sfx_burst": bool(np.random.rand() < SFX_BURST_PROB),
            "sfx_double_outline": bool(np.random.rand() < SFX_DOUBLE_OUTLINE_PROB),
            "sfx_block_rotate": (float(np.random.uniform(*SFX_BLOCK_ROTATE_RANGE)) if np.random.rand() < 0.55 else 0.0),
        }
        if np.random.rand() < SFX_BLOCK_ROTATE_90_PROB:
            params["sfx_block_rotate"] = float(np.random.uniform(80, 100) * np.random.choice([-1.0, 1.0]))
        if np.random.rand() < HOLLOW_PROB:
            params["hollow"] = True
        r = np.random.rand()
        if r < GRADIENT_FILL_PROB:
            params["ink_texture"] = "gradient"
        elif r < GRADIENT_FILL_PROB + STRIPE_FILL_PROB:
            params["ink_texture"] = "stripe"
        if np.random.rand() < EXTRUDE_PROB:
            params["extrude"] = True
        if np.random.rand() < DISTRESS_PROB:
            params["distress"] = True
            params["distress_strength"] = float(np.random.uniform(0.3, 0.9))
        if np.random.rand() < FAINT_PROB:
            params["faint"] = True
            params["faint_alpha"] = float(np.random.uniform(0.45, 0.75))
        if np.random.rand() < 0.7:
            params["text_orientation"] = "upright"
        if text_color == "black" and np.random.rand() < 0.2:
            params["text_color"] = "white"
            params["stroke_color"] = "black"
        if np.random.rand() < 0.15:
            params["slant"] = float(np.random.uniform(*SLANT_MILD_RANGE) * np.random.choice([-1.0, 1.0]))
        _maybe_multicolor(params, text_color, sfx=True)
        if np.random.rand() < BG_CAMO_PROB:
            return params
        return _ensure_legible_outline(params)

    @staticmethod
    def get_random_css_params(sfx=False, lang="ja"):
        from manga_ocr_dev.synthetic_data_generator.lang import (
            DIALOGUE_VERTICAL_PROB,
            normalize_lang,
        )

        lang = normalize_lang(lang)
        if sfx:
            return Renderer.get_random_sfx_params(lang=lang)
        text_color = _sample_text_color(sfx=False)
        params = {
            "lang": lang,
            "font_size": int(np.random.uniform(*DIALOGUE_FONT_SIZE_RANGE)),
            "vertical": bool(np.random.rand() < DIALOGUE_VERTICAL_PROB.get(lang, 0.7)),
            "line_height": float(np.random.uniform(*DIALOGUE_LINE_HEIGHT_RANGE)),
            "background_color": "transparent",
            "text_color": text_color,
            "color_mode": "solid",
        }

        if np.random.rand() < 0.7:
            params["text_orientation"] = "upright"

        stroke_variant = np.random.choice(
            ["stroke", "shadow", "none"], p=[STROKE_PROB, SHADOW_PROB, 1.0 - STROKE_PROB - SHADOW_PROB]
        )
        if stroke_variant == "stroke":
            params["stroke_size"] = int(np.random.choice(STROKE_SIZES, p=STROKE_SIZE_P))
            params["stroke_color"] = "white"
        elif stroke_variant == "shadow":
            params["shadow_size"] = np.random.choice([1, 2, 3])
            params["shadow_color"] = "white" if np.random.rand() < 0.8 else "black"

        if np.random.rand() < IRREGULAR_PROB:
            params["irregular"] = True
            if np.random.rand() < 0.4:
                params.update(
                    line_scale_min=0.8,
                    line_scale_max=1.25,
                    line_shift=0.8,
                    char_jitter_prob=0.0,
                    wobble=0.05,
                    rotate=0.0,
                    small_kana=False,
                )
            else:
                params.update(
                    line_scale_min=0.55,
                    line_scale_max=1.65,
                    line_shift=1.6,
                    char_jitter_prob=0.65,
                    char_scale_min=0.6,
                    char_scale_max=1.65,
                    wobble=0.10,
                    rotate=float(np.random.uniform(2, 8)),
                    small_kana=(lang == "ja"),
                )

        if np.random.rand() < SLANT_PROB:
            if np.random.rand() < SLANT_STRONG_PROB:
                low, high = SLANT_STRONG_RANGE
            else:
                low, high = SLANT_MILD_RANGE
            params["slant"] = float(np.random.uniform(low, high) * np.random.choice([-1.0, 1.0]))

        if np.random.rand() < HOLLOW_PROB * 0.4:
            params["hollow"] = True
        if np.random.rand() < DISTRESS_PROB * 0.6:
            params["distress"] = True
            params["distress_strength"] = float(np.random.uniform(0.25, 0.6))
        if np.random.rand() < FAINT_PROB * 0.7:
            params["faint"] = True
            params["faint_alpha"] = float(np.random.uniform(0.5, 0.8))
        if np.random.rand() < EXTRUDE_PROB * 0.4:
            params["extrude"] = True

        _maybe_multicolor(params, text_color, sfx=False)

        if np.random.rand() < BG_CAMO_PROB * 0.7:
            return params
        return _ensure_legible_outline(params)

    def _load_background(self, h, w):
        df = self.background_df
        lr = getattr(self, "_bg_logratio", None)
        if lr is None or len(lr) != len(df):
            lr = np.log(df["w"].to_numpy(dtype=np.float64) / df["h"].to_numpy(dtype=np.float64))
            self._bg_logratio = lr
        d = np.abs(lr - np.log(w / h))
        k = min(len(df), 64)
        near = np.argpartition(d, k - 1)[:k]
        bg = None
        for _ in range(8):
            row = df.iloc[int(np.random.choice(near))]
            bg = _read_bg_cached(row.path)
            if bg is not None:
                break
            near = np.arange(len(df))
        if bg is None:
            raise FileNotFoundError("no readable background image")

        bh, bw = bg.shape[:2]
        if bw / bh > w / h:
            wh, ww = bh, bh * w / h
        else:
            ww, wh = bw, bw * h / w
        s_min = min(1.0, max(0.45, 0.6 * max(w / max(ww, 1), h / max(wh, 1))))
        sc = np.random.uniform(s_min, 1.0)
        ww, wh = max(8, int(ww * sc)), max(8, int(wh * sc))
        ww, wh = min(ww, bw), min(wh, bh)
        y0 = int(np.random.randint(0, bh - wh + 1))
        x0 = int(np.random.randint(0, bw - ww + 1))
        patch = bg[y0 : y0 + wh, x0 : x0 + ww]
        if np.random.rand() < 0.5:
            patch = patch[:, ::-1]
        if np.random.rand() < 0.5:
            patch = patch[::-1]
        interp = cv2.INTER_AREA if ww * wh > w * h else cv2.INTER_LINEAR
        patch = cv2.resize(np.ascontiguousarray(patch), (w, h), interpolation=interp)
        if np.random.rand() < 0.5:
            patch = 255 - patch
        if np.random.rand() < BG_BLUR_PROB:
            patch = cv2.GaussianBlur(patch, (3, 3), 0)
        return patch

    @staticmethod
    def _ink_edge(img):
        solid = (img[:, :, 3] > 200).astype(np.uint8)
        if not solid.any():
            return None, None
        k = np.ones((3, 3), np.uint8)
        edge = (solid - cv2.erode(solid, k)).astype(bool)
        if not edge.any():
            edge = solid.astype(bool)
        bgr = img[:, :, :3][edge].astype(np.float32)
        lum = float((0.114 * bgr[:, 0] + 0.587 * bgr[:, 1] + 0.299 * bgr[:, 2]).mean())
        ring = cv2.dilate(solid, np.ones((7, 7), np.uint8)).astype(bool) & ~solid.astype(bool)
        return lum, ring

    @staticmethod
    def _low_contrast_frac(bg, edge_lum, ring):
        if edge_lum is None or ring is None or not ring.any():
            return 0.0
        g = bg.astype(np.float32).mean(axis=2)[ring]
        return float((np.abs(g - edge_lum) < BG_MIN_EDGE_DIFF).mean())

    @staticmethod
    def _ink_mean_bgr(img):
        try:
            solid = img[:, :, 3] > 200
            if not solid.any():
                return None
            return img[:, :, :3][solid].mean(axis=0)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _ink_is_chromatic_bgra(img):
        mean = Renderer._ink_mean_bgr(img)
        if mean is None:
            return False
        return float(mean.max() - mean.min()) > 25.0

    def _make_background(self, bg0, img):
        edge_lum, ring = self._ink_edge(img)
        ink_lum = 0.0 if edge_lum is None else edge_lum
        chromatic_ink = self._ink_is_chromatic_bgra(img)
        for _ in range(BG_MAX_TRIES):
            bg = self._decorate_background(self._adjust_background(bg0, ink_lum))
            if chromatic_ink and np.random.rand() < BG_COLOR_ART_PROB:
                ref = self._ink_mean_bgr(img)
                if ref is not None:
                    bg = _colorize_background(bg, ref, strength=float(np.random.uniform(0.35, 0.7)))
            if self._low_contrast_frac(bg, edge_lum, ring) <= BG_MAX_LOW_CONTRAST:
                return bg
        if np.random.rand() < BG_CAMO_PROB:
            bg = self._decorate_background(self._adjust_background(bg0, ink_lum))
            if chromatic_ink:
                ref = self._ink_mean_bgr(img)
                if ref is not None:
                    bg = _colorize_background(bg, ref, strength=0.6)
            return bg
        far = 70.0 if ink_lum >= 128 else 195.0
        f = bg0.astype(np.float32)
        f = (f - f.mean()) * 1.0 + far
        bg = np.clip(f, 0, 255).astype(np.uint8)
        if self._low_contrast_frac(bg, edge_lum, ring) > BG_MAX_LOW_CONTRAST:
            bg = np.full_like(bg0, int(far))
        return bg

    @staticmethod
    def _adjust_background(bg, ink_lum):
        f = bg.astype(np.float32)
        gray = f.mean(axis=2)
        m, sd = float(gray.mean()), float(gray.std())
        mode = str(np.random.choice(BG_TONE_MODES, p=BG_TONE_P))
        if mode == "midtone":
            target = np.random.uniform(85, 175)
        elif mode == "ink_match":
            off = np.random.uniform(*BG_INK_OFFSET) * np.random.choice([-1.0, 1.0])
            target = float(np.clip(ink_lum + off, 35, 220))
        else:
            target = m
        sd_t = max(sd, np.random.uniform(*BG_MIN_STD))
        gain = float(np.clip(sd_t / max(sd, 1.0), 1.0, 3.0))
        f = (f - m) * gain + target
        return np.clip(f, 0, 255).astype(np.uint8)

    @staticmethod
    def _decorate_background(bg):
        h, w = bg.shape[:2]
        f = bg.astype(np.float32)
        if np.random.rand() < BG_GRADIENT_PROB:
            ang = np.random.uniform(0, 2 * np.pi)
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            g = (xx - w / 2) * np.cos(ang) + (yy - h / 2) * np.sin(ang)
            g /= max(1.0, float(np.abs(g).max()))
            f += (np.random.uniform(25, 70) * g)[..., None]
        if np.random.rand() < BG_SCREENTONE_PROB:
            dots = _screentone(h, w)
            f *= (1.0 - np.random.uniform(0.2, 0.55) * dots)[..., None]
        if np.random.rand() < BG_HATCH_PROB:
            f *= (1.0 - _hatch_overlay(h, w, strength=float(np.random.uniform(0.2, 0.45))))[..., None]
        if np.random.rand() < BG_NOISE_PROB:
            f += np.random.normal(0, np.random.uniform(3, 10), (h, w, 1)).astype(np.float32)
        return np.clip(f, 0, 255).astype(np.uint8)

    @staticmethod
    def _sample_bubble_fill():
        mode = str(np.random.choice(BUBBLE_FILL_MODES, p=BUBBLE_FILL_P))
        black, white = (0, 0, 0), (255, 255, 255)
        if mode == "gray":
            v = int(np.random.uniform(195, 243))
            return mode, (v, v, v), black
        if mode == "tint":
            c = np.random.uniform(200, 255, 3)
            c[int(np.argmax(c))] = 255
            return mode, tuple(int(x) for x in c), black
        if mode == "dark":
            v = int(np.random.uniform(15, 70))
            w = int(np.random.uniform(190, 255))
            return mode, (v, v, v), (w, w, w)
        return mode, white, black

    def render_background(self, img, draw_bubble=None, tight=False, return_ink_box=False):
        if draw_bubble is None:
            draw_bubble = np.random.random() < 0.7

        if tight:
            draw_bubble = False
            side = min(img.shape[:2])
            m0 = max(SFX_CROP_MARGIN_MIN, int(side * np.random.uniform(*SFX_CROP_MARGIN_FRAC)))
        else:
            m0 = int(min(img.shape[:2]) * 0.3)
        img = crop_by_alpha(img, m0)
        ink_box = (m0, m0, img.shape[1] - m0, img.shape[0] - m0)

        fill_mode, fill_bgr, contour_bgr = "white", (255, 255, 255), (0, 0, 0)
        if draw_bubble:
            edge_lum, _ = self._ink_edge(img)
            edge_lum = 0.0 if edge_lum is None else edge_lum
            for _ in range(BG_MAX_TRIES):
                fill_mode, fill_bgr, contour_bgr = self._sample_bubble_fill()
                ink = 255.0 - edge_lum if fill_mode == "dark" else edge_lum
                fl = float(np.mean(fill_bgr))
                lums = [fl, 255.0 * (1.0 - 0.45)] if fill_mode == "tone" else [fl]
                if all(abs(ink - L) >= BG_MIN_EDGE_DIFF for L in lums):
                    break
            else:
                fill_mode, fill_bgr, contour_bgr = ("white", (255, 255, 255), (0, 0, 0))
                if edge_lum > 150:
                    fill_mode, fill_bgr, contour_bgr = ("dark", (30, 30, 30), (235, 235, 235))
            if fill_mode == "dark":
                img[:, :, :3] = 255 - img[:, :, :3]
        elif np.random.rand() < 0.5:
            img[:, :, :3] = 255 - img[:, :, :3]

        background = self._load_background(img.shape[0], img.shape[1])
        if draw_bubble:
            background = self._decorate_background(self._adjust_background(background, 128.0))
        else:
            background = self._make_background(background, img)

        if draw_bubble:
            radius = np.random.uniform(0.7, 1.0)
            thickness = np.random.choice([1, 2, 3])
            alpha = np.random.randint(60, 100)
            sigma = np.random.randint(10, 15)

            ymin = m0 - int(min(img.shape[:2]) * np.random.uniform(0.07, 0.12))
            ymax = img.shape[0] - m0 + int(min(img.shape[:2]) * np.random.uniform(0.07, 0.12))
            xmin = m0 - int(min(img.shape[:2]) * np.random.uniform(0.07, 0.12))
            xmax = img.shape[1] - m0 + int(min(img.shape[:2]) * np.random.uniform(0.07, 0.12))

            bubble_fill_color = (*fill_bgr, 255)
            bubble_contour_color = (*contour_bgr, 255)
            bubble = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            bubble = rounded_rectangle(
                bubble,
                (xmin, ymin),
                (xmax, ymax),
                radius=radius,
                color=bubble_fill_color,
                thickness=-1,
            )
            bubble = rounded_rectangle(
                bubble,
                (xmin, ymin),
                (xmax, ymax),
                radius=radius,
                color=bubble_contour_color,
                thickness=thickness,
            )

            t = [
                A.ElasticTransform(alpha=alpha, sigma=sigma, p=0.8),
            ]
            bubble = A.Compose(t)(image=bubble)["image"]

            if fill_mode == "tone":
                dots = _screentone(bubble.shape[0], bubble.shape[1])
                shade = 1.0 - np.random.uniform(0.15, 0.45) * dots
                bubble[:, :, :3] = (bubble[:, :, :3].astype(np.float32) * shade[..., None]).astype(np.uint8)

            background = blend(bubble, background)

        img = blend(img, background)

        if tight:
            return (img, ink_box) if return_ink_box else img

        ymin = m0 - int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        ymax = img.shape[0] - m0 + int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        xmin = m0 - int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        xmax = img.shape[1] - m0 + int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        ymin = max(0, ymin)
        xmin = max(0, xmin)
        ymax = min(img.shape[0], ymax)
        xmax = min(img.shape[1], xmax)
        if ymax <= ymin:
            ymin, ymax = 0, img.shape[0]
        if xmax <= xmin:
            xmin, xmax = 0, img.shape[1]
        img = img[ymin:ymax, xmin:xmax]
        if return_ink_box:
            bx0, by0, bx1, by1 = ink_box
            ink_box = (
                max(0, bx0 - xmin),
                max(0, by0 - ymin),
                min(img.shape[1], bx1 - xmin),
                min(img.shape[0], by1 - ymin),
            )
            return img, ink_box
        return img


def _advance(font, ch, letter_spacing_px):
    try:
        w = font.getlength(ch)
    except (OSError, ValueError):
        w = 0
    if not w:
        w = font.size
    return w + letter_spacing_px


def _char_op(ch, cx, cy, font, vertical, upright, cell=None, angle=0.0):
    if vertical and is_ascii(ch) and not upright and not ch.isspace():
        return {
            "kind": "char",
            "ch": ch,
            "cx": cx,
            "cy": cy,
            "font": font,
            "cell": cell,
            "angle": angle,
            "rotate": True,
            "ttb": False,
        }
    return {
        "kind": "char",
        "ch": ch,
        "cx": cx,
        "cy": cy,
        "font": font,
        "cell": cell,
        "angle": angle,
        "rotate": False,
        "ttb": vertical and not is_ascii(ch),
    }


def _char_op_ex(ch, cx, cy, font, cell, angle, vertical, upright):
    return _char_op(ch, cx, cy, font, vertical, upright, cell=cell, angle=angle)


def _draw_rotated_char(layer, ch, cx, cy, font, cell, fill, stroke_width, stroke_fill, angle=0.0):
    cell = int(cell)
    tile = Image.new("RGBA", (cell, cell), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    kw = {"font": font, "fill": fill, "anchor": "mm"}
    if stroke_width:
        kw.update(stroke_width=stroke_width, stroke_fill=stroke_fill)
    draw.text((cell / 2, cell / 2), ch, **kw)
    if angle:
        tile = tile.rotate(270 + angle, resample=Image.BICUBIC)
    else:
        tile = tile.rotate(270, resample=Image.BICUBIC)
    layer.paste(tile, (round(cx - cell / 2), round(cy - cell / 2)), tile)


def _draw_tilted_char(layer, ch, cx, cy, font, cell, angle, fill, stroke_width, stroke_fill):
    cell = int(cell)
    pad = int(stroke_width or 0) + 2
    size = math.ceil(cell * 1.5) + 2 * pad
    size += size & 1
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    kw = {"font": font, "fill": fill, "anchor": "mm"}
    if stroke_width:
        kw.update(stroke_width=stroke_width, stroke_fill=stroke_fill)
    draw.text((size / 2, size / 2), ch, **kw)
    tile = tile.rotate(angle, resample=Image.BICUBIC)
    layer.paste(tile, (round(cx) - size // 2, round(cy) - size // 2), tile)


def _draw_perspective_char(
    layer,
    ch,
    cx,
    cy,
    font,
    cell,
    angle,
    persp,
    fill,
    stroke_width,
    stroke_fill,
    rotate_mixed=False,
    ttb=False,
    lang="ja",
    aspect=None,
    pcoef=None,
):
    cell = int(cell)
    pad = int(stroke_width or 0) + 4
    p = abs(float(persp))
    ax, ay = (float(aspect[0]), float(aspect[1])) if aspect else (1.0, 1.0)
    total_angle = (270.0 if rotate_mixed else 0.0) + float(angle or 0.0)
    rad = math.radians(total_angle)
    rot_f = abs(math.cos(rad)) + abs(math.sin(rad))
    ink = math.ceil(cell * 1.25) + 2 * pad
    size = max(8, math.ceil(ink * max(ax, ay, 1.0) * rot_f * (1.0 + p)) + 6)
    size += size & 1
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    kw = {"font": font, "fill": fill, "anchor": "mm"}
    if ttb:
        kw.update(direction="ttb", language=str(lang or "ja"))
    if stroke_width:
        kw.update(stroke_width=stroke_width, stroke_fill=stroke_fill)
    draw.text((size / 2, size / 2), ch, **kw)
    if aspect:
        nw, nh = max(2, round(size * ax)), max(2, round(size * ay))
        nw += (size - nw) & 1
        nh += (size - nh) & 1
        stretched = tile.resize((nw, nh), Image.BILINEAR)
        tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        tile.paste(stretched, ((size - nw) // 2, (size - nh) // 2))
    if total_angle:
        tile = tile.rotate(total_angle, resample=Image.BILINEAR)
    if p > 1e-6:
        u = pcoef if pcoef is not None else tuple(float(v) for v in np.random.uniform(-1.0, 1.0, 8))
        ref = cell * 2.2
        a, b, c0 = 1.0 + u[0] * p, u[1] * p, u[2] * p * ref * 0.1
        d, e, f0 = u[3] * p, 1.0 + u[4] * p, u[5] * p * ref * 0.1
        g, h = u[6] * p / ref, u[7] * p / ref
        tx = ty = size / 2.0
        d0 = 1.0 - g * tx - h * ty
        coeffs = (
            (a + tx * g) / d0,
            (b + tx * h) / d0,
            (-a * tx - b * ty + c0 + tx * d0) / d0,
            (d + ty * g) / d0,
            (e + ty * h) / d0,
            (-d * tx - e * ty + f0 + ty * d0) / d0,
            g / d0,
            h / d0,
        )
        tile = tile.transform((size, size), Image.PERSPECTIVE, coeffs, resample=Image.BILINEAR)
    layer.paste(tile, (round(cx) - size // 2, round(cy) - size // 2), tile)


def _draw_speedlines_layer(width, height, n_lines=40, color=(0, 0, 0, 255)):
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx, cy = width / 2, height / 2
    max_r = math.hypot(width, height) / 2
    inner = min(width, height) * 0.28
    for _ in range(max(1, int(n_lines))):
        ang = float(np.random.uniform(0.0, 2.0 * math.pi))
        if np.random.rand() < 0.18:
            continue
        r0 = inner * float(np.random.uniform(0.6, 1.3))
        r1 = max_r * float(np.random.uniform(0.75, 1.0))
        x0, y0 = cx + r0 * math.cos(ang), cy + r0 * math.sin(ang)
        x1, y1 = cx + r1 * math.cos(ang), cy + r1 * math.sin(ang)
        w = int(np.random.choice([1, 2, 2, 3]))
        draw.line([(x0, y0), (x1, y1)], fill=color, width=w)
    return layer


def _draw_parallel_speedlines_layer(width, height, n_lines=40, color=(0, 0, 0, 255)):
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    ang = float(np.random.choice([0.0, 90.0, np.random.uniform(0, 180)])) * math.pi / 180.0
    dx, dy = math.cos(ang), math.sin(ang)
    nx, ny = -dy, dx
    diag = math.hypot(width, height)
    cx, cy = width / 2, height / 2
    for _ in range(max(1, int(n_lines))):
        if np.random.rand() < 0.18:
            continue
        off = float(np.random.uniform(-0.5, 0.5)) * diag
        length = diag * float(np.random.uniform(0.25, 0.7))
        mid = float(np.random.uniform(-0.35, 0.35)) * diag
        x0, y0 = cx + nx * off + dx * (mid - length / 2), cy + ny * off + dy * (mid - length / 2)
        x1, y1 = cx + nx * off + dx * (mid + length / 2), cy + ny * off + dy * (mid + length / 2)
        draw.line([(x0, y0), (x1, y1)], fill=color, width=int(np.random.choice([1, 2, 2, 3])))
    return layer


def _draw_burst_layer(width, height, ops):
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if not ops:
        return layer
    xs = [float(op["cx"]) for op in ops]
    ys = [float(op["cy"]) for op in ops]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    base_r = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 + 20
    base_r = max(30.0, float(base_r))
    n_spikes = int(np.random.randint(10, 17))
    points = []
    for i in range(n_spikes * 2):
        ang = 2.0 * math.pi * i / (n_spikes * 2) + float(np.random.uniform(-0.12, 0.12))
        r = base_r * (float(np.random.uniform(1.05, 1.35)) if i % 2 == 0 else float(np.random.uniform(0.68, 0.85)))
        points.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    draw = ImageDraw.Draw(layer)
    draw.polygon(points, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
    return layer


def _shear_layer(img, slant_deg):
    k = math.tan(math.radians(slant_deg))
    w, h = img.size
    shift = k * h
    new_w = max(1, math.ceil(w + abs(shift)))
    tx = -min(0.0, shift)
    return img.transform(
        (new_w, h), Image.AFFINE, (1, -k, -tx, 0, 1, 0), resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0)
    )


def _draw_tcy_tile(layer, text, cx, cy, font_path, font_size, fill, stroke_width, stroke_fill, angle=0.0):
    cell = int(font_size)
    if not text:
        return
    font = _get_font(font_path, cell)
    try:
        width = font.getlength(text)
    except (OSError, ValueError):
        width = 0
    if width and width > cell:
        shrunk = max(6, int(cell * cell / width))
        font = _get_font(font_path, shrunk)
    tile = Image.new("RGBA", (cell, cell), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    kw = {"font": font, "fill": fill, "anchor": "mm"}
    if stroke_width:
        kw.update(stroke_width=stroke_width, stroke_fill=stroke_fill)
    draw.text((cell / 2, cell / 2), text, **kw)
    if angle:
        tile = tile.rotate(angle, resample=Image.BICUBIC)
    layer.paste(tile, (round(cx - cell / 2), round(cy - cell / 2)), tile)


def _screentone(h, w):
    period = np.random.uniform(3.0, 8.0)
    ang = np.random.uniform(0, np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx * np.cos(ang) + yy * np.sin(ang)) * (2 * np.pi / period)
    v = (-xx * np.sin(ang) + yy * np.cos(ang)) * (2 * np.pi / period)
    pat = 0.5 * (np.cos(u) + np.cos(v))
    return np.clip((pat - np.random.uniform(0.0, 0.6)) * 4.0, 0.0, 1.0)


def crop_by_alpha(img, margin):
    y, x = np.where(img[:, :, 3] > 0)
    if y.size == 0:
        raise ValueError("rendered text layer is empty (nothing to crop)")
    ymin = y.min()
    ymax = y.max()
    xmin = x.min()
    xmax = x.max()
    img = img[ymin : ymax + 1, xmin : xmax + 1]
    img = np.pad(img, ((margin, margin), (margin, margin), (0, 0)))
    return img


def blend(img, background):
    alpha = (img[:, :, 3] / 255)[:, :, np.newaxis]
    img = img[:, :, :3]
    img = (background * (1 - alpha) + img * alpha).astype(np.uint8)
    return img


def rounded_rectangle(src, top_left, bottom_right, radius=1, color=255, thickness=1, line_type=cv2.LINE_AA):
    p1 = top_left
    p2 = (bottom_right[0], top_left[1])
    p3 = bottom_right
    p4 = (top_left[0], bottom_right[1])

    height = abs(bottom_right[1] - top_left[1])
    width = abs(bottom_right[0] - top_left[0])

    radius = min(radius, 1)
    corner_radius = int(radius * (min(height, width) / 2))

    if thickness < 0:
        top_left_main_rect = (int(p1[0] + corner_radius), int(p1[1]))
        bottom_right_main_rect = (int(p3[0] - corner_radius), int(p3[1]))

        top_left_rect_left = (p1[0], p1[1] + corner_radius)
        bottom_right_rect_left = (p4[0] + corner_radius, p4[1] - corner_radius)

        top_left_rect_right = (p2[0] - corner_radius, p2[1] + corner_radius)
        bottom_right_rect_right = (p3[0], p3[1] - corner_radius)

        all_rects = [
            [top_left_main_rect, bottom_right_main_rect],
            [top_left_rect_left, bottom_right_rect_left],
            [top_left_rect_right, bottom_right_rect_right],
        ]

        [cv2.rectangle(src, rect[0], rect[1], color, thickness) for rect in all_rects]

    cv2.line(
        src,
        (p1[0] + corner_radius, p1[1]),
        (p2[0] - corner_radius, p2[1]),
        color,
        abs(thickness),
        line_type,
    )
    cv2.line(
        src,
        (p2[0], p2[1] + corner_radius),
        (p3[0], p3[1] - corner_radius),
        color,
        abs(thickness),
        line_type,
    )
    cv2.line(
        src,
        (p3[0] - corner_radius, p4[1]),
        (p4[0] + corner_radius, p3[1]),
        color,
        abs(thickness),
        line_type,
    )
    cv2.line(
        src,
        (p4[0], p4[1] - corner_radius),
        (p1[0], p1[1] + corner_radius),
        color,
        abs(thickness),
        line_type,
    )

    cv2.ellipse(
        src,
        (p1[0] + corner_radius, p1[1] + corner_radius),
        (corner_radius, corner_radius),
        180.0,
        0,
        90,
        color,
        thickness,
        line_type,
    )
    cv2.ellipse(
        src,
        (p2[0] - corner_radius, p2[1] + corner_radius),
        (corner_radius, corner_radius),
        270.0,
        0,
        90,
        color,
        thickness,
        line_type,
    )
    cv2.ellipse(
        src,
        (p3[0] - corner_radius, p3[1] - corner_radius),
        (corner_radius, corner_radius),
        0.0,
        0,
        90,
        color,
        thickness,
        line_type,
    )
    cv2.ellipse(
        src,
        (p4[0] + corner_radius, p4[1] - corner_radius),
        (corner_radius, corner_radius),
        90.0,
        0,
        90,
        color,
        thickness,
        line_type,
    )

    return src
