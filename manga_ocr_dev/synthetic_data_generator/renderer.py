"""Fast Pillow-based text renderer for synthetic manga OCR data.

The previous implementation used ``html2image`` (a wrapper around a headless
Chrome/Chromium browser) to typeset text. Spawning a browser process per image
is very slow and requires a Chrome installation. This module renders the same
text layouts (vertical/horizontal text, furigana/ruby, tate-chu-yoko,
stroke/shadow styles) directly with Pillow, with no browser, no temp files
and no subprocesses.

Public API is intentionally identical to the old browser renderer, so
``SyntheticDataGenerator`` works unchanged::

    Renderer().render(lines, override_css_params) -> (img_bgra, params)

``img_bgra`` is a ``uint8`` BGRA numpy array in BGR channel order (same as
``cv2.imread(..., cv2.IMREAD_UNCHANGED)`` returned before), so the downstream
``render_background`` pipeline behaves exactly as before.

A note on "identical" images: pixel-for-pixel identical output across two
different rendering engines (Blink vs. FreeType/HarfBuzz) is not achievable -
glyph rasterization, shaping and line breaking differ by design. What this
renderer preserves is the training distribution: same fonts, same texts, same
layout semantics (vertical ratio, ruby, tate-chu-yoko), same style
probabilities (stroke/shadow/none) and the unchanged background/bubble
pipeline. The legacy Chrome renderer is kept as ``BrowserRenderer`` below for
side-by-side checks where Chrome is available.

Besides the uniform grid layout (every line the same size, aligned to a
regular grid), the renderer supports an ``irregular`` layout mode that mimics
challenging hand-lettered manga crops: each line (column in vertical text,
row in horizontal text) gets its own font size and its own start offset, and
optionally per-character emphasis sizing, baseline wobble, slight per-glyph
tilt and shrunken small kana. See ``IRREGULAR_PROB`` and
``get_random_css_params`` for the sampling distribution.

Besides dialogue, the renderer supports an ``sfx`` mode for sound-effect /
onomatopoeia lettering: huge type, per-glyph size/tilt variation, curved
(arc/wave/diagonal) baselines, true per-glyph perspective warps, thick
outlines with optional double (black-outer/white-inner) edges, radiating
speedlines and jagged burst balloons behind the text. SFX is a separate
dataset split (see ``get_random_sfx_params``); decorative shapes are never
part of the ground truth - only font-rendered glyphs (including ``♡♥★`` when
they are characters in the string) are labeled.

Besides black ink, the renderer samples coloured text fills for the majority
of images: red, hot pink (weighted heavily), orange, yellow, green, lime,
cyan, blue, purple, magenta, brown and light pink (see ``COLOR_PROB``,
``SFX_COLOR_PROB`` and ``_sample_text_color``). A good share of coloured
samples mixes several colours within a single image - ``per_line`` paints
each column/row its own colour (emphasis lettering), ``per_char`` paints
every glyph its own colour (rainbow SFX) - see ``MULTICOLOR_PROB``,
``SFX_MULTICOLOR_PROB`` and ``color_mode``/``text_colors`` params. Coloured
samples are returned by ``render`` as 3-channel BGR images so the hue
survives; black/white samples stay grayscale, exactly as before.

Legibility is computed programmatically, not by restricting the palette:
``_relative_luminance``/``_contrast_ratio`` implement WCAG contrast, and
``_ensure_legible_outline`` plus per-glyph edges in ``_assign_op_colors``
give light fills (yellow, cyan, light pink, ...) a black outline and dark
fills a white one, so even neon ink stays readable on white bubbles and
light art. Force colours with
``override_css_params={"text_color": "pink"}`` (hex like ``"#ff5090"``
works too), force mixed lettering with
``{"text_color": "red", "color_mode": "per_char",
"text_colors": ["red", "yellow", "cyan", "pink"]}``, and force grayscale
with ``{"text_color": "black", "color_mode": "solid"}``.

Why not Tauri (or another webview)? A webview is still a full browser engine
(WebKit/WebView2/WKWebView) behind IPC + native build tooling, with no Python
screenshot API. It would keep the per-image engine overhead while adding a
Rust toolchain dependency, so it is slower to install, slower to run and
harder to parallelize than plain Pillow.
"""

import math
from collections import OrderedDict
import re
import threading

import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from manga_ocr_dev.env import BACKGROUND_ANIMETEXT_DIR, BACKGROUND_DIR
from manga_ocr_dev.synthetic_data_generator.utils import (
    MissingGlyphError,
    get_background_df,
    is_ascii,
    is_drawable,
)


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

# Irregular ("challenging crop") layout defaults. Mild settings reproduce
# realscan-style crops (per-line sizes + staggered line starts); strong
# settings add enhanced-style per-character emphasis sizing, baseline wobble,
# slight tilt and shrunken small kana.
IRREGULAR_PROB = 0.35
LINE_SCALE_RANGE = (0.65, 1.45)
LINE_SHIFT_EM = 1.2  # max per-line start offset, in units of the base font size
CHAR_SCALE_RANGE = (0.7, 1.5)
CHAR_JITTER_PROB = 0.5  # per-line probability of enabling per-character sizing
WOBBLE_EM = 0.07  # max perpendicular baseline wobble, in units of the line font size
ROTATE_DEG = 5.0  # max absolute per-character tilt, in degrees
SMALL_KANA = frozenset("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ")
SMALL_KANA_SCALE = (0.55, 0.8)
MIN_CHAR_SIZE = 8  # px floor for scaled glyphs (same floor as furigana)

# Whole-block slant (shear) probabilities. Slanted emphasis lettering shows up
# occasionally in manga, from a mild italic-like lean to a dramatic skew.
SLANT_PROB = 0.12
SLANT_MILD_RANGE = (8.0, 15.0)  # degrees
SLANT_STRONG_RANGE = (20.0, 32.0)  # degrees ("very slanted")
SLANT_STRONG_PROB = 0.33  # fraction of slanted samples that go strong

# SFX (sound-effect) lettering defaults. SFX is a separate dataset split with
# huge type, curved baselines, per-glyph perspective warps, thick outlines,
# speedlines and burst balloons. No decorative (non-font) symbols are drawn -
# only font glyphs count as text.
SFX_FONT_SIZE_RANGE = (90, 150)
SFX_CHAR_SCALE_RANGE = (0.8, 1.9)
SFX_LINE_SCALE_RANGE = (0.85, 1.2)
SFX_PERSP_RANGE = (0.12, 0.28)
SFX_ROTATE_DEG = 22.0  # max absolute per-glyph tilt before curve tangent
SFX_BLOCK_ROTATE_RANGE = (-18.0, 18.0)
SFX_CURVE_MODES = ("arc", "wave", "diagonal", "straight", "zigzag", "bounce", "hill")
SFX_CURVE_P = (0.22, 0.18, 0.16, 0.14, 0.10, 0.10, 0.10)
SFX_ADJACENT_SCALE_MAX = 1.35  # max size ratio between neighbouring glyphs (legibility)
SFX_ASPECT_MAX = 0.4  # max per-glyph non-uniform stretch (tall/wide glyphs)
SFX_SPACING_RANGE = (-0.06, 0.25)  # letter spacing in em (tight overlap .. airy)
SFX_CURVE_AMP_RANGE = (0.15, 0.45)  # in em of the line font size
SFX_SPEEDLINE_PROB = 0.6
SFX_SPEEDLINE_N_RANGE = (24, 64)
SFX_BURST_PROB = 0.35
SFX_DOUBLE_OUTLINE_PROB = 0.3
SFX_VERTICAL_PROB = 0.4
# Tight SFX crops: the crop box hugs the lettering (fill + stroke/outline),
# never the decor (speedlines, burst balloon, shadow blur), which is simply
# clipped by it. Margin around the ink as a fraction of the shorter side.
SFX_CROP_MARGIN_FRAC = (0.02, 0.06)
SFX_CROP_MARGIN_MIN = 2  # px
SFX_INK_ALPHA_THRESHOLD = 8  # ignore near-invisible anti-alias fringe

# Coloured text defaults. Colour pages, covers and SFX use chromatic ink all
# the time (red emphasis, hot pink, blue, orange, ...), so the majority of
# samples is rendered with a chromatic fill and a good share of those mixes
# several colours within a single image (per-line or per-glyph). Coloured
# samples are returned as 3-channel BGR images; black/white samples stay
# grayscale, exactly as before.
COLOR_PROB = 0.6  # dialogue fraction with chromatic text fill
SFX_COLOR_PROB = 0.7  # SFX fraction with chromatic text fill
MULTICOLOR_PROB = 0.4  # fraction of coloured dialogue with mixed colours
SFX_MULTICOLOR_PROB = 0.5  # fraction of coloured SFX with mixed colours
# Per-line (each column/row its own colour) reads like emphasis lettering;
# per-char (rainbow glyphs) reads like playful SFX.
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

# Text border (outline) sampling. Outlines used to be applied to ~80% of
# dialogue samples with sizes up to 8 px on 48 px type (a fat halo that makes
# every glyph trivially separable from the art). Now most samples have no or a
# hairline edge; fat edges are rare.
STROKE_PROB = 0.40  # dialogue: fraction with an outline
SHADOW_PROB = 0.08  # dialogue: fraction with a soft glow
STROKE_SIZES = (1, 2, 3, 4, 6)
STROKE_SIZE_P = (0.35, 0.30, 0.20, 0.10, 0.05)
SFX_STROKE_FRAC = 0.05  # SFX outline cap, fraction of font size
SFX_OUTER_RING = 2  # px added by the black outer ring of a double outline

# Background difficulty. The old pipeline crushed contrast on every no-bubble
# sample (contrast_limit=(-0.8,-0.3), p=1), blurred 30% of backgrounds, and
# stretched mismatched-aspect crops with Resize, which blurs them further. The
# result was flat, very dark or very light backgrounds. Now backgrounds keep
# their texture and their mean brightness is pushed towards mid-tones or
# towards the ink's own brightness (the hard case for an OCR model).
BG_BLUR_PROB = 0.08
BG_TONE_MODES = ("natural", "midtone", "ink_match")
BG_TONE_P = (0.25, 0.35, 0.40)
BG_MIN_STD = (22.0, 60.0)  # background texture std is lifted to at least this
BG_INK_OFFSET = (60.0, 110.0)  # ink_match: |bg_mean - edge_lum| range
# Hard legibility guarantee. Around the outermost ink pixels (the visible
# glyph edge, outline included) at most BG_MAX_LOW_CONTRAST of the surrounding
# background pixels may be within BG_MIN_EDGE_DIFF grey levels of the edge
# colour. Anything worse is regenerated, so white-on-white can never ship.
BG_MIN_EDGE_DIFF = 75.0
BG_MAX_LOW_CONTRAST = 0.20
BG_MAX_TRIES = 6
BG_SCREENTONE_PROB = 0.25
BG_GRADIENT_PROB = 0.25
BG_NOISE_PROB = 0.35
# Bubble fill variants: white used to be the only one.
BUBBLE_FILL_MODES = ("white", "gray", "tint", "tone", "dark")
BUBBLE_FILL_P = (0.45, 0.20, 0.10, 0.15, 0.10)

RUBY_RE = re.compile(r"<ruby>(.*?)<rt>(.*?)</rt></ruby>")
TCY_RE = re.compile(r'<span[^>]*?text-combine-upright[^>]*?>(.*?)</span>')
TAG_RE = re.compile(r"<[^>]+>")

_NAMED_COLORS = {
    "black": (0, 0, 0, 255),
    "white": (255, 255, 255, 255),
    "transparent": (0, 0, 0, 0),
    # Chromatic ink. Vivid print-like tones, including bright yellow/cyan and
    # light pinks - legibility is guaranteed programmatically (see
    # _relative_luminance/_contrast_ratio/_ensure_legible_outline), which picks
    # a contrasting outline per glyph instead of restricting the palette.
    # Hex codes still allow any exact shade via override_css_params.
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

# Bounded LRU. Every (font, size) pair is a separate FreeType face costing
# several MB; SFX draws each glyph at its own size, so an unbounded dict grows
# with (#fonts x #sizes) and eats all RAM (-> swap thrash that looks like a hang).
_FONT_CACHE_MAX = 48
_FONT_CACHE = OrderedDict()
_FONT_CACHE_LOCK = threading.Lock()


def _normalize_color(value, default="black"):
    """Map a CSS color name (or hex, or legacy tuple bug) to an RGBA tuple."""
    if value is None:
        value = default
    if isinstance(value, (tuple, list)):
        # legacy get_random_css_params accidentally produced ("white",)
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
    """Sample a text fill color name: often chromatic, else black."""
    prob = SFX_COLOR_PROB if sfx else COLOR_PROB
    if np.random.rand() >= prob:
        return "black"
    return str(np.random.choice(TEXT_COLOR_NAMES, p=TEXT_COLOR_P))


def _sample_text_colors(n, sfx=False):
    """Sample ``n`` distinct chromatic color names for mixed-colour lettering."""
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
    """Sample a mixed-colour layout: per-line emphasis vs per-glyph rainbow."""
    if sfx:
        return str(np.random.choice(MULTICOLOR_MODES, p=[0.4, 0.6]))
    return str(np.random.choice(MULTICOLOR_MODES, p=[0.7, 0.3]))


def _params_are_color(params):
    """True when params describe chromatic ink (solid or mixed)."""
    mode = str(params.get("color_mode", "solid"))
    if mode in ("per_line", "per_char"):
        return True
    colors = params.get("text_colors", None)
    if isinstance(colors, (list, tuple)) and len(colors):
        return any(_is_chromatic_color(c) for c in colors)
    return _is_chromatic_color(params.get("text_color", "black"))


def _assign_op_colors(ops, params):
    """Assign per-op ``fill``/``stroke``/``shadow`` RGBA tuples.

    ``solid`` (default) paints every glyph with ``text_color``. ``per_line``
    paints each line (column in vertical text, row in horizontal text) with
    its own colour from ``text_colors``. ``per_char`` paints every glyph with
    its own colour from ``text_colors`` (rainbow SFX / emphasis lettering).
    Ruby (furigana) shares its base line's colour in ``per_line`` mode and
    gets its own colour in ``per_char`` mode.

    Legibility is computed per glyph (see ``_needs_dark_outline``): light
    fills (yellow, cyan, light pink, ...) get a black outline, dark fills a
    white one, so every glyph stays readable on white bubbles and light art
    regardless of the sampled hue.
    """
    shadow_fallback = _normalize_color(params.get("shadow_color", "white"), "white")
    black = (0, 0, 0, 255)
    white = (255, 255, 255, 255)

    def contrasting_edge(fill):
        return black if _needs_dark_outline(fill) else white

    mode = str(params.get("color_mode", "solid"))
    if mode not in ("per_line", "per_char"):
        fill = _normalize_color(params.get("text_color", "black"), "black")
        for op in ops:
            op["fill"] = fill
            op["stroke"] = contrasting_edge(fill)
            op["shadow"] = black if _needs_dark_outline(fill) else shadow_fallback
        return ops
    colors = params.get("text_colors", None)
    if not isinstance(colors, (list, tuple)) or not len(colors):
        colors = _sample_text_colors(3)
    fills = [_normalize_color(c, "black") for c in colors]
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
    """True when a color spec is a chromatic (non-gray, non-transparent) ink."""
    try:
        r, g, b, a = _normalize_color(value, "black")
    except (TypeError, ValueError):
        return False
    if a == 0:
        return False
    return not (r == g == b)


def _relative_luminance(rgba):
    """WCAG relative luminance of an RGB(A) color, in 0..1."""
    r, g, b = (c / 255.0 for c in rgba[:3])

    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _contrast_ratio(a, b):
    """WCAG contrast ratio between two RGB(A) colors, in 1..21."""
    la = _relative_luminance(_normalize_color(a, "black"))
    lb = _relative_luminance(_normalize_color(b, "black"))
    hi, lo = (la, lb) if la >= lb else (lb, la)
    return (hi + 0.05) / (lo + 0.05)


# Fill-vs-white contrast below this gets a dark outline: the fill alone is too
# light to read on a white bubble (yellow/cyan/light pink), so the glyph shape
# is defined by a black edge instead. 3.0 is the WCAG large-text threshold;
# our body type (48px) and SFX (90-150px) both count as large text.
MIN_FILL_CONTRAST = 3.0


def _needs_dark_outline(fill_rgba):
    """True when a fill needs a black (not white) outline to stay legible."""
    white = (255, 255, 255, 255)
    la = _relative_luminance(fill_rgba)
    lw = _relative_luminance(white)
    hi, lo = (lw, la) if lw >= la else (la, lw)
    return (hi + 0.05) / (lo + 0.05) < MIN_FILL_CONTRAST


def _ensure_legible_outline(params):
    """Force a contrasting outline/shadow when the fill is too light.

    Operates on a sampled ``params`` dict in place: light fills (yellow,
    cyan, light pink, ...) get a black stroke (or black shadow when the
    variant is a glow) at least 2px wide, so the glyph stays readable on
    white bubbles and light art. Dark fills keep the sampled white halo,
    which pops on dark art. Mixed-colour samples are handled per glyph at
    render time (see ``_assign_op_colors``); here we only guarantee the
    shared ``stroke_size`` is non-zero so those per-glyph outlines can draw.
    """
    mode = str(params.get("color_mode", "solid"))
    if mode in ("per_line", "per_char"):
        if int(params.get("stroke_size", 0) or 0) <= 0 and int(params.get("shadow_size", 0) or 0) <= 0:
            params["stroke_size"] = 2
            params["stroke_color"] = "white"  # fallback; per-glyph strokes override
        elif int(params.get("stroke_size", 0) or 0) <= 0:
            # Glow-only mixed lettering: make the glow dark so light glyphs
            # keep an edge; dark glyphs still read via their fill. Kept small
            # so it reads as a glow, not a double character.
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
    # SFX without its black outer edge would leave a light fill in a white
    # halo; the outer edge guarantees the silhouette on any background.
    if params.get("sfx") and int(params.get("stroke_size", 0) or 0) >= 5:
        params["sfx_double_outline"] = True
    return params


def _get_font(font_path, size):
    """Load (and cache) a TrueType/OpenType font. Thread-safe."""
    key = (str(font_path), int(size))
    with _FONT_CACHE_LOCK:
        font = _FONT_CACHE.get(key)
        if font is None:
            font = ImageFont.truetype(str(font_path), int(size))
            _FONT_CACHE[key] = font
            while len(_FONT_CACHE) > _FONT_CACHE_MAX:
                _FONT_CACHE.popitem(last=False)  # evict least recently used
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
    """Hard guarantee: never rasterize a character the font cannot draw.

    A missing glyph would come out as a "tofu" box while the label still names
    the real character. Raising here makes that impossible.
    """
    for tokens in parsed:
        for tok in tokens:
            text = "".join(_strip_tags(part) for part in tok[1:] if isinstance(part, str))
            for ch in text:
                if not is_drawable(font_path, ch):
                    raise MissingGlyphError(f"{font_path!r} cannot draw {ch!r} (U+{ord(ch):04X})")


def _strip_tags(s):
    return TAG_RE.sub("", s)


def _parse_inline(s):
    """Split a string without ruby into text/tcy tokens."""
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
    """Split a markup line into text/ruby/tcy tokens.

    Returns a list of ("text", str) | ("ruby", base, rt) | ("tcy", str).
    """
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


class Renderer:
    """Pillow-based renderer. Drop-in replacement for the old browser one."""

    def __init__(self, background_df=None):
        self.background_df = background_df if background_df is not None else get_default_background_df()
        self.max_size = 600

    def render(self, lines, override_css_params=None):
        img, params = self.render_text(lines, override_css_params)
        # SFX sits directly on art: never draw a dialogue bubble around it.
        draw_bubble = None if not params.get("sfx") else False
        img = self.render_background(img, draw_bubble=draw_bubble, tight=bool(params.get("sfx")))
        # Native scan-like sizes: small crops stay small (never upscaled),
        # only oversized ones are shrunk to the ceiling. Any size and any
        # aspect ratio may come out, which is what variable-resolution
        # (NaFlex-style) training wants.
        if max(img.shape[:2]) > self.max_size:
            img = A.LongestMaxSize(self.max_size)(image=img)["image"]
        if _params_are_color(params):
            # Coloured ink: keep the 3-channel BGR image so the hue survives.
            # Black/white samples stay grayscale, exactly as before.
            return img, params
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, params

    def render_text(self, lines, override_css_params=None):
        """Render text on a transparent background and return as BGRA image."""

        sfx_hint = bool((override_css_params or {}).get("sfx", False))
        params = self.get_random_css_params(sfx=sfx_hint)
        if override_css_params:
            params.update(override_css_params)
            # An explicit single colour wins over sampled mixing: drop the
            # stale colour list so a pinned black/white override can't leave
            # a chromatic text_colors behind (which would flip the output to
            # colour while rendering gray ink).
            if (
                "text_color" in override_css_params
                and "color_mode" not in override_css_params
                and "text_colors" not in override_css_params
            ):
                params["color_mode"] = "solid"
            if params.get("color_mode", "solid") == "solid":
                params.pop("text_colors", None)
        # Guarantee a contrasting outline even for hand-picked colours
        # (e.g. yellow fill with a white stroke becomes a black stroke).
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
        pad = max(stroke_size, shadow_size, 0) + MIN_PAD

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
            # Non-text decor behind the ink: speedlines and/or burst balloon.
            # Never part of the GT (only font glyphs are labeled).
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

        # Per-op ink colours for mixed-colour lettering. Solid mode paints
        # every glyph with text_color; per_line/per_char vary it (see
        # _assign_op_colors). Shadow keeps its own single colour, so it is
        # rendered before assignment kicks in for the text layers.
        main_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        if shadow_size > 0:
            shadow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            self._execute_ops(
                shadow_layer,
                ops,
                vertical=vertical,
                fill=shadow_rgba,
                stroke_width=0,
                stroke_fill=None,
                font_path=font_path,
                font_size=font_size,
                use_op_fill=False,
            )
            shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=shadow_size))
            base = Image.alpha_composite(base, shadow_layer)

        _assign_op_colors(ops, params)

        outer_layer = None
        if sfx and bool(params.get("sfx_double_outline", False)) and stroke_size > 0:
            # Classic SFX edge: thick black outer + contrasting inner around the
            # (possibly coloured) fill. The outer ring stays black for every
            # glyph; the inner ring adapts per glyph (see _assign_op_colors).
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
            )
            base = Image.alpha_composite(base, outer_layer)

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
        )
        base = Image.alpha_composite(base, main_layer)

        # SFX: remember where the *lettering* is (fill + stroke + outer outline,
        # no speedlines / burst / shadow) so the final crop can hug it.
        ink = None
        if sfx:
            ink_alpha = main_layer.getchannel("A")
            if outer_layer is not None:
                ink_alpha = ImageChops.lighter(ink_alpha, outer_layer.getchannel("A"))
            ink = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            ink.putalpha(ink_alpha)

        if sfx:
            block_rotate = float(params.get("sfx_block_rotate", 0.0))
            if block_rotate:
                base = base.rotate(
                    block_rotate, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0)
                )
                ink = ink.rotate(block_rotate, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0))

        slant = float(params.get("slant", 0.0))
        if slant:
            base = _shear_layer(base, slant)
            if ink is not None:
                ink = _shear_layer(ink, slant)

        if ink is not None:
            # Clip everything (decor included) to the ink bounding box.
            a = ink.getchannel("A").point(lambda v: 255 if v > SFX_INK_ALPHA_THRESHOLD else 0)
            bbox = a.getbbox()
            if bbox is not None:
                base = base.crop(bbox)

        img = np.array(base)
        img = img[:, :, [2, 1, 0, 3]]  # RGBA -> BGRA (BGR channel order like cv2)
        return np.ascontiguousarray(img), params

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
        """Uniform grid layout: every line shares one font size and one origin.

        Each op is a dict:
          {"kind": "char", "ch", "cx", "cy", "font", "rotate", "ttb", "line"}
          {"kind": "tcy", "text", "cx", "cy", "line"}
        Coordinates are cell centers in canvas pixels. ``line`` is the
        zero-based line index used for per-line multicolour lettering.
        """
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
            pitches = [
                max(pitch_base, font_size + (ruby_size + RUBY_GAP if r else 0) + RUBY_GAP)
                for r in has_ruby
            ]
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
            pitches = [
                max(pitch_base, font_size + (ruby_size + RUBY_GAP if r else 0) + RUBY_GAP)
                for r in has_ruby
            ]
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
                        # text-combine-upright only applies to vertical text;
                        # in horizontal mode render the characters normally.
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
        """Challenging hand-lettered layout: per-line sizes and start offsets.

        Each line (column for vertical text, row for horizontal text) is
        typeset at its own font size and starts at its own offset along the
        reading direction, so columns/rows differ in glyph size and are
        staggered instead of grid-aligned. Optionally, characters within a
        line vary in size (emphasis lettering), wobble perpendicular to the
        baseline, tilt slightly, and small kana render shrunken.

        Ops carry per-glyph ``font``/``cell``/``angle`` keys understood by
        ``_execute_ops``. Ruby bases stay uniform within their line (scaled by
        the line size) so furigana remains legible; tcy runs are squeezed into
        a single cell of the line size.
        """
        n_lines = len(parsed)
        line_scales = [float(np.random.uniform(line_scale_min, line_scale_max)) for _ in range(n_lines)]
        line_fs = [max(MIN_CHAR_SIZE, round(font_size * s)) for s in line_scales]
        line_jitter = [bool(np.random.rand() < char_jitter_prob) for _ in range(n_lines)]
        raw_starts = [float(np.random.uniform(-line_shift, line_shift)) * font_size for _ in range(n_lines)]

        spacing_em = float(letter_spacing) if letter_spacing else 0.0

        def scaled_font(size):
            return _get_font(font_path, max(MIN_CHAR_SIZE, int(size)))

        def char_spec(ch, fs, jitter_on):
            """Per-glyph size/tilt/wobble for one base character."""
            scale = 1.0
            if small_kana and ch in SMALL_KANA:
                scale *= float(np.random.uniform(*SMALL_KANA_SCALE))
            if jitter_on:
                scale *= float(np.random.uniform(char_scale_min, char_scale_max))
            size = max(MIN_CHAR_SIZE, round(fs * scale))
            angle = float(np.random.uniform(-rotate, rotate)) if rotate else 0.0
            cross = float(np.random.uniform(-wobble, wobble)) * fs if wobble else 0.0
            return scaled_font(size), size, angle, cross

        # Pass 1: per-line metrics and per-glyph specs (needed for canvas size).
        # Segments are ("chars", [(ch, font, cell, adv, angle, cross)], seg_len),
        # ("ruby", base, rt, [base_adv], seg_len, rt_len) or
        # ("tcy", text, adv, angle, cross). Ruby bases stay uniform at the line
        # size so furigana remains legible.
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
                        # text-combine-upright only applies to vertical text;
                        # in horizontal mode render the characters normally.
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
        """SFX layout: huge glyphs on a curved baseline with per-glyph perspective.

        Each glyph gets its own size, tilt (plus curve-tangent rotation),
        baseline wobble and perspective strength. Ruby is flattened (base text
        only) since SFX never carries furigana; ``♡♥★`` survive only as normal
        font glyphs. Ops carry a ``persp`` key understood by ``_execute_ops``.
        Positions are built in local coords then shifted to fit the canvas, so
        wild curves/rotations never clip.
        """
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
                # Neighbours may differ in size, but not so much that the small
                # one is swallowed by the big one's outline (unreadable = wrong label).
                scale = float(np.clip(scale, prev_scale / SFX_ADJACENT_SCALE_MAX, prev_scale * SFX_ADJACENT_SCALE_MAX))
            base_scale = scale
            is_small = bool(small_kana and ch in SMALL_KANA)
            if is_small:
                scale *= float(np.random.uniform(*SMALL_KANA_SCALE))
                if cap_ref is not None:
                    # ォ/オ, ッ/ツ, ャ/ヤ ... differ mainly by size: stay clearly smaller
                    # than the glyph before (or no larger than a small kana before it),
                    # otherwise the label is ambiguous.
                    scale = min(scale, cap_ref * cap_mult)
            else:
                # Consume the same RNG draws so seeds stay comparable when
                # small_kana is toggled.
                np.random.uniform(*SMALL_KANA_SCALE)
            size = max(MIN_CHAR_SIZE, round(fs * scale))
            # Always consume RNG (even when the magnitude is 0) so identical
            # seeds give identical layouts apart from the varied parameter.
            angle = float(np.random.uniform(-1.0, 1.0)) * rot_max
            cross = float(np.random.uniform(-1.0, 1.0)) * wob_em * fs
            p = float(np.random.uniform(0.0, 1.0)) * persp_max
            ax = 1.0 + float(np.random.uniform(-1.0, 1.0)) * aspect_max
            ay = 1.0 + float(np.random.uniform(-1.0, 1.0)) * aspect_max
            return scaled_font(size), size, angle, cross, p, (ax, ay), base_scale, scale, is_small

        def flat_chars(tokens):
            out = []
            for tok in tokens:
                if tok[0] in ("text", "tcy"):
                    out.extend(list(_strip_tags(tok[1])))
                elif tok[0] == "ruby":
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
                # asymmetric: rises slowly, drops fast (like a thrown punch)
                return sign * amp_em * (math.sin(math.pi * min(1.0, t ** 0.6)) * 1.1 - 0.4)
            return 0.0

        def curve_slope(t):
            e = 1e-3
            return (curve_offset(min(1.0, t + e)) - curve_offset(max(0.0, t - e))) / (2.0 * e)

        # Pass 1: per-glyph specs and line totals.
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
                # size chain uses the unshrunk scale (so a small kana doesn't shrink the
                # whole word); the small-kana cap uses what was actually drawn.
                cap_ref, cap_mult = (final_scale, 1.0) if was_small else (final_scale, SMALL_KANA_SCALE[1])
                adv = (cell + sp) if vertical else _advance(font, ch, sp)
                # A tilted / perspective-warped glyph occupies more than its em
                # box; give it most of that growth as extra pitch so neighbours
                # overlap only slightly (touching is still allowed, burying is not).
                rad = math.radians(angle)
                adv = adv * (1.0 + 0.8 * (abs(math.cos(rad)) + abs(math.sin(rad)) - 1.0) + 0.6 * p)
                # keep the layout pitch honest when a glyph is stretched
                adv = adv * (asp[1] if vertical else asp[0]) if aspect_max else adv
                specs.append([ch, font, cell, adv, angle, cross, p, asp])
                total += adv
            specs_per_line.append((fs, sp, specs))
            totals.append(total)
            pitches.append(round(fs * (1.0 + line_height)) + RUBY_GAP)

        # Pass 2: positions in local coords (origin at 0,0).
        ops = []
        line_origins = []
        acc = 0
        for pitch in pitches:
            line_origins.append(acc)
            acc += pitch
        if vertical:
            # vertical-rl: the first line is the RIGHTMOST column. Mirror the
            # slot origins so the GT ("\n"-joined lines) matches reading order.
            line_origins = [acc - o - p for o, p in zip(line_origins, pitches)]
        for li, (fs, sp, specs) in enumerate(specs_per_line):
            total = totals[li] or 1
            pos = 0
            for ch, font, cell, adv, angle, cross, p, asp in specs:
                t = (pos + adv / 2) / total
                off_px = curve_offset(t) * fs
                slope = curve_slope(t)
                tangent_deg = math.degrees(math.atan(slope * fs / max(total, 1)))
                # For vertical text the reading axis is y; transpose the offset.
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
                        # Unit warp coefficients, drawn ONCE per glyph. Every render pass
                        # (black outer edge, coloured fill, shadow) must reuse the same
                        # warp, otherwise the outline is warped differently from the
                        # glyph and shows up as an offset "ghost" copy of it.
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
        extra = int(math.ceil(max_cell * 0.9 + amp_em * max(line_fs)))
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
        layer, ops, vertical, fill, stroke_width, stroke_fill, font_path, font_size,
        use_op_fill=True, edge_override=None,
    ):
        draw = ImageDraw.Draw(layer)
        if any("persp" in op for op in ops):
            # SFX: paint big glyphs first so small ones (and their outlines) stay on top.
            ops = sorted(ops, key=lambda o: -float(o.get("cell", 0) or 0))
        for op in ops:
            angle = float(op.get("angle", 0) or 0)
            persp = float(op.get("persp", 0) or 0)
            if use_op_fill:
                op_fill = op.get("fill", fill)
                op_edge = edge_override if edge_override is not None else op.get("stroke", stroke_fill)
            else:
                # Shadow/glow pass: each glyph glows in its own shadow tone
                # (black for light fills, sampled glow otherwise).
                op_fill = op.get("shadow", fill)
                op_edge = None
            if op["kind"] == "tcy":
                _draw_tcy_tile(
                    layer,
                    op["text"],
                    op["cx"],
                    op["cy"],
                    font_path=font_path,
                    font_size=op.get("cell", font_size),
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
                    cell=op.get("cell", font_size),
                    angle=angle,
                    persp=persp,
                    fill=op_fill,
                    stroke_width=stroke_width,
                    stroke_fill=op_edge,
                    rotate_mixed=bool(op.get("rotate", False)),
                    ttb=bool(op.get("ttb", False)),
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
                    cell=op.get("cell", font_size),
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
                    cell=op.get("cell", font_size),
                    angle=angle,
                    fill=op_fill,
                    stroke_width=stroke_width,
                    stroke_fill=op_edge,
                )
            else:
                kw = {"font": op["font"], "fill": op_fill, "anchor": "mm"}
                if op.get("ttb"):
                    kw.update(direction="ttb", language="ja")
                if stroke_width:
                    kw.update(stroke_width=stroke_width, stroke_fill=op_edge)
                draw.text((op["cx"], op["cy"]), op["ch"], **kw)

    @staticmethod
    def get_random_sfx_params():
        """Sample SFX lettering params (separate dataset split, not dialogue)."""
        font_size = int(np.random.uniform(*SFX_FONT_SIZE_RANGE))
        text_color = _sample_text_color(sfx=True)
        params = {
            "sfx": True,
            "font_size": font_size,
            "vertical": True if np.random.rand() < SFX_VERTICAL_PROB else False,
            "line_height": 0.3,
            "background_color": "transparent",
            "text_color": text_color,
            "color_mode": "solid",
            "irregular": False,
            # Cap the edge relative to the glyph size: a 10-14 px double outline on a
            # ~70 px thin hiragana fills its counters and the glyph turns into a blob.
            "stroke_size": min(int(np.random.choice([2, 3, 4, 5, 6])), max(2, round(SFX_STROKE_FRAC * font_size))),
            "stroke_color": "white",
            "sfx_curve": str(np.random.choice(SFX_CURVE_MODES, p=SFX_CURVE_P)),
            "sfx_aspect": float(np.random.uniform(0.0, SFX_ASPECT_MAX)) if np.random.rand() < 0.6 else 0.0,
            "sfx_speedline_style": str(np.random.choice(["radial", "parallel"], p=[0.6, 0.4])),
            "letter_spacing": float(np.random.uniform(*SFX_SPACING_RANGE)),
            "sfx_curve_amp": float(np.random.uniform(*SFX_CURVE_AMP_RANGE)),
            "sfx_curve_sign": float(np.random.choice([-1.0, 1.0])),
            "sfx_persp": float(np.random.uniform(*SFX_PERSP_RANGE)),
            "sfx_rotate": float(np.random.uniform(8.0, SFX_ROTATE_DEG)),
            "sfx_char_scale_min": float(SFX_CHAR_SCALE_RANGE[0]),
            "sfx_char_scale_max": float(np.random.uniform(1.4, SFX_CHAR_SCALE_RANGE[1])),
            "sfx_wobble": 0.05,
            "sfx_speedlines": bool(np.random.rand() < SFX_SPEEDLINE_PROB),
            "sfx_speedline_n": int(np.random.randint(*SFX_SPEEDLINE_N_RANGE)),
            "sfx_burst": bool(np.random.rand() < SFX_BURST_PROB),
            "sfx_double_outline": bool(np.random.rand() < SFX_DOUBLE_OUTLINE_PROB),
            "sfx_block_rotate": (
                float(np.random.uniform(*SFX_BLOCK_ROTATE_RANGE)) if np.random.rand() < 0.5 else 0.0
            ),
        }
        if np.random.rand() < 0.7:
            params["text_orientation"] = "upright"
        if text_color == "black" and np.random.rand() < 0.2:
            # white ink with a heavy black edge (very common SFX look)
            params["text_color"] = "white"
            params["stroke_color"] = "black"
        if np.random.rand() < 0.08:
            params["slant"] = float(np.random.uniform(*SLANT_MILD_RANGE) * np.random.choice([-1.0, 1.0]))
        if text_color != "black" and np.random.rand() < SFX_MULTICOLOR_PROB:
            params["color_mode"] = _sample_color_mode(sfx=True)
            params["text_colors"] = _sample_text_colors(int(np.random.randint(2, 5)), sfx=True)
            if text_color not in params["text_colors"]:
                params["text_colors"][0] = text_color
        return _ensure_legible_outline(params)

    @staticmethod
    def get_random_css_params(sfx=False):
        if sfx:
            return Renderer.get_random_sfx_params()
        text_color = _sample_text_color(sfx=False)
        params = {
            "font_size": 48,
            "vertical": True if np.random.rand() < 0.7 else False,
            "line_height": 0.5,
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
            # Small glow only: large blurs read as ghost/double characters.
            params["shadow_size"] = np.random.choice([1, 2, 3])
            params["shadow_color"] = "white" if np.random.rand() < 0.8 else "black"
        elif stroke_variant == "none":
            pass

        if np.random.rand() < IRREGULAR_PROB:
            # Challenging hand-lettered crops: per-line sizes + staggered line
            # starts. Half of them stay mild (realscan style), the other half
            # adds per-character emphasis sizing, wobble, tilt and small kana.
            params["irregular"] = True
            if np.random.rand() < 0.5:
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
                    line_scale_min=0.6,
                    line_scale_max=1.6,
                    line_shift=1.5,
                    char_jitter_prob=0.6,
                    char_scale_min=0.65,
                    char_scale_max=1.6,
                    wobble=0.08,
                    rotate=float(np.random.uniform(2, 7)),
                    small_kana=True,
                )

        if np.random.rand() < SLANT_PROB:
            # Slanted emphasis lettering: usually a mild italic-like lean,
            # occasionally a dramatic skew.
            if np.random.rand() < SLANT_STRONG_PROB:
                low, high = SLANT_STRONG_RANGE
            else:
                low, high = SLANT_MILD_RANGE
            params["slant"] = float(np.random.uniform(low, high) * np.random.choice([-1.0, 1.0]))

        if text_color != "black" and np.random.rand() < MULTICOLOR_PROB:
            # Mixed-colour lettering within one image: each line (emphasis) or
            # each glyph (rainbow) gets its own colour from the list.
            params["color_mode"] = _sample_color_mode(sfx=False)
            params["text_colors"] = _sample_text_colors(int(np.random.randint(2, 5)), sfx=False)
            if text_color not in params["text_colors"]:
                params["text_colors"][0] = text_color

        return _ensure_legible_outline(params)

    # ------------------------------------------------------------ backgrounds
    def _load_background(self, h, w):
        """Random background patch of size (h, w), BGR uint8.

        Picks among the sources whose aspect ratio is closest to the target and
        cuts a random window of exactly the target aspect, so nothing is
        stretched (the old ``A.Resize`` squashed arbitrary-ratio crops, which
        reads as blur) and the same source gives many different patches.
        """
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
            bg = cv2.imread(row.path)
            if bg is not None:
                break
            near = np.arange(len(df))  # unreadable file: fall back to anything
        if bg is None:
            raise FileNotFoundError("no readable background image")

        bh, bw = bg.shape[:2]
        if bw / bh > w / h:
            wh, ww = bh, bh * w / h
        else:
            ww, wh = bw, bw * h / w
        # Random zoom into the source, but never so far that it must be
        # upscaled much (upscaling is the other source of blur).
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
        """(edge_lum, ring_mask) of a BGRA text image.

        ``edge_lum`` is the mean grey level of the outermost opaque ink pixels
        (what actually touches the background: the outline if there is one,
        else the fill). ``ring_mask`` marks the background pixels just outside
        the ink, where legibility is decided. ``edge_lum`` is None without ink.
        """
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
        """Fraction of ring pixels whose grey level is too close to the ink edge."""
        if edge_lum is None or ring is None or not ring.any():
            return 0.0
        g = bg.astype(np.float32).mean(axis=2)[ring]
        return float((np.abs(g - edge_lum) < BG_MIN_EDGE_DIFF).mean())

    def _make_background(self, bg0, img):
        """Tone + decorate ``bg0`` until the ink is legible against it."""
        edge_lum, ring = self._ink_edge(img)
        ink_lum = 0.0 if edge_lum is None else edge_lum
        for _ in range(BG_MAX_TRIES):
            bg = self._decorate_background(self._adjust_background(bg0, ink_lum))
            if self._low_contrast_frac(bg, edge_lum, ring) <= BG_MAX_LOW_CONTRAST:
                return bg
        # Deterministic fallback: push the mean to the far side of the ink.
        far = 70.0 if ink_lum >= 128 else 195.0
        f = bg0.astype(np.float32)
        f = (f - f.mean()) * 1.0 + far
        bg = np.clip(f, 0, 255).astype(np.uint8)
        if self._low_contrast_frac(bg, edge_lum, ring) > BG_MAX_LOW_CONTRAST:
            bg = np.full_like(bg0, int(far))  # last resort: flat but legible
        return bg

    @staticmethod
    def _adjust_background(bg, ink_lum):
        """Make a background harder: keep texture, move its brightness off the extremes.

        ``natural`` keeps the source brightness, ``midtone`` moves it to a mid
        grey, ``ink_match`` moves it close to the ink's own brightness (dark
        art behind black ink, light art behind white ink), which is the case an
        OCR model finds hardest. Texture contrast is never reduced - it is
        lifted to a minimum std - so the background stays busy rather than flat.
        """
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
        """Optional screentone / gradient / sensor noise on a background."""
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
        if np.random.rand() < BG_NOISE_PROB:
            f += np.random.normal(0, np.random.uniform(3, 10), (h, w, 1)).astype(np.float32)
        return np.clip(f, 0, 255).astype(np.uint8)

    @staticmethod
    def _sample_bubble_fill():
        """(mode, fill_bgr, contour_bgr). Bubbles used to be pure white only."""
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
        return mode, white, black  # "white" and "tone" (tone gets a texture later)

    def render_background(self, img, draw_bubble=None, tight=False):
        """Add background and/or text bubble to a BGRA image, crop and return as BGR image.

        ``tight=True`` (SFX) keeps only a thin margin around the ink and skips
        the random loosening crop, so the result hugs the lettering.
        """
        if draw_bubble is None:
            draw_bubble = np.random.random() < 0.7

        if tight:
            draw_bubble = False
            side = min(img.shape[:2])
            m0 = max(SFX_CROP_MARGIN_MIN, int(side * np.random.uniform(*SFX_CROP_MARGIN_FRAC)))
        else:
            m0 = int(min(img.shape[:2]) * 0.3)
        img = crop_by_alpha(img, m0)

        fill_mode, fill_bgr, contour_bgr = "white", (255, 255, 255), (0, 0, 0)
        if draw_bubble:
            edge_lum, _ = self._ink_edge(img)
            edge_lum = 0.0 if edge_lum is None else edge_lum
            # Resample the fill until the ink edge is clearly distinct from it
            # (screentone fills are checked at their darkest shade). Light ink
            # on a light gray bubble, or dark ink on a dark box, never ships.
            for _ in range(BG_MAX_TRIES):
                fill_mode, fill_bgr, contour_bgr = self._sample_bubble_fill()
                ink = 255.0 - edge_lum if fill_mode == "dark" else edge_lum
                fl = float(np.mean(fill_bgr))
                lums = [fl, 255.0 * (1.0 - 0.45)] if fill_mode == "tone" else [fl]
                if all(abs(ink - L) >= BG_MIN_EDGE_DIFF for L in lums):
                    break
            else:
                fill_mode, fill_bgr, contour_bgr = ("white", (255, 255, 255), (0, 0, 0))
                if edge_lum > 150:  # light ink: use a dark caption box instead
                    fill_mode, fill_bgr, contour_bgr = ("dark", (30, 30, 30), (235, 235, 235))
            if fill_mode == "dark":  # light lettering on a dark caption box
                img[:, :, :3] = 255 - img[:, :, :3]
        elif np.random.rand() < 0.5:
            img[:, :, :3] = 255 - img[:, :, :3]

        background = self._load_background(img.shape[0], img.shape[1])
        if draw_bubble:
            # The lettering sits on the bubble fill, not on this art, so the art
            # only needs to be varied (the bubble contrast is set by the fill).
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
                # Screentone-filled bubble (dotted/shaded interior). The black
                # contour stays black because it is multiplied by a value in (0, 1].
                dots = _screentone(bubble.shape[0], bubble.shape[1])
                shade = 1.0 - np.random.uniform(0.15, 0.45) * dots
                bubble[:, :, :3] = (bubble[:, :, :3].astype(np.float32) * shade[..., None]).astype(np.uint8)

            background = blend(bubble, background)

        img = blend(img, background)

        if tight:
            return img

        ymin = m0 - int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        ymax = img.shape[0] - m0 + int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        xmin = m0 - int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        xmax = img.shape[1] - m0 + int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        # Clamp: full-bleed ink (e.g. SFX speedlines spanning the canvas) can
        # otherwise push ymin/xmin negative, which numpy interprets as
        # wrap-around and yields an empty crop.
        ymin = max(0, ymin)
        xmin = max(0, xmin)
        ymax = min(img.shape[0], ymax)
        xmax = min(img.shape[1], xmax)
        if ymax <= ymin:
            ymin, ymax = 0, img.shape[0]
        if xmax <= xmin:
            xmin, xmax = 0, img.shape[1]
        img = img[ymin:ymax, xmin:xmax]
        return img

    def lines_to_html(self, lines):
        lines_str = "\n".join(["<p>" + line + "</p>" for line in lines])
        html = f"<html><body>\n{lines_str}\n</body></html>"
        return html


def _advance(font, ch, letter_spacing_px):
    try:
        w = font.getlength(ch)
    except (OSError, ValueError):
        w = 0
    if not w:
        w = font.size
    return w + letter_spacing_px


def _char_op(ch, cx, cy, font, vertical, upright):
    if vertical and is_ascii(ch) and not upright and not ch.isspace():
        # 'mixed' orientation: latin runs sideways in vertical text.
        return {"kind": "char", "ch": ch, "cx": cx, "cy": cy, "font": font, "rotate": True, "ttb": False}
    return {
        "kind": "char",
        "ch": ch,
        "cx": cx,
        "cy": cy,
        "font": font,
        "rotate": False,
        "ttb": vertical and not is_ascii(ch),
    }


def _char_op_ex(ch, cx, cy, font, cell, angle, vertical, upright):
    """Irregular-layout variant of ``_char_op`` with per-glyph cell/tilt."""
    if vertical and is_ascii(ch) and not upright and not ch.isspace():
        # 'mixed' orientation: latin runs sideways in vertical text.
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


def _draw_rotated_char(layer, ch, cx, cy, font, cell, fill, stroke_width, stroke_fill, angle=0.0):
    """Draw a latin char rotated 90 deg clockwise (vertical 'mixed' mode).

    ``angle`` adds a small extra tilt (counter-clockwise degrees, as in
    ``Image.rotate``) used by the irregular layout mode.
    """
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
    """Draw a glyph with a slight tilt (irregular layout mode).

    The tile is oversized so small tilts never clip the glyph or its stroke.
    """
    cell = int(cell)
    pad = int(stroke_width or 0) + 2
    size = math.ceil(cell * 1.5) + 2 * pad
    size += size & 1  # even tile: centre on a pixel edge, identical across passes
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    kw = {"font": font, "fill": fill, "anchor": "mm"}
    if stroke_width:
        kw.update(stroke_width=stroke_width, stroke_fill=stroke_fill)
    draw.text((size / 2, size / 2), ch, **kw)
    tile = tile.rotate(angle, resample=Image.BICUBIC)
    layer.paste(tile, (int(round(cx)) - size // 2, int(round(cy)) - size // 2), tile)


def _draw_perspective_char(
    layer, ch, cx, cy, font, cell, angle, persp, fill, stroke_width, stroke_fill,
    rotate_mixed=False, ttb=False, aspect=None, pcoef=None,
):
    """Draw one SFX glyph with stretch, tilt and a true perspective warp.

    The glyph (with its stroke) is rasterized once on a tile that is sized from
    the actual worst case (glyph ink x stretch x rotation x perspective) rather
    than a fixed oversize, so the outline stays aligned, nothing clips and the
    expensive rotate/warp passes touch as few pixels as possible. ``persp`` is
    the max coefficient magnitude (typically 0.12-0.28): visible skew without
    destroying legibility.
    """
    cell = int(cell)
    pad = int(stroke_width or 0) + 4
    p = abs(float(persp))
    ax, ay = (float(aspect[0]), float(aspect[1])) if aspect else (1.0, 1.0)
    total_angle = (270.0 if rotate_mixed else 0.0) + float(angle or 0.0)
    rad = math.radians(total_angle)
    rot_f = abs(math.cos(rad)) + abs(math.sin(rad))  # bounding-box growth under rotation
    ink = math.ceil(cell * 1.25) + 2 * pad  # em box + overshoot + outline
    size = max(8, math.ceil(ink * max(ax, ay, 1.0) * rot_f * (1.0 + p)) + 6)
    size += size & 1  # even => the glyph centre sits on a pixel edge in every pass
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    kw = {"font": font, "fill": fill, "anchor": "mm"}
    if ttb:
        kw.update(direction="ttb", language="ja")
    if stroke_width:
        kw.update(stroke_width=stroke_width, stroke_fill=stroke_fill)
    draw.text((size / 2, size / 2), ch, **kw)
    if aspect:
        nw, nh = max(2, int(round(size * ax))), max(2, int(round(size * ay)))
        nw += (size - nw) & 1  # keep the stretched glyph exactly centred (same parity as tile)
        nh += (size - nh) & 1
        stretched = tile.resize((nw, nh), Image.BILINEAR)
        tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        tile.paste(stretched, ((size - nw) // 2, (size - nh) // 2))
    if total_angle:
        tile = tile.rotate(total_angle, resample=Image.BILINEAR)
    if p > 1e-6:
        u = pcoef if pcoef is not None else tuple(float(v) for v in np.random.uniform(-1.0, 1.0, 8))
        # Warp is defined around the tile CENTRE and scaled by the glyph's em size,
        # so it is identical no matter how big the tile is (the outer-edge pass
        # uses a thicker stroke, hence a bigger tile, than the fill pass).
        ref = cell * 2.2
        a, b, c0 = 1.0 + u[0] * p, u[1] * p, u[2] * p * ref * 0.1
        d, e, f0 = u[3] * p, 1.0 + u[4] * p, u[5] * p * ref * 0.1
        g, h = u[6] * p / ref, u[7] * p / ref
        tx = ty = size / 2.0  # tile centre (NOT the glyph's layer position cx, cy)
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
    layer.paste(tile, (int(round(cx)) - size // 2, int(round(cy)) - size // 2), tile)


def _draw_speedlines_layer(width, height, n_lines=40, color=(0, 0, 0, 255)):
    """Radiating speedlines behind SFX text (non-text decor, excluded from GT)."""
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx, cy = width / 2, height / 2
    max_r = math.hypot(width, height) / 2
    inner = min(width, height) * 0.28
    for _ in range(max(1, int(n_lines))):
        ang = float(np.random.uniform(0.0, 2.0 * math.pi))
        # Random gaps keep the burst readable and varied.
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
    """Parallel motion lines (all at one angle) behind SFX text; non-GT decor."""
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
    """Jagged explosion balloon behind SFX text (non-text decor, excluded from GT)."""
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
    """Shear an RGBA text layer horizontally (whole-block slant).

    Applied after shadow/text compositing so outlines and glows stay aligned.
    The canvas is widened to fit the skewed content, so no ink is clipped.
    """
    k = math.tan(math.radians(slant_deg))
    w, h = img.size
    shift = k * h
    new_w = max(1, math.ceil(w + abs(shift)))
    tx = -min(0.0, shift)
    # PIL AFFINE maps output coords back to input coords:
    # forward x_out = x_in + k * y_in + tx  =>  x_in = x_out - k * y_out - tx.
    return img.transform(
        (new_w, h), Image.AFFINE, (1, -k, -tx, 0, 1, 0), resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0)
    )


def _draw_tcy_tile(layer, text, cx, cy, font_path, font_size, fill, stroke_width, stroke_fill, angle=0.0):
    """Squeeze a short ascii run into a single em square (tate-chu-yoko)."""
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


class BrowserRenderer(Renderer):
    """Legacy Chrome-based renderer, kept for side-by-side verification only.

    Requires ``html2image`` + Chrome/Chromium. Not used by the generator.
    """

    def __init__(self, background_df=None):
        from html2image import Html2Image

        self.hti = Html2Image()
        self.background_df = background_df if background_df is not None else get_default_background_df()
        self.max_size = 600

    def render_text(self, lines, override_css_params=None):
        """Render text on transparent background and return as BGRA image."""
        import os
        import uuid

        params = self.get_random_css_params()
        if override_css_params:
            params.update(override_css_params)

        css = get_css(**params)

        # this is just a rough estimate, image is cropped later anyway
        size = (
            int(max(len(line) for line in lines) * params["font_size"] * 1.5),
            int(len(lines) * params["font_size"] * (3 + params["line_height"])),
        )
        if params["vertical"]:
            size = size[::-1]
        html = self.lines_to_html(lines)

        filename = str(uuid.uuid4()) + ".png"
        self.hti.screenshot(html_str=html, css_str=css, save_as=filename, size=size)
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        os.remove(filename)
        return img, params


def _screentone(h, w):
    """Halftone dot pattern in 0..1 (1 = dot), random pitch and angle."""
    period = np.random.uniform(3.0, 8.0)
    ang = np.random.uniform(0, np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx * np.cos(ang) + yy * np.sin(ang)) * (2 * np.pi / period)
    v = (-xx * np.sin(ang) + yy * np.cos(ang)) * (2 * np.pi / period)
    pat = 0.5 * (np.cos(u) + np.cos(v))  # -1..1, peaks at dot centres
    return np.clip((pat - np.random.uniform(0.0, 0.6)) * 4.0, 0.0, 1.0)


def crop_by_alpha(img, margin):
    y, x = np.where(img[:, :, 3] > 0)
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
    """From https://stackoverflow.com/a/60210706"""

    #  corners:
    #  p1 - p2
    #  |     |
    #  p4 - p3

    p1 = top_left
    p2 = (bottom_right[0], top_left[1])
    p3 = bottom_right
    p4 = (top_left[0], bottom_right[1])

    height = abs(bottom_right[1] - top_left[1])
    width = abs(bottom_right[0] - top_left[0])

    if radius > 1:
        radius = 1

    corner_radius = int(radius * (min(height, width) / 2))

    if thickness < 0:
        # big rect
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

    # draw straight lines
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

    # draw arcs
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


def get_css(
    font_size,
    font_path,
    vertical=True,
    background_color="white",
    text_color="black",
    shadow_size=0,
    shadow_color="black",
    stroke_size=0,
    stroke_color="black",
    letter_spacing=None,
    line_height=0.5,
    text_orientation=None,
    **_ignored,
):
    styles = [
        f"background-color: {background_color};",
        f"font-size: {font_size}px;",
        f"color: {text_color};",
        "font-family: custom;",
        f"line-height: {line_height};",
        "margin: 20px;",
    ]

    if text_orientation:
        styles.append(f"text-orientation: {text_orientation};")

    if vertical:
        styles.append("writing-mode: vertical-rl;")

    if shadow_size > 0:
        styles.append(f"text-shadow: 0 0 {shadow_size}px {shadow_color};")

    if stroke_size > 0:
        # stroke is simulated by shadow overlaid multiple times
        styles.extend(
            [
                "text-shadow: " + ",".join([f"0 0 {stroke_size}px {stroke_color}"] * 10 * stroke_size) + ";",
                "-webkit-font-smoothing: antialiased;",
            ]
        )

    if letter_spacing:
        styles.append(f"letter-spacing: {letter_spacing}em;")

    font_path = font_path.replace("\\", "/")

    styles_str = "\n".join(styles)
    css = ""
    css += '\n@font-face {\nfont-family: custom;\nsrc: url("' + font_path + '");\n}\n'
    css += "body {\n" + styles_str + "\n}"
    return css
