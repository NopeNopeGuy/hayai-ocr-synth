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

Why not Tauri (or another webview)? A webview is still a full browser engine
(WebKit/WebView2/WKWebView) behind IPC + native build tooling, with no Python
screenshot API. It would keep the per-image engine overhead while adding a
Rust toolchain dependency, so it is slower to install, slower to run and
harder to parallelize than plain Pillow.
"""

import math
import re
import threading

import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from manga_ocr_dev.env import BACKGROUND_DIR
from manga_ocr_dev.synthetic_data_generator.utils import get_background_df, is_ascii

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

RUBY_RE = re.compile(r"<ruby>(.*?)<rt>(.*?)</rt></ruby>")
TCY_RE = re.compile(r'<span[^>]*?text-combine-upright[^>]*?>(.*?)</span>')
TAG_RE = re.compile(r"<[^>]+>")

_NAMED_COLORS = {
    "black": (0, 0, 0, 255),
    "white": (255, 255, 255, 255),
    "transparent": (0, 0, 0, 0),
}

_FONT_CACHE = {}
_FONT_CACHE_LOCK = threading.Lock()


def _normalize_color(value, default="black"):
    """Map a CSS color name (or legacy tuple bug) to an RGBA tuple."""
    if value is None:
        value = default
    if isinstance(value, (tuple, list)):
        # legacy get_random_css_params accidentally produced ("white",)
        value = value[0] if len(value) > 0 else default
    if not isinstance(value, str):
        value = default
    return _NAMED_COLORS.get(value.strip().lower(), _NAMED_COLORS[default])


def _get_font(font_path, size):
    """Load (and cache) a TrueType/OpenType font. Thread-safe."""
    key = (str(font_path), int(size))
    with _FONT_CACHE_LOCK:
        font = _FONT_CACHE.get(key)
        if font is None:
            font = ImageFont.truetype(str(font_path), int(size))
            _FONT_CACHE[key] = font
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
        self.background_df = background_df if background_df is not None else get_background_df(BACKGROUND_DIR)
        self.max_size = 600

    def render(self, lines, override_css_params=None):
        img, params = self.render_text(lines, override_css_params)
        img = self.render_background(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = A.LongestMaxSize(self.max_size)(image=img)["image"]
        return img, params

    def render_text(self, lines, override_css_params=None):
        """Render text on a transparent background and return as BGRA image."""

        params = self.get_random_css_params()
        if override_css_params:
            params.update(override_css_params)

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

        irregular = bool(params.get("irregular", False))
        pad = max(stroke_size, shadow_size, 0) + MIN_PAD

        parsed = [_parse_line(line) for line in lines]

        if irregular:
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
            )
            shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=shadow_size))
            base = Image.alpha_composite(base, shadow_layer)

        self._execute_ops(
            main_layer,
            ops,
            vertical=vertical,
            fill=text_rgba,
            stroke_width=stroke_size,
            stroke_fill=stroke_rgba if stroke_size > 0 else None,
            font_path=font_path,
            font_size=font_size,
        )
        base = Image.alpha_composite(base, main_layer)

        slant = float(params.get("slant", 0.0))
        if slant:
            base = _shear_layer(base, slant)

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
          {"kind": "char", "ch", "cx", "cy", "font", "rotate", "ttb"}
          {"kind": "tcy", "text", "cx", "cy"}
        Coordinates are cell centers in canvas pixels.
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
            for tokens, pitch, ruby in zip(parsed, pitches, has_ruby):
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
            for tokens, pitch, ruby in zip(parsed, pitches, has_ruby):
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
                slot_right = slot_left
        else:
            width = max(1, math.ceil(inner) + 2 * margin + 2 * pad_irr)
            height = max(1, sum(pitches) + 2 * margin + 2 * pad_irr)
            x_left = margin + pad_irr
            slot_top = margin + pad_irr
            for li, (pitch, fs, ruby_size, start) in enumerate(zip(pitches, line_fs, ruby_sizes, starts)):
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
                slot_top += pitch

        return width, height, ops

    @staticmethod
    def _execute_ops(layer, ops, vertical, fill, stroke_width, stroke_fill, font_path, font_size):
        draw = ImageDraw.Draw(layer)
        for op in ops:
            angle = float(op.get("angle", 0) or 0)
            if op["kind"] == "tcy":
                _draw_tcy_tile(
                    layer,
                    op["text"],
                    op["cx"],
                    op["cy"],
                    font_path=font_path,
                    font_size=op.get("cell", font_size),
                    fill=fill,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill,
                    angle=angle,
                )
            elif op.get("rotate"):
                _draw_rotated_char(
                    layer,
                    op["ch"],
                    op["cx"],
                    op["cy"],
                    font=op["font"],
                    cell=op.get("cell", font_size),
                    fill=fill,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill,
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
                    fill=fill,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_fill,
                )
            else:
                kw = {"font": op["font"], "fill": fill, "anchor": "mm"}
                if op.get("ttb"):
                    kw.update(direction="ttb", language="ja")
                if stroke_width:
                    kw.update(stroke_width=stroke_width, stroke_fill=stroke_fill)
                draw.text((op["cx"], op["cy"]), op["ch"], **kw)

    @staticmethod
    def get_random_css_params():
        params = {
            "font_size": 48,
            "vertical": True if np.random.rand() < 0.7 else False,
            "line_height": 0.5,
            "background_color": "transparent",
            "text_color": "black",
        }

        if np.random.rand() < 0.7:
            params["text_orientation"] = "upright"

        stroke_variant = np.random.choice(["stroke", "shadow", "none"], p=[0.8, 0.15, 0.05])
        if stroke_variant == "stroke":
            params["stroke_size"] = np.random.choice([1, 2, 3, 4, 8])
            params["stroke_color"] = "white"
        elif stroke_variant == "shadow":
            params["shadow_size"] = np.random.choice([2, 5, 10])
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

        return params

    def render_background(self, img):
        """Add background and/or text bubble to a BGRA image, crop and return as BGR image."""
        draw_bubble = np.random.random() < 0.7

        m0 = int(min(img.shape[:2]) * 0.3)
        img = crop_by_alpha(img, m0)

        background_path = self.background_df.sample(1).iloc[0].path
        background = cv2.imread(background_path)

        t = [
            A.HorizontalFlip(),
            A.RandomRotate90(),
            A.InvertImg(),
            A.RandomBrightnessContrast((-0.2, 0.4), (-0.8, -0.3), p=0.5 if draw_bubble else 1),
            A.Blur((3, 5), p=0.3),
            A.Resize(img.shape[0], img.shape[1]),
        ]

        background = A.Compose(t)(image=background)["image"]

        if not draw_bubble:
            if np.random.rand() < 0.5:
                img[:, :, :3] = 255 - img[:, :, :3]

        else:
            radius = np.random.uniform(0.7, 1.0)
            thickness = np.random.choice([1, 2, 3])
            alpha = np.random.randint(60, 100)
            sigma = np.random.randint(10, 15)

            ymin = m0 - int(min(img.shape[:2]) * np.random.uniform(0.07, 0.12))
            ymax = img.shape[0] - m0 + int(min(img.shape[:2]) * np.random.uniform(0.07, 0.12))
            xmin = m0 - int(min(img.shape[:2]) * np.random.uniform(0.07, 0.12))
            xmax = img.shape[1] - m0 + int(min(img.shape[:2]) * np.random.uniform(0.07, 0.12))

            bubble_fill_color = (255, 255, 255, 255)
            bubble_contour_color = (0, 0, 0, 255)
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

            background = blend(bubble, background)

        img = blend(img, background)

        ymin = m0 - int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        ymax = img.shape[0] - m0 + int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        xmin = m0 - int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
        xmax = img.shape[1] - m0 + int(min(img.shape[:2]) * np.random.uniform(0.01, 0.2))
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
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    kw = {"font": font, "fill": fill, "anchor": "mm"}
    if stroke_width:
        kw.update(stroke_width=stroke_width, stroke_fill=stroke_fill)
    draw.text((size / 2, size / 2), ch, **kw)
    tile = tile.rotate(angle, resample=Image.BICUBIC)
    layer.paste(tile, (round(cx - size / 2), round(cy - size / 2)), tile)


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
        self.background_df = background_df if background_df is not None else get_background_df(BACKGROUND_DIR)
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


def crop_by_alpha(img, margin):
    y, x = np.where(img[:, :, 3] > 0)
    ymin = y.min()
    ymax = y.max()
    xmin = x.min()
    xmax = x.max()
    img = img[ymin:ymax, xmin:xmax]
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
