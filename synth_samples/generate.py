"""Regenerate the showcase images in this directory (text layer only, RGBA PNGs).

Run from the repo root: ``python3 synth_samples/generate.py``.
Layouts are seeded, so re-running reproduces the same files.

Every sample pins *all* style knobs it depends on. ``Renderer.render_text``
fills any key that is not given with a random draw (slant, shadow, text
orientation, letter spacing, ...), so an un-pinned knob would silently change
what a file named e.g. ``04_vertical_mixed_tcy`` actually shows.

Each text is also checked against the font's cmap before rendering, so a
showcase image can never contain tofu boxes.
"""

import re
import sys
from pathlib import Path

import fire
import numpy as np
from fontTools.ttLib import TTCollection, TTFont
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manga_ocr_dev.synthetic_data_generator.renderer import Renderer  # noqa: E402

OUT_DIR = Path(__file__).parent

FONT_CANDIDATES = [
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
# Heavier face for SFX when available (falls back to the regular one).
BOLD_CANDIDATES = [p.replace("Regular", "Black") for p in FONT_CANDIDATES] + [
    p.replace("Regular", "Bold") for p in FONT_CANDIDATES
]

_CMAP_CACHE = {}


def _first_existing(paths, what):
    for path in paths:
        if Path(path).is_file():
            return path
    raise OSError(f"No {what} font found; edit FONT_CANDIDATES.")


def _resolve_font():
    return _first_existing(FONT_CANDIDATES, "Japanese")


def _resolve_sfx_font():
    for path in BOLD_CANDIDATES + FONT_CANDIDATES:
        if Path(path).is_file():
            return path
    raise OSError("No Japanese font found; edit FONT_CANDIDATES.")


def _cmap(font_path):
    if font_path not in _CMAP_CACHE:
        if font_path.lower().endswith((".ttc", ".otc")):
            font = TTCollection(font_path, lazy=True).fonts[0]  # face 0 = JP in Noto CJK
        else:
            font = TTFont(font_path, lazy=True)
        chars = set()
        for table in font["cmap"].tables:
            if table.isUnicode():
                chars.update(table.cmap.keys())
        _CMAP_CACHE[font_path] = chars
    return _CMAP_CACHE[font_path]


def _visible_text(lines):
    """Characters that are actually drawn (markup stripped)."""
    text = "".join(lines)
    text = re.sub(r"<rt>.*?</rt>", lambda m: re.sub(r"<[^>]+>", "", m.group(0)), text)
    return re.sub(r"<[^>]+>", "", text)


def _check_glyphs(name, lines, font_path):
    cmap = _cmap(font_path)
    missing = sorted({c for c in _visible_text(lines) if not c.isspace() and ord(c) not in cmap})
    if missing:
        raise ValueError(f"{name}: font has no glyph for {missing}; pick another text or font")


def _base(vertical=True, **kwargs):
    params = {
        "font_path": _resolve_font(),
        "vertical": vertical,
        "font_size": 48,
        "line_height": 0.5,
        "background_color": "transparent",
        "irregular": False,
        "text_color": "black",
        "color_mode": "solid",
        # pinned so nothing is left to the random sampler:
        "text_orientation": "upright",
        "stroke_size": 0,
        "shadow_size": 0,
        "letter_spacing": 0.0,
        "slant": 0.0,
    }
    params.update(kwargs)
    return params


def _sfx_base(**kwargs):
    params = {
        "font_path": _resolve_sfx_font(),
        "sfx": True,
        "font_size": 120,
        "vertical": False,
        "line_height": 0.3,
        "background_color": "transparent",
        "text_color": "black",
        "color_mode": "solid",
        "irregular": False,
        "stroke_size": 8,
        "stroke_color": "white",
        "letter_spacing": 0.0,
        "text_orientation": "upright",
        "sfx_curve": "arc",
        "sfx_curve_amp": 0.3,
        "sfx_curve_sign": 1.0,
        "sfx_persp": 0.2,
        "sfx_rotate": 14.0,
        "sfx_aspect": 0.0,
        "sfx_char_scale_min": 0.8,
        "sfx_char_scale_max": 1.9,
        "sfx_wobble": 0.05,
        "sfx_speedlines": False,
        "sfx_speedline_style": "radial",
        "sfx_speedline_n": 40,
        "sfx_burst": False,
        "sfx_double_outline": True,
        "sfx_block_rotate": 0.0,
        "slant": 0.0,
        "small_kana": True,
    }
    params.update(kwargs)
    return params


_IRR_MILD = dict(
    irregular=True, line_scale_min=0.8, line_scale_max=1.25, line_shift=0.8,
    char_jitter_prob=0.0, wobble=0.05, rotate=0.0, small_kana=False,
)
_IRR_STRONG = dict(
    irregular=True, line_scale_min=0.6, line_scale_max=1.6, line_shift=1.5,
    char_jitter_prob=0.6, char_scale_min=0.65, char_scale_max=1.6, wobble=0.08,
    rotate=5.0, small_kana=True,
)


def _samples():
    """(file name, lines, params, seed). Numbering of 01-14 is kept stable."""
    tcy = '<span style="text-combine-upright: all">12</span>'
    stroke = dict(stroke_size=2, stroke_color="white")
    return [
        # ---- dialogue -----------------------------------------------------
        ("01_vertical_stroke.png", ["約束だよ", "必ず帰る"], _base(**stroke), 0),
        (
            "02_vertical_ruby.png",
            ["<ruby>昨日<rt>きのう</rt></ruby>の夜", "不思議な夢を見た"],
            _base(**stroke),
            1,
        ),
        (
            "03_horizontal_shadow.png",
            ["水平線の向こう側"],
            _base(vertical=False, shadow_size=3, shadow_color="black"),
            2,
        ),
        # "mixed" orientation: Latin runs sideways, tate-chu-yoko for the digits.
        (
            "04_vertical_mixed_tcy.png",
            [f"出発は{tcy}時", "ABC遅刻厳禁"],
            _base(text_orientation="mixed"),
            3,
        ),
        ("05_horizontal_plain.png", ["今日はいい天気だ"], _base(vertical=False), 4),
        ("06_irregular_mild.png", ["お前との婚約", "は破棄する"], _base(**_IRR_MILD), 5),
        ("07_irregular_strong.png", ["お前との婚約", "は破棄する"], _base(**_IRR_STRONG), 6),
        (
            "08_irregular_horizontal.png",
            ["彼女は勇者", "パーティの仲間"],
            _base(vertical=False, **_IRR_MILD),
            7,
        ),
        (
            "09_slanted_irregular.png",
            ["彼女は勇者", "パーティの"],
            _base(slant=25.0, **_IRR_STRONG),
            8,
        ),
        # ---- SFX (original five) ------------------------------------------
        ("10_sfx_arc_speedlines.png", ["ドドドン"], _sfx_base(sfx_curve="arc", sfx_speedlines=True), 9),
        ("11_sfx_wave_burst.png", ["バゴォン"], _sfx_base(sfx_curve="wave", sfx_burst=True), 10),
        (
            "12_sfx_diagonal_persp.png",
            ["ズドン"],
            _sfx_base(sfx_curve="diagonal", sfx_persp=0.28, sfx_double_outline=False),
            11,
        ),
        ("13_sfx_vertical_hearts.png", ["ドキドキ♡"], _sfx_base(vertical=True, sfx_curve="arc"), 12),
        (
            "14_sfx_ecchi_burst.png",
            ["あんっ"],
            # thin hiragana: keep the edge light so the counters stay open
            _sfx_base(
                sfx_curve="wave", sfx_burst=True, sfx_speedlines=True, stroke_size=4, sfx_persp=0.12,
                font_path=_resolve_font(), sfx_char_scale_max=1.5,
            ),
            13,
        ),
        # ---- SFX (new curves / styling) -----------------------------------
        (
            "15_sfx_zigzag_parallel_lines.png",
            ["ガタガタ"],
            _sfx_base(sfx_curve="zigzag", sfx_speedlines=True, sfx_speedline_style="parallel"),
            14,
        ),
        (
            "16_sfx_bounce_stretched.png",
            ["ピョンピョン"],
            _sfx_base(sfx_curve="bounce", sfx_aspect=0.35, sfx_persp=0.15),
            15,
        ),
        (
            "17_sfx_hill_white_ink.png",
            ["ズガーン！！"],
            _sfx_base(
                sfx_curve="hill", text_color="white", stroke_color="black", stroke_size=6,
                sfx_double_outline=False, sfx_block_rotate=-10.0, sfx_persp=0.12, sfx_rotate=8.0,
            ),
            16,
        ),
        # Two lines in vertical text: the FIRST label line is the RIGHT column.
        (
            "18_sfx_vertical_two_columns.png",
            ["ワーゴ", "ロゴロ"],
            _sfx_base(vertical=True, sfx_curve="straight", sfx_persp=0.1, sfx_rotate=6.0),
            17,
        ),
        # Stretched vowels, tight spacing (glyphs may touch), hiragana SFX.
        (
            "19_sfx_tight_spacing_stretched_vowel.png",
            ["ドォォン"],
            _sfx_base(letter_spacing=-0.06, sfx_curve="arc", sfx_burst=True),
            18,
        ),
        (
            "20_sfx_soft_hiragana_airy.png",
            ["ぷにぷに♡"],
            _sfx_base(letter_spacing=0.22, sfx_curve="wave", sfx_persp=0.1, sfx_rotate=8.0),
            19,
        ),
        # ---- colour --------------------------------------------------------
        (
            "21_vertical_per_line_colour.png",
            ["絶対に許さない", "覚悟しろ"],
            _base(color_mode="per_line", text_colors=["red", "pink", "orange"], **stroke),
            20,
        ),
        (
            "22_sfx_per_char_rainbow.png",
            ["キラキラ"],
            _sfx_base(color_mode="per_char", text_colors=["red", "yellow", "cyan", "pink"], sfx_curve="wave"),
            21,
        ),
    ]


def _save(name, lines, params, seed):
    _check_glyphs(name, lines, params["font_path"])
    np.random.seed(seed)
    r = Renderer.__new__(Renderer)
    r.max_size = 600
    img, _ = r.render_text(lines, params)
    # render_text returns BGRA (BGR channel order); flip back to RGBA for PNG.
    img = img[:, :, [2, 1, 0, 3]]
    Image.fromarray(img).save(OUT_DIR / name)
    print(f"{name}: {img.shape[1]}x{img.shape[0]}")


def main(only=None):
    """Render all showcase images. ``--only=sfx`` renders just files whose name contains 'sfx'."""
    for name, lines, params, seed in _samples():
        if only and str(only) not in name:
            continue
        _save(name, lines, params, seed)


if __name__ == "__main__":
    fire.Fire(main)
