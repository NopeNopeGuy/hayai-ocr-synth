"""Regenerate the showcase images in this directory.

Run from the repo root: ``python3 synth_samples/generate.py``.
Layouts are seeded, so re-running reproduces the same files (font and
background choice depend on the installed font/background datasets).

Every sample pins *all* style knobs it depends on. ``Renderer.render`` fills
any key that is not given with a random draw (slant, shadow, text orientation,
letter spacing, ...), so an un-pinned knob would silently change what a file
named e.g. ``04_vertical_mixed_tcy`` actually shows.

Fonts come from the local font library (``assets/fonts.csv`` +
``assets/font_profiles.json``, rooted at ``FONTS_ROOT``), not a single Noto
face: dialogue uses the generator's weighted manga-font sampler, SFX uses the
SFX profile weights. Images are composited over the real background dataset
(AnimeText preferred, Manga109-s fallback) through the production renderer,
including bubbles/degradation. Without a background dataset the samples fall
back to transparent text layers.

Each text is also checked against the chosen font's cmap before rendering, so
a showcase image can never contain tofu boxes.
"""

import sys
from pathlib import Path

import fire
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from manga_ocr_dev.synthetic_data_generator.renderer import (
    _get_real_kana_bank,
    get_default_background_df,
)
from manga_ocr_dev.synthetic_data_generator.utils import is_drawable

OUT_DIR = Path(__file__).parent
FONT_ATTEMPTS = 24

_GENERATOR = None
_HAS_BACKGROUNDS = True
_EMPTY_BACKGROUND_DF = None


def _empty_background_df():
    global _EMPTY_BACKGROUND_DF
    if _EMPTY_BACKGROUND_DF is None:
        import pandas as pd

        _EMPTY_BACKGROUND_DF = pd.DataFrame(
            columns=[
                "source", "image_id", "ymin", "ymax", "xmin", "xmax",
                "h", "w", "n_boxes", "path", "ratio",
            ]
        )
    return _EMPTY_BACKGROUND_DF


def _backgrounds_available():
    try:
        return len(get_default_background_df()) > 0
    except Exception:  # noqa: BLE001 - missing dataset is not fatal
        return False


def _generator():
    """Lazily build the production generator used for font sampling."""
    global _GENERATOR, _HAS_BACKGROUNDS
    if _GENERATOR is None:
        from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator

        _HAS_BACKGROUNDS = _backgrounds_available()
        _GENERATOR = SyntheticDataGenerator(
            background_df=None if _HAS_BACKGROUNDS else _empty_background_df()
        )
    return _GENERATOR


def _visible_text(lines):
    """Characters that are actually drawn (markup stripped)."""
    from manga_ocr_dev.synthetic_data_generator.renderer import _parse_line, _strip_tags

    out = []
    for line in lines:
        for tok in _parse_line(line):
            if tok[0] == "ruby":
                out.append(_strip_tags(tok[1]))
            else:
                out.append(_strip_tags(tok[1] if len(tok) > 1 else ""))
    return "".join(out)


def _check_glyphs(name, lines, font_path):
    # Raster check (not just cmap): never ship a tofu box in showcase images.
    missing = sorted({c for c in _visible_text(lines) if not c.isspace() and not is_drawable(font_path, c)})
    if missing:
        raise ValueError(f"{name}: font cannot draw {missing}; pick another text or font")


def _base(vertical=True, **kwargs):
    params = {
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
        "hollow": False,
        "ink_texture": "solid",
        "extrude": False,
        "distress": False,
        "distress_strength": 0.5,
        "faint": False,
        "faint_alpha": 0.65,
    }
    params.update(kwargs)
    return params


_IRR_MILD = {
    "irregular": True,
    "line_scale_min": 0.8,
    "line_scale_max": 1.25,
    "line_shift": 0.8,
    "char_jitter_prob": 0.0,
    "wobble": 0.05,
    "rotate": 0.0,
    "small_kana": False,
}
_IRR_STRONG = {
    "irregular": True,
    "line_scale_min": 0.6,
    "line_scale_max": 1.6,
    "line_shift": 1.5,
    "char_jitter_prob": 0.6,
    "char_scale_min": 0.65,
    "char_scale_max": 1.6,
    "wobble": 0.08,
    "rotate": 5.0,
    "small_kana": True,
}


def _samples():
    """(file name, lines, params, seed). Numbering of 01-14 is kept stable."""
    tcy = '<span style="text-combine-upright: all">12</span>'
    stroke = {"stroke_size": 2, "stroke_color": "white"}
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
                sfx_curve="wave",
                sfx_burst=True,
                sfx_speedlines=True,
                stroke_size=4,
                sfx_persp=0.12,
                sfx_char_scale_max=1.5,
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
                sfx_curve="hill",
                text_color="white",
                stroke_color="black",
                stroke_size=6,
                sfx_double_outline=False,
                sfx_block_rotate=-10.0,
                sfx_persp=0.12,
                sfx_rotate=8.0,
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
        # ---- hard SFX (outline-only, textured, extruded, steep/circle) ------
        (
            "23_sfx_hollow_steep.png",
            ["ガガガン"],
            _sfx_base(sfx_curve="steep", sfx_curve_amp=0.45, hollow=True, stroke_size=5, sfx_persp=0.3),
            22,
        ),
        (
            "24_sfx_circle_extrude.png",
            ["ドォォン"],
            _sfx_base(sfx_curve="circle", sfx_curve_amp=0.4, extrude=True, sfx_persp=0.3, sfx_rotate=20.0),
            23,
        ),
        (
            "25_sfx_gradient_distress.png",
            ["バゴォン"],
            _sfx_base(
                sfx_curve="wave",
                ink_texture="gradient",
                distress=True,
                distress_strength=0.6,
                sfx_persp=0.32,
                letter_spacing=-0.1,
            ),
            24,
        ),
        (
            "26_sfx_stripe_tiny_overlap.png",
            ["ガタガタ"],
            _sfx_base(
                font_size=48,
                sfx_curve="zigzag",
                ink_texture="stripe",
                letter_spacing=-0.14,
                sfx_char_scale_max=2.0,
                sfx_persp=0.35,
                sfx_double_outline=False,
                stroke_size=3,
            ),
            25,
        ),
        (
            "27_sfx_huge_faint_burst.png",
            ["ズドン"],
            _sfx_base(
                font_size=185,
                sfx_curve="arc",
                faint=True,
                faint_alpha=0.6,
                sfx_burst=True,
                sfx_speedlines=True,
                sfx_persp=0.4,
            ),
            26,
        ),
        (
            "28_dialogue_distress_small.png",
            ["小さな声で言った", "もう帰ろう"],
            _base(
                font_size=28,
                line_height=0.25,
                stroke_size=1,
                stroke_color="white",
                distress=True,
                distress_strength=0.45,
                irregular=True,
                line_scale_min=0.55,
                line_scale_max=1.65,
                line_shift=1.6,
                char_jitter_prob=0.65,
                char_scale_min=0.6,
                char_scale_max=1.65,
                wobble=0.10,
                rotate=6.0,
                small_kana=True,
                slant=18.0,
            ),
            27,
        ),
        # ---- real handwritten kana (opt-in, needs a local bank) ------------
        (
            "29_dialogue_real_kana.png",
            ["お前との婚約は破棄する"],
            _base(real_kana=True, real_kana_prob=1.0, real_kana_seed=28),
            28,
        ),
        (
            "30_dialogue_real_kana_mixed.png",
            ["これは本当に手書きなのか"],
            _base(vertical=False, real_kana=True, real_kana_prob=0.5, real_kana_seed=29),
            29,
        ),
        (
            "31_irregular_real_kana.png",
            ["もう逃げられないよ"],
            _base(real_kana=True, real_kana_prob=1.0, real_kana_seed=30, **_IRR_MILD),
            30,
        ),
        # SFX keeps its fonts unless ``real_kana_sfx`` is set; warped SFX
        # (persp/aspect) still falls back because the bank cannot be matched
        # to a perspective transform, so this one stays flat.
        (
            "32_sfx_real_kana.png",
            ["ドキドキ"],
            _sfx_base(
                font_size=72,
                vertical=True,
                sfx_curve="straight",
                sfx_curve_amp=0.18,
                sfx_persp=0.0,
                sfx_aspect=0.0,
                sfx_rotate=6.0,
                sfx_wobble=0.03,
                sfx_speedlines=False,
                sfx_burst=False,
                sfx_double_outline=True,
                stroke_size=4,
                real_kana=True,
                real_kana_sfx=True,
                real_kana_prob=1.0,
                real_kana_seed=31,
            ),
            31,
        ),
    ]


def _pick_font(text, *, sfx=False):
    """Sample a covering font from the installed library (never a fixed Noto face)."""

    gen = _generator()
    chars = [c for c in text if not c.isspace()]
    for _ in range(FONT_ATTEMPTS):
        if sfx:
            font = str(gen.get_random_sfx_font("ja"))
        else:
            font = str(gen.get_random_font(text, "ja"))
        if all(is_drawable(font, c) for c in chars):
            return font
    raise RuntimeError(
        f"no installed font covers {''.join(sorted(set(chars)))}; "
        "check FONTS_ROOT / assets/fonts.csv"
    )


def _to_image(img):
    """Convert the renderer's BGRA/BGR/gray array to a PIL image."""

    if img.ndim == 2:
        return Image.fromarray(np.ascontiguousarray(img), "L")
    return Image.fromarray(np.ascontiguousarray(img[..., [2, 1, 0]]), "RGB")


def _is_clipped(layer):
    """True when ink touches the text-layer border (the layout overflowed)."""

    alpha = layer[..., 3]
    return bool(alpha[0].any() or alpha[-1].any() or alpha[:, 0].any() or alpha[:, -1].any())


def _font_for(seed, text, *, sfx, attempt=0):
    """Deterministically sample a font, independent of the layout RNG."""

    np.random.seed(seed + attempt)
    return _pick_font(text, sfx=sfx)


def _save(name, lines, params, seed):
    gen = _generator()
    text = _visible_text(lines)
    sfx = bool(params.get("sfx"))

    # A wide/dense face can overflow the layout canvas; re-sample until the
    # text layer has padding on every side. Each attempt uses a fixed seed so
    # the chosen font and the layout stay reproducible. SFX is exempt: its
    # speedlines intentionally reach the canvas edge.
    font = None
    for attempt in range(FONT_ATTEMPTS):
        font = _font_for(seed, text, sfx=sfx, attempt=attempt)
        if sfx:
            break
        np.random.seed(seed)
        trial, _ = gen.renderer.render_text(lines, {**params, "font_path": font})
        if not _is_clipped(trial):
            break
    else:
        raise RuntimeError(f"{name}: every sampled font overflowed the canvas")

    np.random.seed(seed)
    params = {**params, "font_path": font}
    _check_glyphs(name, lines, font)
    if _HAS_BACKGROUNDS:
        img, used = gen.renderer.render(lines, params)
    else:
        img, used = gen.renderer.render_text(lines, params)
    if params.get("real_kana"):
        stats = used.get("real_kana_stats") or {}
        if not stats.get("attached"):
            raise RuntimeError(
                f"{name}: real kana enabled but no glyph was attached "
                "(bank missing or glyphs rejected); see docs/real-kana.md"
            )
    _to_image(img).save(OUT_DIR / name)
    mode = "bg" if _HAS_BACKGROUNDS else "text layer"
    print(f"{name}: {img.shape[1]}x{img.shape[0]}  {mode}  {Path(font).name}")


def main(only=None):
    """Render all showcase images. ``--only=sfx`` renders just files whose name contains 'sfx'."""

    bank_ready = None
    for name, lines, params, seed in _samples():
        if only and str(only) not in name:
            continue
        if params.get("real_kana"):
            if bank_ready is None:
                bank_ready = _get_real_kana_bank() is not None
            if not bank_ready:
                print(f"{name}: skipped (no real-kana bank installed; see docs/real-kana.md)")
                continue
        _save(name, lines, params, seed)


if __name__ == "__main__":
    fire.Fire(main)
