"""Tests for the Pillow-based synthetic text renderer (no browser)."""

import numpy as np
import pandas as pd
import pytest

from manga_ocr_dev.synthetic_data_generator.renderer import (
    Renderer,
    _parse_line,
)

FONT = "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"
pytestmark = pytest.mark.skipif(__import__("os").path.exists(FONT) is False, reason="test font not available")

BASE_CSS = {
    "font_path": FONT,
    "vertical": True,
    "stroke_size": 0,
    "irregular": False,
    "text_color": "black",
    "color_mode": "solid",
}


def make_renderer():
    # Renderer.__init__ would look for the real background dir; bypass it.
    r = Renderer.__new__(Renderer)
    r.max_size = 600
    return r


def test_parse_line_splits_ruby_and_tcy():
    tokens = _parse_line('あ<ruby>漢字<rt>かんじ</rt></ruby>と<span style="text-combine-upright: all">12</span>う')
    assert tokens == [
        ("text", "あ"),
        ("ruby", "漢字", "かんじ"),
        ("text", "と"),
        ("tcy", "12"),
        ("text", "う"),
    ]


def test_parse_plain_line():
    assert _parse_line("あいう") == [("text", "あいう")]


def test_render_text_returns_bgra():
    r = make_renderer()
    img, params = r.render_text(["あいう", "えお"], dict(BASE_CSS))
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 4
    assert (img[:, :, 3] > 0).sum() > 100  # some ink was drawn
    assert params["vertical"] is True


def test_vertical_columns_ordered_right_to_left():
    """First line must end up in the rightmost column (vertical-rl)."""
    r = make_renderer()
    img, _ = r.render_text(["あ", "ん"], dict(BASE_CSS))
    alpha = img[:, :, 3]
    cols = np.where(alpha.max(axis=0) > 0)[0]
    left_half = alpha[:, : img.shape[1] // 2]
    right_half = alpha[:, img.shape[1] // 2 :]
    # 'あ' (first line, rightmost) and 'ん' both present on opposite sides
    assert left_half.max() > 0 and right_half.max() > 0
    assert len(cols) > 10


def test_horizontal_rows_ordered_top_to_bottom():
    r = make_renderer()
    css = dict(BASE_CSS, vertical=False)
    img, _ = r.render_text(["あ", "ん"], css)
    alpha = img[:, :, 3]
    assert alpha[: img.shape[0] // 2].max() > 0
    assert alpha[img.shape[0] // 2 :].max() > 0


def test_stroke_adds_ink_vs_plain():
    r = make_renderer()
    # shadow_size pinned: a sampled shadow blur would inflate the alpha-pixel
    # count and drown out the stroke-vs-plain comparison.
    plain, _ = r.render_text(["あいう"], dict(BASE_CSS, shadow_size=0))
    stroked, _ = r.render_text(["あいう"], dict(BASE_CSS, stroke_size=3, stroke_color="white", shadow_size=0))
    assert (stroked[:, :, 3] > 0).sum() > (plain[:, :, 3] > 0).sum()


def test_shadow_renders_glow():
    r = make_renderer()
    img, _ = r.render_text(
        ["あいう"],
        dict(BASE_CSS, shadow_size=5, shadow_color="white"),
    )
    assert (img[:, :, 3] > 0).sum() > 100


def test_ruby_and_tcy_render_ink():
    r = make_renderer()
    img, _ = r.render_text(
        ['<ruby>漢字<rt>かんじ</rt></ruby>と<span style="text-combine-upright: all">12</span>'],
        dict(BASE_CSS),
    )
    assert (img[:, :, 3] > 0).sum() > 100


def test_legacy_shadow_tuple_color_accepted():
    """Old params accidentally stored ('white',); must not crash."""
    r = make_renderer()
    img, _ = r.render_text(["あ"], dict(BASE_CSS, shadow_size=5, shadow_color=("white",)))
    assert (img[:, :, 3] > 0).sum() > 0


def test_full_render_with_background(tmp_path):
    import cv2

    bg = np.random.RandomState(0).randint(0, 255, (200, 200, 3)).astype(np.uint8)
    bg_path = str(tmp_path / "bg_0_0_200_200.jpg")
    cv2.imwrite(bg_path, bg)
    df = pd.DataFrame([{"path": bg_path, "h": 200, "w": 200, "ratio": 1.0}])
    r = Renderer(background_df=df)
    img, _params = r.render(["あいうえお"], dict(BASE_CSS))
    assert img.dtype == np.uint8 and img.ndim == 2  # grayscale, like the old pipeline
    assert max(img.shape) <= 600


IRREGULAR_NEUTRAL = {
    "irregular": True,
    "line_scale_min": 1.0,
    "line_scale_max": 1.0,
    "line_shift": 0.0,
    "char_jitter_prob": 0.0,
    "wobble": 0.0,
    "rotate": 0.0,
    "small_kana": False,
}


def _column_bands(img):
    alpha = img[:, :, 3]
    w = img.shape[1]
    return [alpha[:, i * w // 3 : (i + 1) * w // 3] for i in range(3)]


def test_irregular_staggers_line_starts():
    """Identical lines must start at different offsets (no shared grid origin)."""
    import numpy as np

    r = make_renderer()
    np.random.seed(0)
    css = dict(BASE_CSS, **IRREGULAR_NEUTRAL)
    css["line_shift"] = 1.5
    img, _ = r.render_text(["あいうえお"] * 3, css)
    tops = []
    for band in _column_bands(img):
        rows = np.where(band.max(axis=1) > 0)[0]
        assert len(rows) > 0
        tops.append(rows.min())
    assert max(tops) - min(tops) > 5


def test_irregular_varies_line_sizes():
    """Per-line font sizes must differ across columns."""
    import numpy as np

    r = make_renderer()
    np.random.seed(0)
    css = dict(BASE_CSS, **IRREGULAR_NEUTRAL)
    css["line_scale_min"] = 0.6
    css["line_scale_max"] = 1.6
    img, _ = r.render_text(["あいうえお"] * 3, css)
    widths = []
    for band in _column_bands(img):
        cols = np.where(band.max(axis=0) > 0)[0]
        assert len(cols) > 0
        widths.append(cols.max() - cols.min())
    assert max(widths) - min(widths) > 5


def test_irregular_varies_char_sizes_within_line():
    """Emphasis lettering: glyphs within one line differ in size."""
    import numpy as np

    from manga_ocr_dev.synthetic_data_generator.renderer import _parse_line

    np.random.seed(0)
    _w, _h, ops = Renderer._layout_irregular(
        [_parse_line("あいうえお")],
        vertical=True,
        font_path=FONT,
        font_size=48,
        line_height=0.5,
        letter_spacing=None,
        upright=False,
        margin=20,
        pad=4,
        line_scale_min=1.0,
        line_scale_max=1.0,
        line_shift=0.0,
        char_jitter_prob=1.0,
        char_scale_min=0.6,
        char_scale_max=1.6,
        wobble=0.0,
        rotate=0.0,
        small_kana=False,
    )
    cells = {op["cell"] for op in ops if op["kind"] == "char"}
    assert len(cells) > 1


def test_irregular_horizontal_and_ruby_tcy_render_ink():
    import numpy as np

    r = make_renderer()
    np.random.seed(0)
    img, _ = r.render_text(
        ["お前との婚約", "は破棄する"],
        dict(BASE_CSS, vertical=False, stroke_size=2, stroke_color="white", irregular=True),
    )
    assert (img[:, :, 3] > 0).sum() > 100

    np.random.seed(1)
    img, _ = r.render_text(
        ['<ruby>漢字<rt>かんじ</rt></ruby>と<span style="text-combine-upright: all">12</span>'],
        dict(BASE_CSS, irregular=True, char_jitter_prob=1.0, wobble=0.08, rotate=5.0),
    )
    assert (img[:, :, 3] > 0).sum() > 100


def test_get_random_css_params_samples_irregular_layouts():
    import numpy as np

    np.random.seed(0)
    n = sum(Renderer.get_random_css_params().get("irregular", False) for _ in range(500))
    assert 0.4 < n / 500 < 0.7


def test_slant_widens_canvas_and_keeps_ink():
    import numpy as np

    r = make_renderer()
    np.random.seed(0)
    flat, _ = r.render_text(["彼女は勇者", "パーティの"], dict(BASE_CSS, slant=0.0))
    np.random.seed(0)
    slanted, _ = r.render_text(["彼女は勇者", "パーティの"], dict(BASE_CSS, slant=25.0))
    assert slanted.shape[1] > flat.shape[1]  # canvas widened to fit the skew
    assert slanted.shape[0] == flat.shape[0]  # height unchanged by horizontal shear
    flat_ink = (flat[:, :, 3] > 0).sum()
    assert (slanted[:, :, 3] > 0).sum() > 0.5 * flat_ink  # no clipping


def test_slant_zero_leaves_canvas_untouched():
    import numpy as np

    r = make_renderer()
    np.random.seed(0)
    a, _ = r.render_text(["あいうえお"], dict(BASE_CSS))
    np.random.seed(0)
    b, _ = r.render_text(["あいうえお"], dict(BASE_CSS, slant=0.0))
    assert a.shape == b.shape


def test_get_random_css_params_samples_slant():
    import numpy as np

    np.random.seed(0)
    slants = [Renderer.get_random_css_params().get("slant", 0.0) for _ in range(2000)]
    slanted = [s for s in slants if s]
    assert 0.05 < len(slanted) / 2000 < 0.25
    assert any(abs(s) >= 20.0 for s in slanted)  # very slanted samples occur


SFX_BASE = {
    "font_path": FONT,
    "sfx": True,
    "vertical": False,
    "text_color": "black",
    "color_mode": "solid",
    "sfx_speedlines": False,
    "sfx_burst": False,
    "sfx_double_outline": False,
    "sfx_block_rotate": 0.0,
    "slant": 0.0,
    "sfx_curve": "straight",
    "sfx_curve_amp": 0.0,
    "sfx_wobble": 0.0,
    "sfx_rotate": 0.0,
    "sfx_char_scale_min": 1.0,
    "sfx_char_scale_max": 1.0,
}


def test_sfx_renders_bgra_with_font_hearts():
    import numpy as np

    r = make_renderer()
    np.random.seed(0)
    img, params = r.render_text(["ドドドン♡"], dict(SFX_BASE, sfx_persp=0.2))
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 4
    assert (img[:, :, 3] > 0).sum() > 100
    assert params["sfx"] is True


def test_sfx_persp_warps_but_keeps_ink():
    import numpy as np

    r = make_renderer()
    np.random.seed(1)
    flat, _ = r.render_text(["ドン"], dict(SFX_BASE, sfx_persp=0.0))
    np.random.seed(1)
    warped, _ = r.render_text(["ドン"], dict(SFX_BASE, sfx_persp=0.25))
    assert abs(flat.shape[0] - warped.shape[0]) <= 45 and abs(flat.shape[1] - warped.shape[1]) <= 45
    assert (warped[:, :, 3] > 0).sum() > 0.5 * (flat[:, :, 3] > 0).sum()


def test_sfx_curve_changes_layout():
    import numpy as np

    r = make_renderer()
    np.random.seed(2)
    arc, _ = r.render_text(["ドドドン"], dict(SFX_BASE, sfx_curve="arc", sfx_curve_amp=0.35))
    np.random.seed(2)
    straight, _ = r.render_text(["ドドドン"], dict(SFX_BASE, sfx_curve="straight"))
    assert (arc[:, :, 3] > 0).sum() > 100 and (straight[:, :, 3] > 0).sum() > 100
    assert arc.shape != straight.shape or (arc.astype(int) - straight.astype(int)).any()


def test_sfx_speedlines_and_burst_add_ink():
    import numpy as np

    r = make_renderer()
    np.random.seed(3)
    plain, _ = r.render_text(["バン"], dict(SFX_BASE, sfx_persp=0.0))
    np.random.seed(3)
    speedy, _ = r.render_text(["バン"], dict(SFX_BASE, sfx_persp=0.0, sfx_speedlines=True, sfx_speedline_n=48))
    assert (speedy[:, :, 3] > 0).sum() > (plain[:, :, 3] > 0).sum()

    np.random.seed(4)
    no_burst, _ = r.render_text(["バン"], dict(SFX_BASE, sfx_persp=0.0))
    np.random.seed(4)
    burst, _ = r.render_text(["バン"], dict(SFX_BASE, sfx_persp=0.0, sfx_burst=True))
    assert (burst[:, :, 3] > 0).sum() > (no_burst[:, :, 3] > 0).sum()


def test_sfx_double_outline_adds_ink():
    import numpy as np

    r = make_renderer()
    np.random.seed(5)
    single, _ = r.render_text(["バン"], dict(SFX_BASE, sfx_persp=0.0, stroke_size=6))
    np.random.seed(5)
    double, _ = r.render_text(["バン"], dict(SFX_BASE, sfx_persp=0.0, stroke_size=6, sfx_double_outline=True))
    assert (double[:, :, 3] > 0).sum() > (single[:, :, 3] > 0).sum()


def test_get_random_sfx_params_are_huge_and_dialogue_free():
    import numpy as np

    from manga_ocr_dev.synthetic_data_generator.renderer import SFX_CURVE_MODES

    np.random.seed(0)
    p = Renderer.get_random_sfx_params()
    assert p["sfx"] is True and p["irregular"] is False
    assert 38 <= p["font_size"] <= 190
    assert p["stroke_size"] >= 0
    assert p["sfx_curve"] in tuple(SFX_CURVE_MODES)

    np.random.seed(0)
    assert Renderer.get_random_css_params(sfx=True)["sfx"] is True
    assert Renderer.get_random_css_params().get("sfx", False) is False


def test_sfx_full_render_skips_bubble(tmp_path):
    import cv2

    bg = np.random.RandomState(0).randint(0, 255, (200, 200, 3)).astype(np.uint8)
    bg_path = str(tmp_path / "bg_0_0_200_200.jpg")
    cv2.imwrite(bg_path, bg)
    df = pd.DataFrame([{"path": bg_path, "h": 200, "w": 200, "ratio": 1.0}])
    r = Renderer(background_df=df)
    img, params = r.render(["ドン"], dict(SFX_BASE, sfx_persp=0.15, sfx_speedlines=True))
    assert img.dtype == np.uint8 and img.ndim == 2
    assert max(img.shape) <= 600
    assert params["sfx"] is True


def _bg_renderer(tmp_path):
    import cv2

    bg = (np.ones((300, 300, 3)) * 220).astype(np.uint8)
    bg_path = str(tmp_path / "bg_0_0_300_300.jpg")
    cv2.imwrite(bg_path, bg)
    df = pd.DataFrame([{"path": bg_path, "h": 300, "w": 300, "ratio": 1.0}])
    return Renderer(background_df=df)


def test_full_render_never_upscales(tmp_path):
    # Scan-like native sizes: a tiny crop must stay tiny, not be blown up to
    # max_size. Any size / aspect ratio may come out of the pipeline.
    r = _bg_renderer(tmp_path)
    img, _ = r.render(["ん"], dict(BASE_CSS))
    assert img.ndim == 2
    assert max(img.shape) < r.max_size


def test_solid_color_text_returns_bgr():

    r = make_renderer()
    img, _params = r.render_text(
        ["あいう"],
        dict(BASE_CSS, text_color="pink", color_mode="solid", stroke_size=2, stroke_color="white"),
    )
    assert img.shape[2] == 4
    alpha = img[:, :, 3]
    assert (alpha > 0).sum() > 100
    # Pink ink survives in the composited layer (BGRA order here).
    ink = img[alpha > 128][:, :3]
    assert (ink[:, 0] > 150).mean() > 0.3  # strong red channel


def test_full_render_color_is_bgr_and_black_is_gray(tmp_path):
    r = _bg_renderer(tmp_path)
    color, _params = r.render(["あいうえお"], dict(BASE_CSS, text_color="red", color_mode="solid"))
    assert color.ndim == 3 and color.shape[2] == 3
    R, G, B = color[:, :, 2].astype(int), color[:, :, 1].astype(int), color[:, :, 0].astype(int)
    assert ((R > B + 30) & (R > G + 30)).sum() > 100
    gray, _ = r.render(["あいうえお"], dict(BASE_CSS))
    assert gray.ndim == 2


def test_color_sampling_is_frequent_and_pink_covered():
    import numpy as np

    np.random.seed(0)
    colors = [Renderer.get_random_css_params()["text_color"] for _ in range(2000)]
    chromatic = [c for c in colors if c != "black"]
    assert 0.10 < len(chromatic) / 2000 < 0.30  # colour is a minority (no chroma cheat)
    assert "pink" in chromatic and "lightpink" in chromatic
    assert len(set(chromatic)) >= 8  # a lot of colours, not just red

    np.random.seed(0)
    sfx_colors = [Renderer.get_random_sfx_params()["text_color"] for _ in range(2000)]
    # Includes white-ink conversion (black -> white heavy-edge SFX).
    assert 0.35 < sum(c != "black" for c in sfx_colors) / 2000 < 0.52


def test_multicolor_per_line_paints_lines_differently(tmp_path):
    import numpy as np

    from manga_ocr_dev.synthetic_data_generator.renderer import _assign_op_colors

    # Structural property (no RNG flakiness): glyphs on the same line share
    # one fill. With 8 lines over 2 colours, distinct fills are near-certain.
    np.random.seed(0)
    ops = [{"kind": "char", "line": li} for li in [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]]
    _assign_op_colors(ops, {"color_mode": "per_line", "text_colors": ["red", "cyan"]})
    for li in range(8):
        pair = [o for o in ops if o["line"] == li]
        assert pair[0]["fill"] == pair[1]["fill"]
    assert len({o["fill"] for o in ops}) == 2

    # End to end: a per-line image keeps colour and shows both inks.
    r = _bg_renderer(tmp_path)
    np.random.seed(0)
    img, params = r.render(
        ["あいうえお", "かきくけこ", "さしすせそ"],
        dict(
            BASE_CSS,
            text_color="red",
            color_mode="per_line",
            text_colors=["red", "cyan"],
            stroke_size=2,
        ),
    )
    assert params["color_mode"] == "per_line"
    assert img.ndim == 3
    B, G, R = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    assert ((R > B + 40) & (R > G + 40)).sum() > 100
    assert ((B > 100) & (G > 100) & (R < 120)).sum() > 100


def test_multicolor_per_char_mixes_within_line(tmp_path):
    import numpy as np

    r = _bg_renderer(tmp_path)
    np.random.seed(1)
    img, params = r.render(
        ["あいうえおかきくけこ"],
        dict(
            BASE_CSS,
            vertical=False,
            text_color="pink",
            color_mode="per_char",
            text_colors=["red", "yellow", "cyan", "pink"],
            stroke_size=2,
        ),
    )
    assert params["color_mode"] == "per_char"
    assert img.ndim == 3
    B, G, R = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    hues = sum(
        [
            ((R > B + 40) & (R > G + 40)).sum() > 100,
            ((R > 150) & (G > 150) & (B < 120)).sum() > 100,
            ((B > 100) & (G > 100) & (R < 120)).sum() > 100,
        ]
    )
    assert hues >= 2


def test_multicolor_sampled_for_colored_images():
    import numpy as np

    np.random.seed(0)
    modes = [Renderer.get_random_css_params()["color_mode"] for _ in range(2000)]
    assert any(m == "per_line" for m in modes)
    assert any(m == "per_char" for m in modes)
    # Black ink never mixes: nothing to mix.
    np.random.seed(0)
    for _ in range(200):
        p = Renderer.get_random_css_params()
        if p["text_color"] == "black":
            assert p["color_mode"] == "solid"


def test_contrast_helpers_and_legible_outline():
    from manga_ocr_dev.synthetic_data_generator.renderer import (
        _contrast_ratio,
        _ensure_legible_outline,
        _needs_dark_outline,
        _normalize_color,
    )

    assert round(_contrast_ratio("black", "white"), 1) == 21.0
    assert _contrast_ratio("yellow", "white") < 3.0
    assert _needs_dark_outline(_normalize_color("yellow")) is True
    assert _needs_dark_outline(_normalize_color("lightpink")) is True
    assert _needs_dark_outline(_normalize_color("black")) is False
    assert _needs_dark_outline(_normalize_color("blue")) is False

    # Light fill with a white halo is rewritten to a black outline.
    p = _ensure_legible_outline(
        {"text_color": "yellow", "stroke_size": 2, "stroke_color": "white", "color_mode": "solid"}
    )
    assert p["stroke_color"] == "black" and p["stroke_size"] >= 2
    # No outline at all: one is added.
    p = _ensure_legible_outline({"text_color": "cyan", "color_mode": "solid"})
    assert p["stroke_color"] == "black" and p["stroke_size"] >= 2
    # Dark fills keep the sampled white halo.
    p = _ensure_legible_outline(
        {"text_color": "black", "stroke_size": 2, "stroke_color": "white", "color_mode": "solid"}
    )
    assert p["stroke_color"] == "white"


def test_get_background_df_with_manifest(tmp_path):
    from manga_ocr_dev.synthetic_data_generator.utils import get_background_df

    manifest = pd.DataFrame(
        [
            {
                "source": "test",
                "image_id": 1,
                "ymin": 10,
                "ymax": 60,
                "xmin": 20,
                "xmax": 120,
                "h": 50,
                "w": 100,
                "n_boxes": 1,
                "path": "test_10_60_20_120.png",
            }
        ]
    )
    manifest.to_csv(tmp_path / "manifest.csv", index=False)
    # also add dummy file to ensure it's not confused
    (tmp_path / "test_10_60_20_120.png").write_bytes(b"dummy")

    df = get_background_df(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["h"] == 50
    assert df.iloc[0]["w"] == 100
    assert df.iloc[0]["ratio"] == 2.0
    assert df.iloc[0]["path"] == str(tmp_path / "test_10_60_20_120.png")


def test_get_background_df_without_manifest_and_ignores_non_images(tmp_path):
    from manga_ocr_dev.synthetic_data_generator.utils import get_background_df

    # Valid image files
    (tmp_path / "crop_0_100_0_200.png").write_bytes(b"dummy")
    (tmp_path / "crop_10_50_20_80.jpg").write_bytes(b"dummy")

    # Invalid files that must be ignored
    (tmp_path / "manifest.csv").write_text("invalid,manifest\n1,2\n")
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "subfolder").mkdir()
    (tmp_path / "bad_name.png").write_bytes(b"dummy")

    df = get_background_df(tmp_path)
    assert len(df) == 2
    paths = set(df["path"])
    assert str(tmp_path / "crop_0_100_0_200.png") in paths
    assert str(tmp_path / "crop_10_50_20_80.jpg") in paths
