"""Accuracy tests for the synthetic generator (labels must match the drawn glyphs).

Needs the same environment as the generator itself (fonts.csv, vocab.csv,
backgrounds). Run: ``pytest -q test_generator_accuracy.py``.
"""

import numpy as np
import pytest

from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
from manga_ocr_dev.synthetic_data_generator.renderer import Renderer, _parse_line


@pytest.fixture(scope="module")
def gen():
    return SyntheticDataGenerator()


def _layout(vertical, lines):
    font_path = next(iter(SyntheticDataGenerator().sfx_fonts))
    _, _, ops = Renderer._layout_sfx(
        [_parse_line(l) for l in lines],
        vertical=vertical,
        font_path=font_path,
        font_size=100,
        line_height=0.3,
        letter_spacing=0.0,
        upright=True,
        margin=20,
        pad=4,
        curve="straight",
        persp=0.0,
        rotate=0.0,
        wobble=0.0,
    )
    return ops


def test_sfx_vertical_first_line_is_rightmost():
    ops = _layout(True, ["ドン", "バン"])
    x0 = np.mean([o["cx"] for o in ops if o["line"] == 0])
    x1 = np.mean([o["cx"] for o in ops if o["line"] == 1])
    assert x0 > x1, "vertical-rl: first GT line must be the right-hand column"


def test_sfx_horizontal_first_line_is_top():
    ops = _layout(False, ["ドン", "バン"])
    assert np.mean([o["cy"] for o in ops if o["line"] == 0]) < np.mean([o["cy"] for o in ops if o["line"] == 1])


@pytest.mark.parametrize("sfx", [False, True])
def test_gt_only_contains_glyphs_of_the_font(gen, sfx):
    for i in range(60):
        np.random.seed(i)
        _, gt, params = gen.process(None, sfx=sfx)
        vocab = gen.font_map[params["font_path"]]
        assert gt.strip()
        assert all(c in vocab or c.isspace() for c in gt), (gt, params["font_path"])


def test_sfx_fonts_cover_kana(gen):
    need = set("ドンバガッー")
    for path in gen.sfx_fonts:
        assert need <= gen.font_map[path]


def test_sfx_is_diverse(gen):
    texts = set()
    for i in range(200):
        np.random.seed(i)
        texts.add(gen.process(None, sfx=True)[1])
    assert len(texts) > 150


# ------------------------------------------------------------------ tofu guard
def test_fonts_csv_has_no_tofu(gen):
    """fonts.csv must not list characters that a font draws as a box.

    Fails on a stale fonts.csv: re-run ``scan_fonts.py`` to fix it.
    """
    import random

    from manga_ocr_dev.synthetic_data_generator.utils import find_bad_glyphs

    paths = sorted(gen.font_map)
    random.Random(0).shuffle(paths)
    for path in paths[:25]:
        bad = find_bad_glyphs(path, gen.font_map[path])
        assert not bad, (path, "".join(sorted(bad))[:50])


@pytest.mark.parametrize("sfx", [False, True])
def test_generated_text_is_drawable(gen, sfx):
    from manga_ocr_dev.synthetic_data_generator.utils import is_drawable

    for i in range(60):
        np.random.seed(1000 + i)
        _, gt, params = gen.process(None, sfx=sfx)
        assert all(is_drawable(params["font_path"], c) for c in gt), (gt, params["font_path"])


def test_renderer_refuses_tofu(gen):
    from manga_ocr_dev.synthetic_data_generator.utils import MissingGlyphError, is_drawable

    path = next(iter(gen.font_map))
    missing = next(c for c in "\U0010fffd\ue0ff" if not is_drawable(path, c) or True)
    if is_drawable(path, missing):
        pytest.skip("font unexpectedly draws the probe character")
    with pytest.raises(MissingGlyphError):
        gen.renderer.render([missing], {"font_path": path})
