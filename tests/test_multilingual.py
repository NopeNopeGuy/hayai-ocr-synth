"""Tests for multilingual (ko/zh/en) synthetic text generation."""

import numpy as np
import pytest

from manga_ocr_dev.synthetic_data_generator.lang import (
    DIALOGUE_VERTICAL_PROB,
    SFX_VERTICAL_PROB,
    SUPPORTED_LANGS,
    detect_lang,
    normalize_lang,
)
from manga_ocr_dev.synthetic_data_generator.sfx_text import (
    ChineseSFXComposer,
    EnglishSFXComposer,
    KoreanSFXComposer,
)
from manga_ocr_dev.synthetic_data_generator.lang import (
    SFX_BASES as LANG_SFX_BASES,
)
from manga_ocr_dev.synthetic_data_generator.lang import (
    SFX_ELONG_TAILS as LANG_SFX_ELONG,
)
from manga_ocr_dev.synthetic_data_generator.lang import (
    SFX_MARK_TAILS as LANG_SFX_MARK,
)

FONT = "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"
HAS_FONT = __import__("os").path.exists(FONT)


def test_supported_langs():
    assert set(SUPPORTED_LANGS) == {"ja", "ko", "zh", "en"}


def test_normalize_lang_aliases():
    assert normalize_lang(None) == "ja"
    assert normalize_lang("KR") == "ko"
    assert normalize_lang("korean") == "ko"
    assert normalize_lang("cn") == "zh"
    assert normalize_lang("chinese") == "zh"
    assert normalize_lang("ENGLISH") == "en"
    assert normalize_lang("jp") == "ja"
    with pytest.raises(ValueError):
        normalize_lang("fr")


def test_detect_lang_scripts():
    assert detect_lang("안녕하세요 감사합니다") == "ko"
    assert detect_lang("你好世界我爱你") == "zh"
    assert detect_lang("Hello world, how are you?") == "en"
    assert detect_lang("こんにちは漢字テスト") == "ja"
    # Kana presence decides Japanese even with heavy kanji mixing.
    assert detect_lang("漢字テストです") == "ja"
    # Han without kana is Chinese.
    assert detect_lang("漢字测试") == "zh"


def test_vertical_probs_ordering():
    # Japanese loves vertical; English never; ko/zh in between.
    assert DIALOGUE_VERTICAL_PROB["ja"] > DIALOGUE_VERTICAL_PROB["zh"] > DIALOGUE_VERTICAL_PROB["ko"]
    assert DIALOGUE_VERTICAL_PROB["en"] == 0.0
    assert SFX_VERTICAL_PROB["en"] == 0.0
    assert SFX_VERTICAL_PROB["ja"] > SFX_VERTICAL_PROB["ko"]


def _vocab_of(*chars_lists):
    import string

    base = set(string.ascii_letters + string.digits + " !?,....!?")
    for chars in chars_lists:
        base.update(chars)
    return base


def test_korean_sfx_composer_vocab_only():
    vocab = _vocab_of("가나다는라마바사아자차카타파하거너더러머버서어저처커터퍼허고노도로모보소오조초코토포호" "ㅋㅎ~!.?")
    comp = KoreanSFXComposer(LANG_SFX_BASES["ko"], LANG_SFX_ELONG["ko"], LANG_SFX_MARK["ko"])
    np.random.seed(0)
    texts = {comp.sample(vocab) for _ in range(50)}
    texts.discard(None)
    assert len(texts) > 10
    for t in texts:
        assert all(c in vocab or c.isspace() for c in t)


def test_chinese_sfx_composer_vocab_only():
    vocab = _vocab_of("哈哈呵嘿哼啊呀哇呜呼轰砰啪嘭咚嗖飕隆哗啦劈啪滴答嘀嗒叮咚咔嚓哐当乒乓噼啪扑哧嘿咻冲杀嗷汪喵咩哞喔咯嘎呱吱嗡!！？?")
    comp = ChineseSFXComposer(LANG_SFX_BASES["zh"], LANG_SFX_ELONG["zh"], LANG_SFX_MARK["zh"])
    np.random.seed(0)
    texts = {comp.sample(vocab) for _ in range(50)}
    texts.discard(None)
    assert len(texts) > 10
    for t in texts:
        assert all(c in vocab or c.isspace() for c in t)


def test_english_sfx_composer_vocab_only():
    import string

    vocab = set(string.ascii_letters + string.digits + " !?.,")
    comp = EnglishSFXComposer(LANG_SFX_BASES["en"], LANG_SFX_ELONG["en"], LANG_SFX_MARK["en"])
    np.random.seed(0)
    texts = {comp.sample(vocab) for _ in range(50)}
    texts.discard(None)
    assert len(texts) > 10
    for t in texts:
        assert all(c in vocab or c.isspace() for c in t)


def test_detect_line_lang_wrapper():
    from manga_ocr_dev.data.process_text import detect_line_lang, split_novel_chapter

    assert detect_line_lang("안녕") == "ko"
    assert detect_line_lang("你好") == "zh"
    assert detect_line_lang("hi there") == "en"
    # English chapter splitting handles ASCII quotes and .!? boundaries.
    lines = split_novel_chapter('"Hello!" she said. "How are you?"\nI am fine.', lang="en")
    assert any("Hello" in ln for ln in lines)
    assert any("How are you" in ln for ln in lines)


def test_export_lines_lang_filter(tmp_path):
    from manga_ocr_dev.data.process_text import export_lines

    f = tmp_path / "mixed.txt"
    f.write_text("こんにちは世界\n안녕하세요 세계\n你好世界\nHello world\n", encoding="utf-8")
    out = tmp_path / "lines_ko"
    created = export_lines(
        num_lines_in_each_package=2,
        num_packages=1,
        start_package=0,
        source_type="dialogue",
        dialogue_data_dir=str(f),
        out_dir=out,
        streaming=False,
        min_len=2,
        dedup_window=10,
        lang="ko",
    )
    assert len(created) == 1
    import pandas as pd

    df = pd.read_csv(created[0])
    assert len(df) == 1  # only the Korean line survives the filter
    assert "안녕하세요" in str(df["line"].iloc[0])


@pytest.mark.skipif(not HAS_FONT, reason="test font not available")
def test_renderer_renders_each_language():
    from manga_ocr_dev.synthetic_data_generator.renderer import Renderer

    r = Renderer.__new__(Renderer)
    r.max_size = 600
    cases = {
        "ko": ["안녕하세요", "감사합니다"],
        "zh": ["你好世界", "谢谢大家"],
        "en": ["Hello world", "How are you?"],
        "ja": ["こんにちは", "漢字テスト"],
    }
    for lang, lines in cases.items():
        img, params = r.render_text(
            lines,
            {
                "font_path": FONT,
                "vertical": False,
                "font_size": 40,
                "line_height": 0.5,
                "irregular": False,
                "text_color": "black",
                "color_mode": "solid",
                "stroke_size": 0,
                "shadow_size": 0,
                "letter_spacing": 0.0,
                "slant": 0.0,
                "text_orientation": "upright",
                "lang": lang,
            },
        )
        assert (img[:, :, 3] > 0).sum() > 100, lang
        assert params["lang"] == lang


@pytest.mark.skipif(not HAS_FONT, reason="test font not available")
def test_renderer_vertical_distribution_per_lang():
    from manga_ocr_dev.synthetic_data_generator.renderer import Renderer

    np.random.seed(0)
    ja = sum(Renderer.get_random_css_params(lang="ja")["vertical"] for _ in range(300))
    en = sum(Renderer.get_random_css_params(lang="en")["vertical"] for _ in range(300))
    assert ja > 150
    assert en == 0


@pytest.mark.skipif(not HAS_FONT, reason="test font not available")
def test_sfx_renders_each_language():
    from manga_ocr_dev.synthetic_data_generator.renderer import Renderer

    r = Renderer.__new__(Renderer)
    r.max_size = 600
    cases = {
        "ko": ["두근두근"],
        "zh": ["哈哈"],
        "en": ["BOOM"],
        "ja": ["ドン"],
    }
    for lang, lines in cases.items():
        np.random.seed(0)
        img, params = r.render_text(
            lines,
            {
                "font_path": FONT,
                "sfx": True,
                "vertical": False,
                "font_size": 80,
                "line_height": 0.3,
                "text_color": "black",
                "color_mode": "solid",
                "stroke_size": 4,
                "stroke_color": "white",
                "letter_spacing": 0.0,
                "text_orientation": "upright",
                "sfx_curve": "straight",
                "sfx_curve_amp": 0.0,
                "sfx_curve_sign": 1.0,
                "sfx_persp": 0.0,
                "sfx_rotate": 0.0,
                "sfx_aspect": 0.0,
                "sfx_char_scale_min": 1.0,
                "sfx_char_scale_max": 1.0,
                "sfx_wobble": 0.0,
                "sfx_speedlines": False,
                "sfx_burst": False,
                "sfx_double_outline": False,
                "sfx_block_rotate": 0.0,
                "slant": 0.0,
                "small_kana": False,
                "lang": lang,
            },
        )
        assert (img[:, :, 3] > 0).sum() > 100, lang


@pytest.fixture(scope="module")
def gen():
    from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator

    try:
        return SyntheticDataGenerator()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"generator env not available: {e}")


@pytest.mark.parametrize(
    "lang,text,must_contain",
    [
        ("ko", "안녕하세요 감사합니다", "안녕"),
        ("zh", "你好世界哈哈", "你好"),
        ("en", "Hello world, how are you?", "Hello"),
        ("ja", "こんにちは漢字テスト", "こんにちは"),
    ],
)
def test_generator_corpus_text_per_lang(gen, lang, text, must_contain):
    np.random.seed(0)
    _img, gt, params = gen.process(text, lang=lang)
    assert params["lang"] == lang
    assert must_contain in gt.replace("\n", "")
    # GT only contains drawable glyphs of the chosen font.
    from manga_ocr_dev.synthetic_data_generator.utils import is_drawable

    assert all(is_drawable(params["font_path"], c) for c in gt if not c.isspace())


@pytest.mark.parametrize("lang", ["ko", "zh", "en"])
def test_generator_random_text_script_dominant(gen, lang):
    from manga_ocr_dev.synthetic_data_generator.lang import detect_lang

    gts = []
    # Note: 'ja' is excluded: random ja fonts include Latin-only faces (a
    # pre-existing fallback), whose ASCII output detects as English.
    for i in range(6):
        np.random.seed(100 + i)
        _img, gt, params = gen.process(None, lang=lang)
        assert params["lang"] == lang
        assert gt.strip()
        gts.append(gt)
    # Most random samples are detected as the requested language. Mixed
    # punctuation/digits are allowed; the dominant script must win.
    detected = [detect_lang(g) for g in gts]
    assert sum(d == lang for d in detected) >= 4, detected


@pytest.mark.parametrize("lang", ["ko", "zh", "en", "ja"])
def test_generator_sfx_per_lang(gen, lang):
    texts = set()
    for i in range(6):
        np.random.seed(i)
        _img, gt, params = gen.process(None, sfx=True, lang=lang)
        assert params["lang"] == lang
        assert gt.strip()
        texts.add(gt)
    assert len(texts) >= 4  # procedural variety, not one fixed word


def test_generator_auto_detects_corpus_script(gen):
    np.random.seed(0)
    _img, _gt, params = gen.process("안녕하세요")
    assert params["lang"] == "ko"
    np.random.seed(0)
    _img, _gt, params = gen.process("Hello world")
    assert params["lang"] == "en"
