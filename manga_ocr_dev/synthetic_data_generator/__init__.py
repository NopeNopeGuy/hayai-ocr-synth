"""Synthetic manga OCR data generator and rendering engine (ja/ko/zh/en)."""

from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
from manga_ocr_dev.synthetic_data_generator.lang import SUPPORTED_LANGS, detect_lang, normalize_lang
from manga_ocr_dev.synthetic_data_generator.renderer import Renderer
from manga_ocr_dev.synthetic_data_generator.sfx_text import (
    ChineseSFXComposer,
    EnglishSFXComposer,
    KoreanSFXComposer,
    SFXComposer,
)

__all__ = [
    "ChineseSFXComposer",
    "EnglishSFXComposer",
    "KoreanSFXComposer",
    "Renderer",
    "SFXComposer",
    "SUPPORTED_LANGS",
    "SyntheticDataGenerator",
    "detect_lang",
    "normalize_lang",
]
