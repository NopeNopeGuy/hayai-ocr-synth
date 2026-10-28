"""Real handwritten kana bank for synthetic manga OCR data.

This package implements the *real kana* half of the synthetic handwriting
pipeline described in ``docs/real-kana.md``:

1. harvest candidate crops from AnimeText-OCR (handwriting decisions can come
   from a trained DINO head, Mercury Decide, or the local heuristic),
2. cut kana locally with ``manga-ocr-charspot-small`` (detection + recognition),
3. optionally verify doubtful glyphs with a vision model (Gemini Flash-Lite or
   an equivalent),
4. clean, upscale and upright the pixels,
5. build a small bank and use it from the existing Pillow renderer.

The bank only ever replaces kana.  Kanji, Latin, digits and punctuation keep
coming from the selected handwriting font.
"""

from manga_ocr_dev.real_kana.bank import KanaBank, KanaGlyph, StrokeBudget, attach_real_kana
from manga_ocr_dev.real_kana.config import KanaConfig

__all__ = [
    "KanaBank",
    "KanaConfig",
    "KanaGlyph",
    "StrokeBudget",
    "attach_real_kana",
]
