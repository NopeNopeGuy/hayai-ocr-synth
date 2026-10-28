"""Harder dialogue sampling on top of the legacy synthetic OCR generator.

Builds on ``generator_legacy.SyntheticDataGenerator`` (font selection, SFX,
furigana, line wrapping, rendering) and changes:

* Hard rendering conditions (distress, faint ink, extrusion, hollow outlines,
  irregular per-line sizing) are sampled for a share of dialogue, so clean
  crops stop being the default. Explicit overrides from the caller always win.
* Japanese lines get furigana more often (70% vs 50%).

Random-text sampling is unchanged from the legacy generator.
"""

from __future__ import annotations

import numpy as np

from manga_ocr_dev.synthetic_data_generator.generator_legacy import (
    SyntheticDataGenerator as _LegacyGenerator,
)
from manga_ocr_dev.synthetic_data_generator.lang import detect_lang, normalize_lang

FURIGANA_PROB_JA = 0.70  # legacy: 0.50


def sample_hard_ink() -> dict:
    """Independent hard rendering conditions for dialogue (empty for easy crops)."""
    o = {}
    if np.random.rand() < 0.30:
        o["distress"] = True
        o["distress_strength"] = float(np.random.uniform(0.3, 0.7))
    if np.random.rand() < 0.20:
        o["faint"] = True
        o["faint_alpha"] = float(np.random.uniform(0.4, 0.7))
    if np.random.rand() < 0.12:
        o["extrude"] = True
    if np.random.rand() < 0.10:
        o["hollow"] = True
    if np.random.rand() < 0.12:
        o["irregular"] = True
    return o


class SyntheticDataGenerator(_LegacyGenerator):
    def process(self, text=None, override_css_params=None, sfx=False, lang=None):
        override = dict(override_css_params or {})
        if bool(sfx or override.get("sfx", False)):
            return super().process(text, override, sfx=True, lang=lang)

        if lang is None:
            lang = override.get("lang")
        if lang is None and text is not None:
            lang = detect_lang(text)
        lang = normalize_lang(lang)

        for key, value in sample_hard_ink().items():
            override.setdefault(key, value)  # explicit caller settings win
        return self._process_dialogue(text, override, lang)

    def _process_dialogue(self, text, override, lang):
        # Same flow as the legacy dialogue path, with the furigana rate raised.
        user_font = "font_path" in override
        params_in, lines, vocab = dict(override), [], None
        for _ in range(5):
            params_in = dict(override)
            params_in["lang"] = lang
            lines, vocab = self._prepare_dialogue_lines(text, params_in, lang=lang)
            if lines or user_font:
                break
        if not lines:
            raise ValueError("no renderable text after filtering by font support")

        text_gt = "\n".join(lines)  # ground truth is taken before furigana is added
        if lang == "ja" and np.random.random() < FURIGANA_PROB_JA:
            word_prob = np.random.choice([0.33, 1.0], p=[0.3, 0.7])
            font_path = params_in["font_path"]
            lines = [self.add_random_furigana(line, word_prob, vocab, font_path=font_path) for line in lines]

        img, params = self.renderer.render(lines, params_in)
        params["lang"] = lang
        return img, text_gt, params
