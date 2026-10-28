# Multilingual support (`manga_ocr_dev/synthetic_data_generator/lang.py`, 233 lines)

Typographic config for `ja/ko/zh/en`. Japanese is the original pipeline (vertical + ruby + tcy + kana mora SFX); others adapt layout and sampling.

Module docstring: `lang.py:1-13`.

## API

- `SUPPORTED_LANGS = ("ja","ko","zh","en")`
- `normalize_lang(lang: str | None) -> str` (`lang.py`) — `None→ja`; aliases `jp/jpn/japanese→ja`, `kr/kor/korean→ko`, `cn/chs/cht/chinese→zh`, `eng/english→en`; raises `ValueError` otherwise.
- `detect_lang(text: str) -> str` — counts kana vs hangul vs han vs latin; kana→`ja`, hangul-dominant→`ko`, han-only→`zh`, latin-dominant→`en`.
- `DIALOGUE_VERTICAL_PROB = {ja:0.70, zh:0.12, ko:0.02, en:0.0}` — Japanese manga mixes vertical/horizontal; Chinese mostly horizontal; Korean/English (near-)always horizontal.
- `SFX_VERTICAL_PROB = {ja:0.40, zh:0.15, ko:0.05, en:0.0}`
- `TTB_LANGUAGE = {ja:ja, zh:zh, ko:ko, en:en}` — OpenType `language` tag for vertical shaping in Pillow.
- `DIALOGUE_PUNCT{lang: (chars, weights)}` — only font-drawable entries used (see `generator_legacy._punct_for_lang`).
- `SFX_BASES{lang: [...]}` — SFX seed lexicons; `SFX_ELONG_TAILS`, `SFX_MARK_TAILS`.
- `EN_WORDS([...])` — English word list; `LANG_CORE_CHARS` — font-test sets; `COMMON_HAN_FREQ` — frequent Han pool (simplified + traditional, 80/20 note).
- `zh` covers simplified + traditional: the generator samples Han from the font's own coverage, so variant follows installed fonts.

Consumed by `renderer.get_random_*_params` (vertical draws), `generator_legacy` (word pools, wrapping, font choice), `process_text.detect_line_lang` wrapper, `run_generate --lang`.

`detect_line_lang(text)` in `process_text.py` is a thin wrapper around `lang.detect_lang`.
