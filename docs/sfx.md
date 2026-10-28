# SFX text (`manga_ocr_dev/synthetic_data_generator/sfx_text.py`, 676 lines)

Procedural unbounded onomatopoeia generator — avoids fixed-lexicon repetition. Every output is vocab/`is_drawable`-gated and tail-decorated, `max_len`-enforced, returning `None` on failure.

## Classes

- `SFXComposer(bases, elong_tails, mark_tails, lexicon_p=0.3, max_len=10)` — Japanese mora-grammar sampler.
- `_Pool(hira, vocab)` — per-script/per-vocab mora pool (`plain/yoon/foreign`, `tsu/n/long/ri/smalls`); `usable()`, `m()`, `small_after(mora)`.
- `_BaseSFXComposer(...)` — shared `_init_patterns`, `_random_pattern`, `_lexicon_cands/_fallback`, `_add_tails(s, ok, elong_p=0.25, mark_p=0.35)`, `_mutate_generic(s, pool, repeat_p, swap_p, compound_p)`.
- `KoreanSFXComposer(lexicon_p=0.45, max_len=10)`, `ChineseSFXComposer(lexicon_p=0.5, max_len=10)`, `EnglishSFXComposer(lexicon_p=0.55, max_len=16)` — `sample(vocab, font_path=None, max_len=None) -> str | None`.

## Japanese sampling (`SFXComposer.sample`)

12% hiragana else katakana; `lexicon_p` mutate vs 10 grammar patterns:

`_burst/_impact/_crash/_repeat/_alt_repeat/_geminate/_tail_ri/_stretch/_breath/_stutter` + compound 12%, then `_add_tails`. Validates `1<=len<=max_len` and `all(c in vocab)`.

- `_mutate(s, pool)` — 1–3 edits: voicing flip, stretch, repeat, ending swap, geminate, compound, truncate.
- Mora tables `_ROWS/_ONSET_W/_VOWEL_W` with yoon/foreign/small-vowel handling; `_long_or`, `_get_pool`.
- Helpers: `_to_hira/_to_kata`, `_vowel_of`, `_voice/_devoice`, `_rand`, `_ok_checker(vocab, font_path)` (vocab or `is_drawable` fallback), `_filter_tails`.

Tails: `SFX_ELONG_TAILS` (`ー/ッ/〜`), `SFX_MARK_TAILS` (`♡♥★`, `~~~`, `……`, `!!!`) — see `lang.py`.

## Multilingual subclasses

Share tail/mutation infra with script-appropriate pools:

- Korean: frequent Hangul syllables; `_burst/_repeat/_stutter/_stretch/_double/_mutate`.
- Chinese: onomatopoeic Han subset `_SFX_HAN`; `_single/_repeat/_aabb/_abab/_mutate` (`_pool` prefers `SFX_HAN`-first).
- English: onset-nucleus-coda synthesis (`ONSETS/NUCLEI/CODAS` + stretch/repeat); `_generate(ok)`, `_mutate`, `_case` (75% UPPER), `_add_tails`.

Seed lexicons `SFX_BASES{lang}` live in `lang.py`.
