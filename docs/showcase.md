# Showcase (`synth_samples/generate.py`, 539 lines)

Reproducible showcase images rendered through the production pipeline: fonts are sampled from the local manga-font library and the text is composited over real manga backgrounds. Run from repo root: `python3 synth_samples/generate.py [--only=<substring>]` (`--only=sfx` renders SFX files only). `fire.Fire(main)`.

## How it works

- `_generator()` (`generate.py:68-79`) — lazily builds the production `SyntheticDataGenerator`; used for font sampling and rendering. If no background dataset is installed it passes an empty `background_df` and the samples become transparent text layers instead.
- `_backgrounds_available()` (`generate.py:61-65`) — `get_default_background_df()` (AnimeText preferred, Manga109-s fallback); missing datasets are not fatal.
- `_visible_text(lines)` (`generate.py:81-93`) — markup-stripped visible text, used for font coverage checks.
- `_check_glyphs(name, lines, font_path)` (`generate.py:95-100`) — raster `is_drawable` gate; never ships tofu.
- `_base(vertical=True, **kwargs)` (`generate.py:102-120`) — pinned dialogue defaults (`font_size=48, line_height=0.5, transparent, irregular=False, solid black, upright, stroke/shadow 0, spacing 0, slant 0`).
- `_sfx_base(**kwargs)` (`generate.py:122-163`) — pinned SFX defaults (`sfx=True, font_size=120, arc/0.3/+1, persp 0.2, rotate 14, aspect 0, char_scale 0.8–1.9, wobble 0.05, double_outline True, block_rotate 0, small_kana True, solid/gradient/stripe, extrude/distress/faint off`).
- `_IRR_MILD/_IRR_STRONG` (`generate.py:165-187`) — mild (line 0.8–1.25, shift 0.8, wobble 0.05) vs strong (0.6–1.6, shift 1.5, jitter 0.6, 0.65–1.6, wobble 0.08, rotate 5, small_kana).
- `_samples() -> [(filename, lines, params, seed)]` (`generate.py:189-440`) — 32 entries, numbering 01–14 stable; every style knob pinned so filenames faithfully represent effects. No sample pins `font_path`.
- `_pick_font(text, sfx=...)` (`generate.py:442-458`) — samples a covering font from the installed library: `get_random_font(text, "ja")` (weighted manga dialogue profiles) or `get_random_sfx_font("ja")` (SFX profile weights). Retries until every visible character is drawable.
- `_font_for(seed, text, ...)` (`generate.py:475-480`) — font sampling on `seed + attempt`, independent of the layout RNG.
- `_save(name, lines, params, seed)` (`generate.py:482-520`) — picks a font, rejects fonts whose text layer overflows the canvas (`_is_clipped`, dialogue only), then renders with `Renderer.render` (`background → bubble → degradation → scan model`) and saves a grayscale/RGB PNG; prints `WxH`, mode and font name. For `real_kana` samples it raises unless at least one bank glyph was attached.
- `main(only=None)` (`generate.py:522-535`) — loads the real-kana bank lazily (only when a `real_kana` sample is selected) and skips those files with a message when no bank is installed.

Re-running reproduces the same layouts for a fixed font library and background dataset; different `FONTS_ROOT`/background data will pick different fonts/panels.

## Real-kana samples

`29`–`32` exercise the opt-in handwriting bank (`docs/real-kana.md`). They need a local bank
(`$REAL_KANA_ROOT`, `assets/handwriting`, `/mnt/real-kana-final`, or `/mnt/real-kana-bank`); without
one `main()` prints `skipped` for them and renders the other 28 files normally. The bank attachment
has its own `real_kana_seed`, so a given bank reproduces the same kana placement.

- `29` replaces 100% of eligible kana; kanji stay font-rendered.
- `30` uses `real_kana_prob=0.5`, so real and font kana mix inside one line.
- `31` combines the bank with the mild hand-lettered (`irregular`) layout.
- `32` opts SFX in with `real_kana_sfx=True`; it stays flat because warped/gradient SFX that carry
  `persp`/`aspect` cannot be matched to a flat bank raster and keep their decorative font.

## Catalog (32 files, seeded 0–31)

| Files | Content |
|---|---|
| `01_vertical_stroke.png` | vertical + white stroke |
| `02_vertical_ruby.png` | `<ruby>昨日<rt>きのう</rt></ruby>` |
| `03_horizontal_shadow.png` | horizontal + shadow |
| `04_vertical_mixed_tcy.png` | `text-combine-upright` digits + sideways Latin |
| `05_horizontal_plain.png` | plain horizontal |
| `06_irregular_mild.png`, `07_irregular_strong.png`, `08_irregular_horizontal.png`, `09_slanted_irregular.png` (slant 25°) | hand-lettered variations |
| `10_sfx_arc_speedlines.png`, `11_sfx_wave_burst.png`, `12_sfx_diagonal_persp.png`, `13_sfx_vertical_hearts.png`, `14_sfx_ecchi_burst.png` | original SFX five |
| `15_sfx_zigzag_parallel_lines.png` … `20_sfx_soft_hiragana_airy.png` | zigzag/bounce/hill/2-column/tight-spacing/airy |
| `21_vertical_per_line_colour.png`, `22_sfx_per_char_rainbow.png` | color modes |
| `23_sfx_hollow_steep.png`, `24_sfx_circle_extrude.png`, `25_sfx_gradient_distress.png`, `26_sfx_stripe_tiny_overlap.png`, `27_sfx_huge_faint_burst.png`, `28_dialogue_distress_small.png` | hard inks |
| `29_dialogue_real_kana.png`, `30_dialogue_real_kana_mixed.png`, `31_irregular_real_kana.png`, `32_sfx_real_kana.png` | real handwritten kana (`real_kana`, `real_kana_prob`, `real_kana_sfx`) |
