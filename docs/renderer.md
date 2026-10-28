# Renderer (`manga_ocr_dev/synthetic_data_generator/renderer.py`, 3006 lines)

Pure-Pillow replacement for the legacy headless-Chrome (`html2image`) screenshotting pipeline. Renders BGRA/BGR `uint8` numpy arrays (BGR channel order, like `cv2.imread(..., IMREAD_UNCHANGED)`). Module docstring `renderer.py:1-68` is the design rationale.

## Public API

- `get_default_background_df()` (`renderer.py:89-97`) — AnimeText dir preferred, Manga109-s fallback.
- `class Renderer(background_df=None)` — `max_size=600`; defaults to `get_default_background_df()`. Bounded LRU font cache (48 faces) via `_get_font()`.
- `Renderer.render(lines, override_css_params=None) -> (img, params)` — full pipeline: `render_text` → `render_background` (bubble or art) → truncation → foreground occluders → `LongestMaxSize(600)` → `_scan_model_bgr` → `_degrade_bgr`. Returns 3-ch BGR if `_params_are_color(params)` else grayscale.
- `Renderer.render_text(lines, override_css_params=None) -> (img_bgra, params)` — transparent BGRA text layer. Dispatches `sfx → _layout_sfx`, `irregular → _layout_irregular`, else `_layout_uniform`. Applies shadow/glow, hollow, extrude, double-outline, gradient/stripe, distress, faint, slant/block-rotate, SFX tight crop.
- `Renderer.get_random_css_params(sfx=False, lang="ja") -> dict` — dialogue sampling: `font_size 26–56`, `vertical` per `DIALOGUE_VERTICAL_PROB`, `line_height 0.22–0.62`, `text_color`/`stroke`/`shadow`/`none`, `irregular` (`IRREGULAR_PROB=0.55`), `slant`, `hollow/distress/faint/extrude`, multicolor.
- `Renderer.get_random_sfx_params(lang="ja") -> dict` — SFX sampling: `font_size 38–190`, `vertical` per `SFX_VERTICAL_PROB`, `stroke_size` capped by `SFX_STROKE_FRAC=0.05`, `sfx_curve/curve_amp/curve_sign/persp/rotate/char_scale/wobble/aspect/speedlines/burst/double_outline/block_rotate`, hard inks.
- `Renderer.render_background(img, draw_bubble=None, tight=False, return_ink_box=False)` — `draw_bubble` defaults `p≈0.7` (forced `False` if `tight`); bubble fill via `_sample_bubble_fill` + `rounded_rectangle` + `ElasticTransform` blend; non-bubble via `_make_background`.
- Helpers: `crop_by_alpha(img, margin)`, `blend(img, background)`, `rounded_rectangle(...)`.

## Layout modes

1. **Uniform** (`_layout_uniform`) — every line same size on a regular grid. Supports ruby `<ruby>漢字<rt>かんじ</rt></ruby>` (`RUBY_SCALE=0.45`, `RUBY_GAP=6`) and tate-chū-yoko `<span style="text-combine-upright: all">12</span>`, parsed by `_parse_line` → `[("text",str)|("ruby",base,rt)|("tcy",str)]`. Markup stripped for GT by `_strip_tags`.
2. **Irregular** (`_layout_irregular`) — hand-lettered crops: per-line font size (`LINE_SCALE_RANGE=(0.6,1.55)`), start offset (`LINE_SHIFT_EM=1.5`), per-char emphasis (`CHAR_SCALE_RANGE`, `CHAR_JITTER_PROB=0.55`), baseline wobble (`WOBBLE_EM=0.09`), tilt (`ROTATE_DEG=7.0`), shrunken small kana (`SMALL_KANA`, scale 0.55–0.8, floor `MIN_CHAR_SIZE=8`).
3. **SFX** (`_layout_sfx`) — huge type on 8+ baselines (`arc/wave/diagonal/straight/zigzag/bounce/hill/steep/circle`), per-glyph size/tilt, true per-glyph perspective (`_draw_perspective_char`), thick strokes + double edges, radiating `speedlines` / `parallel` lines + `burst` balloons. Decor is never in GT — only font glyphs (incl. `♡♥★` when in the string). Vertical SFX: first label line is the rightmost column.

Reading order guarantee (tested in `test_generator_accuracy.py`): vertical-rl first line rightmost, horizontal first line topmost.

## Color & legibility

Palette sampled by `_sample_text_color(sfx)` / `_sample_text_colors(n)` (`COLOR_PROB`, `SFX_COLOR_PROB`): red, hot pink (heavy), orange, yellow, green, lime, cyan, blue, purple, magenta, brown, light pink. Modes (`_sample_color_mode`, `MULTICOLOR_PROB`): `solid`, `per_line`, `per_char`. Chromatic outputs return BGR; black/white stay grayscale.

Legibility is programmatic (WCAG): `_relative_luminance` / `_contrast_ratio` + `_ensure_legible_outline` + per-glyph `_assign_op_colors` give light fills a dark outline and vice versa. Helpers: `_needs_dark_outline`, `_desaturate_rgba`, `_maybe_mute_fill`, `_normalize_color`, `_is_chromatic_color`.

Force: `{"text_color": "pink"}` (hex `"#ff5090"` ok), mixed `{"text_color":"red","color_mode":"per_char","text_colors":[...]}`, grayscale `{"text_color":"black","color_mode":"solid"}`.

## Background / bubble / degradation

- `_load_background(h,w)`, `_make_background`, `_adjust_background` (tone-match ink luminance), `_decorate_background` (screentones), `_sample_bubble_fill`, `_colorize_background`, `_hatch_overlay`.
- Truncation `_truncate_bgr` (`TRUNC_FRAC_RANGE`), occluders `_draw_foreground_occluders_bgr`, ink realism `_ink_realism` / `_apply_ink_modulation` / `_distress_rgba` / `_smooth_field`, scan/capture `_scan_model_bgr` + `_degrade_bgr` (JPEG, blur, tilt, noise), `_blur_rgba_premultiplied`, `_sweep_layer`, `_scale_layer_alpha`, `_shear_layer(slant_deg)`, `_ink_edge/_ink_mean_bgr/_ink_is_chromatic_bgra/_low_contrast_frac`, `_screentone`, `_draw_speedlines_layer/_draw_parallel_speedlines_layer/_draw_burst_layer`, `_draw_rotated/tilted/perspective_char`, `_draw_tcy_tile`, `_char_op/_char_op_ex/_advance`, `_execute_ops`.

`BODY_MARGIN=20`, `MIN_PAD=4` prevent stroke/blur clipping.
