# Synthetic data generator

Generation of synthetic image-text pairs imitating Japanese manga for the purpose of training OCR.

Features:
- using either text from corpus or random text
- text overlaid on background images
- drawing text bubbles
- various fonts and font styles
- variety of text layouts:
  - vertical and horizontal text
  - multi-line text
  - [furigana](https://en.wikipedia.org/wiki/Furigana) (added randomly)
  - [tate chū yoko](https://www.w3.org/International/articles/vertical-text/#tcy)
  - irregular hand-lettered crops (sampled for ~35% of images, split evenly
    between mild and strong): per-line font sizes with staggered line starts,
    plus optional per-character emphasis sizing, baseline wobble, slight tilt
    and shrunken small kana. Force with `override_css_params={"irregular": ...}`
    and tune via `line_scale_min/max`, `line_shift`, `char_scale_min/max`,
    `char_jitter_prob`, `wobble`, `rotate`, `small_kana`.
  - whole-block slant (sampled for ~12% of images, a third of them strongly
    at 20–32°): italic-like skew applied to the composited text layer, so
    stroke outlines and shadows stay aligned. Force/tune with
    `override_css_params={"slant": degrees}`.


Text rendering is done directly with [Pillow](https://pillow.readthedocs.io/)
(no browser, no subprocesses, no temp files). The renderer typesets the same
layouts the old Chrome-based pipeline produced: vertical (`vertical-rl`) and
horizontal text, furigana (`<ruby>`), tate-chu-yoko
(`text-combine-upright`), `upright` vs. `mixed` Latin orientation, white
stroke outlines and glow shadows, on a transparent background. Fonts are
cached in memory, so rendering one image takes ~10-25 ms for the text layer
(~45 ms end-to-end incl. background/bubble on an i5-6500T), versus ~1.6 s per
image with the old per-image headless-Chrome screenshots. Generation is
parallelized with processes (`process_map`), since Pillow rasterization holds
the GIL and threads no longer scale; expect roughly a 30x throughput gain on
a 16-core machine compared to the old ~17 min per 10000 images.

A note on pixel identity: two different text engines (Blink vs.
FreeType/HarfBuzz) can never produce bit-for-bit identical rasters, so exact
pixel equality with the old pipeline is not achievable by any reimplementation
short of running the same browser. What is preserved is the training
distribution: same fonts, texts, layout semantics, style probabilities and the
unchanged background/bubble code. (A webview-based renderer such as Tauri was
deliberately not used: it is still a full browser engine behind IPC plus a
Rust toolchain, i.e. all of the old overhead with none of the speedup and no
Python screenshot API.) The legacy Chrome renderer is kept as
`BrowserRenderer` in `renderer.py` for side-by-side comparison where Chrome is
available, and `tests/test_pil_renderer.py` covers the new renderer.

High-level generation pipeline is as follows:
1. Preprocess text (truncate and/or split into lines, add random furigana).
2. Render text on a transparent background, using HTML engine.
3. Select background image from backgrounds dataset.
4. Overlay the text on the background, optionally drawing a bubble around the text.

# Examples

## Images generated with text from [CC-100 Japanese corpus](https://data.statmt.org/cc-100/)
![](../../assets/examples/cc-100.jpg)

## Images generated with random text
![](../../assets/examples/random.jpg)