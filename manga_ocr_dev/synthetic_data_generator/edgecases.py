"""Edge-case rendering helpers and targeted 100k synthetic generator.

Covers:
  1. Attached dakuten / handakuten on arbitrary kana (ん゛, ん\u3099, あ゛, っ゛, ン゛, etc.)
     positioned at the upper-right shoulder with unified stroke + fill and
     font / thick-brush / G-pen visual styles.
  2. Clean/hard flat-page multi-line paragraphs (white, paper bleed-through,
     narration boxes, dark pages, down to ~10.5px effective font) to break OCR looping.
  3. Hard colliding manga line-art backgrounds and raw 3-level manga scan crops.
  4. Sharded HF Parquet generation (--edgecase-mode in modal/modal_app.py) pulling
     real afterwords, prose & dakuten vocalizations from Japanese-Novels-23M and
     Japanese-Dialogue-Dataset.
"""

from __future__ import annotations

import math
import os
import re
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

CLEAN_BG_MIN_EFFECTIVE_FONT = 10.5
VOICING_MARKS = frozenset("\u3099\u309a\u309b\u309c\uff9e\uff9f")
DAKUTEN_MARKS = frozenset("\u3099\u309b\uff9e")
HANDAKUTEN_MARKS = frozenset("\u309a\u309c\uff9f")

# Native hardness defaults used by renderer.py even outside --edgecase-mode
NATIVE_CLEAN_BG_PROB = 0.08
NATIVE_HARD_MANGA_BG_PROB = 0.35
NATIVE_RAW_SCAN_PROB = 0.25


# -----------------------------------------------------------------------------
# Grapheme & voicing helpers
# -----------------------------------------------------------------------------
def is_voicable_kana(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch[0])
    return (0x3041 <= cp <= 0x3096) or (0x30A1 <= cp <= 0x30FA) or cp == 0x30FC


def is_attached_voicing_cluster(cluster: str) -> bool:
    return (
        isinstance(cluster, str)
        and len(cluster) >= 2
        and cluster[-1] in VOICING_MARKS
        and is_voicable_kana(cluster[0])
    )


def split_graphemes_with_voicing(text: str) -> list[str]:
    """Attach combining marks AND standalone manga voicing marks (゛/゜/ﾞ/ﾟ) onto kana."""
    clusters: list[str] = []
    for ch in text:
        if clusters and unicodedata.category(ch).startswith("M"):
            clusters[-1] += ch
        elif clusters and ch in VOICING_MARKS and len(clusters[-1]) == 1 and is_voicable_kana(clusters[-1]):
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


# -----------------------------------------------------------------------------
# Unified upper-right dakuten / handakuten glyph compositing
# -----------------------------------------------------------------------------
def _render_voicing_mark_mask(font, cell: int, is_handakuten: bool, rng, *, get_font_fn, is_drawable_fn, safe_draw_fn, style: str = "auto"):
    """Build a tightly cropped grayscale ('L') mask for a dakuten/handakuten mark."""
    if style == "auto" and not is_handakuten:
        chosen = str(rng.choice(["font", "brush", "pen"], p=[0.40, 0.40, 0.20]))
    elif is_handakuten:
        chosen = "font" if rng.random() < 0.65 else "brush"
    else:
        chosen = style

    font_path = getattr(font, "path", None)

    if chosen == "font" and font_path:
        candidates = ("゜", "ﾟ") if is_handakuten else ("゛", "ﾞ")
        for mark_ch in candidates:
            if not is_drawable_fn(font_path, mark_ch):
                continue
            try:
                m_sz = max(8, int(round(cell * float(rng.uniform(0.85, 1.32)))))
                m_font = get_font_fn(font_path, m_sz)
                canvas = max(32, m_sz * 2)
                tmp = Image.new("L", (canvas, canvas), 0)
                d = ImageDraw.Draw(tmp)
                safe_draw_fn(d, (canvas / 2.0, canvas / 2.0), mark_ch, font=m_font, fill=255, anchor="mm")
                bbox = tmp.getbbox()
                if bbox is None:
                    continue
                crop = tmp.crop(bbox)
                cw, ch = crop.size
                target_h = int(round(cell * float(rng.uniform(0.24, 0.44))))
                if ch > 0 and (ch < cell * 0.20 or ch > cell * 0.50):
                    scale = max(0.35, min(2.6, target_h / float(ch)))
                    crop = crop.resize((max(3, int(round(cw * scale))), max(3, int(round(ch * scale)))), Image.BICUBIC)
                rot = float(rng.uniform(-14.0, 14.0))
                if abs(rot) > 1.0:
                    crop = crop.rotate(rot, resample=Image.BICUBIC, expand=True, fillcolor=0)
                    bb2 = crop.getbbox()
                    if bb2 is not None:
                        crop = crop.crop(bb2)
                if not is_handakuten and rng.random() < 0.40 and cell >= 32:
                    k_sz = int(rng.choice([2, 3])) if cell >= 64 else 2
                    arr = cv2.dilate(np.asarray(crop), np.ones((k_sz, k_sz), np.uint8), iterations=1)
                    crop = Image.fromarray(arr, mode="L")
                return crop
            except Exception:
                continue

    ss = 2
    if is_handakuten:
        diam = max(6, int(round(cell * float(rng.uniform(0.22, 0.36)))))
        thick = max(2, int(round(diam * float(rng.uniform(0.18, 0.30)))))
        c_sz = (diam + thick * 2 + 8) * ss
        tmp = Image.new("L", (c_sz, c_sz), 0)
        d = ImageDraw.Draw(tmp)
        pad = (thick + 4) * ss
        d.ellipse([pad, pad, c_sz - pad, c_sz - pad], outline=255, width=max(2, thick * ss))
        tmp = tmp.resize((c_sz // ss, c_sz // ss), Image.LANCZOS)
        bb = tmp.getbbox()
        return tmp.crop(bb) if bb else tmp

    # Expressive brush ('||', '\\') or rough hand-lettered G-pen dakuten strokes
    mh = max(8, int(round(cell * float(rng.uniform(0.27, 0.46)))))
    tick_w = max(2, int(round(cell * float(rng.uniform(0.072, 0.138) if chosen == "brush" else rng.uniform(0.045, 0.085)))))
    gap = max(tick_w + 2, int(round(cell * float(rng.uniform(0.12, 0.22)))))
    slant_dx = int(round(mh * float(rng.uniform(-0.10, 0.36))))
    cw = (abs(slant_dx) + gap + tick_w * 4 + 14) * ss
    ch = (mh + tick_w * 4 + 14) * ss
    tmp = Image.new("L", (cw, ch), 0)
    d = ImageDraw.Draw(tmp)
    x_start = (tick_w + 7 + max(0, -slant_dx)) * ss
    y_start = (tick_w + 7) * ss

    for i in range(2):
        jx0 = int(round(float(rng.uniform(-0.07, 0.07)) * mh * ss))
        jy0 = int(round(float(rng.uniform(-0.10, 0.10)) * mh * ss))
        jy1 = int(round(float(rng.uniform(-0.10, 0.10)) * mh * ss))
        x0 = x_start + i * gap * ss + jx0
        y0 = y_start + jy0
        x1 = x0 + slant_dx * ss
        y1 = y_start + mh * ss + jy1
        w_ss = max(2, int(round(tick_w * ss * float(rng.uniform(0.85, 1.20)))))
        if chosen == "pen":
            xm = int((x0 + x1) / 2 + float(rng.uniform(-0.14, 0.14)) * mh * ss)
            ym = int((y0 + y1) / 2)
            d.line([(x0, y0), (xm, ym), (x1, y1)], fill=255, width=w_ss, joint="curve")
        else:
            hw0 = max(1, int(round(w_ss * float(rng.uniform(0.42, 0.58)))))
            hw1 = max(1, int(round(w_ss * float(rng.uniform(0.32, 0.52)))))
            poly = [(x0 - hw0, y0), (x0 + hw0, y0), (x1 + hw1, y1), (x1 - hw1, y1)]
            d.polygon(poly, fill=255)
        r = max(1, w_ss // 3)
        d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=255)
        d.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=255)

    tmp = tmp.resize((max(4, cw // ss), max(4, ch // ss)), Image.LANCZOS)
    bb = tmp.getbbox()
    return tmp.crop(bb) if bb else tmp


def draw_attached_voicing_char(
    layer,
    cluster: str,
    cx: float,
    cy: float,
    *,
    font,
    cell: int,
    fill,
    stroke_width: int,
    stroke_fill,
    get_font_fn,
    is_drawable_fn,
    safe_draw_fn,
    angle: float = 0.0,
    persp: float = 0.0,
    rotate_mixed: bool = False,
    ttb: bool = False,
    lang: str = "ja",
    aspect=None,
    pcoef=None,
    voicing_seed=None,
    dakuten_style: str = "auto",
) -> None:
    """Render kana + attached dakuten/handakuten (e.g. ん゛, ん\u3099, あ゛) with unified stroke & fill."""
    base = cluster[:-1]
    mark = cluster[-1]
    is_handakuten = mark in HANDAKUTEN_MARKS
    rng = np.random.default_rng(int(voicing_seed)) if voicing_seed is not None else np.random.default_rng()

    cell = int(cell or getattr(font, "size", 48))
    sw = max(0, int(stroke_width or 0))
    p = abs(float(persp or 0.0))
    ax, ay = (float(aspect[0]), float(aspect[1])) if aspect else (1.0, 1.0)
    total_angle = (270.0 if rotate_mixed else 0.0) + float(angle or 0.0)
    rad = math.radians(total_angle)
    rot_f = abs(math.cos(rad)) + abs(math.sin(rad))
    pad = sw + max(10, int(round(cell * 0.50)))
    ink = math.ceil(cell * 1.48) + 2 * pad
    size = max(24, math.ceil(ink * max(ax, ay, 1.0) * rot_f * (1.0 + p)) + 8)
    size += size & 1

    tcx = size / 2.0 - cell * 0.06
    tcy = size / 2.0 + cell * 0.04

    base_kw = {"font": font, "anchor": "mm"}
    if ttb:
        base_kw.update(direction="ttb", language=str(lang or "ja"))

    probe = Image.new("L", (size, size), 0)
    safe_draw_fn(ImageDraw.Draw(probe), (tcx, tcy), base, fill=255, **base_kw)
    bb = probe.getbbox()
    if bb is None:
        bx0, by0, bx1, by1 = (
            int(tcx - cell * 0.35),
            int(tcy - cell * 0.35),
            int(tcx + cell * 0.35),
            int(tcy + cell * 0.35),
        )
    else:
        bx0, by0, bx1, by1 = bb
    bh = max(4, by1 - by0)

    mark_mask = _render_voicing_mark_mask(
        font,
        cell,
        is_handakuten,
        rng,
        get_font_fn=get_font_fn,
        is_drawable_fn=is_drawable_fn,
        safe_draw_fn=safe_draw_fn,
        style=dakuten_style,
    )
    mw, mh = mark_mask.size

    place_roll = float(rng.random())
    if place_roll < 0.48:
        # Tucked tightly into the upper-right pocket (exact Image 2 style for ん゛/い゛/あ゛)
        mx = int(round(bx1 - mw * float(rng.uniform(0.22, 0.55)) + cell * float(rng.uniform(0.0, 0.08))))
        my = int(round(by0 - mh * float(rng.uniform(-0.08, 0.22)) + bh * float(rng.uniform(0.0, 0.08))))
    elif place_roll < 0.82:
        mx = int(round(bx1 - mw * float(rng.uniform(0.02, 0.24)) + cell * float(rng.uniform(0.02, 0.11))))
        my = int(round(by0 - mh * float(rng.uniform(0.04, 0.28)) + bh * float(rng.uniform(0.0, 0.10))))
    else:
        mx = int(round(bx1 + cell * float(rng.uniform(0.03, 0.14))))
        my = int(round(by0 + bh * float(rng.uniform(-0.06, 0.14))))

    margin_clamp = sw + 2
    mx = max(margin_clamp, min(size - mw - margin_clamp, mx))
    my = max(margin_clamp, min(size - mh - margin_clamp, my))

    mark_full = Image.new("L", (size, size), 0)
    mark_full.paste(mark_mask, (mx, my))

    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    fill_rgba = fill if isinstance(fill, tuple) and len(fill) == 4 else (0, 0, 0, 255)
    edge_rgba = stroke_fill if isinstance(stroke_fill, tuple) and len(stroke_fill) == 4 else fill_rgba
    is_hollow = fill_rgba[3] == 0

    # Pass 1: Unified outer stroke for both base glyph and attached mark.
    if sw > 0 and stroke_fill is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * sw + 1, 2 * sw + 1))
        dil = cv2.dilate(np.asarray(mark_full), kernel, iterations=1)
        if is_hollow:
            safe_draw_fn(draw, (tcx, tcy), base, fill=(0, 0, 0, 0), stroke_width=sw, stroke_fill=edge_rgba, **base_kw)
            ring = np.clip(dil.astype(np.int16) - np.asarray(mark_full).astype(np.int16), 0, 255).astype(np.uint8)
            mark_layer = Image.new("RGBA", (size, size), edge_rgba)
            ring_alpha = (ring.astype(np.float32) * (edge_rgba[3] / 255.0)).astype(np.uint8)
            mark_layer.putalpha(Image.fromarray(ring_alpha, mode="L"))
            tile = Image.alpha_composite(tile, mark_layer)
        else:
            safe_draw_fn(draw, (tcx, tcy), base, fill=edge_rgba, stroke_width=sw, stroke_fill=edge_rgba, **base_kw)
            mark_stroke = Image.new("RGBA", (size, size), edge_rgba)
            dil_alpha = (dil.astype(np.float32) * (edge_rgba[3] / 255.0)).astype(np.uint8)
            mark_stroke.putalpha(Image.fromarray(dil_alpha, mode="L"))
            tile = Image.alpha_composite(tile, mark_stroke)

    # Pass 2: Unified inner fill for both base glyph and attached mark.
    if not is_hollow:
        draw = ImageDraw.Draw(tile)
        safe_draw_fn(draw, (tcx, tcy), base, fill=fill_rgba, **base_kw)
        mark_fill = Image.new("RGBA", (size, size), fill_rgba)
        mf_alpha = (np.asarray(mark_full).astype(np.float32) * (fill_rgba[3] / 255.0)).astype(np.uint8)
        mark_fill.putalpha(Image.fromarray(mf_alpha, mode="L"))
        tile = Image.alpha_composite(tile, mark_fill)

    if aspect:
        nw, nh = max(2, round(size * ax)), max(2, round(size * ay))
        nw += (size - nw) & 1
        nh += (size - nh) & 1
        stretched = tile.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        canvas.paste(stretched, ((size - nw) // 2, (size - nh) // 2))
        tile = canvas
    if total_angle:
        tile = tile.rotate(total_angle, resample=Image.BICUBIC)
    if p > 1e-6:
        u = pcoef if pcoef is not None else tuple(float(v) for v in rng.uniform(-1.0, 1.0, 8))
        ref = cell * 2.2
        a, b, c0 = 1.0 + u[0] * p, u[1] * p, u[2] * p * ref * 0.1
        d, e, f0 = u[3] * p, 1.0 + u[4] * p, u[5] * p * ref * 0.1
        g, h = u[6] * p / ref, u[7] * p / ref
        tx = ty = size / 2.0
        d0 = 1.0 - g * tx - h * ty
        coeffs = (
            (a + tx * g) / d0,
            (b + tx * h) / d0,
            (-a * tx - b * ty + c0 + tx * d0) / d0,
            (d + ty * g) / d0,
            (e + ty * h) / d0,
            (-d * tx - e * ty + f0 + ty * d0) / d0,
            g / d0,
            h / d0,
        )
        tile = tile.transform((size, size), Image.PERSPECTIVE, coeffs, resample=Image.BILINEAR)

    layer.paste(tile, (round(cx) - size // 2, round(cy) - size // 2), tile)


# -----------------------------------------------------------------------------
# Hard backgrounds & scan degradations
# -----------------------------------------------------------------------------
def render_clean_background(
    renderer,
    img,
    *,
    crop_by_alpha_fn,
    normalize_color_fn,
    blend_fn,
    clean_bg_color=None,
    clean_margin=None,
    clean_box_border: bool = False,
    clean_bleedthrough: bool = False,
):
    """Render text onto a clean/paper/narration-box/dark flat page background."""
    img = crop_by_alpha_fn(img, 0)
    ih, iw = img.shape[:2]
    if isinstance(clean_margin, (tuple, list)) and len(clean_margin) == 4:
        mt, mb, ml, mr = (max(2, int(v)) for v in clean_margin)
    elif clean_margin is not None:
        m = max(2, int(clean_margin))
        mt = mb = ml = mr = m
    else:
        base_m = max(6, min(30, int(min(ih, iw) * np.random.uniform(0.03, 0.14))))
        mt = max(4, int(base_m * np.random.uniform(0.65, 1.35)))
        mb = max(4, int(base_m * np.random.uniform(0.65, 1.35)))
        ml = max(6, int(base_m * np.random.uniform(0.75, 1.45)))
        mr = max(6, int(base_m * np.random.uniform(0.75, 1.45)))
    img = np.pad(img, ((mt, mb), (ml, mr), (0, 0)))
    ink_box = (ml, mt, img.shape[1] - mr, img.shape[0] - mb)

    opaque = img[:, :, 3] > 120
    if opaque.any():
        obgr = img[:, :, :3][opaque].astype(np.float32)
        body_lum = float((0.114 * obgr[:, 0] + 0.587 * obgr[:, 1] + 0.299 * obgr[:, 2]).mean())
    else:
        body_lum = 0.0

    if clean_bg_color is not None:
        if isinstance(clean_bg_color, (tuple, list)) and len(clean_bg_color) >= 3:
            bg_bgr = tuple(int(np.clip(c, 0, 255)) for c in clean_bg_color[:3])
        else:
            rgba = normalize_color_fn(clean_bg_color, "white")
            bg_bgr = (rgba[2], rgba[1], rgba[0])
    elif body_lum > 150.0:
        v = int(np.random.randint(8, 36))
        bg_bgr = (v, v, v)
    else:
        r = np.random.rand()
        if r < 0.58:
            bg_bgr = (255, 255, 255)
        elif r < 0.86:
            v = int(np.random.randint(240, 254))
            bg_bgr = (
                v,
                min(255, v + int(np.random.randint(-2, 3))),
                min(255, v + int(np.random.randint(-2, 3))),
            )
        else:
            v = int(np.random.randint(218, 240))
            bg_bgr = (v, v, v)

    background = np.full((img.shape[0], img.shape[1], 3), bg_bgr, dtype=np.uint8)
    if clean_bleedthrough and body_lum < 140.0:
        try:
            ghost = renderer._load_background(img.shape[0], img.shape[1])
            ghost = cv2.GaussianBlur(ghost, (5, 5), 0).astype(np.float32)
            alpha_g = float(np.random.uniform(0.03, 0.11))
            background = np.clip(
                background.astype(np.float32) * (1.0 - alpha_g) + ghost * alpha_g, 0, 255
            ).astype(np.uint8)
        except Exception:
            pass

    if clean_box_border:
        bh, bw = background.shape[:2]
        bcol = (235, 235, 235) if body_lum > 150.0 else (int(np.random.randint(0, 45)),) * 3
        bth = int(np.random.choice([1, 2, 2, 3]))
        bx0 = max(1, int(ml * np.random.uniform(0.15, 0.55)))
        by0 = max(1, int(mt * np.random.uniform(0.15, 0.55)))
        bx1 = min(bw - 2, bw - max(1, int(mr * np.random.uniform(0.15, 0.55))))
        by1 = min(bh - 2, bh - max(1, int(mb * np.random.uniform(0.15, 0.55))))
        if bx1 > bx0 + 8 and by1 > by0 + 8:
            cv2.rectangle(background, (bx0, by0), (bx1, by1), bcol, bth)

    quality = renderer._text_visibility(img, background)
    out = blend_fn(img, background)
    return out, ink_box, quality


def make_hard_manga_background(bg0, *, screentone_fn, hatch_fn):
    """Compose harsh manga halftone screentone + bold colliding hair/cloth ink lines."""
    h, w = bg0.shape[:2]
    if np.random.rand() < 0.45:
        base_gray = int(np.random.randint(135, 228))
        bg = np.full((h, w, 3), base_gray, dtype=np.float32)
        dots = screentone_fn(h, w)
        bg *= (1.0 - float(np.random.uniform(0.28, 0.62)) * dots)[..., None]
        bg = np.clip(bg, 0, 255).astype(np.uint8)
    else:
        f = bg0.astype(np.float32)
        g = f.mean(axis=2, keepdims=True)
        bg = np.repeat(
            np.clip((g - g.mean()) * float(np.random.uniform(1.1, 2.0)) + float(np.random.uniform(115, 195)), 0, 255),
            3,
            axis=2,
        ).astype(np.uint8)
        if np.random.rand() < 0.70:
            dots = screentone_fn(h, w)
            bg = np.clip(
                bg.astype(np.float32) * (1.0 - float(np.random.uniform(0.22, 0.55)) * dots)[..., None], 0, 255
            ).astype(np.uint8)

    n_strokes = int(np.random.randint(3, 11))
    diag = math.hypot(h, w)
    base_ang = float(np.random.uniform(0, math.pi))
    for _ in range(n_strokes):
        ang = base_ang + float(np.random.uniform(-0.45, 0.45))
        cx = float(np.random.uniform(0, w))
        cy = float(np.random.uniform(0, h))
        dx, dy = math.cos(ang) * diag * 0.8, math.sin(ang) * diag * 0.8
        p0 = (int(cx - dx), int(cy - dy))
        p2 = (int(cx + dx), int(cy + dy))
        bend = float(np.random.uniform(-0.22, 0.22)) * diag
        p1 = (int(cx - math.sin(ang) * bend), int(cy + math.cos(ang) * bend))
        pts = []
        for t in np.linspace(0.0, 1.0, 12):
            u = 1.0 - t
            px = int(u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0])
            py = int(u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1])
            pts.append([px, py])
        th = max(2, int(round(min(h, w) * float(np.random.uniform(0.025, 0.085)))))
        col_v = int(np.random.choice([0, 0, 0, 22, 48, 255]))
        cv2.polylines(bg, [np.array(pts, dtype=np.int32)], False, (col_v, col_v, col_v), th, lineType=cv2.LINE_AA)

    if np.random.rand() < 0.35:
        bg = np.clip(
            bg.astype(np.float32) * (1.0 - hatch_fn(h, w, strength=float(np.random.uniform(0.25, 0.55))))[..., None],
            0,
            255,
        ).astype(np.uint8)
    return bg


def degrade_clean_page_bgr(img, *, effective_font: float, scan_model_fn):
    """Realistic digital/scanned page artifacts without destroying small multi-line text."""
    out = img
    if np.random.rand() < 0.28:
        h, w = out.shape[:2]
        ang = float(np.random.uniform(-2.2, 2.2))
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
        out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if np.random.rand() < 0.40:
        out = scan_model_fn(out, allow_spread=False)
    if effective_font >= 16.0 and np.random.rand() < 0.25 and min(out.shape[:2]) >= 64:
        h, w = out.shape[:2]
        s = float(np.random.uniform(0.76, 0.94))
        small = cv2.resize(out, (max(32, int(w * s)), max(32, int(h * s))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if np.random.rand() < 0.65:
        q = int(np.random.randint(40, 95))
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    return out


def degrade_raw_manga_crop_bgr(img, *, is_color: bool, max_output_aspect: float, scan_model_fn):
    """Simulate raw manga/doujin scan crops (narrow pixel res, crisp 3-level halftone, heavy JPEG)."""
    out = img
    h, w = out.shape[:2]
    if min(h, w) > 90 and np.random.rand() < 0.55:
        target_short = int(np.random.randint(52, min(138, min(h, w))))
        scale = target_short / float(min(h, w))
        nw, nh = max(34, int(round(w * scale))), max(34, int(round(h * scale)))
        if max(nw, nh) / max(1, min(nw, nh)) <= max_output_aspect:
            interp = cv2.INTER_AREA if np.random.rand() < 0.70 else cv2.INTER_NEAREST
            out = cv2.resize(out, (nw, nh), interpolation=interp)

    if not is_color and np.random.rand() < 0.45:
        f = out.astype(np.float32)
        gain = float(np.random.uniform(1.15, 1.58))
        out = np.clip((f - 128.0) * gain + 128.0, 0, 255).astype(np.uint8)
        if np.random.rand() < 0.45:
            levels = int(np.random.choice([4, 6, 8, 12]))
            step = 255.0 / max(2, levels - 1)
            out = np.clip(np.round(out.astype(np.float32) / step) * step, 0, 255).astype(np.uint8)
    else:
        out = scan_model_fn(out, allow_spread=False)

    if np.random.rand() < 0.30:
        out = out.astype(np.float32) + np.random.normal(0.0, float(np.random.uniform(3.0, 11.0)), out.shape).astype(np.float32)
        out = np.clip(out, 0, 255).astype(np.uint8)

    if np.random.rand() < 0.68:
        q = int(np.random.randint(20, 62))
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    return out


# =============================================================================
# Targeted ~100k edge-case dataset generator (called by modal/modal_app.py)
# =============================================================================
_JA_AFTERWORD_TEMPLATES = [
    "ご購読まことにありがとうございました！",
    "お手に取っていただきありがとうございます。",
    "超ひさしぶりの本です。",
    "久しぶりの新刊になりました。",
    "ここまで読んでくださって感謝です！",
    "次回もよろしくお願いします。",
    "またイベントでお会いしましょう！",
    "感想などいただけたら嬉しいです。",
    "メメント杜から長谷川らくに改名したりしました。",
    "ペンネームを少し変更しました。",
    "もう長いことこういうキャラを愛でている気がします。",
    "前から描きたかったネタをやっと形にできました。",
    "そして⇧と同じくらい黒髪センターパートのキャラが好きです。",
    "個人的には→のシーンが一番お気に入りです。",
    "※この物語はフィクションです。",
    "※実在の人物・団体とは一切関係ありません。",
    "ゴムはつけましょう。",
    "体調管理には気をつけましょう。",
    "締切ギリギリでしたが何とか間に合いました！",
    "アシスタントの皆様、本当にありがとうございました。",
    "表紙デザインもすごく可愛く仕上がっています。",
    "ページ数が予定より少し増えてしまいました。",
    "巻末におまけページも収録しています。",
    "誤字脱字があったらごめんなさい！",
    "次巻の発売は来春を予定しています。",
    "アンケートや応援メッセージいつも励みになります。",
    "SNSのフォローもよろしくね♪",
    "それではまた次の作品でお会いしましょう。",
    "描き下ろしイラストもぜひチェックしてください★",
    "設定資料は公式サイトでも公開中です。",
    "少し駆け足になってしまった部分もありますが楽しんで頂けたら幸いです。",
    "キャラクターの表情を描くのがとても楽しかったです。",
    "背景のトーン作業が一番大変でした…！",
    "これからもマイペースに更新していく予定です。",
    "発行日：2026年11月19日 初版発行",
    "無断転載・複製・AI学習への利用を禁じます。",
]

_JA_INLINE_SYMBOLS = ["⇧", "⇨", "→", "←", "↑", "↓", "・", "※", "★", "☆", "♡", "♪", "〜", "…", "●", "◎"]

_JA_NONSTANDARD_VOICED_HIGH = [
    "ん", "あ", "い", "う", "え", "お",
    "ぁ", "ぃ", "ぅ", "ぇ", "ぉ", "っ",
    "ン", "ア", "イ", "ウ", "エ", "オ",
    "ァ", "ィ", "ゥ", "ェ", "ォ", "ッ",
]
_JA_NONSTANDARD_VOICED_EXTRA = [
    "な", "に", "ぬ", "ね", "の", "ま", "み", "む", "め", "も",
    "や", "ゆ", "よ", "ら", "り", "る", "れ", "ろ", "わ", "ー",
    "ナ", "ニ", "ヌ", "ネ", "ノ", "マ", "ミ", "ム", "メ", "モ",
    "ヤ", "ユ", "ヨ", "ラ", "リ", "ル", "レ", "ロ", "ワ",
]
_JA_STANDARD_VOICED_HIRA = list("がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽゔ")
_JA_STANDARD_VOICED_KATA = list("ガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポヴ")
_JA_CONFUSION_PAIRS = [
    "ばぱは", "びぴひ", "ぶぷふ", "べぺへ", "ぼぽほ",
    "がっか", "ぎっき", "ぐっく", "だたった", "どっと", "ずっす",
    "バパハ", "ビピヒ", "ブプフ", "ガッカ", "ダタッ", "ヴウッ", "ゔうっ",
]
_JA_TRAIL_SMALL = ["ッ", "っ", "ぃ", "ぁ", "ぅ", "ぇ", "ぉ", "ィ", "ァ", "ゥ", "ェ", "ォ", "い", "あ", "ん", "ー"]
_JA_END_PUNCT = ["", "", "", "！", "っ", "ッ", "…", "♡", "？", "！？", "っ！", "ッ！"]
_JA_DAKUTEN_WORDS = [
    "ん{d}ッい", "ん{d}っぃ", "ん{d}っ", "ん{d}ッ", "ん{d}ぁ", "ん{d}あ{d}",
    "ん{d}ん{d}", "ん{d}ぐ{d}っ", "ん{d}ふ{d}っ", "ん{d}ぅ{d}",
    "あ{d}ッ", "あ{d}っ", "あ{d}ぁ{d}", "あ{d}あ{d}っ", "あ{d}あ{d}あ{d}っ！",
    "い{d}っ", "い{d}ぃ{d}", "い{d}た{d}い{d}っ", "う{d}っ", "う{d}ぁ{d}っ",
    "え{d}っ", "お{d}っ", "お{d}ぉ{d}っ", "ぁ{d}っ", "ぃ{d}っ", "ぅ{d}っ",
    "ン{d}ッ", "ア{d}ッ", "ア{d}ア{d}ア{d}ッ", "イ{d}ッ", "ウ{d}ッ", "エ{d}ッ", "オ{d}ッ",
    "や{d}め{d}て{d}っ", "ぐ{d}っ", "ぎ{d}ゃ{d}あ{d}", "ご{d}ほ{d}っ", "が{d}は{d}っ",
    "んっ…ん{d}っ", "ああ{d}っ", "ん{d}っい", "ん{d}ッぃ", "あっ…あ{d}ぁっ",
    "はぁっ…ん{d}っ", "ん{d}っ…い{d}っ", "ん{d}ぉ{d}っ♡",
]
_JA_STANDARD_VOICED_SFX = [
    "ガッ", "ゴッ", "ドッ", "バキッ", "ボコッ", "ズバァッ", "グサッ", "ドゴォッ",
    "ビリッ", "ザワッ", "ゾクッ", "ギロッ", "ベチャッ", "ギュッ", "ブワッ", "パキッ",
    "ポキッ", "ピキッ", "ガタッ", "ゴゴゴ", "ドドド", "ダダダ", "バタバタ", "ズキズキ",
    "パァンッ", "バンッ", "ボンッ", "ポンッ", "ピシャッ", "ビシャッ", "プツッ", "ブツッ",
    "ぐっ", "ごほっ", "だめっ", "ばかっ", "びくっ", "どきっ", "ぞくっ", "ぷるっ",
    "ぎゅっ", "ぶわっ", "べたっ", "じわっ", "がたっ", "ぱっ", "ぽっ", "ゔっ", "ヴッ",
]


def import_repo_normalizer():
    """Import repo normalize_text if available, else replicate exact NFKC + CJK space stripping."""
    for mod_name in (
        "manga_ocr_dev.data.process_dialogue_and_novels",
        "manga_ocr_dev.data.generate_lines",
        "manga_ocr_dev.data.process_lines",
    ):
        try:
            mod = __import__(mod_name, fromlist=["normalize_text"])
            fn = getattr(mod, "normalize_text", None)
            if callable(fn):
                return fn
        except Exception:
            pass

    cjk_space = re.compile(
        r"(?<=[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\u3000-\u303f])\s+"
        r"|\s+(?=[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\u3000-\u303f])"
    )
    ws_re = re.compile(r"\s+")

    def _fallback_normalize(text: str | None) -> str:
        if not text:
            return ""
        t = unicodedata.normalize("NFKC", str(text))
        t = cjk_space.sub("", t)
        return ws_re.sub(" ", t).strip()

    return _fallback_normalize


def load_ja_corpora(
    cache_dir: Path,
    *,
    limit: int = 250_000,
    lines_path: str = "",
    commit_fn=None,
) -> tuple[list[str], list[str]]:
    """Return (paragraph_sentences, real_dakuten_lines) harvested from novels & dialogue."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if lines_path and Path(lines_path).is_file():
        p = Path(lines_path)
        if p.suffix == ".parquet":
            tbl = pq.read_table(p)
            col = "line" if "line" in tbl.column_names else tbl.column_names[0]
            rows = [str(x).strip() for x in tbl[col].to_pylist() if x and 4 <= len(str(x).strip()) <= 45]
        else:
            rows = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if 4 <= len(ln.strip()) <= 45]
        if rows:
            print(f"Loaded {len(rows):,} sentences from local lines_path={p}")
            return rows, []

    cache_dir.mkdir(parents=True, exist_ok=True)
    sent_file = cache_dir / "ja_edgecase_sentences_v2.parquet"
    dak_file = cache_dir / "ja_edgecase_dakuten_v2.parquet"
    if sent_file.is_file():
        rows = [str(x) for x in pq.read_table(sent_file)["line"].to_pylist() if x]
        d_rows = [str(x) for x in pq.read_table(dak_file)["line"].to_pylist() if x] if dak_file.is_file() else []
        if len(rows) >= 5_000:
            print(f"Reusing cached corpora: {len(rows):,} sentences, {len(d_rows):,} real dakuten lines")
            return rows, d_rows

    sentences: list[str] = []
    real_dakuten: list[str] = []
    sent_split_re = re.compile(r"(?<=[。！？!?\n])")
    ja_re = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
    aozora_ruby = re.compile(r"｜?([^《\r\n]+)《[^》\r\n]+》")
    aozora_notes = re.compile(r"［＃[^］\r\n]+］")
    nonstd_dak_re = re.compile(r"[んあいうえおぁぃぅぇぉっンアイウエオァィゥェォッ][゛\u3099ﾞ゜\u309a]")

    def _harvest(raw_text: str, cap: int) -> None:
        if not raw_text:
            return
        cleaned = aozora_notes.sub("", aozora_ruby.sub(r"\1", raw_text))
        for raw in sent_split_re.split(cleaned):
            s = raw.strip().strip("\u3000「」『』").replace("\t", " ")
            if not s or "http" in s or "==" in s:
                continue
            if 2 <= len(s) <= 16 and len(real_dakuten) < 40_000 and nonstd_dak_re.search(s):
                real_dakuten.append(s)
            if len(sentences) < cap and 5 <= len(s) <= 38 and ja_re.search(s):
                sentences.append(s)

    # Tier 1: OmniAICreator/Japanese-Novels-23M (afterwords + NSFW/SFW web novels)
    try:
        from datasets import load_dataset

        novel_cap = int(limit * 0.62)
        configs = ("Narou_NSFW", "Kakuyomu", "Narou_SFW", "pixiv", "Hameln")
        per_cfg = max(5_000, novel_cap // len(configs))
        for cfg in configs:
            target = min(novel_cap, len(sentences) + per_cfg)
            try:
                print(f"Streaming Japanese-Novels-23M ({cfg}) [afterwords + prose + dakuten]...")
                ds = load_dataset("OmniAICreator/Japanese-Novels-23M", cfg, split="train", streaming=True)
                for row in ds:
                    for field in ("afterword", "atogaki", "preface", "maegaki", "synopsis"):
                        val = row.get(field)
                        if isinstance(val, str) and val:
                            _harvest(val, target)
                    body = row.get("main_text") or row.get("text")
                    if isinstance(body, str) and body:
                        _harvest(body[:2200], target)
                    if len(sentences) >= target:
                        break
            except Exception as exc:
                print(f"Japanese-Novels-23M ({cfg}) notice: {exc}")
        print(f"Collected {len(sentences):,} novel lines & {len(real_dakuten):,} real dakuten lines.")
    except Exception as exc:
        print(f"Japanese-Novels-23M streaming notice: {exc}")

    # Tier 2: JustANormalTinkerer/Japanese-Dialogue-Dataset
    try:
        from datasets import load_dataset

        print("Streaming JustANormalTinkerer/Japanese-Dialogue-Dataset...")
        for sp in ("anime_tv", "anime_movie", "drama_tv"):
            if len(sentences) >= limit:
                break
            try:
                ds = load_dataset("JustANormalTinkerer/Japanese-Dialogue-Dataset", split=sp, streaming=True)
                for row in ds:
                    val = row.get("dialogue") or row.get("text") or ""
                    if isinstance(val, str):
                        _harvest(val, limit)
                    if len(sentences) >= limit:
                        break
            except Exception as exc:
                print(f"Japanese-Dialogue-Dataset ({sp}) notice: {exc}")
        print(f"Total sentences after dialogue dataset: {len(sentences):,}")
    except Exception as exc:
        print(f"Japanese-Dialogue-Dataset streaming notice: {exc}")

    # Tier 3: Wikipedia fallback
    if len(sentences) < 5_000:
        try:
            from datasets import load_dataset

            print("Streaming fallback sentences from wikimedia/wikipedia (20231101.ja)...")
            ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
            for item in ds:
                _harvest(item.get("text") or "", limit)
                if len(sentences) >= limit:
                    break
        except Exception as exc:
            print(f"Wikipedia ja fallback notice: {exc}")

    sentences.extend(_JA_AFTERWORD_TEMPLATES * 25)
    if sentences:
        tmp_f = sent_file.with_suffix(".parquet.tmp")
        pq.write_table(pa.table({"line": sentences}), tmp_f)
        os.replace(tmp_f, sent_file)
    if real_dakuten:
        tmp_d = dak_file.with_suffix(".parquet.tmp")
        pq.write_table(pa.table({"line": real_dakuten}), tmp_d)
        os.replace(tmp_d, dak_file)
    if commit_fn is not None:
        commit_fn()
    return sentences, real_dakuten


def categorize_ja_fonts(fonts_root: Path) -> tuple[list[str], list[str], list[str]]:
    """Return (all_fonts, clean_paragraph_fonts, expressive_dakuten_fonts)."""
    exts = {".ttf", ".otf", ".ttc"}
    all_fonts = sorted(str(p) for p in fonts_root.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if not all_fonts:
        raise RuntimeError(f"No font files found under {fonts_root}")

    expressive_kw = (
        "brush", "fude", "tegaki", "hand", "bold", "heavy", "black", "pop", "comic",
        "manga", "kuro", "kouzan", "aoyagi", "anzu", "851", "shinkai", "kaisen",
        "gyosho", "kaisho", "callig", "marker", "pen", "extra", "ultra", "thick",
    )
    clean_kw = (
        "gothic", "maru", "mincho", "sans", "serif", "noto", "meiryo", "yu",
        "hiragino", "mplus", "rounded", "regular", "medium", "book", "normal",
        "gen", "source", "ipag", "ipam", "sawarabi", "kosugi", "zen", "biz",
    )
    expressive = [f for f in all_fonts if any(k in Path(f).name.lower() for k in expressive_kw)]
    clean = [f for f in all_fonts if any(k in Path(f).name.lower() for k in clean_kw)]
    return all_fonts, (clean if len(clean) >= 3 else all_fonts), (expressive if len(expressive) >= 3 else all_fonts)


def detect_line_join(repo_dir: Path, language: str, line_join: str) -> str:
    if line_join == "none":
        return ""
    if line_join == "newline":
        return "\n"
    if line_join == "space":
        return " "
    if line_join != "auto":
        return line_join
    gen_file = repo_dir / "manga_ocr_dev" / "synthetic_data_generator" / "generator.py"
    if gen_file.is_file():
        src = gen_file.read_text(encoding="utf-8", errors="ignore")
        if '"\\n".join' in src or "'\\n'.join" in src:
            return "\n"
    return " " if language in ("en", "ko") else ""


def build_clean_paragraph(rng, sentences: list[str], font_path: str, is_drawable_fn) -> tuple[list[str], dict]:
    n_lines = int(rng.choice([3, 4, 5, 6, 7, 8, 9, 10, 11], p=[0.06, 0.12, 0.18, 0.20, 0.16, 0.12, 0.08, 0.05, 0.03]))
    use_afterword_bias = rng.random() < 0.30
    repeat_ending = rng.choice(["です。", "ます。", "ました。", "ですね。", "ましょう。", "でした。"]) if rng.random() < 0.20 else None
    list_style = str(rng.choice(["none", "bullet", "numbered", "indented"], p=[0.76, 0.10, 0.07, 0.07]))

    lines: list[str] = []
    for li in range(n_lines):
        if use_afterword_bias and rng.random() < 0.50:
            s = str(_JA_AFTERWORD_TEMPLATES[int(rng.integers(0, len(_JA_AFTERWORD_TEMPLATES)))])
        else:
            s = str(sentences[int(rng.integers(0, len(sentences)))])
        roll_len = rng.random()
        if roll_len < 0.22 and len(s) > 13:
            cut = int(rng.integers(6, min(len(s), 15)))
            s = s[:cut].rstrip("、, ") + str(rng.choice(["。", "！", "？", ""]))
        elif roll_len < 0.48 and len(s) < 22:
            s2 = str(sentences[int(rng.integers(0, len(sentences)))])
            s = (s + s2)[: int(rng.integers(22, 38))]
        if repeat_ending and rng.random() < 0.50 and len(s) >= 6:
            s = s.rstrip("。！？!? ") + str(repeat_ending)
        if list_style == "bullet" and rng.random() < 0.85:
            s = "・" + s
        elif list_style == "numbered":
            s = f"{li + 1}.{s}"
        elif list_style == "indented" and li == 0 and is_drawable_fn(font_path, "\u3000"):
            s = "\u3000" + s
        s = "".join(ch for ch in s if is_drawable_fn(font_path, ch))
        if len(s) < 3:
            s = "".join(ch for ch in "ありがとうございます。" if is_drawable_fn(font_path, ch))
        lines.append(s)

    if rng.random() < 0.35 and lines:
        drawable_syms = [sym for sym in _JA_INLINE_SYMBOLS if is_drawable_fn(font_path, sym)]
        if drawable_syms:
            for _ in range(int(rng.choice([1, 1, 2]))):
                sym = str(drawable_syms[int(rng.integers(0, len(drawable_syms)))])
                li = int(rng.integers(0, len(lines)))
                ln = lines[li]
                if len(ln) >= 4:
                    pos = int(rng.integers(2, len(ln) - 1))
                    lines[li] = ln[:pos] + sym + ln[pos:]

    vertical = bool(rng.random() < 0.08)
    max_len = max(len(ln) for ln in lines)
    font_size = int(rng.integers(22, 38)) if max_len >= 24 else int(rng.integers(26, 46))

    bg_roll = float(rng.random())
    on_busy_art = bg_roll > 0.90
    dark_page = (not on_busy_art) and (bg_roll > 0.78)
    if dark_page:
        v_txt = int(rng.integers(220, 256))
        text_color = f"#{v_txt:02x}{v_txt:02x}{v_txt:02x}"
        bg_val = int(rng.integers(8, 36))
        bg_color = (bg_val, bg_val, bg_val)
    else:
        ink_v = int(rng.integers(0, 62)) if rng.random() < 0.45 else 0
        text_color = f"#{ink_v:02x}{ink_v:02x}{ink_v:02x}"
        bg_color = None

    params = {
        "lang": "ja",
        "font_path": font_path,
        "font_size": font_size,
        "vertical": vertical,
        "line_height": float(rng.uniform(0.10, 0.44)),
        "letter_spacing": float(rng.uniform(-0.03, 0.06)),
        "text_align": str(rng.choice(["left", "center", "right"], p=[0.78, 0.16, 0.06])),
        "text_color": text_color,
        "color_mode": "solid",
        "stroke_size": int(rng.choice([3, 4])) if on_busy_art else 0,
        "stroke_color": "white",
        "shadow_size": 0,
        "irregular": False,
        "sfx": False,
        "draw_bubble": False,
        "clean_bg": not on_busy_art,
        "clean_bg_color": bg_color,
        "clean_box_border": (not on_busy_art) and bool(rng.random() < 0.18),
        "clean_bleedthrough": (not on_busy_art) and (not dark_page) and bool(rng.random() < 0.22),
        "hard_manga_bg": on_busy_art and bool(rng.random() < 0.60),
        "no_invert": True,
        "min_effective_font": CLEAN_BG_MIN_EFFECTIVE_FONT,
        "max_size": 600,
    }
    return lines, params


def build_dakuten_sample(rng, font_path: str, dakuten_char: str, is_drawable_fn, real_dakuten: list[str] | None = None) -> tuple[list[str], dict]:
    d = dakuten_char
    p_mark = "\u309a" if d == "\u3099" else "゜"
    r = float(rng.random())

    if real_dakuten and r < 0.20:
        # Authentic human-written dakuten vocalizations harvested from Narou_NSFW / pixiv
        raw = str(real_dakuten[int(rng.integers(0, len(real_dakuten)))])
        lines = [raw]
    elif r < 0.52:
        # Short tight expressive non-standard dakuten (exact Image 2 style!)
        if rng.random() < 0.45:
            word = str(_JA_DAKUTEN_WORDS[int(rng.integers(0, len(_JA_DAKUTEN_WORDS)))]).format(d=d)
        else:
            lead_pool = _JA_NONSTANDARD_VOICED_HIGH if rng.random() < 0.85 else _JA_NONSTANDARD_VOICED_EXTRA
            lead = str(lead_pool[int(rng.integers(0, len(lead_pool)))])
            mark = p_mark if rng.random() < 0.07 else d
            n_tail = int(rng.choice([1, 2, 3, 4], p=[0.24, 0.40, 0.26, 0.10]))
            pieces = [lead + mark]
            for _ in range(n_tail):
                if rng.random() < 0.28:
                    ch2 = str(_JA_NONSTANDARD_VOICED_HIGH[int(rng.integers(0, len(_JA_NONSTANDARD_VOICED_HIGH)))])
                    pieces.append(ch2 + (d if rng.random() < 0.50 else ""))
                else:
                    pieces.append(str(_JA_TRAIL_SMALL[int(rng.integers(0, len(_JA_TRAIL_SMALL)))]))
            word = "".join(pieces) + str(_JA_END_PUNCT[int(rng.integers(0, len(_JA_END_PUNCT)))])
        lines = [word]
        if rng.random() < 0.24:
            extra = str(_JA_DAKUTEN_WORDS[int(rng.integers(0, len(_JA_DAKUTEN_WORDS)))]).format(d=d)
            lines.append(extra[: max(2, int(rng.integers(2, len(extra) + 1)))])
    elif r < 0.74:
        n_lines = int(rng.choice([1, 2, 3], p=[0.48, 0.36, 0.16]))
        lines = []
        for _ in range(n_lines):
            parts = []
            for _ in range(int(rng.integers(2, 5))):
                if rng.random() < 0.65:
                    parts.append(str(_JA_DAKUTEN_WORDS[int(rng.integers(0, len(_JA_DAKUTEN_WORDS)))]).format(d=d))
                else:
                    c = str(_JA_NONSTANDARD_VOICED_HIGH[int(rng.integers(0, len(_JA_NONSTANDARD_VOICED_HIGH)))])
                    parts.append(c + (d if rng.random() < 0.55 else ""))
            lines.append("".join(parts)[: int(rng.integers(4, 12))])
    elif r < 0.87:
        if rng.random() < 0.60:
            lines = [str(_JA_STANDARD_VOICED_SFX[int(rng.integers(0, len(_JA_STANDARD_VOICED_SFX)))])]
            if rng.random() < 0.25:
                lines.append(str(_JA_STANDARD_VOICED_SFX[int(rng.integers(0, len(_JA_STANDARD_VOICED_SFX)))]))
        else:
            pool = _JA_STANDARD_VOICED_HIRA if rng.random() < 0.55 else _JA_STANDARD_VOICED_KATA
            ln = "".join(str(pool[int(rng.integers(0, len(pool)))]) for _ in range(int(rng.integers(2, 6))))
            lines = [ln + str(_JA_TRAIL_SMALL[int(rng.integers(0, len(_JA_TRAIL_SMALL)))])]
    else:
        pair = str(_JA_CONFUSION_PAIRS[int(rng.integers(0, len(_JA_CONFUSION_PAIRS)))])
        if rng.random() < 0.45:
            pair += str(_JA_TRAIL_SMALL[int(rng.integers(0, len(_JA_TRAIL_SMALL)))])
        lines = [pair]

    clean_lines = []
    for ln in lines:
        s = "".join(ch for ch in ln if ch in VOICING_MARKS or is_drawable_fn(font_path, ch))
        if s:
            clean_lines.append(s)
    if not clean_lines:
        clean_lines = [f"ん{d}ッい"]

    vertical = bool(rng.random() < 0.80)
    style_roll = float(rng.random())
    draw_bubble = style_roll > 0.86
    use_sfx = (not draw_bubble) and (style_roll < 0.36)
    use_irregular = (not draw_bubble) and (not use_sfx)

    font_size = int(rng.integers(52, 124)) if not draw_bubble else int(rng.integers(32, 64))
    stroke_sz = 0 if draw_bubble else int(rng.choice([4, 5, 6, 7, 8, 9], p=[0.16, 0.24, 0.24, 0.16, 0.12, 0.08]))
    invert_white_text = (not draw_bubble) and (rng.random() < 0.18)

    params = {
        "lang": "ja",
        "font_path": font_path,
        "font_size": font_size,
        "vertical": vertical,
        "draw_bubble": draw_bubble,
        "tight": not draw_bubble,
        "clean_bg": False,
        "hard_manga_bg": (not draw_bubble) and bool(rng.random() < 0.65),
        "raw_scan_hard": (not draw_bubble) and bool(rng.random() < 0.60),
        "fg_occlude_prob": 0.22 if not draw_bubble else 0.08,
        "trunc_prob": 0.08,
        "no_invert": True,
        "text_color": "white" if invert_white_text else "black",
        "stroke_color": "black" if invert_white_text else "white",
        "stroke_size": stroke_sz,
        "color_mode": "solid",
        "sfx": use_sfx,
        "irregular": use_irregular,
        "small_kana": True,
        "dakuten_style": str(rng.choice(["auto", "brush", "pen", "font"], p=[0.35, 0.35, 0.15, 0.15])),
    }
    if use_irregular:
        params.update({
            "line_height": float(rng.uniform(0.10, 0.32)),
            "letter_spacing": float(rng.uniform(-0.22, 0.04)),
            "line_scale_min": 0.82,
            "line_scale_max": 1.30,
            "line_shift": 0.50,
            "char_jitter_prob": 0.85,
            "char_scale_min": 0.68,
            "char_scale_max": 1.45,
            "wobble": float(rng.uniform(0.06, 0.18)),
            "rotate": float(rng.uniform(2.0, 11.0)),
        })
    elif use_sfx:
        params.update({
            "letter_spacing": float(rng.uniform(-0.18, 0.06)),
            "sfx_curve": str(rng.choice(["straight", "arc", "diagonal", "wave"], p=[0.42, 0.26, 0.18, 0.14])),
            "sfx_curve_amp": float(rng.uniform(0.04, 0.20)),
            "sfx_rotate": float(rng.uniform(2.0, 12.0)),
            "sfx_block_rotate": float(rng.uniform(-18.0, 18.0)) if rng.random() < 0.42 else 0.0,
            "sfx_speedlines": bool(rng.random() < 0.20),
            "sfx_burst": bool(rng.random() < 0.08),
            "sfx_double_outline": bool(rng.random() < 0.22),
        })
    return clean_lines, params


_WORKER_STATE: dict = {}


def _worker_init(cfg: dict) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    if str(cfg["repo_dir"]) not in sys.path:
        sys.path.insert(0, str(cfg["repo_dir"]))
    from manga_ocr_dev.synthetic_data_generator.renderer import Renderer
    from manga_ocr_dev.synthetic_data_generator.utils import get_background_df

    bg_df = get_background_df(Path(cfg["bg_dir"]))
    _WORKER_STATE["cfg"] = cfg
    _WORKER_STATE["renderer"] = Renderer(background_df=bg_df)
    _WORKER_STATE["normalizer"] = import_repo_normalizer()
    _WORKER_STATE["rng"] = np.random.default_rng(int(cfg["seed"]) + (os.getpid() % 100_000) * 1009)


def _generate_one(task: tuple[int, str]) -> dict | None:
    from manga_ocr_dev.synthetic_data_generator.utils import is_drawable

    idx, kind = task
    cfg = _WORKER_STATE["cfg"]
    renderer = _WORKER_STATE["renderer"]
    normalizer = _WORKER_STATE["normalizer"]
    rng = _WORKER_STATE["rng"]

    for _ in range(16):
        try:
            if kind == "clean_paragraph":
                f_pool = cfg["clean_fonts"] if rng.random() < 0.85 else cfg["all_fonts"]
                font_path = f_pool[int(rng.integers(0, len(f_pool)))]
                lines, params = build_clean_paragraph(rng, cfg["sentences"], font_path, is_drawable)
                source = "edgecase_clean_paragraph"
            else:
                f_pool = cfg["expressive_fonts"] if rng.random() < 0.72 else cfg["all_fonts"]
                font_path = f_pool[int(rng.integers(0, len(f_pool)))]
                lines, params = build_dakuten_sample(
                    rng, font_path, cfg["dakuten_char"], is_drawable, cfg["real_dakuten"]
                )
                source = "edgecase_dakuten"

            img, out_params = renderer.render(lines, override_css_params=params)
            ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg["jpeg_quality"])])
            if not ok:
                continue
            image_id = f"{cfg['split']}_{cfg['run_id']}_{idx:07d}"
            if cfg["apply_normalize"]:
                transcription = cfg["join_str"].join(normalizer(ln) for ln in lines)
            else:
                transcription = cfg["join_str"].join(lines)
            if not transcription:
                continue
            return {
                "image": {"bytes": enc.tobytes(), "path": f"{image_id}.jpg"},
                "image_id": image_id,
                "transcription": transcription,
                "language": "ja",
                "vertical": bool(out_params.get("vertical", False)),
                "source": source,
            }
        except Exception:
            continue
    return None


def _make_preview_png(records: list[dict], out_path: Path) -> bytes | None:
    if not records:
        return None
    tiles = []
    for rec in records:
        arr = np.frombuffer(rec["image"]["bytes"], dtype=np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if im is None:
            continue
        pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        pil.thumbnail((420, 320), Image.LANCZOS)
        tiles.append(pil)
    if not tiles:
        return None
    cols = min(4, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    cell_w, cell_h, pad = 440, 340, 12
    grid = Image.new("RGB", (cols * cell_w + pad, rows * cell_h + pad), (210, 210, 210))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * cell_w + (cell_w - pad - t.width) // 2
        y = pad + r * cell_h + (cell_h - pad - t.height) // 2
        grid.paste(t, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    return out_path.read_bytes()


def _write_hf_parquet_shard(rows: list[dict], shard_path: Path) -> None:
    from datasets import Dataset, Features, Image as HFImage, Value

    features = Features({
        "image": HFImage(decode=False),
        "image_id": Value("string"),
        "transcription": Value("string"),
        "language": Value("string"),
        "vertical": Value("bool"),
        "source": Value("string"),
    })
    data = {k: [r[k] for r in rows] for k in ("image", "image_id", "transcription", "language", "vertical", "source")}
    ds = Dataset.from_dict(data, features=features)
    tmp_path = shard_path.with_suffix(".parquet.tmp")
    ds.to_parquet(str(tmp_path))
    os.replace(tmp_path, shard_path)


def run_edgecase_pipeline(
    *,
    edgecase_mode: str,
    edgecase_paragraph_ratio: float,
    dakuten_char: str,
    line_join: str,
    apply_repo_normalize: bool,
    repo_dir: Path,
    resolved_fonts_root: Path,
    resolved_bg_dir: Path,
    cache_lines_dir: Path,
    out_path: Path,
    hf_dataset: str,
    hf_token: str | None,
    split: str,
    run_id: str,
    n_images: int,
    rows_per_shard: int,
    workers: int,
    seed: int,
    jpeg_quality: int,
    lines_path: str,
    resume: bool,
    override_previous: bool,
    override_scope: str,
    preview_n: int,
    skip_preview: bool,
    preview_only: bool,
    push_to_hub: bool,
    update_card: bool,
    dry_run_card: bool,
    commit_fn=None,
    run_script_fn=None,
) -> dict:
    import multiprocessing as mp
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi
    from tqdm import tqdm

    mode = edgecase_mode.strip().lower()
    if mode in ("both", "all", "edgecases"):
        p_ratio = float(min(1.0, max(0.0, edgecase_paragraph_ratio)))
    elif mode in ("clean_paragraph", "paragraph", "multiline"):
        p_ratio = 1.0
    elif mode in ("dakuten", "handakuten", "voicing"):
        p_ratio = 0.0
    else:
        raise ValueError(f"Unknown edgecase_mode={edgecase_mode!r}. Use 'all', 'clean_paragraph', or 'dakuten'.")

    normalizer = import_repo_normalizer()
    if dakuten_char == "auto":
        probe = normalizer("ん゛") if apply_repo_normalize else "ん゛"
        resolved_dakuten = probe[-1] if len(probe) >= 2 else "゛"
    else:
        resolved_dakuten = dakuten_char
        if dakuten_char == "゛" and apply_repo_normalize:
            apply_repo_normalize = False

    join_str = detect_line_join(repo_dir, "ja", line_join)
    print(
        f"Edge-case pipeline: mode={mode}, paragraph_ratio={p_ratio:.2f}, "
        f"dakuten_char={resolved_dakuten!r} (U+{ord(resolved_dakuten):04X}), "
        f"apply_repo_normalize={apply_repo_normalize}, join={join_str!r}, "
        f"n_images={n_images:,}, workers={workers}"
    )

    sentences, real_dakuten = load_ja_corpora(
        cache_lines_dir, lines_path=lines_path, commit_fn=commit_fn
    )
    all_fonts, clean_fonts, expr_fonts = categorize_ja_fonts(resolved_fonts_root)
    print(
        f"Fonts ready: total={len(all_fonts)}, clean_paragraph={len(clean_fonts)}, "
        f"expressive_dakuten={len(expr_fonts)}"
    )

    worker_cfg = {
        "repo_dir": str(repo_dir),
        "bg_dir": str(resolved_bg_dir),
        "sentences": sentences,
        "real_dakuten": real_dakuten,
        "all_fonts": all_fonts,
        "clean_fonts": clean_fonts,
        "expressive_fonts": expr_fonts,
        "seed": seed,
        "jpeg_quality": jpeg_quality,
        "dakuten_char": resolved_dakuten,
        "join_str": join_str,
        "apply_normalize": apply_repo_normalize,
        "run_id": run_id,
        "split": split,
    }

    def _kind_for(i: int) -> str:
        if p_ratio >= 1.0:
            return "clean_paragraph"
        if p_ratio <= 0.0:
            return "dakuten"
        return "clean_paragraph" if (i % 1000) < int(round(p_ratio * 1000)) else "dakuten"

    preview_bytes = None
    preview_filename = None
    if preview_only or not skip_preview:
        _worker_init(worker_cfg)
        prev_recs = []
        for i in range(max(2, preview_n)):
            kind = "clean_paragraph" if (p_ratio >= 1.0 or (p_ratio > 0.0 and i % 2 == 0)) else "dakuten"
            rec = _generate_one((i, kind))
            if rec is not None:
                prev_recs.append(rec)
                print(f"  [preview {i}] {rec['source']}: {rec['transcription'][:55]!r}")
        preview_png = out_path / f"preview-{run_id}.png"
        preview_bytes = _make_preview_png(prev_recs, preview_png)
        if preview_bytes is not None:
            preview_filename = preview_png.name
    if preview_only:
        return {
            "run_id": run_id,
            "split": split,
            "language": "ja",
            "edgecase_mode": mode,
            "preview_only": True,
            "preview_filename": preview_filename,
            "preview_bytes": preview_bytes,
        }

    data_dir = out_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    n_shards = (n_images + rows_per_shard - 1) // rows_per_shard
    ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")

    with ctx.Pool(processes=max(1, workers), initializer=_worker_init, initargs=(worker_cfg,)) as pool:
        for shard_idx in range(n_shards):
            shard_path = data_dir / f"{split}-append-{run_id}-{shard_idx:05d}.parquet"
            start_i = shard_idx * rows_per_shard
            target_n = min(rows_per_shard, n_images - start_i)
            if resume and shard_path.is_file():
                try:
                    existing = pq.ParquetFile(shard_path).metadata.num_rows
                    if existing >= target_n:
                        print(f"Resuming: skipping existing shard {shard_path.name} ({existing:,} rows)")
                        continue
                except Exception:
                    pass

            tasks = [(start_i + k, _kind_for(start_i + k)) for k in range(target_n)]
            rows = []
            for rec in tqdm(
                pool.imap_unordered(_generate_one, tasks, chunksize=32),
                total=len(tasks),
                desc=f"Shard {shard_idx + 1}/{n_shards}",
                unit="img",
            ):
                if rec is not None:
                    rows.append(rec)

            extra_i = n_images + shard_idx * 10_000
            while len(rows) < target_n:
                need = target_n - len(rows)
                retry_tasks = [(extra_i + k, _kind_for(extra_i + k)) for k in range(need)]
                extra_i += need
                for rec in pool.map(_generate_one, retry_tasks):
                    if rec is not None:
                        rows.append(rec)
                        if len(rows) >= target_n:
                            break

            rows.sort(key=lambda r: r["image_id"])
            _write_hf_parquet_shard(rows, shard_path)
            print(f"Wrote {shard_path.name}: {len(rows):,} rows ({shard_path.stat().st_size / 1e6:.1f} MB)")
            if commit_fn is not None:
                commit_fn()

    shards = sorted(data_dir.glob(f"{split}-append-{run_id}-*.parquet"))
    total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in shards)
    total_size_mb = sum(p.stat().st_size for p in shards) / 1e6
    print(f"Validated {len(shards)} edge-case shard(s): {total_rows:,} rows, {total_size_mb:.1f} MB total.")

    rows_to_add = 0
    if not push_to_hub:
        print(f"push_to_hub is False. Local shards kept at {data_dir}")
    else:
        if not hf_token:
            raise RuntimeError("push_to_hub is True but HF_TOKEN is missing.")
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=hf_dataset, repo_type="dataset", private=True, exist_ok=True)
        remote_files = set(api.list_repo_files(hf_dataset, repo_type="dataset"))
        delete_patterns = None
        if override_previous:
            delete_patterns = (
                f"data/{split}-append-*.parquet"
                if override_scope == "split"
                else f"data/{split}-append-{run_id}-*.parquet"
            )
            new_shards = shards
        else:
            new_shards = [p for p in shards if f"data/{p.name}" not in remote_files]
        rows_to_add = sum(pq.ParquetFile(p).metadata.num_rows for p in new_shards)
        if new_shards:
            print(f"Uploading {len(new_shards)} edge-case shard(s) ({rows_to_add:,} rows) to {hf_dataset}...")
            api.upload_folder(
                repo_id=hf_dataset,
                repo_type="dataset",
                folder_path=str(data_dir),
                path_in_repo="data",
                allow_patterns=[p.name for p in new_shards],
                delete_patterns=delete_patterns,
                commit_message=f"Add {rows_to_add:,} edge-case synthetic crops ({split}, {mode})",
            )

    if push_to_hub and update_card and run_script_fn is not None and (rows_to_add or (override_previous and override_scope == "split")):
        card_script = repo_dir / "tools" / "append_hayai_synth.py"
        card_flag = "--set" if (override_previous and override_scope == "split") else "--add"
        card_val = total_rows if (override_previous and override_scope == "split") else rows_to_add
        card_cmd = ["update-card", "--repo-id", hf_dataset, card_flag, f"{split}={card_val}", "--token", hf_token]
        if dry_run_card:
            card_cmd.append("--dry-run")
        try:
            run_script_fn(card_script, card_cmd)
        except Exception as e:
            print(f"Dataset card update notice: {e}")

    if shards:
        sample_row = pq.read_table(shards[0]).to_pylist()[0]
        print("Sample Edge-Case Shard Record:")
        print(f"  image_id:      {sample_row['image_id']}")
        print(f"  transcription: {sample_row['transcription']!r}")
        print(f"  language:      {sample_row['language']}")
        print(f"  vertical:      {sample_row['vertical']}")
        print(f"  source:        {sample_row['source']}")
        print(f"  image bytes:   {len(sample_row['image']['bytes'])} bytes")

    return {
        "run_id": run_id,
        "split": split,
        "language": "ja",
        "edgecase_mode": mode,
        "shards_generated": len(shards),
        "total_rows": total_rows,
        "rows_uploaded": rows_to_add,
        "total_size_mb": round(total_size_mb, 2),
        "preview_filename": preview_filename,
        "preview_bytes": preview_bytes,
    }
