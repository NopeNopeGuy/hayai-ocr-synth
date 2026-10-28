"""Alpha extraction and component-level cleaning for real kana crops.

A crop is RGB pixels cut from a manga page.  This stage estimates the ink
colour and background, builds a soft alpha mask, and removes only whole
connected components (dirt, neighbour fragments, bubble specks).  It never
repaints or synthesises strokes.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

from manga_ocr_dev.real_kana.config import KanaConfig, ensure_bank_dirs
from manga_ocr_dev.real_kana.segment import iter_stage_groups
from manga_ocr_dev.real_kana.utils import (
    alpha_bbox,
    atomic_write_json,
    colour_separated_alpha,
    estimate_ink_rgb,
    estimate_stroke_width,
    load_image,
    mask_to_rgba,
    trim_alpha,
)

log = logging.getLogger(__name__)


def _component_table(alpha: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    ink = (alpha >= 96).astype(np.uint8)
    return cv2.connectedComponentsWithStats(ink, connectivity=8)


def _select_components(alpha: np.ndarray, cfg: KanaConfig) -> tuple[np.ndarray, dict[str, Any]]:
    n, labels, stats, cents = _component_table(alpha)
    if n <= 1:
        return np.zeros_like(alpha), {"components": 0, "kept": 0, "removed": 0}
    areas = stats[:, cv2.CC_STAT_AREA]
    main = int(np.argmax(areas[1:])) + 1
    main_area = max(1, int(areas[main]))
    h, w = alpha.shape
    main_cx, main_cy = cents[main]
    diag = float(np.hypot(w, h))
    keep: list[int] = [main]
    removed: list[int] = []
    neighbour_like = 0
    for i in range(1, n):
        if i == main:
            continue
        area = int(areas[i])
        x, y, cw, ch = [int(v) for v in stats[i, :4]]
        touches = x <= 0 or y <= 0 or x + cw >= w or y + ch >= h
        dist = float(np.hypot(cents[i][0] - main_cx, cents[i][1] - main_cy))
        if area <= 1:
            removed.append(i)
            continue
        if cfg.remove_border_dirt and touches and area < 0.40 * main_area:
            removed.append(i)
            continue
        if touches and area < 0.15 * main_area and dist > 0.55 * diag:
            removed.append(i)
            continue
        if area < 0.01 * main_area and dist > 0.35 * diag:
            removed.append(i)
            continue
        if dist > 0.90 * diag and area < 0.75 * main_area:
            neighbour_like += 1
        keep.append(i)

    mask = np.isin(labels, keep).astype(np.uint8)
    alpha2 = (alpha * mask).astype(np.uint8)
    return alpha2, {
        "components": int(n - 1),
        "kept": len(keep),
        "removed": len(removed),
        "main_area": main_area,
        "neighbour_like": neighbour_like,
    }


def clean_crop(rgb: np.ndarray, cfg: KanaConfig | None = None) -> tuple[Image.Image, dict[str, Any]]:
    """Return an RGBA glyph and a quality report.  Pure function."""

    cfg = cfg or KanaConfig()
    rgb = np.asarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    alpha, info = colour_separated_alpha(rgb)
    alpha, comp_info = _select_components(alpha, cfg)
    # Drop faint anti-aliasing dust after component filtering.
    alpha = np.where(alpha < 10, 0, alpha).astype(np.uint8)
    bbox = alpha_bbox(alpha, 4)
    flags: list[str] = []
    if bbox is None:
        return Image.new("RGBA", (1, 1)), {"flags": ["empty"], "components": comp_info}
    x0, y0, x1, y1 = bbox
    pad = 1
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    alpha = alpha[y0:y1, x0:x1]
    rgb_crop = rgb[y0:y1, x0:x1]
    coverage = float((alpha >= 96).mean()) if alpha.size else 0.0
    if coverage < cfg.min_alpha_coverage:
        flags.append("too_little_ink")
    if coverage > cfg.max_alpha_coverage:
        flags.append("too_much_ink")
    if comp_info.get("components", 0) > cfg.max_components:
        flags.append("too_many_components")
    if comp_info.get("neighbour_like", 0) > 0:
        flags.append("possible_neighbour")
    if alpha.shape[0] < 3 or alpha.shape[1] < 3:
        flags.append("too_small")

    # Ink at a border usually means a cut stroke or a bit of a neighbour.
    border = np.concatenate([alpha[0, :], alpha[-1, :], alpha[:, 0], alpha[:, -1]])
    border_ink = float((border >= 96).mean()) if border.size else 0.0
    if border_ink > 0.04:
        flags.append("border_ink")

    ink_rgb = estimate_ink_rgb(Image.fromarray(rgb_crop), alpha)
    stroke = estimate_stroke_width(alpha)
    if stroke < 0.8:
        flags.append("thin_stroke")
    if stroke > max(3.0, min(alpha.shape) * 0.35):
        flags.append("blob")

    rgba = mask_to_rgba(alpha, ink_rgb, colour=True, rgb=rgb_crop)
    rgba = trim_alpha(rgba, threshold=4, pad=1)
    report = {
        **info,
        **comp_info,
        "flags": flags,
        "coverage": round(coverage, 4),
        "border_ink": round(border_ink, 4),
        "stroke_width": round(float(stroke), 3),
        "ink_rgb": list(ink_rgb),
        "width": int(rgba.width),
        "height": int(rgba.height),
    }
    return rgba, report


def clean_candidates(
    cfg: KanaConfig | None = None,
    *,
    limit: int | None = None,
    resume: bool = True,
    shard: int = 0,
    shards: int = 1,
) -> dict[str, int]:
    cfg = cfg or KanaConfig()
    shards = max(1, int(shards))
    shard = int(shard)
    if not 0 <= shard < shards:
        raise ValueError(f"shard must be in [0, {shards}), got {shard}")
    paths = ensure_bank_dirs(cfg.root)
    stats = {"sources": 0, "glyphs": 0, "quarantined": 0, "errors": 0}
    for index, (raw_dir, records) in enumerate(iter_stage_groups(cfg, "raw")):
        if shards > 1 and index % shards != shard:
            continue
        if limit is not None and stats["sources"] >= limit:
            break
        out_dir = paths["cleaned"] / raw_dir.name
        meta_path = out_dir / "meta.json"
        if resume and meta_path.is_file():
            continue
        stats["sources"] += 1
        cleaned: list[dict[str, Any]] = []
        for rec in records:
            try:
                raw_path = cfg.root / rec["raw_path"]
                image = load_image(raw_path, "RGB")
                rgba, report = clean_crop(np.asarray(image), cfg)
            except Exception as exc:  # noqa: BLE001
                log.warning("clean failed for %s: %s", rec.get("glyph_id"), exc)
                stats["errors"] += 1
                continue
            if "empty" in report.get("flags", []) or "too_little_ink" in report.get("flags", []):
                rec = {**rec, "status": "quarantined", "clean_report": report}
                _write_quarantine(cfg, rec, rgba)
                stats["quarantined"] += 1
                cleaned.append(rec)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            glyph_path = out_dir / f"{rec['glyph_id']}.png"
            rgba.save(glyph_path)
            rec = {
                **rec,
                "status": "cleaned",
                "cleaned_path": str(glyph_path.relative_to(cfg.root)),
                "clean_report": report,
                "stroke_width": report.get("stroke_width", 0.0),
                "ink_rgb": report.get("ink_rgb", [0, 0, 0]),
                "upscale": 1.0,
            }
            if "border_ink" in report.get("flags", []) or "possible_neighbour" in report.get("flags", []):
                rec["watch"] = True
            cleaned.append(rec)
            stats["glyphs"] += 1
        atomic_write_json(out_dir / "records.json", cleaned)
        atomic_write_json(
            meta_path,
            {
                "crop_id": records[0].get("crop_id", raw_dir.name) if records else raw_dir.name,
                "kept": sum(1 for r in cleaned if r.get("status") == "cleaned"),
                "quarantined": sum(1 for r in cleaned if r.get("status") == "quarantined"),
            },
        )
    return stats


def _write_quarantine(cfg: KanaConfig, rec: dict[str, Any], image: Image.Image) -> None:
    out_dir = cfg.root / "quarantine" / "cleaned" / str(rec.get("crop_id") or "unknown")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        image.save(out_dir / f"{rec.get('glyph_id', 'glyph')}.png")
        atomic_write_json(out_dir / f"{rec.get('glyph_id', 'glyph')}.json", rec)
    except OSError:
        pass
