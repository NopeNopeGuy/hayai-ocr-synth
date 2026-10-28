"""Upright glyphs using one rotation estimate per line.

Real handwriting cut from a slanted line carries the line's slant.  The article
rotates every glyph of a line by the same angle, preserving each glyph's own
natural slant relative to the line.  We use the vision model's per-line
``tilt_deg`` when available and fall back to a PCA estimate from the glyph
centres.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from manga_ocr_dev.real_kana.config import KanaConfig, ensure_bank_dirs
from manga_ocr_dev.real_kana.segment import load_stage_records
from manga_ocr_dev.real_kana.utils import (
    atomic_write_json,
    estimate_stroke_width,
    load_image,
    read_json,
    sha256_file,
    trim_alpha,
)

log = logging.getLogger(__name__)


def estimate_tilt_from_centres(points: Iterable[tuple[float, float]], vertical: bool | None = None) -> float:
    """Return the PIL rotation that makes the point sequence upright."""

    pts = np.asarray(list(points), dtype=np.float64)
    if pts.shape[0] < 2:
        return 0.0
    centre = pts.mean(axis=0)
    centred = pts - centre
    # Principal axis.  The sign of the eigenvector is irrelevant because the
    # angle is reduced to the line's orientation, not a direction.
    cov = centred.T @ centred
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    dx, dy = float(axis[0]), float(axis[1])
    if vertical is None:
        spread = pts.max(axis=0) - pts.min(axis=0)
        vertical = spread[1] >= spread[0]
    if vertical:
        # Ensure the vector points down; atan2(-dx, dy) is the PIL rotation.
        if dy < 0:
            dx, dy = -dx, -dy
        angle = math.degrees(math.atan2(-dx, dy))
    else:
        if dx < 0:
            dx, dy = -dx, -dy
        angle = math.degrees(math.atan2(dy, dx))
    # Normalise to [-45, 45]: the reading axis has no forward direction.
    while angle > 45:
        angle -= 90
    while angle < -45:
        angle += 90
    return float(angle)


def load_angle_overrides(path: str | Path | None) -> dict[str, float]:
    """CSV with ``glyph_id,angle`` or ``crop_id,line_index,angle``."""

    if not path:
        return {}
    out: dict[str, float] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh):
            if not row or row[0].strip().lower() in {"glyph_id", "crop_id"}:
                continue
            try:
                if len(row) == 2:
                    out[row[0].strip()] = float(row[1])
                elif len(row) >= 3:
                    out[f"{row[0].strip()}:{row[1].strip()}"] = float(row[2])
            except ValueError:
                continue
    return out


def _line_angles(
    records: list[dict[str, Any]],
    *,
    use_vision: bool,
    overrides: dict[str, float],
) -> dict[tuple[str, int], float]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault((str(rec.get("crop_id")), int(rec.get("line_index", 0))), []).append(rec)
    angles: dict[tuple[str, int], float] = {}
    for key, rows in groups.items():
        crop_id, line_index = key
        override = overrides.get(f"{crop_id}:{line_index}")
        if override is not None:
            angles[key] = override
            continue
        if use_vision:
            vals = [float(r.get("line_tilt_deg", 0.0) or 0.0) for r in rows]
            angles[key] = float(np.clip(np.median(vals), -45, 45))
            continue
        pts = []
        for r in rows:
            x0, y0, x1, y1 = r["box"]
            pts.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
        angles[key] = estimate_tilt_from_centres(pts)
    return angles


def upright_candidates(
    cfg: KanaConfig | None = None,
    *,
    limit: int | None = None,
    resume: bool = True,
    use_vision_tilt: bool = True,
    angles_csv: str | Path | None = None,
    shard: int = 0,
    shards: int = 1,
) -> dict[str, int]:
    cfg = cfg or KanaConfig()
    shards = max(1, int(shards))
    shard = int(shard)
    if not 0 <= shard < shards:
        raise ValueError(f"shard must be in [0, {shards}), got {shard}")
    paths = ensure_bank_dirs(cfg.root)
    records = [r for r in load_stage_records(cfg, "upscaled") if r.get("status") == "upscaled"]
    # Only rotate complete lines; a single upscaled glyph still uses vision tilt.
    overrides = load_angle_overrides(angles_csv)
    angles = _line_angles(records, use_vision=use_vision_tilt, overrides=overrides)
    stats = {"glyphs": 0, "rotated": 0, "skipped": 0, "errors": 0}
    per_crop: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        per_crop.setdefault(str(rec.get("crop_id") or "unknown"), []).append(rec)

    # Final glyphs from an earlier run may be stale (a verifier removed or
    # relabelled a crop).  Index them so the cleanup below can remove orphans.
    existing_by_crop: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for sidecar_path in paths["glyphs"].glob("*/*.json"):
        data = read_json(sidecar_path, default={}) or {}
        if isinstance(data, dict):
            existing_by_crop.setdefault(str(data.get("crop_id") or ""), []).append((sidecar_path, data))

    for index, (crop_id, rows) in enumerate(per_crop.items()):
        if shards > 1 and index % shards != shard:
            continue
        out_dir = paths["glyphs"]
        out_records: list[dict[str, Any]] = []
        written: dict[str, Path] = {}
        for rec in rows:
            if limit is not None and stats["glyphs"] >= limit:
                break
            try:
                src = cfg.root / rec["upscaled_path"]
                image = load_image(src, "RGBA")
            except Exception as exc:  # noqa: BLE001
                log.warning("cannot load %s: %s", rec.get("glyph_id"), exc)
                stats["errors"] += 1
                continue
            key = (str(rec.get("crop_id")), int(rec.get("line_index", 0)))
            angle = angles.get(key, 0.0)
            if abs(angle) > 1e-4:
                image = image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0))
                image = trim_alpha(image, threshold=4, pad=1)
                stats["rotated"] += 1
            label = str(rec.get("label") or "")
            if not label:
                stats["skipped"] += 1
                continue
            label_dir = out_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            glyph_id = str(rec["glyph_id"])
            glyph_path = label_dir / f"{glyph_id}.png"
            image.save(glyph_path)
            written[glyph_id] = glyph_path
            alpha = np.asarray(image.getchannel("A"))
            stroke = estimate_stroke_width(alpha)
            sidecar = {
                **rec,
                "status": "glyph",
                "glyph_path": str(glyph_path.relative_to(cfg.root)),
                "angle_applied": round(float(angle), 4),
                "stroke_width": round(float(stroke), 4),
                "width": int(image.width),
                "height": int(image.height),
                "sha256": sha256_file(glyph_path),
                "derived": False,
            }
            sidecar.pop("cleaned_path", None)
            sidecar.pop("upscaled_path", None)
            atomic_write_json(label_dir / f"{rec['glyph_id']}.json", sidecar)
            out_records.append(sidecar)
            stats["glyphs"] += 1
        if out_records:
            atomic_write_json(paths["glyphs"] / f".{crop_id}.records.json", out_records)

        valid_ids = {str(rec.get("glyph_id")) for rec in rows}
        for sidecar_path, data in existing_by_crop.get(crop_id, []):
            glyph_id = str(data.get("glyph_id") or sidecar_path.stem)
            png_path = sidecar_path.with_suffix(".png")
            if glyph_id in written:
                # Same glyph, new location: only the freshly written copy stays.
                if png_path.resolve() == written[glyph_id].resolve():
                    continue
            elif glyph_id in valid_ids:
                # A glyph limit stopped before this row; keep the old copy.
                continue
            try:
                sidecar_path.unlink(missing_ok=True)
                png_path.unlink(missing_ok=True)
            except OSError:
                pass
    return stats
