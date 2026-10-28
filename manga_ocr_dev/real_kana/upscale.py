"""Mild upscaling and artefact checks for real kana glyphs.

The reference pipeline used Real-ESRGAN x4 and then reviewed every glyph.  The
user asked for less aggressive upscaling, so this stage defaults to x2 and
leaves glyphs whose short side is already 48 px alone.  Real-ESRGAN is used
only when the package and a model file are available; Lanczos is the fallback.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from manga_ocr_dev.real_kana.config import KanaConfig, ensure_bank_dirs
from manga_ocr_dev.real_kana.segment import iter_stage_groups
from manga_ocr_dev.real_kana.utils import atomic_write_json, load_image, resize_rgba

log = logging.getLogger(__name__)


def _white_core_fraction(rgba: Image.Image) -> float:
    arr = np.asarray(rgba.convert("RGBA"))
    alpha = arr[..., 3]
    ink = alpha >= 96
    if ink.sum() < 20:
        return 0.0
    # White ink is legitimate; the artefact check only applies to dark ink.
    ink_luma = float(np.median(arr[..., :3][ink] @ np.array([0.299, 0.587, 0.114])))
    if ink_luma >= 200:
        return 0.0
    # Erode so the anti-aliased boundary is not counted as a white core.
    core = cv2.erode(ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if core.sum() < 4:
        core = ink
    rgb = arr[..., :3].astype(np.int16)
    white = (rgb >= 245).all(axis=2)
    return float(white[core].mean())


def _speckle_count(rgba: Image.Image) -> int:
    alpha = (np.asarray(rgba.getchannel("A")) >= 96).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(alpha, connectivity=8)
    if n <= 1:
        return 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max()) if len(areas) else 0
    return int(np.sum(areas < max(2, 0.01 * largest)))


def _realesrgan_upscale(
    rgb: np.ndarray,
    alpha: np.ndarray,
    scale: float,
    model_path: Path,
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import torch  # noqa: F401
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        return None
    if not model_path.is_file():
        return None
    try:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path=str(model_path),
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
        )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out, _ = upsampler.enhance(bgr, outscale=max(1.0, scale))
        rgb_out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        alpha_out = cv2.resize(alpha, (rgb_out.shape[1], rgb_out.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        return rgb_out, alpha_out
    except Exception as exc:  # noqa: BLE001
        log.warning("Real-ESRGAN failed (%s); falling back to Lanczos", exc)
        return None


def upscale_glyph(
    rgba: Image.Image,
    *,
    scale: float = 2.0,
    min_side: int = 48,
    realesrgan_model: str | Path | None = None,
) -> tuple[Image.Image, float, dict[str, Any]]:
    """Upscale a glyph and return ``(image, applied_scale, report)``."""

    rgba = rgba.convert("RGBA")
    w, h = rgba.size
    if min(w, h) >= min_side or scale <= 1.0:
        return rgba, 1.0, {"applied": 1.0, "method": "none", "white_core": round(_white_core_fraction(rgba), 5)}

    factor = min(float(scale), max(1.0, float(min_side) / max(1, min(w, h))))
    tw, th = max(1, round(w * factor)), max(1, round(h * factor))
    arr = np.asarray(rgba)
    method = "lanczos"
    if realesrgan_model:
        out = _realesrgan_upscale(arr[..., :3], arr[..., 3], factor, Path(realesrgan_model))
        if out is not None:
            rgb, alpha = out
            if (rgb.shape[1], rgb.shape[0]) != (tw, th):
                rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
                alpha = cv2.resize(alpha, (tw, th), interpolation=cv2.INTER_AREA)
            rgba = Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), mode="RGBA")
            method = "realesrgan"
        else:
            rgba = resize_rgba(rgba, (tw, th))
    else:
        rgba = resize_rgba(rgba, (tw, th))  # type: ignore[assignment]

    white = _white_core_fraction(rgba)
    speckles = _speckle_count(rgba)
    flags = []
    if white > 0.015:
        flags.append("white_cores")
    if speckles > 4:
        flags.append("speckles")
    return rgba, float(factor), {
        "applied": round(float(factor), 4),
        "method": method,
        "white_core": round(white, 5),
        "speckles": speckles,
        "flags": flags,
    }


def upscale_candidates(
    cfg: KanaConfig | None = None,
    *,
    limit: int | None = None,
    resume: bool = True,
    realesrgan_model: str | None = None,
    shard: int = 0,
    shards: int = 1,
) -> dict[str, int]:
    cfg = cfg or KanaConfig()
    shards = max(1, int(shards))
    shard = int(shard)
    if not 0 <= shard < shards:
        raise ValueError(f"shard must be in [0, {shards}), got {shard}")
    model_path = realesrgan_model or os.environ.get("REAL_ESRGAN_MODEL")
    paths = ensure_bank_dirs(cfg.root)
    stats = {"sources": 0, "glyphs": 0, "skipped": 0, "flagged": 0, "errors": 0}
    for index, (clean_dir, records) in enumerate(iter_stage_groups(cfg, "cleaned")):
        if shards > 1 and index % shards != shard:
            continue
        if limit is not None and stats["sources"] >= limit:
            break
        out_dir = paths["upscaled"] / clean_dir.name
        meta_path = out_dir / "meta.json"
        if resume and meta_path.is_file():
            continue
        stats["sources"] += 1
        out_records: list[dict[str, Any]] = []
        for rec in records:
            if rec.get("status") != "cleaned":
                out_records.append(rec)
                stats["skipped"] += 1
                continue
            try:
                image = load_image(cfg.root / rec["cleaned_path"], "RGBA")
                image, factor, report = upscale_glyph(
                    image, scale=cfg.upscale, min_side=cfg.upscale_min_side, realesrgan_model=model_path
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("upscale failed for %s: %s", rec.get("glyph_id"), exc)
                stats["errors"] += 1
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{rec['glyph_id']}.png"
            image.save(out_path)
            total_scale = float(rec.get("upscale", 1.0)) * factor
            rec = {
                **rec,
                "status": "upscaled",
                "upscaled_path": str(out_path.relative_to(cfg.root)),
                "upscale": round(total_scale, 4),
                "upscale_report": report,
            }
            if report.get("flags"):
                rec["watch"] = True
                stats["flagged"] += 1
            out_records.append(rec)
            stats["glyphs"] += 1
        atomic_write_json(out_dir / "records.json", out_records)
        atomic_write_json(meta_path, {"crop_id": clean_dir.name, "glyphs": len(out_records)})
    return stats
