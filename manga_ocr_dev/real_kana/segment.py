"""Turn candidate crops into raw kana crops.

Detection and recognition use the local ``manga-ocr-charspot-small`` model.
The vision model is *not* used here; it is an optional final verifier over the
cleaned/upscaled glyphs (see :func:`verify_candidates`).  Passing an explicit
``client`` keeps the older vision-segmentation path available for tests and for
banks produced before the local model was adopted.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

from manga_ocr_dev.real_kana.config import KanaConfig, ensure_bank_dirs
from manga_ocr_dev.real_kana.contact import contact_sheet, overlay_boxes
from manga_ocr_dev.real_kana.utils import (
    atomic_write_json,
    iter_jsonl,
    load_image,
    read_json,
    safe_id,
)
from manga_ocr_dev.real_kana.vision import (
    REVIEW_SCHEMA,
    SEGMENT_SCHEMA,
    VisionClient,
    apply_review,
    normalise_segmentation,
    prepare_for_vision,
    review_prompt,
    segment_prompt,
)

log = logging.getLogger(__name__)

_PATH_KEYS = {
    "raw": ("raw_path",),
    "cleaned": ("cleaned_path", "raw_path"),
    "upscaled": ("upscaled_path", "cleaned_path", "raw_path"),
}


def _candidate_image(cfg: KanaConfig, rec: dict[str, Any]) -> Image.Image:
    path = Path(rec.get("image_path") or "")
    if not path.is_absolute():
        path = cfg.root / path
    if not path.is_file():
        raise FileNotFoundError(f"candidate image missing: {path}")
    return load_image(path, "RGB")


def _iter_candidates(cfg: KanaConfig) -> Iterator[dict[str, Any]]:
    index = cfg.root / "candidates" / "candidates.jsonl"
    yield from iter_jsonl(index)


def _records_path(cfg: KanaConfig, stage: str, crop_id: str) -> Path:
    return cfg.root / stage / safe_id(crop_id) / "records.json"


def load_stage_records(cfg: KanaConfig, stage: str) -> list[dict[str, Any]]:
    """Load all records of a stage, one records.json per source crop."""

    root = cfg.root / stage
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*/records.json")):
        rows = read_json(path, default=[]) or []
        if isinstance(rows, list):
            out.extend(r for r in rows if isinstance(r, dict))
    return out


def iter_stage_groups(cfg: KanaConfig, stage: str) -> Iterator[tuple[Path, list[dict[str, Any]]]]:
    root = cfg.root / stage
    if not root.is_dir():
        return
    for path in sorted(root.glob("*/records.json")):
        rows = read_json(path, default=[]) or []
        if isinstance(rows, list):
            yield path.parent, [r for r in rows if isinstance(r, dict)]


def segment_candidates(
    cfg: KanaConfig | None = None,
    client: VisionClient | None = None,
    *,
    segmenter: Any | None = None,
    limit: int | None = None,
    resume: bool = True,
    review: bool = False,
    shard: int = 0,
    shards: int = 1,
    log_every: int = 25,
) -> dict[str, int]:
    """Cut kana from harvested candidates with the local charspot model.

    Writes, per candidate, ``raw/<id>/source.png``, ``overlay.png``,
    ``records.json`` and one PNG per kana.  A candidate the model labels as
    printed or SFX is recorded with an empty glyph list and skipped.

    ``client`` is only for the legacy vision path (and tests); when it is
    ``None`` the local model is used.  ``review`` is a compatibility shortcut
    that verifies the raw crops immediately; the normal flow verifies after
    cleaning and upscaling instead.
    """

    cfg = cfg or KanaConfig()
    shards = max(1, int(shards))
    shard = int(shard)
    if not 0 <= shard < shards:
        raise ValueError(f"shard must be in [0, {shards}), got {shard}")
    use_local = client is None
    if use_local:
        if segmenter is None:
            from manga_ocr_dev.real_kana.local_ocr import CharspotOcr

            segmenter = CharspotOcr(
                model_dir=cfg.local_model_dir,
                backend=cfg.local_backend,
                threads=cfg.local_threads,
            )
        segmenter_name = str(getattr(segmenter, "model", "manga-ocr-charspot-small"))
    else:
        segmenter_name = str(getattr(client, "model", cfg.vision_model))

    paths = ensure_bank_dirs(cfg.root)
    raw_root = paths["raw"]
    stats = {"seen": 0, "segmented": 0, "skipped": 0, "glyphs": 0, "errors": 0}

    for index, cand in enumerate(_iter_candidates(cfg)):
        if shards > 1 and index % shards != shard:
            continue
        if limit is not None and stats["segmented"] >= limit:
            break
        stats["seen"] += 1
        crop_id = str(cand.get("crop_id") or "")
        if not crop_id:
            stats["skipped"] += 1
            continue
        out_dir = raw_root / safe_id(crop_id)
        meta_path = out_dir / "meta.json"
        if resume and meta_path.is_file():
            stats["skipped"] += 1
            continue
        try:
            image = _candidate_image(cfg, cand)
        except Exception as exc:  # noqa: BLE001
            log.warning("cannot load candidate %s: %s", crop_id, exc)
            stats["errors"] += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        image.save(out_dir / "source.png")
        transcription = str(cand.get("transcription") or "")
        try:
            if use_local:
                from manga_ocr_dev.real_kana.local_ocr import response_from_read

                read = segmenter.read(image)
                response = response_from_read(
                    read,
                    width=image.width,
                    height=image.height,
                    transcription=transcription,
                )
                parsed = normalise_segmentation(
                    response,
                    width=image.width,
                    height=image.height,
                    transcription=transcription,
                    cfg=cfg,
                    min_confidence=cfg.local_min_confidence,
                )
                model_image = image
                scale_x = scale_y = 1.0
            else:
                # The vision client may enlarge or shrink the image before
                # sending it.  Quote the sent size in the prompt and map boxes
                # back afterwards.
                model_image = prepare_for_vision(image, cfg.vision_upscale_max_side)
                scale_x = image.width / max(1, model_image.width)
                scale_y = image.height / max(1, model_image.height)
                response = client.generate_json(
                    segment_prompt(transcription, model_image.width, model_image.height),
                    [model_image],
                    SEGMENT_SCHEMA,
                )
                parsed = normalise_segmentation(
                    response,
                    width=model_image.width,
                    height=model_image.height,
                    transcription=transcription,
                    cfg=cfg,
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("segmentation failed for %s: %s", crop_id, exc)
            # Do not write meta.json on failure: resume uses it as a
            # completion marker, and a transient model error should be retried.
            atomic_write_json(
                out_dir / "error.json",
                {"crop_id": crop_id, "status": "segment_error", "error": str(exc)[:500]},
            )
            stats["errors"] += 1
            continue

        records: list[dict[str, Any]] = []
        order = 0
        for line in parsed["lines"]:
            for glyph in line["glyphs"]:
                mx0, my0, mx1, my1 = glyph["box"]
                x0 = max(0, min(image.width - 1, int(round(mx0 * scale_x))))
                y0 = max(0, min(image.height - 1, int(round(my0 * scale_y))))
                x1 = max(x0 + 1, min(image.width, int(round(mx1 * scale_x))))
                y1 = max(y0 + 1, min(image.height, int(round(my1 * scale_y))))
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                glyph_id = f"{safe_id(crop_id)}_l{glyph['line_index']}_g{order}"
                crop = image.crop((x0, y0, x1, y1))
                crop_path = out_dir / f"{glyph_id}.png"
                crop.save(crop_path)
                record = {
                    "glyph_id": glyph_id,
                    "crop_id": crop_id,
                    "image_id": cand.get("image_id", ""),
                    "transcription": transcription,
                    "label": glyph["label"],
                    "confidence": glyph["confidence"],
                    "box": [x0, y0, x1, y1],
                    "model_box": [mx0, my0, mx1, my1],
                    "model_size": [model_image.width, model_image.height],
                    "line_index": glyph["line_index"],
                    "line_tilt_deg": glyph["line_tilt_deg"],
                    "order": order,
                    "status": "raw",
                    "reviewed": False,
                    "sequence_match": parsed["sequence_match"],
                    "sequence_ok": parsed["sequence_ok"],
                    "segmenter": "charspot" if use_local else "vision",
                    "vision_model": segmenter_name,
                    "source_image": str((out_dir / "source.png").relative_to(cfg.root)),
                    "raw_path": str(crop_path.relative_to(cfg.root)),
                    "source_dataset": cand.get("dataset", ""),
                    "source_revision": cand.get("revision", ""),
                }
                records.append(record)
                order += 1

        try:
            overlay = overlay_boxes(
                image,
                [r["box"] for r in records],
                labels=[r["label"] for r in records],
            )
            overlay.save(out_dir / "overlay.png")
            if records:
                items = [
                    (load_image(cfg.root / r["raw_path"], "RGBA"), str(i), r["label"]) for i, r in enumerate(records)
                ]
                contact_sheet(items, cols=min(8, max(1, len(items))), cell=96, title=f"{crop_id}").save(
                    out_dir / "crops.png"
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("could not build contact sheet for %s: %s", crop_id, exc)

        atomic_write_json(
            meta_path,
            {
                "crop_id": crop_id,
                "status": parsed["status"],
                "transcription": transcription,
                "sequence_match": parsed["sequence_match"],
                "sequence_ok": parsed["sequence_ok"],
                "expected_kana": parsed["expected_kana"],
                "warnings": parsed["warnings"],
                "segmenter": "charspot" if use_local else "vision",
                "vision_model": segmenter_name,
            },
        )
        atomic_write_json(out_dir / "records.json", records)
        stats["glyphs"] += len(records)
        if records:
            stats["segmented"] += 1
        else:
            stats["skipped"] += 1
        if stats["seen"] % log_every == 0:
            log.info("segmentation: %s", stats)

    if review:
        if client is None:
            from manga_ocr_dev.real_kana.vision import build_vision

            client = build_vision(cfg)
        verify_candidates(cfg, client, stage="raw", limit=limit, resume=resume, uncertain_only=False)
    return stats


def _stage_image_path(rec: dict[str, Any], stage: str) -> str | None:
    for key in _PATH_KEYS.get(stage, _PATH_KEYS["upscaled"]):
        value = rec.get(key)
        if value:
            return str(value)
    return None


def _needs_verification(rec: dict[str, Any], cfg: KanaConfig) -> bool:
    """Conservative triage: only spend a verification call on doubtful crops."""

    if bool(rec.get("reviewed")):
        return False
    if rec.get("watch") or rec.get("review_changed"):
        return True
    try:
        confidence = float(rec.get("confidence", 1.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < cfg.verify_confidence:
        return True
    interesting = {str(flag) for flag in cfg.verify_clean_flags}
    for key in ("clean_report", "upscale_report"):
        report = rec.get(key) or {}
        if not isinstance(report, dict):
            continue
        if interesting.intersection(str(flag) for flag in report.get("flags") or []):
            return True
    return False


def _unlink_record_files(cfg: KanaConfig, rec: dict[str, Any]) -> None:
    for key in ("upscaled_path", "cleaned_path", "raw_path"):
        value = rec.get(key)
        if not value:
            continue
        try:
            (cfg.root / str(value)).unlink(missing_ok=True)
        except OSError:
            pass


def verify_candidates(
    cfg: KanaConfig | None = None,
    client: VisionClient | None = None,
    *,
    stage: str = "upscaled",
    limit: int | None = None,
    resume: bool = True,
    uncertain_only: bool = True,
) -> dict[str, int]:
    """Use the vision model as a final verifier over already-cut glyphs.

    This is deliberately the *last* vision pass: the local model did detection
    and recognition, cleaning removed background, and upscaling made the crop
    readable.  The verifier only corrects/removes glyphs; the stage status is
    preserved so downstream stages still see ``status="upscaled"``.
    """

    if stage not in _PATH_KEYS:
        raise ValueError(f"unknown verification stage: {stage}")
    cfg = cfg or KanaConfig()
    if client is None:
        from manga_ocr_dev.real_kana.vision import build_vision

        client = build_vision(cfg)
    stats = {
        "seen": 0,
        "reviewed": 0,
        "verified": 0,
        "glyphs": 0,
        "removed": 0,
        "changed": 0,
        "no_targets": 0,
        "skipped": 0,
        "errors": 0,
    }
    stage_root = cfg.root / stage
    for records_path in sorted(stage_root.glob("*/records.json")):
        out_dir = records_path.parent
        if limit is not None and stats["reviewed"] >= limit:
            break
        stats["seen"] += 1
        report_path = out_dir / f"verify_{stage}.json"
        if resume and report_path.is_file():
            previous = read_json(report_path, default={}) or {}
            # A doubtful-only pass that found nothing must not block a later
            # explicit `--all` audit of the same group.
            if uncertain_only or not isinstance(previous, dict) or not previous.get("uncertain_only"):
                stats["skipped"] += 1
                continue
        records = read_json(records_path, default=[]) or []
        if not isinstance(records, list) or not records:
            continue
        records = [r for r in records if isinstance(r, dict) and r.get("glyph_id")]
        targets = [r for r in records if not uncertain_only or _needs_verification(r, cfg)]
        if not targets:
            stats["no_targets"] += 1
            atomic_write_json(
                report_path,
                {
                    "stage": stage,
                    "calls": 0,
                    "uncertain_only": uncertain_only,
                    "kept": len(records),
                    "original": len(records),
                    "note": "no targets",
                },
            )
            continue

        usable: list[dict[str, Any]] = []
        items = []
        current = []
        for rec in targets:
            rel = _stage_image_path(rec, stage)
            if not rel:
                continue
            path = cfg.root / rel
            if not path.is_file():
                log.debug("verifier skipped missing crop %s", rel)
                continue
            usable.append(rec)
            index = len(usable) - 1
            items.append((load_image(path, "RGBA"), str(index), str(rec.get("label") or "")))
            current.append((index, str(rec.get("label") or ""), f"{stage} crop"))
        if not items:
            stats["no_targets"] += 1
            atomic_write_json(
                report_path,
                {
                    "stage": stage,
                    "calls": 0,
                    "uncertain_only": uncertain_only,
                    "kept": len(records),
                    "original": len(records),
                    "note": "no crop files",
                },
            )
            continue

        target_ids = {str(rec.get("glyph_id")) for rec in usable}
        try:
            sheet = contact_sheet(
                items,
                cols=min(8, max(1, len(items))),
                cell=112,
                title=str(records[0].get("crop_id") or out_dir.name),
            )
            prompt = review_prompt(str(records[0].get("transcription") or ""), current)
            response = client.generate_json(prompt, [sheet], REVIEW_SCHEMA)
            reviewed, warnings = apply_review(response, usable)
        except Exception as exc:  # noqa: BLE001
            log.warning("verification failed for %s: %s", out_dir.name, exc)
            stats["errors"] += 1
            continue

        kept = {str(rec.get("glyph_id")): rec for rec in reviewed}
        before = {str(rec.get("glyph_id")): rec for rec in records}
        out_records: list[dict[str, Any]] = []
        removed = 0
        changed = 0
        for rec in records:
            glyph_id = str(rec.get("glyph_id"))
            if glyph_id not in target_ids:
                out_records.append(rec)
                continue
            if glyph_id not in kept:
                _unlink_record_files(cfg, rec)
                removed += 1
                continue
            updated = kept[glyph_id]
            if updated.get("label") != before[glyph_id].get("label"):
                changed += 1
            updated["reviewed"] = True
            updated["status"] = rec.get("status", stage)
            updated["verify_stage"] = stage
            updated["verify_warnings"] = warnings
            out_records.append(updated)

        stats["removed"] += removed
        stats["changed"] += changed
        stats["glyphs"] += len(usable)
        stats["reviewed"] += 1
        stats["verified"] += 1
        atomic_write_json(records_path, out_records)
        atomic_write_json(
            report_path,
            {
                "stage": stage,
                "calls": 1,
                "uncertain_only": uncertain_only,
                "checked": len(usable),
                "kept": len(kept),
                "removed": removed,
                "changed": changed,
                "original": len(records),
                "warnings": warnings,
            },
        )
        sheet.save(out_dir / f"verify_{stage}_sheet.png")
    return stats


def review_candidates(
    cfg: KanaConfig | None = None,
    client: VisionClient | None = None,
    *,
    limit: int | None = None,
    resume: bool = True,
) -> dict[str, int]:
    """Backwards-compatible alias for verifying raw crops.

    New runs should verify the cleaned/upscaled stage instead.
    """

    return verify_candidates(cfg, client, stage="raw", limit=limit, resume=resume, uncertain_only=False)
