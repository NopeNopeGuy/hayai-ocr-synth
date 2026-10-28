"""Harvest candidate handwritten crops from AnimeText-OCR."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from manga_ocr_dev.real_kana.config import ANIMETEXT_OCR_REPO, ANIMETEXT_OCR_REVISION, KanaConfig, ensure_bank_dirs
from manga_ocr_dev.real_kana.handwriting import HandwritingScorer
from manga_ocr_dev.real_kana.utils import (
    append_jsonl,
    has_letters,
    image_to_png_bytes,
    is_likely_sfx,
    iter_jsonl,
    kana_sequence,
    safe_id,
    sha256_bytes,
)

log = logging.getLogger(__name__)


@dataclass
class HarvestStats:
    scanned: int = 0
    decoded: int = 0
    language_rejected: int = 0
    length_rejected: int = 0
    kana_rejected: int = 0
    confidence_rejected: int = 0
    side_rejected: int = 0
    sfx_rejected: int = 0
    duplicate_rejected: int = 0
    handwriting_rejected: int = 0
    saved: int = 0
    errors: int = 0
    per_source: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "decoded": self.decoded,
            "language_rejected": self.language_rejected,
            "length_rejected": self.length_rejected,
            "kana_rejected": self.kana_rejected,
            "confidence_rejected": self.confidence_rejected,
            "side_rejected": self.side_rejected,
            "sfx_rejected": self.sfx_rejected,
            "duplicate_rejected": self.duplicate_rejected,
            "handwriting_rejected": self.handwriting_rejected,
            "saved": self.saved,
            "errors": self.errors,
            "per_source": self.per_source,
        }


def _iter_stream(revision: str) -> Iterator[dict[str, Any]]:
    try:
        import datasets
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("harvesting needs the 'datasets' package: pip install datasets") from exc

    ds = datasets.load_dataset(
        ANIMETEXT_OCR_REPO,
        split="train",
        streaming=True,
        revision=revision,
    )
    try:
        yield from ds
    finally:
        # The streaming parquet reader sometimes raises while being closed.
        try:
            del ds
        except Exception as exc:  # noqa: BLE001
            log.debug("ignoring error while closing AnimeText-OCR stream: %s", exc)


def _passes_metadata(row: dict[str, Any], cfg: KanaConfig, stats: HarvestStats) -> str | None:
    text = str(row.get("transcription") or "")
    lang = str(row.get("language") or "")
    if lang != "ja":
        stats.language_rejected += 1
        return None
    if not (cfg.min_chars <= len(text) <= cfg.max_chars):
        stats.length_rejected += 1
        return None
    if not has_letters(text):
        stats.length_rejected += 1
        return None
    kana = kana_sequence(text)
    if len(kana) < cfg.min_kana:
        stats.kana_rejected += 1
        return None
    try:
        conf = min(float(row.get("seq_confidence", 1.0)), float(row.get("min_token_confidence", 1.0)))
    except (TypeError, ValueError):
        conf = 1.0
    if conf < cfg.min_ocr_confidence:
        stats.confidence_rejected += 1
        return None
    if cfg.exclude_likely_sfx and is_likely_sfx(text, cfg.sfx_max_katakana_chars):
        stats.sfx_rejected += 1
        return None
    return text


def harvest_candidates(
    cfg: KanaConfig | None = None,
    *,
    limit: int | None = 5000,
    max_scan: int | None = 250_000,
    revision: str = ANIMETEXT_OCR_REVISION,
    score_threshold: float | None = None,
    head_path: str | Path | None = None,
    handwriting_backend: str = "auto",
    mercury_client: Any | None = None,
    resume: bool = True,
    log_every: int = 1000,
) -> HarvestStats:
    """Stream AnimeText-OCR and save usable candidate crops.

    Filtering follows the article: handwritten score, Japanese, OCR
    confidence, 3--40 characters with at least two kana, and a short side of
    at least 18 px.  Metadata filtering happens before image decoding; the
    handwriting score is computed only for rows that survive it.
    """

    cfg = cfg or KanaConfig()
    cfg.validate()
    paths = ensure_bank_dirs(cfg.root)
    images_dir = paths["candidates"] / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    index_path = paths["candidates"] / "candidates.jsonl"
    stats = HarvestStats()

    existing: set[str] = set()
    if resume and index_path.is_file():
        for row in iter_jsonl(index_path):
            cid = row.get("crop_id")
            if cid:
                existing.add(str(cid))
    log.info("resume: %d candidates already indexed", len(existing))

    scorer = HandwritingScorer(
        head_path,
        backend=handwriting_backend,
        mercury_client=mercury_client,
    )
    if score_threshold is not None:
        threshold = float(score_threshold)
    elif scorer.backend == "dino" and scorer.head is not None:
        # A trained head carries the operating point chosen with --threshold.
        threshold = float(scorer.head.threshold)
    elif scorer.backend == "mercury":
        # Mercury's noul is a probability, so 0.5 is the natural cut; the DINO
        # threshold of 0.95 is far too strict for it.
        threshold = 0.5
    else:
        threshold = cfg.handwriting_threshold
    seen_hashes: set[str] = set()
    if resume and index_path.is_file():
        for row in iter_jsonl(index_path):
            if row.get("image_sha256"):
                seen_hashes.add(str(row["image_sha256"]))

    def _rows():
        for row in _iter_stream(revision):
            stats.scanned += 1
            if max_scan is not None and stats.scanned > max_scan:
                return
            if limit is not None and stats.saved >= limit:
                return
            text = _passes_metadata(row, cfg, stats)
            if text is None:
                if stats.scanned % log_every == 0:
                    log.info("scanned=%d saved=%d", stats.scanned, stats.saved)
                continue
            try:
                image = row["image"]
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(image)
                image = image.convert("RGB")
            except Exception:  # noqa: BLE001
                stats.errors += 1
                continue
            stats.decoded += 1
            if min(image.size) < cfg.min_short_side:
                stats.side_rejected += 1
                continue
            crop_id = str(row.get("crop_id") or "")
            if not crop_id:
                crop_id = safe_id(f"{row.get('image_id', '')}_{row.get('transcription', '')}")
            if resume and crop_id in existing:
                continue
            png = image_to_png_bytes(image)
            digest = sha256_bytes(png)
            if digest in seen_hashes:
                stats.duplicate_rejected += 1
                continue
            try:
                score = scorer.score_image(image)
            except Exception as exc:  # noqa: BLE001
                log.warning("scoring failed for %s: %s", crop_id, exc)
                stats.errors += 1
                continue
            if score < threshold:
                stats.handwriting_rejected += 1
                continue

            file_stem = safe_id(crop_id)
            if (images_dir / f"{file_stem}.png").exists():
                file_stem = f"{file_stem}_{digest[:8]}"
            image_path = images_dir / f"{file_stem}.png"
            image_path.write_bytes(png)
            seen_hashes.add(digest)
            existing.add(crop_id)
            src = str(row.get("source") or "AnimeText")
            stats.per_source[src] = stats.per_source.get(src, 0) + 1
            rec = {
                "crop_id": crop_id,
                "image_id": str(row.get("image_id") or ""),
                "transcription": text,
                "language": "ja",
                "seq_confidence": float(row.get("seq_confidence", 0.0) or 0.0),
                "min_token_confidence": float(row.get("min_token_confidence", 0.0) or 0.0),
                "source": src,
                "dataset": ANIMETEXT_OCR_REPO,
                "revision": revision,
                "width": int(image.width),
                "height": int(image.height),
                "image_sha256": digest,
                "image_path": str(image_path.relative_to(cfg.root)),
                "handwriting_score": round(float(score), 5),
                "handwriting_method": scorer.method,
                "kana_count": len(kana_sequence(text)),
            }
            append_jsonl(index_path, rec)
            stats.saved += 1
            if stats.scanned % log_every == 0:
                log.info("scanned=%d saved=%d", stats.scanned, stats.saved)
            yield rec

    for _ in _rows():
        pass
    log.info("harvest done: %s", stats.as_dict())
    return stats
