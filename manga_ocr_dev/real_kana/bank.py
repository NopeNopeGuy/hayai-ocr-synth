"""The real-kana bank and its renderer hook.

A bank is a directory of tightly cropped kana RGBA images plus a JSONL
manifest.  Selection follows the article:

* the glyph's stroke width, after scaling to the requested height, must be
  within x1.35 of the line's target stroke width;
* all bank glyphs used in one image must stay within x1.65 of each other;
* a glyph is never enlarged beyond x4 of its native crop;
* when a small kana has no native glyph, a full-size counterpart can be used
  as a derived glyph at 50% (the article did this only for ぃ).

Only kana are ever replaced.  Everything else stays with the font.
"""

from __future__ import annotations

import logging
import math
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image

from manga_ocr_dev.real_kana.config import KanaConfig, ensure_bank_dirs
from manga_ocr_dev.real_kana.utils import (
    SMALL_TO_LARGE,
    alpha_bbox,
    atomic_write_json,
    estimate_ink_rgb,
    estimate_stroke_width,
    is_kana,
    iter_jsonl,
    load_image,
    normalize_kana_label,
    read_json,
    relpath,
    sha256_file,
    write_jsonl,
)

log = logging.getLogger(__name__)


@dataclass(slots=True)
class KanaGlyph:
    id: str
    label: str
    path: Path
    width: int
    height: int
    stroke_width: float
    ink_rgb: tuple[int, int, int] = (0, 0, 0)
    upscale: float = 1.0
    max_scale: float = 4.0
    derived_from: str | None = None
    derived_scale: float = 1.0
    watch: bool = False
    reviewed: bool = False
    sha256: str = ""
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def is_derived(self) -> bool:
        return self.derived_from is not None

    def effective_stroke(self, target_height: float) -> float:
        if self.height <= 0:
            return 0.0
        return float(self.stroke_width) * float(target_height) / float(self.height)

    def required_base_scale(self, target_height: float) -> float:
        native = max(1.0, float(self.height) * float(self.derived_scale))
        return float(target_height) / native


class StrokeBudget:
    """Keep all bank glyphs in one image within a stroke-width ratio."""

    def __init__(self, ratio: float = 1.65):
        self.ratio = float(ratio)
        self.values: list[float] = []

    def allows(self, value: float) -> bool:
        if value <= 0:
            return False
        if not self.values:
            return True
        lo = min(min(self.values), value)
        hi = max(max(self.values), value)
        return hi / max(lo, 1e-6) <= self.ratio

    def add(self, value: float) -> None:
        self.values.append(float(value))

    def __len__(self) -> int:
        return len(self.values)


def _sidecar_for(path: Path) -> dict[str, Any]:
    return read_json(path.with_suffix(".json"), default={}) or {}


def glyph_from_record(root: Path, record: dict[str, Any], cfg: KanaConfig) -> KanaGlyph | None:
    """Build a glyph from a manifest/sidecar record; validate the file.

    Startup must stay cheap: the manifest already carries size and stroke, so
    pixels are only opened later, on demand, by :meth:`KanaBank._alpha`.
    """

    raw_path = record.get("glyph_path") or record.get("path")
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    label = normalize_kana_label(str(record.get("label") or path.parent.name))
    if label is None:
        return None
    stroke = float(record.get("stroke_width") or 0.0)
    width = int(record.get("width") or 0)
    height = int(record.get("height") or 0)
    if stroke <= 0 or width <= 0 or height <= 0:
        # Old manifests may lack metrics; measure those few records only.
        try:
            with Image.open(path) as source:
                image = source.convert("RGBA")
                width, height = image.size
                alpha = np.asarray(image.getchannel("A"))
        except OSError:
            return None
        if stroke <= 0:
            stroke = estimate_stroke_width(alpha)
    if stroke <= 0 or width <= 0 or height <= 0:
        return None
    ink = record.get("ink_rgb") or (0, 0, 0)
    try:
        ink_rgb = tuple(int(v) for v in ink[:3])
    except (TypeError, ValueError):
        ink_rgb = (0, 0, 0)
    upscale = float(record.get("upscale", 1.0) or 1.0)
    max_scale = float(record.get("max_scale", cfg.max_native_enlargement / max(upscale, 1e-6)))
    return KanaGlyph(
        id=str(record.get("glyph_id") or path.stem),
        label=label,
        path=path,
        width=int(width),
        height=int(height),
        stroke_width=float(stroke),
        ink_rgb=ink_rgb,  # type: ignore[arg-type]
        upscale=upscale,
        max_scale=max(1.0, max_scale),
        derived_from=record.get("derived_from"),
        derived_scale=float(record.get("derived_scale", 1.0) or 1.0),
        watch=bool(record.get("watch", False)),
        reviewed=bool(record.get("reviewed", False)),
        sha256=str(record.get("sha256") or ""),
        # Keep the object graph small: only provenance needed at render time.
        source={"crop_id": record.get("crop_id"), "image_id": record.get("image_id")},
    )


class KanaBank:
    """In-memory index of real kana glyphs."""

    def __init__(self, glyphs: Iterable[KanaGlyph] = (), cfg: KanaConfig | None = None):
        self.cfg = cfg or KanaConfig()
        self._by_label: dict[str, list[KanaGlyph]] = defaultdict(list)
        self._by_id: dict[str, KanaGlyph] = {}
        self._alpha_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._render_cache: OrderedDict[tuple[Any, ...], Image.Image] = OrderedDict()
        self._cache_limit = 1024
        for glyph in glyphs:
            self._add(glyph)
        self._derive_missing_small_kana()

    # -- construction -----------------------------------------------------

    @classmethod
    def load(cls, root: str | Path | None = None, cfg: KanaConfig | None = None) -> "KanaBank":
        cfg = cfg or KanaConfig()
        root = Path(root).expanduser() if root else cfg.root
        manifest = root / "manifest.jsonl"
        glyphs = [g for row in iter_jsonl(manifest) if (g := glyph_from_record(root, row, cfg)) is not None]
        if not glyphs:
            rows = build_manifest(root, cfg)
            glyphs = [g for row in rows if (g := glyph_from_record(root, row, cfg)) is not None]
        return cls(glyphs, cfg)

    @classmethod
    def from_records(
        cls,
        records: Iterable[dict[str, Any]],
        root: str | Path,
        cfg: KanaConfig | None = None,
    ) -> "KanaBank":
        cfg = cfg or KanaConfig()
        root = Path(root)
        glyphs = [g for r in records if (g := glyph_from_record(root, r, cfg)) is not None]
        return cls(glyphs, cfg)

    def _add(self, glyph: KanaGlyph) -> None:
        if glyph.id in self._by_id:
            return
        self._by_id[glyph.id] = glyph
        self._by_label[glyph.label].append(glyph)

    def _derive_missing_small_kana(self) -> None:
        """Create 50%-scale derived entries for missing small kana."""

        for small, large in SMALL_TO_LARGE.items():
            if self._by_label.get(small):
                continue
            bases = self._by_label.get(large) or []
            for base in bases[: max(8, self.cfg.max_candidates_per_label * 4)]:
                derived = KanaGlyph(
                    id=f"derived:{base.id}:{small}",
                    label=small,
                    path=base.path,
                    width=base.width,
                    height=base.height,
                    stroke_width=base.stroke_width,
                    ink_rgb=base.ink_rgb,
                    upscale=base.upscale,
                    max_scale=base.max_scale,
                    derived_from=base.id,
                    derived_scale=self.cfg.small_kana_scale,
                    watch=base.watch,
                    reviewed=base.reviewed,
                    sha256=base.sha256,
                    source={**base.source, "derived": True, "derived_from_label": large},
                )
                self._add(derived)

    # -- queries ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._by_id)

    def labels(self) -> dict[str, int]:
        return {label: len(rows) for label, rows in sorted(self._by_label.items())}

    def stats(self) -> dict[str, Any]:
        rows = list(self._by_id.values())
        real = [g for g in rows if not g.is_derived]
        strokes = [g.stroke_width for g in real]
        return {
            "glyphs": len(rows),
            "real": len(real),
            "derived": len(rows) - len(real),
            "labels": len(self._by_label),
            "sources": len({g.source.get("crop_id") for g in real if g.source.get("crop_id")}),
            "stroke_min": round(min(strokes), 3) if strokes else 0.0,
            "stroke_median": round(float(np.median(strokes)), 3) if strokes else 0.0,
            "stroke_max": round(max(strokes), 3) if strokes else 0.0,
            "reviewed": sum(g.reviewed for g in real),
            "watch": sum(g.watch for g in real),
            "per_label": self.labels(),
        }

    def candidates(
        self,
        label: str,
        target_stroke: float,
        target_height: float,
        rng: random.Random | np.random.Generator | None = None,
        *,
        budget: StrokeBudget | None = None,
    ) -> list[KanaGlyph]:
        """Return valid glyphs, best (lowest penalty) first."""

        if target_height <= 0 or target_stroke <= 0:
            return []
        tol = self.cfg.stroke_tolerance
        scored: list[tuple[float, KanaGlyph]] = []
        for glyph in self._by_label.get(label, []):
            required = glyph.required_base_scale(target_height)
            if required > glyph.max_scale * 1.001:
                continue
            eff = glyph.effective_stroke(target_height)
            ratio = eff / target_stroke
            if ratio > tol or ratio < 1.0 / tol:
                continue
            if budget is not None and not budget.allows(eff):
                continue
            score = 2.0 * abs(math.log(ratio))
            score += 0.08 * abs(math.log(max(required, 1e-6)))
            if glyph.watch:
                score += 0.35
            if not glyph.reviewed:
                score += 0.12
            if glyph.is_derived:
                score += 0.08
            # A tiny random component keeps repeated kana from looking cloned.
            if rng is not None:
                score += float(rng.random()) * 0.22
            scored.append((score, glyph))
        scored.sort(key=lambda item: item[0])
        return [g for _, g in scored]

    def pick(
        self,
        label: str,
        target_stroke: float,
        target_height: float,
        rng: random.Random | np.random.Generator | None = None,
        *,
        budget: StrokeBudget | None = None,
        top_k: int | None = None,
    ) -> KanaGlyph | None:
        cands = self.candidates(label, target_stroke, target_height, rng, budget=budget)
        if not cands:
            return None
        k = top_k or self.cfg.max_candidates_per_label
        pool = cands[: max(1, k)]
        if len(pool) == 1 or rng is None:
            return pool[0]
        # Prefer the better-fitting glyphs but keep variety.
        weights = np.exp(-0.9 * np.arange(len(pool), dtype=np.float64))
        weights /= weights.sum()
        if isinstance(rng, np.random.Generator):
            index = int(rng.choice(len(pool), p=weights))
        else:
            index = int(rng.choices(range(len(pool)), weights=weights.tolist(), k=1)[0])
        return pool[index]

    # -- rendering --------------------------------------------------------

    def _alpha(self, glyph: KanaGlyph) -> np.ndarray:
        key = str(glyph.path)
        cached = self._alpha_cache.get(key)
        if cached is not None:
            self._alpha_cache.move_to_end(key)
            return cached
        image = load_image(glyph.path, "RGBA")
        arr = np.asarray(image.getchannel("A"))
        # The bank is tightly cropped; trim again in case a sidecar was stale.
        bbox = alpha_bbox(arr, 2)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            arr = arr[y0:y1, x0:x1]
        self._alpha_cache[key] = arr
        while len(self._alpha_cache) > self._cache_limit:
            self._alpha_cache.popitem(last=False)
        return arr

    def render(
        self,
        glyph: KanaGlyph,
        target_height: float,
        ink_rgb: Sequence[int] = (0, 0, 0),
        *,
        outline_width: int = 0,
        outline_rgb: Sequence[int] | None = None,
        filled: bool = True,
    ) -> Image.Image:
        """Render a glyph at the requested ink height, tinted to *ink_rgb*."""

        target_h = max(1, int(round(target_height)))
        ink = tuple(int(np.clip(v, 0, 255)) for v in list(ink_rgb)[:3])
        outline = tuple(int(np.clip(v, 0, 255)) for v in list(outline_rgb or (0, 0, 0))[:3])
        key = (str(glyph.path), target_h, ink, int(outline_width), outline if outline_width else None, bool(filled))
        cached = self._render_cache.get(key)
        if cached is not None:
            self._render_cache.move_to_end(key)
            return cached.copy()
        alpha = self._alpha(glyph)
        if alpha.size == 0:
            return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        # Scaling the base pixels directly is correct for derived glyphs too:
        # the derivation only changes the allowed enlargement, not the shape.
        scale = target_h / max(1, alpha.shape[0])
        target_w = max(1, int(round(alpha.shape[1] * scale)))
        resized = cv2.resize(alpha, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        resized = np.clip(resized, 0, 255).astype(np.uint8)
        if outline_width > 0:
            k = max(1, int(outline_width))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            dilated = cv2.dilate(resized, kernel)
            base = np.zeros((*resized.shape, 4), dtype=np.uint8)
            base[..., :3] = outline
            base[..., 3] = dilated
            out = Image.fromarray(base, mode="RGBA")
            if filled:
                fill = np.zeros_like(base)
                fill[..., :3] = ink
                fill[..., 3] = resized
                out.alpha_composite(Image.fromarray(fill, mode="RGBA"))
        elif filled:
            arr = np.zeros((target_h, target_w, 4), dtype=np.uint8)
            arr[..., :3] = ink
            arr[..., 3] = resized
            out = Image.fromarray(arr, mode="RGBA")
        else:
            out = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        self._render_cache[key] = out
        while len(self._render_cache) > self._cache_limit // 4:
            self._render_cache.popitem(last=False)
        return out.copy()


# ---------------------------------------------------------------------------
# Renderer integration


def _op_cell(op: dict[str, Any]) -> float:
    """Font cell size for an op; uniform layouts only carry the font object."""

    for key in ("cell", "font_size"):
        value = op.get(key)
        if value:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return float(getattr(op.get("font"), "size", 0) or 0)


def _is_identity_aspect(aspect: Any) -> bool:
    """True when an op's aspect transform is a no-op (SFX sets ``(1, 1)``)."""

    if not aspect:
        return True
    try:
        ax, ay = (float(aspect[0]), float(aspect[1]))
    except (TypeError, ValueError, IndexError, KeyError):
        return False
    return abs(ax - 1.0) <= 1e-3 and abs(ay - 1.0) <= 1e-3


def _op_is_eligible(op: dict[str, Any], cfg: KanaConfig) -> bool:
    if op.get("kind") != "char":
        return False
    ch = str(op.get("ch") or "")
    if not is_kana(ch):
        return False
    if op.get("kana"):
        return False
    try:
        persp = abs(float(op.get("persp") or 0.0))
    except (TypeError, ValueError):
        return False
    # Warped glyphs cannot be matched to a flat bank raster, but SFX always
    # carries an identity aspect and a zero perspective, which is harmless.
    if persp > 1e-3 or not _is_identity_aspect(op.get("aspect")):
        return False
    cell = _op_cell(op)
    return cell >= cfg.min_font_size_for_bank


def attach_real_kana(
    ops: Sequence[dict[str, Any]],
    bank: KanaBank,
    *,
    cfg: KanaConfig | None = None,
    rng: np.random.Generator | random.Random | None = None,
    probability: float | None = None,
    lang: str = "ja",
) -> dict[str, Any]:
    """Attach bank glyphs to eligible char ops.  Mutates *ops* in place."""

    cfg = cfg or bank.cfg
    if str(lang).lower() not in {"ja", "japanese", "jp"}:
        return {"attached": 0, "skipped": 0}
    rng = rng or np.random.default_rng()
    probability = cfg.real_kana_prob if probability is None else float(probability)
    stroke_frac = float(rng.uniform(cfg.pen_stroke_frac_min, cfg.pen_stroke_frac_max))
    budget = StrokeBudget(cfg.image_stroke_ratio)
    attached = 0
    skipped = 0
    for op in ops:
        if not _op_is_eligible(op, cfg):
            continue
        if float(rng.random()) > probability:
            skipped += 1
            continue
        cell = _op_cell(op)
        target_h = cell * cfg.kana_height_frac * float(rng.uniform(0.94, 1.06))
        # Small kana are drawn ~50% of the line height, matching the article's
        # derived-kana rule; their strokes are proportionally thicker, so the
        # stroke match still uses the full line target below.
        if str(op["ch"]) in SMALL_TO_LARGE:
            target_h *= cfg.small_kana_scale
        pressure = float(rng.uniform(cfg.pen_pressure_min, cfg.pen_pressure_max))
        # The bank is matched against the line's target stroke, not against
        # the per-glyph pressure (which is a property of the font's ink).  The
        # glyph's own stroke already carries its natural pressure.
        target_stroke = stroke_frac * cell
        glyph = bank.pick(str(op["ch"]), target_stroke, target_h, rng, budget=budget)
        if glyph is None:
            skipped += 1
            continue
        eff = glyph.effective_stroke(target_h)
        budget.add(eff)
        op["kana"] = {
            "glyph": glyph,
            "target_height": float(target_h),
            "target_stroke": float(target_stroke),
            "effective_stroke": float(eff),
            "pressure": pressure,
        }
        attached += 1
    return {"attached": attached, "skipped": skipped, "stroke_span": budget.values}


def draw_bank_op(
    layer: Image.Image,
    op: dict[str, Any],
    bank: KanaBank,
    *,
    fill: Sequence[int],
    stroke_width: int = 0,
    stroke_fill: Sequence[int] | None = None,
    angle: float = 0.0,
) -> bool:
    """Draw one op's real-kana glyph.  Returns False if no glyph is attached."""

    info = op.get("kana")
    if not isinstance(info, dict):
        return False
    glyph = info.get("glyph")
    if not isinstance(glyph, KanaGlyph):
        return False
    target_h = float(info.get("target_height") or glyph.height)
    outline_w = int(stroke_width or 0) if stroke_fill is not None else 0
    filled = not (len(fill) >= 4 and int(fill[3]) == 0)
    image = bank.render(
        glyph,
        target_h,
        ink_rgb=fill[:3] if len(fill) >= 3 else (0, 0, 0),
        outline_width=outline_w,
        outline_rgb=(stroke_fill[:3] if stroke_fill and len(stroke_fill) >= 3 else (0, 0, 0)),
        filled=filled,
    )
    if abs(angle) > 1e-6:
        image = image.rotate(float(angle), resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0))
    cx, cy = float(op.get("cx", 0)), float(op.get("cy", 0))
    x = int(round(cx - image.width / 2.0))
    y = int(round(cy - image.height / 2.0))
    layer.paste(image, (x, y), image)
    return True


# ---------------------------------------------------------------------------
# Manifest construction / auditing


def build_manifest(root: str | Path | None = None, cfg: KanaConfig | None = None) -> list[dict[str, Any]]:
    cfg = cfg or KanaConfig()
    root = Path(root).expanduser() if root else cfg.root
    paths = ensure_bank_dirs(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(paths["glyphs"].rglob("*.png")):
        if path.name.startswith("."):
            continue
        sidecar = _sidecar_for(path)
        label = normalize_kana_label(str(sidecar.get("label") or path.parent.name))
        if label is None:
            log.warning("skipping %s: parent directory is not a kana", path)
            continue
        try:
            with Image.open(path) as source:
                image = source.convert("RGBA")
                width, height = image.size
                alpha = np.asarray(image.getchannel("A"))
        except OSError as exc:
            log.warning("skipping unreadable glyph %s: %s", path, exc)
            continue
        stroke = float(sidecar.get("stroke_width") or 0.0) or estimate_stroke_width(alpha)
        ink = sidecar.get("ink_rgb") or estimate_ink_rgb(image, alpha)
        upscale = float(sidecar.get("upscale", 1.0) or 1.0)
        row = {
            **sidecar,
            "glyph_id": str(sidecar.get("glyph_id") or path.stem),
            "label": label,
            "glyph_path": relpath(path, root),
            "width": int(width),
            "height": int(height),
            "stroke_width": round(float(stroke), 4),
            "ink_rgb": [int(v) for v in ink[:3]],
            "upscale": upscale,
            "max_scale": float(sidecar.get("max_scale", cfg.max_native_enlargement / max(upscale, 1e-6))),
            "sha256": sidecar.get("sha256") or sha256_file(path),
            "derived": False,
            "reviewed": bool(sidecar.get("reviewed", False)),
        }
        rows.append(row)
    rows.sort(key=lambda r: (r["label"], r["glyph_id"]))
    write_jsonl(paths["root"] / "manifest.jsonl", rows)
    stats = _manifest_stats(rows)
    atomic_write_json(
        paths["root"] / "manifest.meta.json",
        {
            "config": cfg.to_dict(),
            "glyph_count": len(rows),
            "labels": stats["labels"],
            "source_crops": stats["source_crops"],
            "real_kana_pipeline": "manga_ocr_dev.real_kana",
        },
    )
    return rows


def _manifest_stats(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels: dict[str, int] = defaultdict(int)
    crops: set[str] = set()
    for row in rows:
        labels[str(row.get("label"))] += 1
        if row.get("crop_id"):
            crops.add(str(row["crop_id"]))
    return {"labels": dict(sorted(labels.items())), "source_crops": len(crops)}


def audit_bank(root: str | Path | None = None, cfg: KanaConfig | None = None) -> dict[str, Any]:
    cfg = cfg or KanaConfig()
    root = Path(root).expanduser() if root else cfg.root
    rows = list(iter_jsonl(root / "manifest.jsonl"))
    if not rows:
        rows = build_manifest(root, cfg)
    problems: list[str] = []
    hashes: dict[str, str] = {}
    labels: dict[str, int] = defaultdict(int)
    strokes: list[float] = []
    sizes: list[int] = []
    source_crops: set[str] = set()
    reviewed = 0
    for row in rows:
        label = row.get("label")
        if not label or normalize_kana_label(str(label)) is None:
            problems.append(f"bad label: {row.get('glyph_id')}")
            continue
        labels[str(label)] += 1
        path = root / str(row.get("glyph_path", ""))
        if not path.is_file():
            problems.append(f"missing file: {row.get('glyph_id')}")
            continue
        digest = str(row.get("sha256") or "")
        if digest and digest in hashes:
            problems.append(f"duplicate hash: {row.get('glyph_id')} == {hashes[digest]}")
        elif digest:
            hashes[digest] = str(row.get("glyph_id"))
        try:
            strokes.append(float(row.get("stroke_width", 0)))
            sizes.append(int(row.get("width", 0)))
        except (TypeError, ValueError):
            problems.append(f"bad metrics: {row.get('glyph_id')}")
        if row.get("crop_id"):
            source_crops.add(str(row["crop_id"]))
        if row.get("reviewed"):
            reviewed += 1
    missing = sorted(set(SMALL_TO_LARGE) - set(labels))
    report = {
        "root": str(root),
        "glyphs": len(rows),
        "labels": len(labels),
        "per_label": dict(sorted(labels.items())),
        "missing_small_kana": missing,
        "source_crops": len(source_crops),
        "reviewed": reviewed,
        "stroke_median": round(float(np.median(strokes)), 3) if strokes else 0.0,
        "width_median": int(np.median(sizes)) if sizes else 0,
        "problems": problems,
    }
    atomic_write_json(root / "audit.json", report)
    return report


def export_review_sheets(
    root: str | Path | None = None,
    out_dir: str | Path | None = None,
    *,
    per_sheet: int = 64,
) -> int:
    from manga_ocr_dev.real_kana.contact import label_sheet

    root = Path(root).expanduser() if root else KanaConfig().root
    out_dir = Path(out_dir) if out_dir else root / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(iter_jsonl(root / "manifest.jsonl"))
    if not rows:
        rows = build_manifest(root)
    by_label: dict[str, list[tuple[str, Image.Image, str]]] = defaultdict(list)
    for row in rows:
        path = root / str(row.get("glyph_path", ""))
        if not path.is_file():
            continue
        try:
            image = load_image(path, "RGBA")
        except OSError:
            continue
        by_label[str(row.get("label"))].append((str(row.get("label")), image, str(row.get("glyph_id"))))
    sheets = 0
    for label, items in sorted(by_label.items()):
        for start in range(0, len(items), per_sheet):
            sheet = label_sheet(items[start : start + per_sheet])
            sheet.save(out_dir / f"{label}_{start // per_sheet:02d}.png")
            sheets += 1
    return sheets
