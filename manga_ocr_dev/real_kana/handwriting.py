"""Handwriting-appearance scoring for candidate crops.

The reference pipeline trained a small classifier on frozen DINOv2 features.
This module implements exactly that: frozen embeddings, a tiny logistic head,
and a deterministic heuristic fallback for when no training labels (or no
torch/transformers installation) are available.  The fallback is deliberately
conservative and should be replaced by a trained head for production runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
from PIL import Image

from manga_ocr_dev.real_kana.utils import (
    atomic_write_json,
    colour_separated_alpha,
    iter_jsonl,
    load_image,
    read_json,
)

DEFAULT_DINO_MODEL = "facebook/dinov2-small"


# ---------------------------------------------------------------------------
# Heuristic fallback


def _ridge_values(alpha: np.ndarray) -> np.ndarray:
    ink = (alpha >= 127).astype(np.uint8)
    if int(ink.sum()) < 4:
        return np.zeros(0, dtype=np.float32)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    dilated = cv2.dilate(dist, np.ones((3, 3), np.uint8))
    ridge = ink.astype(bool) & (dist >= dilated - 1e-4) & (dist > 0.5)
    values = dist[ridge]
    return values if values.size else dist[ink.astype(bool)]


def heuristic_handwriting_score(rgb: np.ndarray | Image.Image) -> float:
    """A 0..1 handwriting-likeness score from simple stroke statistics.

    This is *not* the DINO classifier from the article.  It exists so the
    pipeline can be run end-to-end before a labelled seed set is available.
    Handwriting tends to have a wider stroke-width distribution, rougher
    contours and less regular component sizes than a typeface.
    """

    if isinstance(rgb, Image.Image):
        rgb = np.asarray(rgb.convert("RGB"))
    if rgb.ndim != 3 or min(rgb.shape[:2]) < 6:
        return 0.0

    alpha, _ = colour_separated_alpha(rgb)
    ink = alpha >= 127
    area = float(ink.sum())
    if area < 8:
        return 0.0

    values = _ridge_values(alpha)
    if values.size < 3:
        return 0.0
    stroke_mean = float(values.mean())
    stroke_cv = float(values.std() / max(stroke_mean, 1e-3))

    n, _labels, stats, cents = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 3]
    comp_count = len(keep)
    if comp_count == 0:
        return 0.0

    areas = np.array([stats[i, cv2.CC_STAT_AREA] for i in keep], dtype=np.float32)
    area_cv = float(areas.std() / max(areas.mean(), 1e-3))
    ws = np.array([stats[i, cv2.CC_STAT_WIDTH] for i in keep], dtype=np.float32)
    hs = np.array([stats[i, cv2.CC_STAT_HEIGHT] for i in keep], dtype=np.float32)
    aspect_std = float(np.std(np.log(np.maximum(ws, 1) / np.maximum(hs, 1))))

    # Contour roughness: perimeter normalised by sqrt(area).  A clean glyph
    # is near ~4; broken/rough handwriting has a higher ratio.
    contours, _ = cv2.findContours(ink.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeters = [cv2.arcLength(c, True) for c in contours if cv2.contourArea(c) >= 2]
    roughness = float(np.mean([p / np.sqrt(max(a, 1.0)) for p, a in zip(perimeters, areas)])) if perimeters else 0.0

    # Baseline/centroid wobble.  Printed text sits on a very regular line.
    cy = np.array([cents[i][1] for i in keep], dtype=np.float32)
    baseline_std = float(cy.std() / max(rgb.shape[0], 1))

    coverage = area / float(rgb.shape[0] * rgb.shape[1])
    contrast = float(np.std(values) / max(values.mean(), 1e-3))

    # Hand-tuned logistic; coefficients are intentionally modest so the
    # fallback never looks like a confident production classifier.
    z = (
        3.1 * stroke_cv
        + 1.8 * min(roughness / 6.0, 2.0)
        + 1.2 * min(area_cv, 2.0)
        + 2.0 * min(baseline_std * 20.0, 2.0)
        + 0.8 * min(aspect_std, 2.0)
        + 0.7 * min(contrast, 2.0)
        - 5.4
    )
    score = 1.0 / (1.0 + np.exp(-z))
    # Very sparse or nearly full crops are almost never usable.
    if coverage < 0.015 or coverage > 0.90:
        score *= 0.5
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# DINOv2 embeddings + logistic head


class DinoEmbedder:
    """Frozen DINOv2 CLS embeddings (lazy torch/transformers import)."""

    def __init__(self, model_id: str = DEFAULT_DINO_MODEL, device: str | None = None):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "DINO scoring needs torch and transformers. "
                "Install them with: pip install torch transformers"
            ) from exc
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()

    def embed(self, images: Sequence[Image.Image], batch_size: int = 16) -> np.ndarray:
        self._load()
        import torch

        rows: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[start : start + batch_size]]
            inputs = self._processor(images=batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model(**inputs)
            emb = out.last_hidden_state[:, 0].float().cpu().numpy()
            rows.append(emb)
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(rows, axis=0).astype(np.float32)


@dataclass
class LinearHead:
    coef: np.ndarray
    intercept: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    threshold: float = 0.95
    model_id: str = DEFAULT_DINO_MODEL
    auc: float | None = None

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        z = ((embeddings - self.mean) / np.maximum(self.std, 1e-6)) @ self.coef + float(self.intercept)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def _scalar(data, key: str, default):
    if key not in data:
        return default
    value = np.asarray(data[key]).reshape(-1)
    if value.size == 0:
        return default
    return value[0].item()


def load_head(path: str | Path) -> LinearHead | None:
    path = Path(path)
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=False)
    return LinearHead(
        coef=np.asarray(data["coef"], dtype=np.float32).reshape(-1),
        intercept=np.asarray(data["intercept"], dtype=np.float32).reshape(()),
        mean=np.asarray(data["mean"], dtype=np.float32).reshape(-1),
        std=np.asarray(data["std"], dtype=np.float32).reshape(-1),
        threshold=float(_scalar(data, "threshold", 0.95)),
        model_id=str(_scalar(data, "model_id", DEFAULT_DINO_MODEL)),
        auc=float(_scalar(data, "auc", np.nan)) if "auc" in data else None,
    )


def _read_labels(path: Path) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.lower().startswith("path,"):
                continue
            parts = line.rsplit(",", 1)
            if len(parts) != 2:
                continue
            raw_path, raw_label = parts
            label = raw_label.strip().lower()
            if label in {"1", "hand", "handwritten", "true", "yes"}:
                value = 1
            elif label in {"0", "print", "printed", "false", "no", "typeset"}:
                value = 0
            else:
                continue
            rows.append((Path(raw_path.strip()).expanduser(), value))
    return rows


def train_head(
    labels_csv: str | Path,
    output: str | Path,
    *,
    model_id: str = DEFAULT_DINO_MODEL,
    threshold: float = 0.95,
    embedding_cache: str | Path | None = None,
) -> dict:
    """Train a logistic head on frozen DINO features.

    The CSV has two columns: ``path,label`` where label is 1/handwritten or
    0/printed.  A 5-fold AUC is reported and stored with the head.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    rows = _read_labels(Path(labels_csv))
    if len(rows) < 20:
        raise ValueError("need at least 20 labelled crops to train a stable head")
    y = np.array([label for _, label in rows], dtype=np.int32)
    if len(np.unique(y)) < 2:
        raise ValueError("both handwritten and printed examples are required")

    cache_path = Path(embedding_cache) if embedding_cache else Path(output).with_suffix(".embeddings.npz")
    cache = read_json(str(cache_path) + ".json", default={}) or {}
    embedder = DinoEmbedder(model_id)
    if cache_path.is_file() and cache.get("model_id") == model_id and cache.get("n") == len(rows):
        data = np.load(cache_path)
        X = np.asarray(data["x"], dtype=np.float32)
    else:
        images = [load_image(p, "RGB") for p, _ in rows]
        X = embedder.embed(images)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, x=X)
        atomic_write_json(str(cache_path) + ".json", {"model_id": model_id, "n": len(rows)})

    folds = min(5, int(np.bincount(y).min()))
    auc = None
    if folds >= 2:
        oof = np.zeros(len(y), dtype=np.float32)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(X[train_idx], y[train_idx])
            oof[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        auc = float(roc_auc_score(y, oof))

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X, y)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        coef=clf.coef_.reshape(-1).astype(np.float32),
        intercept=np.asarray([clf.intercept_[0]], dtype=np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        threshold=np.asarray([threshold], dtype=np.float32),
        model_id=np.asarray([model_id]),
        auc=np.asarray([auc if auc is not None else np.nan], dtype=np.float32),
    )
    return {"n": len(rows), "auc": auc, "output": str(out), "model_id": model_id}


class HandwritingScorer:
    """Merge a trained DINO head, the Mercury classifier, or the heuristic."""

    BACKENDS: ClassVar[set[str]] = {"auto", "heuristic", "dino", "mercury"}

    def __init__(
        self,
        head_path: str | Path | None = None,
        model_id: str = DEFAULT_DINO_MODEL,
        *,
        backend: str = "auto",
        mercury_client: Any | None = None,
    ):
        backend = str(backend or "auto").lower()
        if backend not in self.BACKENDS:
            raise ValueError(f"unknown handwriting backend {backend!r}")
        self.head = load_head(head_path) if head_path else None
        if backend == "auto":
            backend = "dino" if self.head is not None else "heuristic"
        if backend == "dino" and self.head is None:
            raise ValueError("the dino handwriting backend needs a trained head (--head)")
        self.backend = backend
        self.model_id = self.head.model_id if self.head else model_id
        self._embedder: DinoEmbedder | None = None
        self._mercury = mercury_client
        if self.backend == "mercury" and self._mercury is None:
            from manga_ocr_dev.real_kana.mercury import MercuryDecisionClient

            self._mercury = MercuryDecisionClient()

    @property
    def method(self) -> str:
        if self.backend == "mercury":
            return "mercury_noul"
        if self.head is not None:
            return "dino_logistic"
        return "heuristic"

    def score_images(self, images: Sequence[Image.Image]) -> list[float]:
        if self.backend == "mercury":
            return [float(self._mercury.score_handwriting(im)) for im in images]
        if self.head is None:
            return [heuristic_handwriting_score(im) for im in images]
        if self._embedder is None:
            self._embedder = DinoEmbedder(self.model_id)
        return self.head.score(self._embedder.embed(images)).tolist()

    def score_image(self, image: Image.Image) -> float:
        if self.backend == "mercury":
            return float(self._mercury.score_handwriting(image))
        return self.score_images([image])[0]


def write_heuristic_labels(images_jsonl: str | Path, output_csv: str | Path, root: str | Path | None = None) -> int:
    """Utility: write a labels CSV with heuristic guesses for manual fixing.

    Paths in ``candidates.jsonl`` are relative to the bank root.  Passing
    ``root`` makes the output CSV contain absolute paths, which is what
    :func:`train_head` expects.
    """

    index_path = Path(images_jsonl)
    root_path = Path(root).expanduser() if root else None
    rows = []
    for row in iter_jsonl(index_path):
        raw_path = row.get("image_path") or row.get("path")
        if not raw_path:
            continue
        path = Path(raw_path).expanduser()
        if not path.is_absolute() and root_path is not None:
            path = root_path / path
        try:
            score = heuristic_handwriting_score(load_image(path))
        except OSError:
            continue
        rows.append(f"{path},{1 if score >= 0.5 else 0}\n")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(output_csv).write_text("path,label\n" + "".join(rows), encoding="utf-8")
    return len(rows)
