"""Fast handwritten-vs-printed crop classifier.

Mercury Decide is accurate but too slow to label hundreds of thousands of
candidates.  This module trains a small MobileNetV3 on a Mercury-labelled
sample and uses it to filter the rest.  The split is grouped by ``image_id`` so
crops from the same source page never leak between train and validation.

The CSV used here has three columns: ``path,label,image_id``.  A two-column
``path,label`` CSV also works; the image id is then derived from the crop file
name (``<image_id>_<crop>``).
"""

from __future__ import annotations

import csv
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from manga_ocr_dev.real_kana.utils import atomic_write_json, iter_jsonl, safe_id

log = logging.getLogger(__name__)

DEFAULT_CROP_SIZE = 96
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CHECKPOINT_VERSION = 1


def prepare_crop(image: Image.Image | str | Path, size: int = DEFAULT_CROP_SIZE) -> Image.Image:
    """Pad a crop to a white square and resize it without distortion."""

    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        image = background
    else:
        image = image.convert("RGB")
    width, height = image.size
    side = max(1, width, height)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def _image_id(path: str | Path, fallback: str = "") -> str:
    if fallback:
        return str(fallback)
    return Path(path).stem.rsplit("_", 1)[0]


def read_label_rows(labels_csv: str | Path) -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []
    with open(labels_csv, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("labels CSV must contain at least 'path' and 'label' columns")
        for row in reader:
            path = str(row.get("path") or "").strip()
            if not path:
                continue
            try:
                label = int(str(row.get("label", "")).strip())
            except ValueError:
                continue
            if label not in (0, 1):
                continue
            rows.append((path, label, _image_id(path, str(row.get("image_id") or "").strip())))
    return rows


def _build_model(pretrained: bool = True):
    from torch import nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = int(model.classifier[-1].in_features)
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


class CropDataset:
    """Torch dataset kept importable without importing torch at module load."""

    def __init__(self, paths: Sequence[str], labels: Sequence[int] | None, size: int, train: bool):
        self.paths = list(paths)
        self.labels = None if labels is None else list(labels)
        self.size = int(size)
        self.train = bool(train)
        self._augment = None
        if self.train:
            from torchvision import transforms

            self._augment = transforms.Compose(
                [
                    transforms.RandomAffine(
                        degrees=3,
                        translate=(0.03, 0.03),
                        scale=(0.95, 1.05),
                        fill=255,
                    ),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15),
                ]
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        import torch
        from torchvision import transforms

        image = prepare_crop(self.paths[index], self.size)
        if self._augment is not None:
            image = self._augment(image)
        tensor = transforms.functional.to_tensor(image)
        tensor = transforms.functional.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        if self.labels is None:
            return tensor
        return tensor, torch.tensor(float(self.labels[index]), dtype=torch.float32)


def _make_loader(dataset: CropDataset, batch_size: int, shuffle: bool, workers: int):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(0, int(workers)),
        pin_memory=False,
        persistent_workers=workers > 0,
        drop_last=False,
    )


def _predict_loader(model: Any, loader: Any) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    probs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images, labels = batch
                targets.append(labels.numpy())
            else:
                images = batch
            logits = model(images).squeeze(1)
            probs.append(torch.sigmoid(logits).numpy())
    if not probs:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    return np.concatenate(probs).astype(np.float32), (
        np.concatenate(targets).astype(np.float32) if targets else np.zeros(0, dtype=np.float32)
    )


def _best_threshold(y: np.ndarray, probs: np.ndarray) -> tuple[float, dict[str, float]]:
    from sklearn.metrics import precision_recall_fscore_support

    best = (0.5, {"precision": 0.0, "recall": 0.0, "f1": 0.0})
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = (probs >= threshold).astype(np.int32)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y.astype(np.int32), pred, average="binary", zero_division=0
        )
        if f1 > best[1]["f1"]:
            best = (
                float(threshold),
                {"precision": float(precision), "recall": float(recall), "f1": float(f1)},
            )
    return best


def train_cnn(
    labels_csv: str | Path,
    output: str | Path,
    *,
    epochs: int = 8,
    batch_size: int = 128,
    lr: float = 3e-4,
    crop_size: int = DEFAULT_CROP_SIZE,
    workers: int = 3,
    pretrained: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Train a Mercury-distilled handwriting classifier with an honest split."""

    import torch
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit

    rows = read_label_rows(labels_csv)
    if len(rows) < 100:
        raise ValueError("need at least 100 labelled crops")
    y = np.array([label for _, label, _ in rows], dtype=np.int32)
    if len(np.unique(y)) < 2:
        raise ValueError("both handwritten and printed examples are required")
    groups = np.array([group for _, _, group in rows])
    if len(np.unique(groups)) < 2:
        raise ValueError("need at least two image_id groups for a grouped split")

    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(split.split(np.zeros(len(rows)), y, groups))
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    if len(np.unique([label for _, label, _ in train_rows])) < 2:
        raise ValueError("grouped split produced a single class in the training set")

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _build_model(pretrained=pretrained)
    train_ds = CropDataset(
        [path for path, _, _ in train_rows],
        [label for _, label, _ in train_rows],
        crop_size,
        train=True,
    )
    val_ds = CropDataset(
        [path for path, _, _ in val_rows],
        [label for _, label, _ in val_rows],
        crop_size,
        train=False,
    )
    train_loader = _make_loader(train_ds, batch_size, True, workers)
    val_loader = _make_loader(val_ds, batch_size, False, workers)

    positives = max(1, int(np.sum([label for _, label, _ in train_rows])))
    negatives = max(1, len(train_rows) - positives)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    best_auc = -1.0
    best_state = None
    history: list[dict[str, float]] = []

    for epoch in range(1, max(1, epochs) + 1):
        model.train()
        total = 0.0
        seen = 0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images).squeeze(1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(labels)
            seen += len(labels)
        scheduler.step()
        val_probs, val_y = _predict_loader(model, val_loader)
        auc = float(roc_auc_score(val_y, val_probs)) if len(np.unique(val_y)) > 1 else float("nan")
        row = {"epoch": epoch, "loss": total / max(1, seen), "val_auc": auc}
        history.append(row)
        log.info("epoch %d/%d loss=%.4f val_auc=%.4f", epoch, epochs, row["loss"], auc)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val_probs, val_y = _predict_loader(model, val_loader)
    auc = float(roc_auc_score(val_y, val_probs)) if len(np.unique(val_y)) > 1 else float("nan")
    threshold, metrics = _best_threshold(val_y, val_probs)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "state_dict": model.state_dict(),
        "crop_size": int(crop_size),
        "threshold": float(threshold),
        "val_auc": auc,
        "val_metrics": metrics,
        "n_labels": len(rows),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "positives": int(y.sum()),
        "history": history,
    }
    torch.save(checkpoint, out)
    atomic_write_json(
        str(out) + ".json",
        {
            "labels": str(labels_csv),
            "n_labels": len(rows),
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "positives": int(y.sum()),
            "val_auc": auc,
            "threshold": threshold,
            "val_metrics": metrics,
            "history": history,
        },
    )
    return {
        "n": len(rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "positives": int(y.sum()),
        "val_auc": auc,
        "threshold": threshold,
        "val_metrics": metrics,
        "output": str(out),
    }


class CnnScorer:
    """Load a trained checkpoint and score crop paths in batches."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        batch_size: int = 128,
        workers: int = 3,
        device: str | None = None,
    ):
        import torch

        self.checkpoint_path = Path(checkpoint)
        data = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if int(data.get("version", 0)) != CHECKPOINT_VERSION:
            raise ValueError(f"unsupported CNN checkpoint version in {self.checkpoint_path}")
        self.device = device or "cpu"
        self.model = _build_model(pretrained=False)
        self.model.load_state_dict(data["state_dict"])
        self.model.to(self.device).eval()
        self.crop_size = int(data.get("crop_size", DEFAULT_CROP_SIZE))
        self.threshold = float(data.get("threshold", 0.5))
        self.val_auc = float(data.get("val_auc", float("nan")))
        self.batch_size = int(batch_size)
        self.workers = int(workers)

    def score_paths(self, paths: Sequence[str | Path]) -> np.ndarray:
        dataset = CropDataset([str(p) for p in paths], None, self.crop_size, train=False)
        loader = _make_loader(dataset, self.batch_size, False, self.workers)
        probs, _ = _predict_loader(self.model, loader)
        return probs

    def score_path(self, path: str | Path) -> float:
        return float(self.score_paths([path])[0])


def score_candidates(
    candidates_jsonl: str | Path,
    checkpoint: str | Path,
    *,
    batch_size: int = 128,
    workers: int = 3,
    cache_path: str | Path | None = None,
    chunk_size: int = 4000,
) -> list[dict[str, Any]]:
    """Score every candidate and return rows with ``cnn_score`` added.

    When ``cache_path`` is given, scores are appended there as they are computed
    and reused on the next call.  That keeps a 400k-crop scoring pass resumable
    and lets a second run only score crops added since the first one.
    """

    rows = list(iter_jsonl(candidates_jsonl))
    if not rows:
        return []
    root = Path(candidates_jsonl).parent.parent
    cache: dict[str, float] = {}
    if cache_path is not None and Path(cache_path).is_file():
        for record in iter_jsonl(cache_path):
            name = str(record.get("image_path") or "")
            if name and record.get("cnn_score") is not None:
                cache[name] = float(record["cnn_score"])
        log.info("score cache: %d crops already scored", len(cache))
    pending = [row for row in rows if str(row["image_path"]) not in cache]
    if pending:
        scorer = CnnScorer(checkpoint, batch_size=batch_size, workers=workers)
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(pending), max(1, int(chunk_size))):
            chunk = pending[start : start + max(1, int(chunk_size))]
            paths = [str(root / row["image_path"]) for row in chunk]
            scores = scorer.score_paths(paths)
            fresh: list[str] = []
            for row, score in zip(chunk, scores):
                value = round(float(score), 6)
                cache[str(row["image_path"])] = value
                fresh.append(
                    json.dumps(
                        {"image_path": row["image_path"], "cnn_score": value},
                        ensure_ascii=False,
                    )
                )
            if cache_path is not None:
                with open(cache_path, "a", encoding="utf-8") as fh:
                    fh.write("\n".join(fresh) + "\n")
            log.info(
                "scored %d/%d pending (%d/%d pool)",
                min(start + len(chunk), len(pending)),
                len(pending),
                len(cache),
                len(rows),
            )
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append({**row, "cnn_score": cache[str(row["image_path"])]})
    return out


def _selection_score(row: dict[str, Any]) -> float:
    if "mercury_noul" in row:
        return float(row["mercury_noul"])
    return float(row.get("cnn_score", 0.0))


def mercury_refine_uncertain(
    scored_rows: Sequence[dict[str, Any]],
    source_root: str | Path,
    client: Any,
    *,
    low: float = 0.35,
    high: float = 0.65,
    max_calls: int = 2000,
    threshold: float = 0.5,
    center: float | None = None,
) -> dict[str, Any]:
    """Ask Mercury only about CNN decisions near the boundary.

    Rows outside ``[low, high]`` keep their CNN score.  Rows inside the band are
    sorted by closeness to ``center`` (default: the selection cutoff) and capped
    by ``max_calls``; their Mercury ``noul`` becomes the score used for selection.
    """

    source_root = Path(source_root)
    uncertain = [
        row
        for row in scored_rows
        if low <= float(row.get("cnn_score", 0.0)) <= high
    ]
    pivot = float(threshold if center is None else center)
    uncertain.sort(key=lambda row: abs(float(row.get("cnn_score", 0.0)) - pivot))
    chosen = uncertain[: max(0, int(max_calls))]
    labelled = 0
    handwritten = 0
    errors = 0
    for row in chosen:
        path = source_root / row["image_path"]
        try:
            result = client.label(path, threshold=threshold)
        except Exception as exc:  # noqa: BLE001
            log.warning("Mercury refinement failed for %s: %s", path, exc)
            errors += 1
            continue
        row["mercury_noul"] = round(float(result["noul"]), 6)
        row["mercury_label"] = int(bool(result["handwritten"]))
        labelled += 1
        handwritten += int(bool(result["handwritten"]))
    return {
        "uncertain": len(uncertain),
        "labelled": labelled,
        "handwritten": handwritten,
        "skipped_over_cap": max(0, len(uncertain) - len(chosen)),
        "errors": errors,
        "low": float(low),
        "high": float(high),
        "center": pivot,
    }


def write_selected_bank(
    scored_rows: Sequence[dict[str, Any]],
    source_root: str | Path,
    target_root: str | Path,
    *,
    limit: int = 100_000,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Hard-link the top-scoring candidates into a fresh bank root."""

    source_root = Path(source_root)
    target_root = Path(target_root)
    selected = [row for row in scored_rows if _selection_score(row) >= threshold]
    selected.sort(key=_selection_score, reverse=True)
    selected = selected[: max(0, int(limit))]
    images_dir = target_root / "candidates" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    index_path = target_root / "candidates" / "candidates.jsonl"
    index_path.write_text("", encoding="utf-8")
    linked = 0
    reused = 0
    missing = 0
    for row in selected:
        src = source_root / row["image_path"]
        if not src.is_file():
            missing += 1
            continue
        name = src.name
        dst = images_dir / name
        keep = False
        if dst.exists():
            try:
                keep = os.path.samefile(src, dst)
            except OSError:
                keep = False
        if keep:
            reused += 1
        else:
            stem = Path(name).stem
            suffix = 1
            while dst.exists():
                dst = images_dir / f"{stem}_{suffix}{src.suffix}"
                suffix += 1
            try:
                os.link(src, dst)
            except OSError:
                import shutil

                shutil.copy2(src, dst)
            linked += 1
        record = dict(row)
        record["image_path"] = str(dst.relative_to(target_root))
        record["selection_method"] = "mercury_distilled_cnn"
        with open(index_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "pool": len(scored_rows),
        "above_threshold": int(sum(_selection_score(r) >= threshold for r in scored_rows)),
        "mercury_refined": int(sum("mercury_noul" in r for r in scored_rows)),
        "selected": linked,
        "reused": reused,
        "missing": missing,
        "threshold": float(threshold),
        "target": str(target_root),
    }


def prune_unselected_stages(
    target_root: str | Path,
    crop_ids: Sequence[str],
    *,
    stages: Sequence[str] = ("raw", "cleaned", "upscaled"),
) -> dict[str, int]:
    """Drop per-crop stage directories that are no longer in the selection.

    Re-running ``filter-cnn`` on an existing bank replaces the candidate index.
    Any glyph crops cut from a candidate that fell out of the new selection must
    be removed before the next stage, otherwise they would leak into the bank.
    """

    import shutil

    target_root = Path(target_root)
    keep = {safe_id(str(cid)) for cid in crop_ids}
    removed: dict[str, int] = {}
    for stage in stages:
        root = target_root / stage
        if not root.is_dir():
            removed[stage] = 0
            continue
        count = 0
        for path in root.iterdir():
            if path.is_dir() and path.name not in keep:
                shutil.rmtree(path, ignore_errors=True)
                count += 1
        removed[stage] = count
    return removed
