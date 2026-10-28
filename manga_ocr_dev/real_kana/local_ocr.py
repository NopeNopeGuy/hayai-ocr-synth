"""Local character detection + recognition with ``manga-ocr-charspot-small``.

The model returns every character of a cropped text box in reading order,
with a tight box and a confidence score.  This replaces the vision-model
segmentation step; Gemini is only used afterwards, as a verifier.

The checkout may live next to the project root (``manga-ocr-charspot-small/``),
be installed with ``pip install -e manga-ocr-charspot-small``, or be pointed at
with ``MANGA_OCR_CHARSPOT_SMALL_DIR``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from manga_ocr_dev.real_kana.upright import estimate_tilt_from_centres

DEFAULT_MODEL_NAME = "manga-ocr-charspot-small"
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = _REPO_ROOT / DEFAULT_MODEL_NAME


class LocalOcrUnavailable(RuntimeError):
    """Raised when the local charspot model cannot be found or loaded."""


def find_model_dir(explicit: str | Path | None = None) -> Path:
    candidates = [explicit, os.environ.get("MANGA_OCR_CHARSPOT_SMALL_DIR"), DEFAULT_MODEL_DIR]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if (path / "config.json").is_file():
            return path
    raise LocalOcrUnavailable(
        "manga-ocr-charspot-small was not found. Clone it to "
        f"{DEFAULT_MODEL_DIR}, set MANGA_OCR_CHARSPOT_SMALL_DIR, or pass --local-model."
    )


def _load_class(model_dir: Path):
    try:
        from manga_ocr_charspot_small import MangaOcrCharspotSmall
    except ModuleNotFoundError:
        # A plain checkout of the model repository: its package directory is
        # the model directory itself.
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        try:
            from manga_ocr_charspot_small import MangaOcrCharspotSmall
        except ModuleNotFoundError as exc:  # pragma: no cover - broken checkout
            raise LocalOcrUnavailable(
                f"cannot import manga_ocr_charspot_small from {model_dir}: {exc}"
            ) from exc
    return MangaOcrCharspotSmall


class CharspotOcr:
    """Thin wrapper so the rest of the pipeline does not depend on the package."""

    model = DEFAULT_MODEL_NAME

    def __init__(
        self,
        model_dir: str | Path | None = None,
        backend: str = "auto",
        threads: int | None = None,
    ):
        self.model_dir = find_model_dir(model_dir)
        cls = _load_class(self.model_dir)
        try:
            self._ocr = cls(model_dir=self.model_dir, backend=backend, threads=threads)
        except TypeError:
            # Older revisions only accepted model_dir/backend.
            self._ocr = cls(model_dir=self.model_dir, backend=backend)
        except Exception as exc:  # noqa: BLE001
            raise LocalOcrUnavailable(f"cannot load {self.model}: {exc}") from exc

    def read(self, image: Image.Image | np.ndarray | str | Path) -> dict[str, Any]:
        return self._ocr.read(image)

    def read_many(self, images: Sequence[Any]) -> list[dict[str, Any]]:
        return self._ocr.read(list(images))

    def __call__(self, image: Image.Image | np.ndarray | str | Path) -> str:
        return self._ocr(image)


@dataclass
class _Char:
    char: str
    box: list[int]
    score: float
    order: int


def _median(values: Sequence[float], default: float = 1.0) -> float:
    return float(np.median(values)) if values else default


def _clamp_box(box: Sequence[float], width: int, height: int) -> list[int] | None:
    if len(box) != 4:
        return None
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    ix0 = max(0, min(width - 1, int(round(x0))))
    iy0 = max(0, min(height - 1, int(round(y0))))
    ix1 = max(ix0 + 1, min(width, int(round(x1))))
    iy1 = max(iy0 + 1, min(height, int(round(y1))))
    if ix1 - ix0 < 2 or iy1 - iy0 < 2:
        return None
    return [ix0, iy0, ix1, iy1]


def group_chars(
    chars: Sequence[dict[str, Any]],
    width: int,
    height: int,
    *,
    vertical: bool | None = None,
) -> list[dict[str, Any]]:
    """Group reading-order characters into lines/columns with a tilt estimate."""

    parsed: list[_Char] = []
    for index, raw in enumerate(chars):
        if not isinstance(raw, dict):
            continue
        box = _clamp_box(raw.get("box") or [], width, height)
        if box is None:
            continue
        parsed.append(
            _Char(
                char=str(raw.get("char") or ""),
                box=box,
                score=float(raw.get("score") or 0.0),
                order=index,
            )
        )
    if not parsed:
        return []

    if vertical is None:
        union_w = max(c.box[2] for c in parsed) - min(c.box[0] for c in parsed)
        union_h = max(c.box[3] for c in parsed) - min(c.box[1] for c in parsed)
        vertical = union_h >= union_w

    axis = 0 if vertical else 1  # x for vertical columns, y for horizontal lines
    centres = [((c.box[axis] + c.box[axis + 2]) / 2.0) for c in parsed]
    sizes = [(c.box[axis + 2] - c.box[axis]) for c in parsed]
    tolerance = max(2.0, 0.55 * _median(sizes))

    groups: list[list[_Char]] = []
    for char, centre in zip(parsed, centres):
        if groups:
            previous = groups[-1]
            previous_centre = _median([(g.box[axis] + g.box[axis + 2]) / 2.0 for g in previous])
            if abs(centre - previous_centre) <= tolerance:
                previous.append(char)
                continue
        groups.append([char])

    lines: list[dict[str, Any]] = []
    for group in groups:
        points = [((c.box[0] + c.box[2]) / 2.0, (c.box[1] + c.box[3]) / 2.0) for c in group]
        tilt = estimate_tilt_from_centres(points, vertical=vertical)
        glyphs = [
            {
                "label": c.char,
                "x0": c.box[0],
                "y0": c.box[1],
                "x1": c.box[2],
                "y1": c.box[3],
                "confidence": c.score,
            }
            for c in group
        ]
        lines.append({"tilt_deg": tilt, "glyphs": glyphs})
    return lines


def response_from_read(
    result: dict[str, Any],
    *,
    width: int,
    height: int,
    transcription: str,
    vertical: bool | None = None,
) -> dict[str, Any]:
    """Convert one ``ocr.read`` result into the segmentation response shape."""

    from manga_ocr_dev.real_kana.utils import is_likely_sfx

    lines = group_chars(result.get("chars") or [], width, height, vertical=vertical)
    return {
        "is_handwritten": True,
        "is_sound_effect": is_likely_sfx(transcription),
        "lines": lines,
    }
