"""Vision-model access for kana segmentation and review.

The article used several independent GPT sessions.  This implementation uses a
single configurable model (Gemini Flash-Lite by default, or any OpenAI-
compatible vision endpoint) with three narrowly-scoped prompts:

* ``segment``: line tilt + tight boxes + one kana label per box,
* ``review``: correct labels/removals on a numbered contact sheet,
* ``handwriting``: weak labels used to train the DINO head.

All calls are cached by a hash of model, prompt and image bytes, so re-running
a stage is cheap and deterministic.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import random
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
from PIL import Image

from manga_ocr_dev.real_kana.config import KanaConfig, ensure_bank_dirs
from manga_ocr_dev.real_kana.utils import (
    atomic_write_json,
    kana_sequence,
    normalize_kana_label,
    sha256_bytes,
)

log = logging.getLogger(__name__)

PROMPT_VERSION = "real-kana-v1"


class VisionUnavailable(RuntimeError):
    """Raised when no usable credentials or SDK are configured."""


class VisionClient(Protocol):  # pragma: no cover - structural type
    model: str

    def generate_json(self, prompt: str, images: Sequence[Image.Image], schema: dict[str, Any]) -> dict[str, Any]:
        ...


def _json_schema() -> dict[str, Any]:
    glyph = {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "x0": {"type": "integer"},
            "y0": {"type": "integer"},
            "x1": {"type": "integer"},
            "y1": {"type": "integer"},
            "confidence": {"type": "number"},
        },
        "required": ["label", "x0", "y0", "x1", "y1", "confidence"],
    }
    return {
        "type": "object",
        "properties": {
            "is_handwritten": {"type": "boolean"},
            "is_sound_effect": {"type": "boolean"},
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tilt_deg": {"type": "number"},
                        "glyphs": {"type": "array", "items": glyph},
                    },
                    "required": ["tilt_deg", "glyphs"],
                },
            },
        },
        "required": ["is_handwritten", "is_sound_effect", "lines"],
    }


SEGMENT_SCHEMA = _json_schema()

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "glyphs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "label": {"type": "string"},
                    "remove": {"type": "boolean"},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "remove", "confidence", "reason"],
            },
        }
    },
    "required": ["glyphs"],
}

HANDWRITING_SCHEMA = {
    "type": "object",
    "properties": {
        "handwritten": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["handwritten", "confidence", "reason"],
}


def segment_prompt(transcription: str, width: int, height: int) -> str:
    return f"""You are labeling a crop from a Japanese manga page for an OCR dataset.

Transcription given by a previous OCR step: {transcription!r}
The image is {width} x {height} pixels. The crop may contain one or more text
lines or columns, and may contain both kana and kanji.

Task:
1. Decide whether this is handwritten/lettered manga text (irregular strokes)
   or typeset printed text. Sound effects are separate.
2. For every **kana** (hiragana or katakana) that is clearly readable, return a
   tight bounding box around its ink and exactly one kana as its label.
   - Do NOT box kanji, Latin letters, digits, punctuation, long-dash/wave
     marks, hearts, or other symbols.
   - Do NOT box sound effects or decorative lettering.
   - If two kana touch, still give one box per kana.
   - Coordinates are in the ORIGINAL image, origin top-left, x right, y down.
     They must satisfy 0 <= x0 < x1 <= {width} and 0 <= y0 < y1 <= {height}.
   - Boxes may overlap only when the glyphs genuinely overlap; otherwise keep
     a small margin of background around the ink, never a neighbouring glyph.
3. For each line, estimate `tilt_deg`: the counter-clockwise rotation (PIL
   convention, positive = counter-clockwise) needed to make that line's
   reading axis upright. Use 0 if it is already upright. Small values are
   normal; never exceed 45 degrees.
4. Reading order: top-to-bottom for vertical lines, left-to-right for
   horizontal lines. Return lines in reading order (rightmost column first for
   vertical Japanese).

If the crop is not handwritten dialogue, set `is_handwritten` false.
If it is a sound effect, set `is_sound_effect` true.
Omit unreadable or uncertain kana rather than guessing. Return JSON only.
"""


def review_prompt(transcription: str, current: Sequence[tuple[int, str, str]]) -> str:
    rows = "\n".join(f"{i}: current label {label!r} ({reason})" for i, label, reason in current)
    return f"""You are auditing already-cut kana crops from a Japanese manga page.

Source OCR transcription: {transcription!r}
The contact sheet shows the crops in reading order, each with its index drawn
above it on a light checkerboard. Current records:
{rows}

For every index, decide:
- Is the crop really the kana written in `current label`? Watch for common
  confusions such as ん/人, せ/た, ユ/エ, ぷ/ぶ, わ/ね, ロ/口.
- Does the crop include a piece of a neighbouring character, a speech-bubble
  border, a sound effect, or an unrelated mark? If so, mark remove=true.
- If the label is wrong but the pixels are a clean single kana, return the
  corrected label (one hiragana or katakana character).
- If the crop is a kanji, symbol, punctuation or unreadable, remove it.
- Keep a crop when it is a clean, recognisable kana even if the OCR line does
  not contain that exact character; the OCR may be wrong.

Return one entry per index. `remove` must be boolean. Do not invent indices.
"""


def handwriting_prompt() -> str:
    return """Is this crop handwritten / hand-lettered Japanese manga text, or
typeset printed text? Handwriting has irregular stroke width, uneven spacing,
wobbling baselines and idiosyncratic shapes. Printed text is mechanically
regular. Ignore whether the characters are kana or kanji. Answer with a
boolean `handwritten`, a confidence, and a short reason. Return JSON only."""


def _encode_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def vision_scale(width: int, height: int, max_side: int) -> float:
    """Scale applied before sending an image to a vision model."""

    longest = max(1, int(max(width, height)))
    if longest >= 96:
        return min(float(max_side) / longest, 1.0)
    # Tiny crops are hard to inspect; enlarge them so the model sees crisp
    # pixels rather than inventing detail.
    return min(float(max_side) / longest, max(1.0, 96.0 / longest))


def prepare_for_vision(image: Image.Image, max_side: int = 1400) -> Image.Image:
    """Return the exact image sent to the model.

    Segmentation prompts must quote ``prepared.size``, not the original size,
    and boxes must be mapped back by the caller.
    """

    image = image.convert("RGB")
    width, height = image.size
    scale = vision_scale(width, height, max_side)
    if abs(scale - 1.0) < 1e-9:
        return image
    resample = Image.NEAREST if scale > 1.0 else Image.LANCZOS
    return image.resize((max(1, round(width * scale)), max(1, round(height * scale))), resample)



class CachedVision:
    """Disk-cached wrapper around a vision backend."""

    def __init__(
        self,
        backend: VisionClient,
        cache_dir: str | Path | None = None,
        *,
        max_side: int = 1400,
        use_cache: bool = True,
    ):
        self.backend = backend
        self.model = backend.model
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_side = max_side
        self.use_cache = use_cache
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_json(self, prompt: str, images: Sequence[Image.Image], schema: dict[str, Any]) -> dict[str, Any]:
        prepared = [prepare_for_vision(im, self.max_side) for im in images]
        blobs = [_encode_png(im) for im in prepared]
        key = sha256_bytes(
            json.dumps(
                {
                    "version": PROMPT_VERSION,
                    "model": self.model,
                    "prompt": prompt,
                    "schema": schema,
                    "images": [sha256_bytes(b) for b in blobs],
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        )
        cache_path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if cache_path and self.use_cache and cache_path.is_file():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))["response"]
            except (json.JSONDecodeError, KeyError):
                pass
        response = self.backend.generate_json(prompt, prepared, schema)
        if cache_path:
            atomic_write_json(cache_path, {"model": self.model, "prompt_version": PROMPT_VERSION, "response": response})
        return response


class GeminiVision:
    """Gemini via google-genai (API key or Vertex AI)."""

    def __init__(
        self,
        model: str = "gemini-3.5-flash-lite",
        *,
        api_key: str | None = None,
        project: str | None = None,
        location: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 5,
        retry_base_s: float = 2.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_base_s = retry_base_s
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise VisionUnavailable("install google-genai: pip install google-genai") from exc

        import os

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = location or os.environ.get("GOOGLE_CLOUD_LOCATION") or "us-central1"
        if key:
            self.client = genai.Client(api_key=key)
        elif project:
            self.client = genai.Client(vertexai=True, project=project, location=location)
        else:
            raise VisionUnavailable(
                "Set GEMINI_API_KEY (or GOOGLE_API_KEY), or GOOGLE_CLOUD_PROJECT for Vertex AI. "
                "For a local test, pass a fake client or use --offline."
            )
        from google.genai import types

        self._types = types

    def generate_json(self, prompt: str, images: Sequence[Image.Image], schema: dict[str, Any]) -> dict[str, Any]:
        parts: list[Any] = [self._types.Part.from_text(text=prompt)]
        for image in images:
            parts.append(self._types.Part.from_bytes(data=_encode_png(image), mime_type="image/png"))
        config = self._types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=8192,
        )
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.generate_content(model=self.model, contents=parts, config=config)
                text = getattr(response, "text", None)
                if not text:
                    raise RuntimeError("vision model returned an empty response")
                return _parse_json(text)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not _retryable(exc) or attempt >= self.max_retries:
                    raise
                delay = min(60.0, self.retry_base_s * (2**attempt)) * (0.7 + 0.6 * random.random())
                log.warning("vision call failed (%s); retrying in %.1fs", exc, delay)
                time.sleep(delay)
        raise RuntimeError(f"vision call failed: {last_exc}")


class OpenAICompatVision:
    """Any OpenAI-compatible vision endpoint (e.g. a local vLLM server)."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 5,
        retry_base_s: float = 2.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_base_s = retry_base_s
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise VisionUnavailable("install openai: pip install openai") from exc
        import os

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key and not base_url:
            raise VisionUnavailable("set OPENAI_API_KEY or --base-url for an OpenAI-compatible vision model")
        self.client = OpenAI(api_key=key or "local", base_url=base_url)

    def generate_json(self, prompt: str, images: Sequence[Image.Image], schema: dict[str, Any]) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt + "\n\nJSON schema:\n" + json.dumps(schema)}]
        for image in images:
            b64 = base64.b64encode(_encode_png(image)).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                return _parse_json(response.choices[0].message.content or "")
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not _retryable(exc) or attempt >= self.max_retries:
                    raise
                delay = min(60.0, self.retry_base_s * (2**attempt)) * (0.7 + 0.6 * random.random())
                log.warning("vision call failed (%s); retrying in %.1fs", exc, delay)
                time.sleep(delay)
        raise RuntimeError(f"vision call failed: {last_exc}")


def _retryable(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(word in text for word in ("api key", "unauthenticated", "permission", "invalid_argument", "not found")):
        return False
    return any(word in text for word in ("429", "500", "502", "503", "504", "timeout", "temporar", "rate", "overload"))


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("vision response is not a JSON object")
    return obj


def build_vision(cfg: KanaConfig, *, use_cache: bool = True, **kwargs: Any) -> CachedVision:
    provider = str(kwargs.pop("provider", cfg.vision_provider)).lower()
    if provider in {"gemini", "google", "vertex"}:
        backend: VisionClient = GeminiVision(
            kwargs.pop("model", cfg.vision_model),
            temperature=cfg.vision_temperature,
            max_retries=cfg.vision_max_retries,
            retry_base_s=cfg.vision_retry_base_s,
            **kwargs,
        )
    elif provider in {"openai", "local", "compatible"}:
        backend = OpenAICompatVision(
            kwargs.pop("model", cfg.vision_model),
            temperature=cfg.vision_temperature,
            max_retries=cfg.vision_max_retries,
            retry_base_s=cfg.vision_retry_base_s,
            **kwargs,
        )
    else:
        raise ValueError(f"unknown vision provider: {provider}")
    paths = ensure_bank_dirs(cfg.root)
    return CachedVision(backend, cache_dir=paths["cache"], max_side=cfg.vision_upscale_max_side, use_cache=use_cache)


# ---------------------------------------------------------------------------
# Validation


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    return inter / float(area_a + area_b - inter)


def normalise_segmentation(
    response: dict[str, Any],
    *,
    width: int,
    height: int,
    transcription: str,
    cfg: KanaConfig,
    min_confidence: float | None = None,
) -> dict[str, Any]:
    """Validate and clamp a raw model response.  Pure function, easy to test."""

    if not isinstance(response, dict):
        raise ValueError("response must be an object")
    threshold = cfg.min_glyph_confidence if min_confidence is None else float(min_confidence)
    is_handwritten = bool(response.get("is_handwritten", True))
    is_sfx = bool(response.get("is_sound_effect", False))
    warnings: list[str] = []
    lines_out: list[dict[str, Any]] = []
    total = 0
    for line_index, line in enumerate(response.get("lines") or []):
        if not isinstance(line, dict):
            continue
        try:
            tilt = float(line.get("tilt_deg", 0.0) or 0.0)
        except (TypeError, ValueError):
            tilt = 0.0
        tilt = float(np.clip(tilt, -45.0, 45.0))
        glyphs: list[dict[str, Any]] = []
        seen_boxes: list[tuple[int, int, int, int]] = []
        for raw in line.get("glyphs") or []:
            if not isinstance(raw, dict):
                continue
            label = normalize_kana_label(str(raw.get("label") or ""))
            if label is None:
                warnings.append(f"dropped non-kana label {raw.get('label')!r}")
                continue
            try:
                x0 = int(round(float(raw.get("x0", 0))))
                y0 = int(round(float(raw.get("y0", 0))))
                x1 = int(round(float(raw.get("x1", 0))))
                y1 = int(round(float(raw.get("y1", 0))))
                conf = float(raw.get("confidence", 0.0))
            except (TypeError, ValueError):
                warnings.append(f"dropped malformed box for {label}")
                continue
            x0, x1 = sorted((max(0, min(width - 1, x0)), max(1, min(width, x1))))
            y0, y1 = sorted((max(0, min(height - 1, y0)), max(1, min(height, y1))))
            if x1 - x0 < cfg.min_glyph_box_side or y1 - y0 < cfg.min_glyph_box_side:
                warnings.append(f"dropped tiny box for {label}")
                continue
            if conf < threshold:
                warnings.append(f"dropped low-confidence {label} ({conf:.2f})")
                continue
            box = (x0, y0, x1, y1)
            if any(_iou(box, prev) > 0.60 for prev in seen_boxes):
                warnings.append(f"dropped duplicate/overlapping box for {label}")
                continue
            seen_boxes.append(box)
            glyphs.append(
                {
                    "label": label,
                    "box": [x0, y0, x1, y1],
                    "confidence": float(np.clip(conf, 0.0, 1.0)),
                    "line_index": line_index,
                    "line_tilt_deg": tilt,
                }
            )
        if glyphs:
            lines_out.append({"tilt_deg": tilt, "glyphs": glyphs})
            total += len(glyphs)

    expected = kana_sequence(transcription)
    got = [g["label"] for line in lines_out for g in line["glyphs"]]
    match = SequenceMatcher(a=expected, b=got).ratio() if expected else 0.0
    status = "ok"
    if is_sfx:
        status = "sound_effect"
    elif not is_handwritten:
        status = "not_handwritten"
    elif total == 0:
        status = "no_glyphs"
    return {
        "status": status,
        "is_handwritten": is_handwritten,
        "is_sound_effect": is_sfx,
        "lines": lines_out,
        "glyph_count": total,
        "expected_kana": expected,
        "sequence_match": round(float(match), 4),
        "sequence_ok": bool(expected and match >= 0.80),
        "warnings": warnings,
    }


def apply_review(
    response: dict[str, Any],
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Apply a review response to glyph records.  Pure function."""

    by_index: dict[int, dict[str, Any]] = {}
    for raw in response.get("glyphs") or []:
        if not isinstance(raw, dict):
            continue
        try:
            index = int(raw.get("index"))
        except (TypeError, ValueError):
            continue
        by_index[index] = raw
    out: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, rec in enumerate(records):
        correction = by_index.get(index)
        if correction is None:
            out.append(rec)
            continue
        if bool(correction.get("remove", False)):
            warnings.append(f"review removed index {index}")
            continue
        label = normalize_kana_label(str(correction.get("label") or ""))
        if label and label != rec.get("label"):
            warnings.append(f"review changed {rec.get('label')} -> {label} at index {index}")
            rec = {**rec, "label": label, "review_changed": True}
        try:
            conf = float(correction.get("confidence", rec.get("confidence", 0.0)))
            rec = {**rec, "review_confidence": float(np.clip(conf, 0.0, 1.0))}
        except (TypeError, ValueError):
            pass
        if correction.get("reason"):
            rec = {**rec, "review_reason": str(correction["reason"])[:300]}
        out.append(rec)
    return out, warnings


def label_handwriting(client: CachedVision, image: Image.Image) -> dict[str, Any]:
    response = client.generate_json(handwriting_prompt(), [image], HANDWRITING_SCHEMA)
    return {
        "handwritten": bool(response.get("handwritten", False)),
        "confidence": float(response.get("confidence", 0.0) or 0.0),
        "reason": str(response.get("reason", ""))[:300],
    }
