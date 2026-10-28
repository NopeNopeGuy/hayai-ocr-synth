"""Handwriting classification with OpenRouter's Mercury Decide model.

Mercury Decide is not a generative vision model: it answers narrow, typed
questions about a state.  For this pipeline we ask one ``noul`` question
(probability of "yes") and use the result as a handwritten-vs-printed score.

The API key is read from ``OPENROUTER_API_KEY`` by default.  Responses are
cached by model/question/image hash so a resumed labelling or harvest run does
not pay for the same crop twice.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from manga_ocr_dev.real_kana.utils import atomic_write_json, load_image, read_json, sha256_bytes

log = logging.getLogger(__name__)

DEFAULT_MODEL = "inception/mercury-decide:free"
DEFAULT_URL = "https://openrouter.ai/api/alpha/decisions"
QUESTION_KEY = "is_handwritten"
QUESTION_VERSION = "handwriting-v1"
STATE_TEXT = "A crop of Japanese manga text."

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class MercuryUnavailable(RuntimeError):
    """Raised when the Mercury endpoint cannot be used."""


def _prepare_png(image: Image.Image | str | Path, max_side: int) -> bytes:
    """Return a compact PNG, enlarging tiny crops and downscaling huge ones."""

    if isinstance(image, (str, Path)):
        image = load_image(image, "RGB")
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image or a path")
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    longest = max(1, int(max(width, height)))
    target_min = min(96, max(1, int(max_side)))
    if longest < target_min:
        scale = target_min / longest
    elif longest > max_side:
        scale = max_side / longest
    else:
        scale = 1.0
    if abs(scale - 1.0) > 1e-6:
        image = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _extract_noul(payload: dict[str, Any]) -> float:
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise MercuryUnavailable("Mercury response has no 'answers' object")
    answer = answers.get(QUESTION_KEY)
    if not isinstance(answer, dict):
        raise MercuryUnavailable(f"Mercury response has no {QUESTION_KEY!r} answer")
    value = answer.get("noul")
    if not isinstance(value, (int, float)):
        raise MercuryUnavailable(f"Mercury answer has no numeric 'noul': {answer!r}")
    return float(min(1.0, max(0.0, float(value))))


class MercuryDecisionClient:
    """Small client for one binary Mercury question: handwritten or printed."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        url: str | None = None,
        timeout_s: float = 120.0,
        retries: int = 4,
        retry_base_s: float = 2.0,
        max_side: int = 768,
        cache_dir: str | Path | None = None,
        referer: str = "https://github.com/manga-ocr-better-synth",
        title: str = "manga-ocr-better-synth",
        session: requests.Session | None = None,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise MercuryUnavailable(
                "OPENROUTER_API_KEY is not set; export it or pass --openrouter-api-key"
            )
        self.model = model or os.environ.get("REAL_KANA_MERCURY_MODEL") or DEFAULT_MODEL
        self.url = url or os.environ.get("REAL_KANA_MERCURY_URL") or DEFAULT_URL
        self.timeout_s = float(timeout_s)
        self.retries = max(0, int(retries))
        self.retry_base_s = float(retry_base_s)
        self.max_side = max(32, int(max_side))
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self.referer = referer
        self.title = title
        self._session = session or requests.Session()

    @property
    def method(self) -> str:
        return "mercury_noul"

    def _payload(self, png: bytes) -> dict[str, Any]:
        encoded = base64.b64encode(png).decode("ascii")
        return {
            "model": self.model,
            "state": [
                STATE_TEXT,
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            ],
            "questions": {
                QUESTION_KEY: {
                    "type": "noul",
                    "instructions": (
                        "Is the text in this manga crop handwritten or hand-lettered, "
                        "rather than typeset/printed?"
                    ),
                    "criteria": {
                        "true": "Irregular, hand-drawn strokes or hand-lettered text",
                        "false": "Mechanically regular typeset or printed text",
                    },
                }
            },
        }

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.referer,
            "X-OpenRouter-Title": self.title,
        }
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self._session.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_s,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.retries:
                    delay = self.retry_base_s * (2**attempt)
                    log.warning("Mercury request failed (%s); retrying in %.1fs", exc, delay)
                    time.sleep(delay)
                    continue
                raise MercuryUnavailable(f"Mercury request failed: {exc}") from exc

            if response.status_code in _RETRYABLE_STATUS and attempt < self.retries:
                delay = self.retry_base_s * (2**attempt)
                log.warning("Mercury HTTP %s; retrying in %.1fs", response.status_code, delay)
                time.sleep(delay)
                continue
            if response.status_code >= 400:
                raise MercuryUnavailable(
                    f"Mercury HTTP {response.status_code}: {response.text[:500]}"
                )
            try:
                data = response.json()
            except ValueError as exc:
                raise MercuryUnavailable(f"Mercury returned invalid JSON: {response.text[:300]}") from exc
            if not isinstance(data, dict):
                raise MercuryUnavailable("Mercury response is not a JSON object")
            return data

        raise MercuryUnavailable(f"Mercury request failed: {last_error}")

    def _cache_path(self, png: bytes) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = sha256_bytes(
            png
            + b"\0"
            + self.model.encode("utf-8")
            + b"\0"
            + QUESTION_VERSION.encode("utf-8")
            + b"\0"
            + self.url.encode("utf-8")
        )
        return self.cache_dir / f"{digest}.json"

    def label(self, image: Image.Image | str | Path, *, threshold: float = 0.5) -> dict[str, Any]:
        """Return the handwriting decision plus the raw ``noul`` probability."""

        png = _prepare_png(image, self.max_side)
        cache_path = self._cache_path(png)
        if cache_path is not None and cache_path.is_file():
            cached = read_json(cache_path, default=None)
            if isinstance(cached, dict) and isinstance(cached.get("noul"), (int, float)):
                noul = float(cached["noul"])
                return {**cached, "handwritten": bool(noul >= float(threshold))}

        data = self._request(self._payload(png))
        noul = _extract_noul(data)
        result = {
            "handwritten": bool(noul >= float(threshold)),
            "confidence": noul,
            "noul": noul,
            "method": self.method,
            "model": str(data.get("model") or self.model),
            "usage": data.get("usage") or {},
        }
        if cache_path is not None:
            atomic_write_json(cache_path, result)
        return result

    def score_handwriting(self, image: Image.Image | str | Path) -> float:
        """Return the probability that the crop is handwritten (0..1)."""

        return float(self.label(image)["noul"])

    def close(self) -> None:
        self._session.close()
