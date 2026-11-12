#!/usr/bin/env python3
"""Re-score every glyph in the real-kana bank with MiniCPM-OCR-Jev.

The bank is a directory with ``manifest.jsonl`` and tightly cropped kana PNGs
(see ``manga_ocr_dev.real_kana``).  This tool sends each glyph image together
with its manifest label to ModelBest's MiniCPM-OCR-Jev scoring endpoint and
records ``prob_yes`` -- the model's estimate that the label matches the image.
It does not re-recognise the glyph and does not change the label.

Typical use::

    export JEV_API_KEY='...'
    python tools/recheck_real_kana_jev.py score --limit 128   # smoke test
    python tools/recheck_real_kana_jev.py score               # all glyphs
    python tools/recheck_real_kana_jev.py report --montage 300
    python tools/recheck_real_kana_jev.py apply --min-score 0.5

``score`` is resumable: successfully scored glyphs are appended to
``<out>/scores.jsonl`` and skipped on the next run.  ``apply`` materialises a
filtered bank (hardlinks by default) whose manifest is loadable by
``KanaBank.load``.  Nothing is deleted from the source bank.

The API key is read from ``JEV_API_KEY`` (or ``--api-key-file``); it is never
written to disk or to the log.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont

try:
    from manga_ocr_dev.real_kana.utils import (
        atomic_write_json,
        iter_jsonl,
        read_json,
        sha256_file,
    )
except ModuleNotFoundError:  # running the script directly from a source checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from manga_ocr_dev.real_kana.utils import (
        atomic_write_json,
        iter_jsonl,
        read_json,
        sha256_file,
    )

DEFAULT_BASE_URL = "https://api.modelbest.ai/v1"
DEFAULT_MODEL = "MiniCPM-OCR-Jev"
DEFAULT_BANK = os.environ.get("REAL_KANA_BANK", "/mnt/real-kana-final")
MAX_BATCH = 64
PNG_PREFIX = "data:image/png;base64,"

log = logging.getLogger("recheck")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def batched(iterable: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def percentile(sorted_values: Sequence[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = pos - lower
    return float(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * frac)


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def setup_logging(log_path: Path | None) -> None:
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    log.addHandler(stream)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        log.addHandler(file_handler)


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key.strip()
    if args.api_key_file:
        try:
            return Path(args.api_key_file).expanduser().read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SystemExit(f"cannot read --api-key-file: {exc}") from exc
    key = os.environ.get("JEV_API_KEY", "").strip()
    if key:
        return key
    raise SystemExit(
        "no API key: set JEV_API_KEY, or pass --api-key-file, or --api-key "
        "(the key is never logged or written to the output directory)"
    )


# ---------------------------------------------------------------------------
# work items and image preparation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WorkItem:
    glyph_id: str
    label: str
    crop_id: str
    path: Path
    attempts: int = 0


@dataclass(slots=True)
class Outcome:
    glyph_id: str
    label: str
    crop_id: str
    verdict: str
    prob_yes: float | None
    prob_no: float | None
    detail: str = ""

    @property
    def scored(self) -> bool:
        return (
            self.verdict == "scored"
            and self.prob_yes is not None
            and 0.0 <= self.prob_yes <= 1.0
        )


class ScoreError(Exception):
    """A batch could not be scored for a non-fatal reason."""


class TransientError(ScoreError):
    """Retries were exhausted; treat the items as request errors."""


class SplitBatch(ScoreError):
    """The server rejected this batch; retry its items in smaller pieces."""


class FatalAPIError(Exception):
    """Abort the run (auth, billing, wrong endpoint, malformed request...)."""


def encode_glyph(path: Path, args: argparse.Namespace) -> str:
    """Flatten a glyph onto white, pad it, scale it and return a PNG data URI."""

    try:
        with Image.open(path) as source:
            image = source.convert("RGBA")
    except OSError as exc:
        raise ScoreError(f"cannot read image: {exc}") from exc
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(background, image).convert("RGB")

    if args.pad > 0:
        pad = int(args.pad)
        padded = Image.new("RGB", (image.width + 2 * pad, image.height + 2 * pad), "white")
        padded.paste(image, (pad, pad))
        image = padded

    width, height = image.size
    scale = 1.0
    short_side = min(width, height)
    if short_side < args.min_side:
        scale = min(args.min_side / short_side, args.max_upscale)
    if max(width, height) * scale > args.max_side:
        scale = args.max_side / max(width, height)
    if abs(scale - 1.0) > 1e-3:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=6)
    return PNG_PREFIX + base64.b64encode(buffer.getvalue()).decode("ascii")


def make_outcome(item: WorkItem, verdict: str, prob: Any, detail: str = "", prob_no: Any = None) -> Outcome:
    def as_prob(value: Any) -> float | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(number):
            return None
        return round(number, 6)

    return Outcome(
        glyph_id=item.glyph_id,
        label=item.label,
        crop_id=item.crop_id,
        verdict=verdict,
        prob_yes=as_prob(prob),
        prob_no=as_prob(prob_no),
        detail=(detail or "")[:400],
    )


def outcome_to_record(outcome: Outcome, attempts: int, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "glyph_id": outcome.glyph_id,
        "label": outcome.label,
        "crop_id": outcome.crop_id,
        "verdict": outcome.verdict,
        "prob_yes": outcome.prob_yes,
        "prob_no": outcome.prob_no,
        "detail": outcome.detail,
        "attempts": attempts,
        "threshold": args.threshold,
        "model": args.model,
        "scored_at": utc_now(),
    }


def score_value(record: dict[str, Any] | None) -> float | None:
    """Return a valid 0..1 JEV score, or None for a failed/invalid record."""
    if not record or record.get("verdict") != "scored":
        return None
    try:
        value = float(record.get("prob_yes"))
    except (TypeError, ValueError):
        return None
    if not 0.0 <= value <= 1.0:
        return None
    return value


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


class Context:
    def __init__(self, args: argparse.Namespace, api_key: str):
        self.args = args
        self.api_key = api_key
        self.url = args.base_url.rstrip("/") + "/chat/completions"
        self.local = threading.local()

    def session(self) -> requests.Session:
        session = getattr(self.local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }
            )
            self.local.session = session
        return session


def backoff_seconds(attempt: int, response: requests.Response | None = None) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, min(120.0, float(retry_after)))
            except ValueError:
                try:
                    when = parsedate_to_datetime(retry_after)
                    delta = (when - datetime.now(timezone.utc)).total_seconds()
                    return max(0.0, min(120.0, delta))
                except (TypeError, ValueError):
                    pass
    return min(60.0, (2.0 ** attempt)) * (0.75 + 0.5 * random.random())


def short_body(response: requests.Response) -> str:
    try:
        text = response.text
    except (requests.RequestException, UnicodeError):
        return ""
    text = " ".join(text.split())
    return text[:500]


def parse_response(body: dict[str, Any], case_ids: Sequence[str]) -> list[Outcome]:
    content = None
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            content = message.get("content")
    data: Any = None
    if isinstance(content, dict):
        data = content
    elif isinstance(content, str):
        try:
            data = json.loads(strip_code_fence(content))
        except json.JSONDecodeError:
            data = None
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        data = body if isinstance(body.get("results"), list) else None
    if not isinstance(data, dict):
        raise TransientError("response has no score payload")

    results = [row for row in data["results"] if isinstance(row, dict)]
    by_id = {str(row.get("case_id")): row for row in results if row.get("case_id") is not None}
    aligned: list[dict[str, Any] | None] = []
    for index, case_id in enumerate(case_ids):
        row = by_id.get(case_id)
        if row is None and len(results) == len(case_ids):
            row = results[index]
        aligned.append(row)

    outcomes: list[Outcome] = []
    for row in aligned:
        if row is None:
            outcomes.append(
                Outcome("", "", "", "missing_result", None, None, "no result for this case_id")
            )
            continue
        verdict = str(row.get("verdict") or "missing_verdict")
        outcomes.append(
            Outcome(
                "",
                "",
                "",
                verdict,
                row.get("prob_yes"),
                row.get("prob_no"),
                str(row.get("detail") or ""),
            )
        )
    return outcomes


def post_batch(ctx: Context, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    args = ctx.args
    attempts = max(1, int(args.batch_retries) + 1)
    last_error = ""
    for attempt in range(attempts):
        response: requests.Response | None = None
        try:
            response = ctx.session().post(
                ctx.url,
                json=payload,
                timeout=(min(30.0, args.timeout), args.timeout),
            )
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 >= attempts:
                raise TransientError(last_error) from exc
            delay = backoff_seconds(attempt)
            log.warning("transport error (%s); retrying in %.1fs", last_error, delay)
            time.sleep(delay)
            continue

        status = response.status_code
        if status == 413:
            raise SplitBatch("HTTP 413: payload too large")
        if status in (401, 402, 403, 404):
            raise FatalAPIError(f"HTTP {status}: {short_body(response)}")
        if status == 429 or 500 <= status < 600:
            last_error = f"HTTP {status}: {short_body(response)}"
            if attempt + 1 >= attempts:
                raise TransientError(last_error)
            delay = backoff_seconds(attempt, response)
            log.warning("%s; retrying in %.1fs", last_error, delay)
            time.sleep(delay)
            continue
        if status == 400 and "moderation_invalid_input" in response.text:
            raise SplitBatch("HTTP 400: moderation_invalid_input")
        if status in (400, 422):
            raise FatalAPIError(f"HTTP {status}: {short_body(response)}")
        if not response.ok:
            raise FatalAPIError(f"HTTP {status}: {short_body(response)}")

        try:
            body = response.json()
        except ValueError as exc:
            last_error = "response is not JSON"
            if attempt + 1 >= attempts:
                raise TransientError(last_error) from exc
            time.sleep(backoff_seconds(attempt))
            continue

        usage = 0
        try:
            usage = int((body.get("usage") or {}).get("total_tokens") or 0)
        except (TypeError, ValueError):
            usage = 0
        return body, usage
    raise TransientError(last_error or "request failed")


def score_once(items: Sequence[WorkItem], ctx: Context) -> tuple[list[Outcome], int]:
    outcomes: list[Outcome | None] = [None] * len(items)
    prepared: list[tuple[WorkItem, str]] = []
    positions: list[int] = []
    for index, item in enumerate(items):
        try:
            uri = encode_glyph(item.path, ctx.args)
        except ScoreError as exc:
            outcomes[index] = make_outcome(item, "image_error", None, str(exc))
            continue
        prepared.append((item, uri))
        positions.append(index)
    if not prepared:
        return [row for row in outcomes if row is not None], 0

    content: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for item, uri in prepared:
        content.append({"type": "image_url", "image_url": {"url": uri}})
        content.append({"type": "text", "text": item.label})
        cases.append(
            {
                "case_id": item.glyph_id,
                "image": uri,
                "ocr_text": item.label,
                "category": "text",
            }
        )
    payload = {
        "model": ctx.args.model,
        "messages": [{"role": "user", "content": content}],
        "minicpm_ocr_jev": {
            "mode": "score",
            "category": "text",
            "threshold": ctx.args.threshold,
            "items": cases,
        },
        "stream": False,
    }

    body, usage = post_batch(ctx, payload)
    parsed = parse_response(body, [item.glyph_id for item, _ in prepared])
    for offset, ((item, _), result) in enumerate(zip(prepared, parsed)):
        outcomes[positions[offset]] = make_outcome(
            item, result.verdict, result.prob_yes, result.detail, result.prob_no
        )

    filled = [
        row if row is not None else make_outcome(items[index], "missing_result", None, "")
        for index, row in enumerate(outcomes)
    ]
    return filled, usage


def run_items(items: Sequence[WorkItem], ctx: Context) -> tuple[list[Outcome], int]:
    if not items:
        return [], 0
    try:
        return score_once(items, ctx)
    except SplitBatch as exc:
        if len(items) == 1:
            return [make_outcome(items[0], "rejected", None, str(exc))], 0
        middle = len(items) // 2
        left, left_usage = run_items(items[:middle], ctx)
        right, right_usage = run_items(items[middle:], ctx)
        return left + right, left_usage + right_usage
    except TransientError as exc:
        return [make_outcome(item, "request_error", None, str(exc)) for item in items], 0


def check_model(ctx: Context) -> None:
    url = ctx.args.base_url.rstrip("/") + "/models"
    try:
        response = ctx.session().get(url, timeout=(15.0, 60.0))
    except requests.RequestException as exc:
        raise FatalAPIError(f"cannot reach {url}: {exc}") from exc
    if response.status_code in (401, 402, 403):
        raise FatalAPIError(f"model check failed: HTTP {response.status_code}: {short_body(response)}")
    if not response.ok:
        log.warning("model list returned HTTP %s; continuing anyway", response.status_code)
        return
    try:
        body = response.json()
    except ValueError:
        log.warning("model list is not JSON; continuing anyway")
        return
    if not isinstance(body, dict):
        log.warning("model list has an unexpected shape; continuing anyway")
        return
    ids = [str(row.get("id")) for row in body.get("data", []) if isinstance(row, dict)]
    if ids and ctx.args.model not in ids and ctx.args.model.lower() not in {value.lower() for value in ids}:
        raise FatalAPIError(
            f"model {ctx.args.model!r} is not in the key's model list; available: {', '.join(ids[:20])}"
        )
    log.info("model check ok: %s", ctx.args.model)


# ---------------------------------------------------------------------------
# score command
# ---------------------------------------------------------------------------


def load_scored_ids(path: Path) -> set[str]:
    scored: set[str] = set()
    if not path.is_file():
        return scored
    for row in iter_jsonl(path):
        if row.get("verdict") != "scored" or not row.get("glyph_id"):
            continue
        if score_value(row) is not None:
            scored.add(str(row["glyph_id"]))
    return scored


def iter_work(bank: Path, skip: set[str], limit: int) -> Iterator[WorkItem]:
    """Yield unscored glyphs; ``limit`` caps the manifest rows inspected."""
    examined = 0
    for row in iter_jsonl(bank / "manifest.jsonl"):
        if limit and examined >= limit:
            return
        examined += 1
        glyph_id = str(row.get("glyph_id") or "")
        if not glyph_id or glyph_id in skip:
            continue
        raw_path = row.get("glyph_path") or row.get("path")
        if not raw_path:
            log.warning("skipping %s: no glyph_path", glyph_id)
            continue
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = bank / path
        yield WorkItem(
            glyph_id=glyph_id,
            label=str(row.get("label") or ""),
            crop_id=str(row.get("crop_id") or ""),
            path=path,
        )


def write_record(handle: io.TextIOBase, record: dict[str, Any], sync: bool = False) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    if sync:
        handle.flush()
        os.fsync(handle.fileno())


def prepare_run_meta(out: Path, bank: Path, fingerprint: str, args: argparse.Namespace) -> dict[str, Any]:
    meta_path = out / "run.meta.json"
    expected = {
        "bank": str(bank.resolve()),
        "manifest_sha256": fingerprint,
        "base_url": args.base_url,
        "model": args.model,
    }
    if meta_path.is_file():
        existing = read_json(meta_path, {}) or {}
        if existing.get("manifest_sha256") not in (None, fingerprint):
            raise SystemExit(
                f"{meta_path} belongs to a different manifest "
                f"({existing.get('manifest_sha256')} != {fingerprint}); use a fresh --out"
            )
        for key in ("base_url", "model"):
            if existing.get(key) not in (None, expected[key]):
                raise SystemExit(
                    f"{meta_path} was written for {key}={existing.get(key)!r}, "
                    f"not {expected[key]!r}; use a fresh --out"
                )
        return existing
    atomic_write_json(meta_path, {**expected, "created_at": utc_now()})
    return expected


def summarize_scores(path: Path, threshold: float = 0.5) -> dict[str, Any]:
    verdicts: Counter[str] = Counter()
    scored = 0
    passed = 0
    for row in load_scores(path).values():
        verdict = str(row.get("verdict") or "unknown")
        verdicts[verdict] += 1
        value = score_value(row)
        if value is not None:
            scored += 1
            passed += value >= threshold
    return {
        "scored": scored,
        f"pass_at_{threshold:g}": passed,
        "verdicts": dict(sorted(verdicts.items())),
    }


def cmd_score(args: argparse.Namespace, bank: Path, out: Path) -> int:
    manifest = bank / "manifest.jsonl"
    if not manifest.is_file():
        raise SystemExit(f"no manifest at {manifest}")
    out.mkdir(parents=True, exist_ok=True)
    scores_path = out / "scores.jsonl"

    skip = load_scored_ids(scores_path)
    with manifest.open(encoding="utf-8") as handle:
        total_glyphs = sum(1 for _ in handle)
    log.info("bank %s: %d glyphs, %d already scored", bank, total_glyphs, len(skip))

    work = list(iter_work(bank, skip, args.limit))
    if not work:
        log.info("nothing to do: no unscored glyphs in the selected scope")
        atomic_write_json(
            out / "summary.json",
            {
                "finished_at": utc_now(),
                "total": total_glyphs,
                **summarize_scores(scores_path, args.threshold),
            },
        )
        return 0
    total_work = len(work)
    log.info(
        "this run: %d glyphs, %d per request, concurrency %d",
        total_work,
        args.batch_size,
        args.concurrency,
    )

    if args.dry_run:
        sample = work[: args.batch_size]
        sizes = []
        for item in sample:
            uri = encode_glyph(item.path, args)
            sizes.append((len(uri), item.glyph_id, item.label))
        sizes.sort(reverse=True)
        payload_bytes = sum(size for size, _, _ in sizes) * 2  # messages + items
        log.info("dry run: %d images, largest data URIs:", len(sizes))
        for size, glyph_id, label in sizes[:5]:
            log.info("  %7d bytes  %s  %s", size, glyph_id, label)
        log.info("estimated JSON body for one batch: ~%.1f KiB", payload_bytes / 1024)
        try:
            first = Image.open(sample[0].path).convert("RGBA")
            background = Image.new("RGBA", first.size, (255, 255, 255, 255))
            Image.alpha_composite(background, first).convert("RGB").save(out / "dry_run_sample.png")
            log.info("wrote %s for a visual sanity check", out / "dry_run_sample.png")
        except OSError as exc:
            log.warning("could not write dry_run_sample.png: %s", exc)
        return 0

    fingerprint = sha256_file(manifest)
    prepare_run_meta(out, bank, fingerprint, args)
    ctx = Context(args, resolve_api_key(args))
    if not args.skip_model_check:
        check_model(ctx)

    batches = list(batched(work, args.batch_size))
    started = time.monotonic()
    scored_this_run = 0
    usage_total = 0
    done_items = 0
    completed_batches = 0
    pass_count = 0
    fatal: str | None = None
    internal_errors = 0
    # (item, last outcome) for glyphs that did not reach ``scored``.
    retry: list[tuple[WorkItem, Outcome]] = []

    scores_handle = scores_path.open("a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(run_items, batch, ctx): batch for batch in batches}
            try:
                for future in as_completed(futures):
                    batch = futures[future]
                    done_items += len(batch)
                    completed_batches += 1
                    try:
                        outcomes, usage = future.result()
                    except FatalAPIError as exc:
                        fatal = str(exc)
                        for pending in futures:
                            pending.cancel()
                        break
                    except Exception as exc:
                        internal_errors += 1
                        log.exception("batch crashed; will retry its %d glyphs", len(batch))
                        for item in batch:
                            item.attempts = max(1, item.attempts)
                            retry.append((item, make_outcome(item, "internal_error", None, str(exc))))
                        continue

                    usage_total += usage
                    for item, outcome in zip(batch, outcomes):
                        item.attempts = max(1, item.attempts)
                        if outcome.scored:
                            write_record(scores_handle, outcome_to_record(outcome, item.attempts, args))
                            scored_this_run += 1
                            if (outcome.prob_yes or 0.0) >= args.threshold:
                                pass_count += 1
                        else:
                            retry.append((item, outcome))

                    if completed_batches % max(1, args.log_every) == 0:
                        elapsed = max(1e-6, time.monotonic() - started)
                        rate = done_items / elapsed
                        remaining = max(0, total_work - done_items)
                        eta = remaining / rate if rate > 0 else 0.0
                        log.info(
                            "progress %d/%d (%.1f%%) scored=%d unscored=%d pass=%.1f%% %.1f glyph/s ETA %s",
                            done_items,
                            total_work,
                            100.0 * done_items / total_work,
                            scored_this_run,
                            len(retry),
                            100.0 * pass_count / max(1, scored_this_run),
                            rate,
                            time.strftime("%H:%M:%S", time.gmtime(eta)),
                        )
            except KeyboardInterrupt:
                for pending in futures:
                    pending.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise

        for round_index in range(args.item_retries):
            if fatal is not None or not retry:
                break
            log.info("retry round %d: %d unscored items", round_index + 1, len(retry))
            pending = retry
            retry = []
            for batch_items in batched([item for item, _ in pending], args.batch_size):
                try:
                    outcomes, usage = run_items(batch_items, ctx)
                except FatalAPIError as exc:
                    fatal = str(exc)
                    retry.extend(
                        (item, make_outcome(item, "request_error", None, "aborted by a fatal API error"))
                        for item in batch_items
                    )
                    break
                usage_total += usage
                for item, outcome in zip(batch_items, outcomes):
                    item.attempts += 1
                    if outcome.scored:
                        write_record(scores_handle, outcome_to_record(outcome, item.attempts, args))
                        scored_this_run += 1
                        if (outcome.prob_yes or 0.0) >= args.threshold:
                            pass_count += 1
                    else:
                        retry.append((item, outcome))

        # Persist final non-scored outcomes so the report can show them.
        # Rerunning ``score`` retries them; only ``scored`` is skipped.
        if fatal is None:
            for item, outcome in retry:
                write_record(scores_handle, outcome_to_record(outcome, max(1, item.attempts), args))
        scores_handle.flush()
        os.fsync(scores_handle.fileno())
    except KeyboardInterrupt:
        log.warning(
            "interrupted; %d scored results are safe, rerun the same command to resume",
            scored_this_run,
        )
        return 130
    finally:
        scores_handle.close()

    summary = {
        "finished_at": utc_now(),
        "bank": str(bank),
        "manifest_sha256": fingerprint,
        "model": args.model,
        "base_url": args.base_url,
        "threshold": args.threshold,
        "total_glyphs": total_glyphs,
        "work_this_run": total_work,
        "scored_this_run": scored_this_run,
        "pass_this_run": pass_count,
        "unscored_this_run": len(retry),
        "usage_total_tokens": usage_total,
        "internal_errors": internal_errors,
        "elapsed_s": round(time.monotonic() - started, 1),
        "fatal": fatal,
        "overall": summarize_scores(scores_path, args.threshold),
    }
    atomic_write_json(out / "summary.json", summary)
    if fatal:
        log.error("fatal API error: %s", fatal)
        log.error("partial results are in %s; fix the problem and rerun to resume", scores_path)
        return 2
    log.info(
        "done: %d scored this run, %.1f%% passed >= %.2f, %d still unscored, %.1fs, %d tokens",
        scored_this_run,
        100.0 * pass_count / max(1, scored_this_run),
        args.threshold,
        len(retry),
        time.monotonic() - started,
        usage_total,
    )
    if retry:
        log.info("rerun the same command to retry the %d unscored glyphs", len(retry))
    return 0



# ---------------------------------------------------------------------------
# report command
# ---------------------------------------------------------------------------


def load_scores(path: Path) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return best
    for row in iter_jsonl(path):
        glyph_id = str(row.get("glyph_id") or "")
        if not glyph_id:
            continue
        previous = best.get(glyph_id)
        if previous is None or row.get("verdict") == "scored":
            best[glyph_id] = row
        # never replace a scored result with a failed one
    return best


def find_japanese_font() -> str | None:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    try:
        result = subprocess.run(
            ["fc-match", "-f", "%{file}", "Noto Sans CJK JP"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        path = result.stdout.strip()
        if path and Path(path).is_file():
            return path
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = find_japanese_font()
    if path:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def write_montage(
    path: Path,
    rows: Sequence[dict[str, Any]],
    bank: Path,
    threshold: float,
    columns: int = 8,
    cell: int = 128,
) -> None:
    if not rows:
        return
    font = load_font(14)
    rows_total = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows_total * cell), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        column = index % columns
        line = index // columns
        x = column * cell
        y = line * cell
        draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline=(210, 210, 210))
        try:
            with Image.open(bank / str(row["glyph_path"])) as source:
                glyph = source.convert("RGBA")
                background = Image.new("RGBA", glyph.size, (255, 255, 255, 255))
                glyph = Image.alpha_composite(background, glyph).convert("RGB")
        except OSError:
            continue
        glyph.thumbnail((cell - 14, cell - 34), Image.Resampling.LANCZOS)
        sheet.paste(glyph, (x + (cell - glyph.width) // 2, y + 6 + (cell - 34 - glyph.height) // 2))
        score = row.get("prob_yes")
        color = (180, 0, 0) if score is not None and float(score) < threshold else (20, 20, 20)
        draw.text((x + 4, y + cell - 26), f"{row.get('label', '?')} {float(score):.3f}", fill=color, font=font)
        draw.text((x + 4, y + cell - 12), str(row.get("glyph_id", ""))[-16:], fill=(120, 120, 120), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    log.info("wrote %s (%d lowest-scoring glyphs)", path, len(rows))


def cmd_report(args: argparse.Namespace, bank: Path, out: Path) -> int:
    scores_path = out / "scores.jsonl"
    if not scores_path.is_file():
        raise SystemExit(f"no scores at {scores_path}; run the score command first")
    scores = load_scores(scores_path)
    rows: dict[str, dict[str, Any]] = {}
    label_counts: Counter[str] = Counter()
    total = 0
    for row in iter_jsonl(bank / "manifest.jsonl"):
        total += 1
        glyph_id = str(row.get("glyph_id") or "")
        label_counts[str(row.get("label"))] += 1
        rows[glyph_id] = row

    scored = [scores[key] for key in rows if score_value(scores.get(key)) is not None]
    values = sorted(value for row in scored if (value := score_value(row)) is not None)
    verdicts = Counter(str(row.get("verdict")) for row in scores.values())
    missing = [key for key in rows if key not in scores]
    unscored = [key for key in rows if key in scores and scores[key].get("verdict") != "scored"]

    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    pass_table = {}
    for threshold in thresholds:
        count = sum(1 for value in values if value >= threshold)
        pass_table[f"{threshold:.2f}"] = count

    by_label: dict[str, dict[str, Any]] = {}
    label_values: dict[str, list[float]] = defaultdict(list)
    for glyph_id, row in rows.items():
        value = score_value(scores.get(glyph_id))
        if value is None:
            continue
        label_values[str(row.get("label"))].append(value)
    for label in sorted(label_counts):
        label_scores = sorted(label_values.get(label, []))
        by_label[label] = {
            "glyphs": label_counts[label],
            "scored": len(label_scores),
            "mean": round(sum(label_scores) / len(label_scores), 4) if label_scores else None,
            "p10": round(percentile(label_scores, 0.10), 4) if label_scores else None,
            "pass": sum(1 for value in label_scores if value >= args.threshold),
        }

    report = {
        "bank": str(bank),
        "scores": str(scores_path),
        "threshold": args.threshold,
        "glyphs_in_manifest": total,
        "scored": len(values),
        "unscored_in_manifest": len(unscored),
        "missing_from_results": len(missing),
        "verdicts": dict(sorted(verdicts.items())),
        "quantiles": {
            "min": round(values[0], 4) if values else None,
            "p01": round(percentile(values, 0.01), 4) if values else None,
            "p05": round(percentile(values, 0.05), 4) if values else None,
            "p10": round(percentile(values, 0.10), 4) if values else None,
            "p25": round(percentile(values, 0.25), 4) if values else None,
            "p50": round(percentile(values, 0.50), 4) if values else None,
            "p75": round(percentile(values, 0.75), 4) if values else None,
            "p90": round(percentile(values, 0.90), 4) if values else None,
            "p95": round(percentile(values, 0.95), 4) if values else None,
            "p99": round(percentile(values, 0.99), 4) if values else None,
            "max": round(values[-1], 4) if values else None,
            "mean": round(sum(values) / len(values), 4) if values else None,
        },
        "pass_at": pass_table,
        "labels": by_label,
    }
    atomic_write_json(out / "report.json", report)

    print()
    print(f"bank:     {bank}")
    print(f"scores:   {scores_path}")
    print(f"manifest: {total} glyphs, {len(label_counts)} labels")
    print(f"scored:   {len(values)}  ({100.0 * len(values) / max(1, total):.1f}%)")
    print(f"unscored: {len(unscored)}   missing: {len(missing)}")
    print(f"verdicts: {dict(sorted(verdicts.items()))}")
    if values:
        print()
        print("prob_yes:", "  ".join(f"{key}={value:.3f}" for key, value in report["quantiles"].items() if value is not None))
        print()
        print("threshold   kept      share")
        for threshold in thresholds:
            count = pass_table[f"{threshold:.2f}"]
            print(f"  >= {threshold:.2f}   {count:6d}   {100.0 * count / len(values):5.1f}%")

    ranked = [label for label in by_label if by_label[label]["scored"] >= 10]
    ranked.sort(key=lambda label: (by_label[label]["mean"] or 0.0))
    if ranked:
        print()
        print(f"worst labels (n>=10), threshold {args.threshold:.2f}:")
        for label in ranked[:15]:
            entry = by_label[label]
            print(f"  {label}  n={entry['scored']:5d}  mean={entry['mean']:.3f}  p10={entry['p10']:.3f}  pass={entry['pass']}")
        print()
        print("best labels (n>=10):")
        for label in reversed(ranked[-8:]):
            entry = by_label[label]
            print(f"  {label}  n={entry['scored']:5d}  mean={entry['mean']:.3f}  pass={entry['pass']}")

    if args.csv:
        csv_path = Path(args.csv).expanduser()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["glyph_id", "label", "crop_id", "verdict", "prob_yes"])
            for glyph_id, row in rows.items():
                score = scores.get(glyph_id, {})
                writer.writerow([
                    glyph_id,
                    row.get("label"),
                    row.get("crop_id"),
                    score.get("verdict"),
                    score.get("prob_yes"),
                ])
        print()
        print(f"wrote {csv_path}")

    if args.montage > 0 and values:
        lowest = sorted(scored, key=lambda row: score_value(row) or 0.0)[: args.montage]
        montage_rows = [
            {**rows[str(row["glyph_id"])], "prob_yes": score_value(row)}
            for row in lowest
            if str(row["glyph_id"]) in rows
        ]
        write_montage(out / "lowest_scores.png", montage_rows, bank, args.threshold)
    print()
    print(f"report written to {out / 'report.json'}")
    return 0


# ---------------------------------------------------------------------------
# apply command
# ---------------------------------------------------------------------------


def cmd_apply(args: argparse.Namespace, bank: Path, out: Path) -> int:
    scores_path = out / "scores.jsonl"
    if not scores_path.is_file():
        raise SystemExit(f"no scores at {scores_path}; run the score command first")
    scores = load_scores(scores_path)
    dest = Path(args.dest).expanduser() if args.dest else out / f"bank-min{args.min_score:g}"
    threshold = float(args.min_score)

    def keep(row: dict[str, Any]) -> bool:
        value = score_value(scores.get(str(row.get("glyph_id") or "")))
        if value is None:
            return bool(args.keep_unscored)
        return value >= threshold

    if args.dry_run:
        kept = dropped = 0
        kept_by_label: Counter[str] = Counter()
        dropped_by_label: Counter[str] = Counter()
        for row in iter_jsonl(bank / "manifest.jsonl"):
            label = str(row.get("label") or "")
            if keep(row) and (row.get("glyph_path") or row.get("path")):
                kept += 1
                kept_by_label[label] += 1
            else:
                dropped += 1
                dropped_by_label[label] += 1
        print(f"would keep {kept} glyphs, drop {dropped} (threshold {threshold:g})")
        print(f"labels: {len(kept_by_label)}; destination would be {dest}")
        if dropped_by_label:
            worst = dropped_by_label.most_common(10)
            print("most-dropped labels: " + ", ".join(f"{label}={count}" for label, count in worst))
        return 0

    if dest.exists() and any(dest.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{dest} is not empty; pass --overwrite to replace its bank files")
        for name in ("glyphs", "manifest.jsonl", "manifest.jsonl.tmp", "manifest.meta.json"):
            target = dest / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "glyphs").mkdir(exist_ok=True)

    kept = 0
    dropped = 0
    dropped_by_label: Counter[str] = Counter()
    kept_by_label: Counter[str] = Counter()
    kept_crops: set[str] = set()
    manifest_path = dest / "manifest.jsonl"
    tmp_path = manifest_path.with_suffix(".jsonl.tmp")
    bank_resolved = bank.resolve()

    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(bank / "manifest.jsonl"):
            label = str(row.get("label") or "")
            raw_path = row.get("glyph_path") or row.get("path")
            if not keep(row) or not raw_path:
                dropped += 1
                dropped_by_label[label] += 1
                continue
            source = Path(str(raw_path))
            if not source.is_absolute():
                source = bank / source
            try:
                relative = source.resolve().relative_to(bank_resolved)
            except ValueError:
                relative = Path("glyphs") / label / source.name
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                if args.copy:
                    shutil.copy2(source, target)
                else:
                    try:
                        os.link(source, target)
                    except OSError:
                        shutil.copy2(source, target)
            score = scores.get(str(row.get("glyph_id") or ""), {})
            updated = dict(row)
            updated["glyph_path"] = relative.as_posix()
            updated["jev_prob_yes"] = score.get("prob_yes")
            updated["jev_verdict"] = score.get("verdict")
            updated["jev_threshold"] = threshold
            handle.write(json.dumps(updated, ensure_ascii=False, separators=(",", ":")) + "\n")
            kept += 1
            kept_by_label[label] += 1
            if row.get("crop_id"):
                kept_crops.add(str(row["crop_id"]))
    os.replace(tmp_path, manifest_path)

    source_meta = read_json(bank / "manifest.meta.json", {}) or {}
    meta = dict(source_meta)
    config = dict(meta.get("config") or {})
    config["root"] = str(dest.resolve())
    meta["config"] = config
    meta["glyph_count"] = kept
    meta["labels"] = dict(sorted(kept_by_label.items()))
    meta["source_crops"] = len(kept_crops)
    meta["real_kana_pipeline"] = "manga_ocr_dev.real_kana"
    meta["jev_filter"] = {
        "source_bank": str(bank.resolve()),
        "scores": str(scores_path.resolve()),
        "min_score": threshold,
        "keep_unscored": bool(args.keep_unscored),
        "kept": kept,
        "dropped": dropped,
        "created_at": utc_now(),
    }
    atomic_write_json(dest / "manifest.meta.json", meta)

    print(f"kept {kept} glyphs, dropped {dropped} (threshold {threshold:g})")
    print(f"labels: {len(kept_by_label)}; source crops: {len(kept_crops)}")
    print(f"bank: {dest}")
    if dropped_by_label:
        worst = dropped_by_label.most_common(10)
        print("most-dropped labels: " + ", ".join(f"{label}={count}" for label, count in worst))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common(parser: argparse.ArgumentParser, suppress: bool = False) -> None:
    default = argparse.SUPPRESS if suppress else None
    parser.add_argument("--bank", default=default, help="bank root with manifest.jsonl")
    parser.add_argument("--out", default=default, help="output directory (default: <bank>/recheck)")
    parser.add_argument("--log", default=default, help="log file (default: <out>/recheck.log)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bank", default=DEFAULT_BANK, help=f"bank root (default: {DEFAULT_BANK})")
    parser.add_argument("--out", default=None, help="output directory (default: <bank>/recheck)")
    parser.add_argument("--log", default=None, help="log file (default: <out>/recheck.log)")

    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="score every glyph with MiniCPM-OCR-Jev (resumable)")
    add_common(score, suppress=True)
    score.add_argument("--api-key", default=None, help="API key (prefer JEV_API_KEY / --api-key-file)")
    score.add_argument("--api-key-file", default=None, help="file containing the API key")
    score.add_argument("--base-url", default=os.environ.get("JEV_BASE_URL", DEFAULT_BASE_URL))
    score.add_argument("--model", default=os.environ.get("JEV_MODEL", DEFAULT_MODEL))
    score.add_argument("--concurrency", type=int, default=8, help="parallel requests (default: 8)")
    score.add_argument("--batch-size", type=int, default=MAX_BATCH, help=f"items per request, max {MAX_BATCH}")
    score.add_argument("--threshold", type=float, default=0.5, help="API threshold and pass statistic (default: 0.5)")
    score.add_argument("--timeout", type=float, default=180.0, help="read timeout per request in seconds (default: 180)")
    score.add_argument("--batch-retries", type=int, default=5, help="retries per HTTP request (default: 5)")
    score.add_argument("--item-retries", type=int, default=2, help="retry rounds for unscored items (default: 2)")
    score.add_argument("--limit", type=int, default=0, help="only inspect the first N manifest rows (0 = all); useful for a smoke test")
    score.add_argument("--min-side", type=int, default=32, help="upscale tiny glyphs to this short side (default: 32)")
    score.add_argument("--max-side", type=int, default=256, help="downscale images above this side (default: 256)")
    score.add_argument("--max-upscale", type=float, default=4.0, help="cap for tiny-glyph upscaling (default: 4)")
    score.add_argument("--pad", type=int, default=2, help="white border in pixels before scaling (default: 2)")
    score.add_argument("--log-every", type=int, default=10, help="progress every N batches (default: 10)")
    score.add_argument("--skip-model-check", action="store_true", help="do not call /models first")
    score.add_argument("--dry-run", action="store_true", help="build one batch of images, call nothing, print sizes")

    report = sub.add_parser("report", help="summarise scores and write a low-score montage")
    add_common(report, suppress=True)
    report.add_argument("--threshold", type=float, default=0.5, help="pass threshold for tables (default: 0.5)")
    report.add_argument("--montage", type=int, default=200, help="number of lowest scores to render (default: 200, 0=off)")
    report.add_argument("--csv", default=None, help="optional CSV output path")

    apply_cmd = sub.add_parser("apply", help="build a filtered bank from the scores")
    add_common(apply_cmd, suppress=True)
    apply_cmd.add_argument("--min-score", type=float, default=0.5, help="keep glyphs with prob_yes >= this (default: 0.5)")
    apply_cmd.add_argument("--dest", default=None, help="destination bank (default: <out>/bank-min<score>)")
    apply_cmd.add_argument("--keep-unscored", action="store_true", help="keep glyphs without a scored result")
    apply_cmd.add_argument("--copy", action="store_true", help="copy files instead of hardlinking")
    apply_cmd.add_argument("--overwrite", action="store_true", help="replace an existing destination bank")
    apply_cmd.add_argument("--dry-run", action="store_true", help="print counts without writing")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "score":
        if not 1 <= args.batch_size <= MAX_BATCH:
            parser.error(f"--batch-size must be between 1 and {MAX_BATCH}")
        if args.concurrency < 1:
            parser.error("--concurrency must be >= 1")
        if not 0.0 <= args.threshold <= 1.0:
            parser.error("--threshold must be between 0 and 1")
        if args.log_every < 1:
            parser.error("--log-every must be >= 1")
        args.base_url = args.base_url.rstrip("/")
        args.model = args.model.strip()
    if args.command == "apply" and not 0.0 <= args.min_score <= 1.0:
        parser.error("--min-score must be between 0 and 1")
    if args.command == "report" and not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    bank = Path(args.bank).expanduser().resolve()
    out = Path(args.out).expanduser().resolve() if args.out else bank / "recheck"
    log_path = Path(args.log).expanduser() if args.log else out / "recheck.log"
    setup_logging(log_path)

    if args.command == "score":
        if args.skip_model_check:
            args._skip_model_check = True
        return cmd_score(args, bank, out)
    if args.command == "report":
        return cmd_report(args, bank, out)
    if args.command == "apply":
        return cmd_apply(args, bank, out)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
