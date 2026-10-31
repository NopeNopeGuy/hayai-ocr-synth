#!/usr/bin/env python3
"""Generate synthetic manga OCR crops and append them to a Hugging Face dataset.

The script is designed to run unchanged on Kaggle or locally.  It reads the
repository's own generator (fonts, backgrounds, real-kana bank) and writes
parquet shards whose schema matches ``JustANormalTinkerer/hayai-synth-dataset``:

    image        Image (raw JPEG bytes)
    transcription string
    language     string
    source       string
    image_id     string
    vertical     bool

Generation is controlled by a *crop preset* (what kind of crop) and a *line
source* (what kind of text):

    python tools/append_hayai_synth.py generate \
      --split handwritten --preset handwritten --n-images 5000 \
      --lines hf:dialogue --out-dir /kaggle/working/hayai_append

    python tools/append_hayai_synth.py generate \
      --split japanese --preset dialogue --n-images 5000 \
      --lines hf:novel:Narou_NSFW --bubble never --vertical auto

    python tools/append_hayai_synth.py upload \
      --out-dir /kaggle/working/hayai_append --split handwritten \
      --run-id 20261031-120000-abc --repo-id JustANormalTinkerer/hayai-synth-dataset

Use ``generate --help`` for the full list of crop, colour and layout controls.
Environment variables (``FONTS_ROOT``, ``MANGA_OCR_ASSETS_PATH``,
``BACKGROUND_ANIMETEXT_DIR``, ``REAL_KANA_ROOT``) are set from the CLI flags
*before* the generator is imported, so they win over the defaults in
``manga_ocr_dev/env.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
import traceback
import zlib
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_REPO_ID = "JustANormalTinkerer/hayai-synth-dataset"
DEFAULT_DIALOGUE_DATASET = "JustANormalTinkerer/Japanese-Dialogue-Dataset"
DEFAULT_NOVEL_DATASET = "OmniAICreator/Japanese-Novels-23M"
SPLITS = ("japanese", "sfx", "handwritten")
PRESETS = (
    "auto",
    "dialogue",
    "dialogue_bubble",
    "dialogue_flat",
    "dialogue_tight",
    "handwritten",
    "sfx",
    "mixed",
    "custom",
)
AUTO_PRESET = {
    "japanese": "dialogue",
    "sfx": "sfx",
    "handwritten": "handwritten",
}
EXPECTED_FEATURES = [
    ("image", "image"),
    ("transcription", "string"),
    ("language", "string"),
    ("source", "string"),
    ("image_id", "string"),
    ("vertical", "bool"),
]

_ENGINE = None


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def _now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + f"{random.randrange(16**4):04x}"


def _sanitize(value: Any, max_len: int = 60) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.-")
    return (text or "x")[:max_len]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _tri(value: str | None) -> str:
    """Normalise ``auto`` / ``true`` / ``false`` style flags."""
    if value is None:
        return "auto"
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return "true"
    if text in {"false", "0", "no", "off"}:
        return "false"
    return "auto"


def _parse_weights(spec: str | None) -> tuple[list[str], list[float]]:
    if not spec:
        spec = "dialogue=0.55,handwritten=0.25,sfx=0.12,dialogue_flat=0.08"
    names: list[str] = []
    weights: list[float] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            name, raw = part.split("=", 1)
        else:
            name, raw = part, "1"
        name = name.strip()
        if name not in PRESETS or name in {"auto", "mixed", "custom"}:
            raise ValueError(f"invalid mix preset: {name!r}")
        names.append(name)
        weights.append(float(raw))
    if not names:
        raise ValueError("empty --preset-mix")
    total = sum(weights)
    if total <= 0:
        raise ValueError("--preset-mix weights must sum to > 0")
    return names, [w / total for w in weights]


def _parse_override(value: str | None) -> dict:
    if not value:
        return {}
    text = value.strip()
    if text.startswith("{"):
        data = json.loads(text)
    else:
        data = json.loads(Path(text).expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("--override-json must be a JSON object")
    return data


def _resolve_files(value: str) -> list[Path]:
    path = Path(value).expanduser()
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in {".csv", ".parquet", ".jsonl", ".json", ".txt"})
        if not files:
            raise FileNotFoundError(f"no csv/parquet/jsonl/txt files under {path}")
        return files
    if any(ch in value for ch in "*?["):
        files = sorted(Path(p) for p in Path().glob(value) if p.is_file())
        if not files:
            raise FileNotFoundError(f"no files matched {value!r}")
        return files
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"no such file or directory: {value}")


def _env_from_args(args: argparse.Namespace) -> dict[str, str]:
    env: dict[str, str] = {}
    if getattr(args, "fonts_root", None):
        env["FONTS_ROOT"] = str(Path(args.fonts_root).expanduser())
    if getattr(args, "assets_dir", None):
        env["MANGA_OCR_ASSETS_PATH"] = str(Path(args.assets_dir).expanduser())
    if getattr(args, "backgrounds", None):
        env["BACKGROUND_ANIMETEXT_DIR"] = str(Path(args.backgrounds).expanduser())
    if getattr(args, "kana_bank", None):
        env["REAL_KANA_ROOT"] = str(Path(args.kana_bank).expanduser())
    return env


def _apply_env(env: dict[str, str]) -> None:
    for key, value in env.items():
        os.environ[key] = value


# ---------------------------------------------------------------------------
# text sources
# ---------------------------------------------------------------------------


def _iter_local_lines(path: str, text_column: str | None, source_label: str | None, max_line_len: int) -> Iterator[dict]:
    import pandas as pd

    from manga_ocr_dev.data.process_text import normalize_text

    text_fields = ("line", "text", "dialogue", "transcription", "Sentences", "sentences")
    counter = 0
    for file in _resolve_files(path):
        suffix = file.suffix.lower()
        default_source = source_label or file.stem
        if suffix == ".csv":
            chunks = pd.read_csv(file, chunksize=10_000)
            for chunk in chunks:
                yield from _iter_dataframe(chunk, file, text_column, default_source, max_line_len, text_fields, normalize_text, counter)
                counter += len(chunk)
        elif suffix == ".parquet":
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(file)
            for batch in pf.iter_batches(batch_size=10_000):
                frame = batch.to_pandas()
                yield from _iter_dataframe(frame, file, text_column, default_source, max_line_len, text_fields, normalize_text, counter)
                counter += len(frame)
        elif suffix in {".jsonl", ".json"}:
            if suffix == ".json":
                data = json.loads(file.read_text(encoding="utf-8"))
                rows = data if isinstance(data, list) else data.get("data", [])
            else:
                rows = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
            for row in rows:
                raw = row.get(text_column or "line") or row.get("text") or row.get("dialogue")
                line = normalize_text(str(raw)) if raw is not None else ""
                if not line or len(line) > max_line_len:
                    continue
                counter += 1
                yield {
                    "source": str(row.get("source") or default_source),
                    "id": str(row.get("id") or f"{default_source}_{counter}"),
                    "line": line,
                }
        else:
            for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
                line = normalize_text(line)
                if not line or len(line) > max_line_len:
                    continue
                counter += 1
                yield {"source": default_source, "id": f"{default_source}_{counter}", "line": line}


def _iter_dataframe(
    frame,
    file: Path,
    text_column: str | None,
    default_source: str,
    max_line_len: int,
    text_fields: Sequence[str],
    normalize_text,
    offset: int,
) -> Iterator[dict]:
    if text_column is not None:
        if text_column not in frame.columns:
            raise KeyError(f"{file} has no column {text_column!r}; columns: {list(frame.columns)}")
        column = text_column
    else:
        column = next((name for name in text_fields if name in frame.columns), None)
        if column is None:
            raise KeyError(f"{file} has no text column (looked for {text_fields}); columns: {list(frame.columns)}")
    has_source = "source" in frame.columns
    has_id = "id" in frame.columns
    for row_idx, row in frame.iterrows():
        raw = row.get(column)
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            continue
        line = normalize_text(str(raw))
        if not line or len(line) > max_line_len:
            continue
        source = str(row["source"]) if has_source and row.get("source") is not None else default_source
        line_id = str(row["id"]) if has_id and row.get("id") is not None else f"{default_source}_{offset + row_idx}"
        yield {"source": source, "id": line_id, "line": line}


def _iter_text_items(cfg: dict) -> Iterator[dict]:
    lines = str(cfg["lines"])
    if lines.startswith("hf:"):
        spec = lines[3:]
        if spec.startswith("dialogue"):
            from manga_ocr_dev.data.process_text import iter_dialogue_stream

            splits = spec.split(":", 1)[1] if ":" in spec else cfg.get("hf_splits") or ""
            yield from iter_dialogue_stream(
                dataset_name=cfg.get("hf_dataset") or DEFAULT_DIALOGUE_DATASET,
                splits=splits,
                streaming=True,
                min_len=2,
                max_len=int(cfg["max_line_len"]),
            )
            return
        if spec.startswith("novel"):
            from manga_ocr_dev.data.process_text import iter_novel_stream

            configs = spec.split(":", 1)[1] if ":" in spec else cfg.get("hf_configs") or ""
            yield from iter_novel_stream(
                dataset_name=cfg.get("hf_dataset") or DEFAULT_NOVEL_DATASET,
                configs=configs,
                streaming=True,
                min_len=2,
                max_len=int(cfg["max_line_len"]),
            )
            return
        raise ValueError("--lines hf:... must be hf:dialogue[:splits] or hf:novel[:configs]")

    yield from _iter_local_lines(
        path=lines,
        text_column=cfg.get("text_column"),
        source_label=cfg.get("source_label"),
        max_line_len=int(cfg["max_line_len"]),
    )


def _iter_jobs(cfg: dict, start_index: int = 0) -> Iterator[tuple]:
    text_source = str(cfg["lines"]).lower()
    item_iter: Iterator[dict] | None = None
    if text_source not in {"random", "sfx"}:
        item_iter = _iter_text_items(cfg)
        dedup = None
        if int(cfg.get("dedup_window") or 0) > 0:
            from manga_ocr_dev.data.process_text import RollingDedup

            dedup = RollingDedup(window_size=int(cfg["dedup_window"]))
        if dedup is not None:
            item_iter = (item for item in item_iter if not dedup.is_duplicate(item["line"]))
        if start_index:
            item_iter = islice(item_iter, start_index, None)

    idx = start_index
    total = int(cfg["n_images"])
    while total < 0 or idx < total:
        if item_iter is None:
            text = None
            if text_source == "sfx" or cfg["preset"] == "sfx":
                source = cfg.get("source_label") or "sfx"
            else:
                source = cfg.get("source_label") or "random"
        else:
            try:
                item = next(item_iter)
            except StopIteration:
                _eprint(f"Lines exhausted after {idx - start_index} samples.")
                break
            text = item["line"]
            source = cfg.get("source_label") or item["source"]

        seed = zlib.crc32(f"{cfg['seed']}:{cfg['run_id']}:{idx}".encode()) % (2**32)
        preset = _resolve_preset(cfg, seed)
        image_id = f"{preset}_{_sanitize(source, 36)}_{cfg['run_id']}_{idx:07d}"
        yield idx, seed, text, str(source), image_id, preset
        idx += 1


def _resolve_preset(cfg: dict, seed: int) -> str:
    preset = cfg["preset"]
    if preset != "mixed":
        return preset
    rng = random.Random(seed)
    return rng.choices(cfg["mix_names"], weights=cfg["mix_weights"], k=1)[0]


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def _build_override(cfg: dict, preset: str, text: str | None) -> dict:
    from manga_ocr_dev.synthetic_data_generator.renderer import TEXT_COLOR_NAMES

    override: dict[str, Any] = dict(cfg.get("override_json") or {})

    if preset == "sfx":
        override["sfx"] = True
    elif preset == "dialogue_bubble":
        override["draw_bubble"] = True
    elif preset == "dialogue_flat":
        override["draw_bubble"] = False
    elif preset == "dialogue_tight":
        override["draw_bubble"] = False
        override["tight"] = True
    elif preset == "handwritten":
        override.setdefault("irregular", True)
        if cfg.get("kana_bank") and cfg["real_kana"] != "false":
            override.setdefault("real_kana", True)
    elif preset == "custom" and not override:
        raise ValueError("--preset custom requires --override-json")

    if cfg["vertical"] != "auto":
        override["vertical"] = cfg["vertical"] == "true"

    if cfg["bubble"] == "always":
        override["draw_bubble"] = True
    elif cfg["bubble"] == "never":
        override["draw_bubble"] = False
    elif cfg.get("bubble_prob") is not None:
        override["bubble_prob"] = float(cfg["bubble_prob"])

    if cfg.get("tight"):
        override["tight"] = True

    if cfg["irregular"] != "auto":
        override["irregular"] = cfg["irregular"] == "true"

    if cfg.get("slant_deg") is not None:
        override["slant"] = float(cfg["slant_deg"])

    if cfg.get("double_outline") not in {None, "auto"}:
        override["double_outline"] = cfg["double_outline"] == "true"

    if cfg["hard_ink"] == "off":
        override.update({"distress": False, "faint": False, "extrude": False, "hollow": False})

    if cfg.get("font_size_min") or cfg.get("font_size_max"):
        default_lo, default_hi = (45, 160) if preset == "sfx" else (28, 72)
        lo = int(cfg.get("font_size_min") or default_lo)
        hi = int(cfg.get("font_size_max") or default_hi)
        if hi < lo:
            lo, hi = hi, lo
        override["font_size"] = int(random.randint(lo, hi))

    if cfg.get("stroke_size") is not None:
        override["stroke_size"] = int(cfg["stroke_size"])

    if cfg["real_kana"] == "true":
        override["real_kana"] = True
    elif cfg["real_kana"] == "false":
        override["real_kana"] = False
    if cfg.get("real_kana_prob") is not None:
        override["real_kana_prob"] = float(cfg["real_kana_prob"])

    if cfg["color"] == "mono":
        override["text_color"] = "black"
        override["color_mode"] = "solid"
        override.pop("text_colors", None)
    elif cfg["color"] == "color":
        if not override.get("text_color"):
            override["text_color"] = str(random.choice(TEXT_COLOR_NAMES))
        if cfg.get("color_mode"):
            override["color_mode"] = cfg["color_mode"]
    elif cfg.get("color_mode"):
        override["color_mode"] = cfg["color_mode"]

    if cfg.get("text_color"):
        override["text_color"] = cfg["text_color"]

    return override


class Engine:
    """Per-process renderer + font picker."""

    def __init__(self, cfg: dict):
        from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
        from manga_ocr_dev.synthetic_data_generator.utils import get_font_meta

        self.cfg = cfg
        self.generator = SyntheticDataGenerator(
            background_df=cfg["background_df"],
            kana_bank=cfg.get("kana_bank"),
            real_kana=False,
            real_kana_prob=cfg.get("real_kana_prob"),
        )
        self.fonts_df, self.font_map = get_font_meta()
        self.hand_fonts = list(cfg.get("hand_fonts") or [])
        self._hand_idx = [self.font_map.index[p] for p in self.hand_fonts if p in self.font_map.index]

    def _pick_handwritten_font(self, text: str | None) -> str | None:
        if not self._hand_idx:
            return None
        if not text:
            return str(random.choice(self.hand_fonts))
        try:
            coverage = self.font_map.coverage(set(text))
        except Exception:  # noqa: BLE001
            return None
        values = coverage[self._hand_idx]
        best = float(values.max()) if len(values) else 0.0
        if best < 0.85:
            # Let the generator pick a font that covers the line instead of
            # dropping most of it.  Rare kanji in novels often hit this path.
            return None
        candidates = [self.hand_fonts[j] for j in self._indices_at_max(values, best)]
        return str(random.choice(candidates))

    @staticmethod
    def _indices_at_max(values, best: float) -> list[int]:
        import numpy as np

        return list(np.flatnonzero(values >= best - 1e-6))

    def render(self, job: tuple) -> dict:
        import cv2
        import numpy as np

        idx, seed, text, source, image_id, preset = job
        np.random.seed(seed)
        random.seed(seed)

        override = _build_override(self.cfg, preset, text)
        if preset == "handwritten":
            font = self._pick_handwritten_font(text)
            if font:
                override.setdefault("font_path", font)

        if preset == "sfx":
            img, text_gt, params = self.generator.process(None, override, sfx=True, lang=self.cfg["language"])
        else:
            img, text_gt, params = self.generator.process(text, override, sfx=False, lang=self.cfg["language"])

        if not text_gt or not str(text_gt).strip():
            raise ValueError("empty transcription")

        ok, encoded = cv2.imencode(
            ".jpg",
            img,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.get("jpeg_quality", 95))],
        )
        if not ok:
            raise RuntimeError("cv2.imencode failed")

        return {
            "image": {"bytes": encoded.tobytes(), "path": f"{image_id}.jpg"},
            "transcription": str(text_gt),
            "language": str(self.cfg["language"]),
            "source": str(source),
            "image_id": str(image_id),
            "vertical": bool(params.get("vertical", False)),
            "preset": str(preset),
            "index": int(idx),
            "real_kana_attached": int((params.get("real_kana_stats") or {}).get("attached", 0)),
        }


def _init_worker(cfg: dict) -> None:
    global _ENGINE
    _apply_env(cfg.get("env") or {})
    import cv2

    cv2.setNumThreads(0)
    _ENGINE = Engine(cfg)


def _worker_generate(job: tuple) -> dict:
    try:
        return _ENGINE.render(job)  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        return {"error": traceback.format_exc(), "image_id": job[4], "index": job[0]}


def _background_df(args: argparse.Namespace):
    from manga_ocr_dev.synthetic_data_generator.utils import get_background_df

    if args.backgrounds:
        path = Path(args.backgrounds).expanduser()
        df = get_background_df(path)
        if df.empty:
            raise RuntimeError(f"no background crops found in {path}")
        return df
    from manga_ocr_dev.synthetic_data_generator.renderer import get_default_background_df

    df = get_default_background_df()
    if df.empty:
        raise RuntimeError("no default backgrounds found; pass --backgrounds")
    return df


def _hand_fonts():
    from manga_ocr_dev.synthetic_data_generator.utils import get_font_meta, get_font_profile

    _, font_map = get_font_meta()
    paths = []
    for path in font_map.paths:
        profile = get_font_profile(path)
        if profile.get("font_type") == "handwritten" and profile.get("is_manga", True):
            paths.append(path)
    return paths


def _check_font_root(font_map) -> None:
    if not font_map.paths:
        raise RuntimeError("fonts.csv is empty: run scan_fonts.py first")
    step = max(1, len(font_map.paths) // 400)
    sample = font_map.paths[::step][:400]
    missing = sum(not Path(path).is_file() for path in sample)
    if missing > len(sample) * 0.5:
        raise RuntimeError(
            f"{missing}/{len(sample)} sampled font files are missing under FONTS_ROOT. "
            "Point --fonts-root at the Kaggle input that preserves the fonts.csv relative layout, "
            "or rescan fonts with `python -m manga_ocr_dev.synthetic_data_generator.scan_fonts`."
        )


def _cfg_from_args(args: argparse.Namespace) -> dict:
    preset = args.preset
    if preset == "auto":
        preset = AUTO_PRESET[args.split]
    if preset == "custom" and not args.override_json:
        raise SystemExit("--preset custom requires --override-json")

    mix_names, mix_weights = _parse_weights(args.preset_mix)
    cfg = {
        "split": args.split,
        "preset": preset,
        "mix_names": mix_names,
        "mix_weights": mix_weights,
        "n_images": int(args.n_images),
        "rows_per_shard": max(1, int(args.rows_per_shard)),
        "run_id": _sanitize(args.run_id or _now_run_id(), 40),
        "out_dir": str(Path(args.out_dir).expanduser()),
        "lines": args.lines,
        "source_label": args.source_label,
        "text_column": args.text_column,
        "hf_dataset": args.hf_dataset,
        "hf_splits": args.hf_splits,
        "hf_configs": args.hf_configs,
        "max_line_len": int(args.max_line_len),
        "dedup_window": int(args.dedup_window),
        "language": args.language,
        "vertical": _tri(args.vertical),
        "bubble": args.bubble,
        "bubble_prob": args.bubble_prob,
        "tight": bool(args.tight),
        "irregular": _tri(args.irregular),
        "slant_deg": args.slant_deg,
        "font_size_min": args.font_size_min,
        "font_size_max": args.font_size_max,
        "stroke_size": args.stroke_size,
        "double_outline": args.double_outline,
        "hard_ink": args.hard_ink,
        "color": args.color,
        "color_mode": args.color_mode,
        "text_color": args.text_color,
        "real_kana": _tri(args.real_kana),
        "real_kana_prob": args.real_kana_prob,
        "kana_bank": str(Path(args.kana_bank).expanduser()) if args.kana_bank else None,
        "backgrounds": str(Path(args.backgrounds).expanduser()) if args.backgrounds else None,
        "fonts_root": str(Path(args.fonts_root).expanduser()) if args.fonts_root else None,
        "assets_dir": str(Path(args.assets_dir).expanduser()) if args.assets_dir else None,
        "jpeg_quality": int(args.jpeg_quality),
        "seed": int(args.seed),
        "workers": int(args.workers),
        "start_method": args.start_method,
        "override_json": _parse_override(args.override_json),
        "env": _env_from_args(args),
    }
    return cfg


def _prepare_cfg(cfg: dict) -> dict:
    _apply_env(cfg.get("env") or {})

    from manga_ocr_dev.synthetic_data_generator.utils import get_font_meta

    background_df = _background_df_from_cfg(cfg)
    cfg["background_df"] = background_df
    hand_fonts = _hand_fonts()
    cfg["hand_fonts"] = hand_fonts
    _, font_map = get_font_meta()
    _check_font_root(font_map)

    if cfg["real_kana"] != "false" and cfg.get("kana_bank"):
        manifest = Path(cfg["kana_bank"]) / "manifest.jsonl"
        if not manifest.is_file():
            raise RuntimeError(f"real-kana manifest not found: {manifest}")
    return cfg


def _background_df_from_cfg(cfg: dict):
    from manga_ocr_dev.synthetic_data_generator.utils import get_background_df

    if cfg.get("backgrounds"):
        df = get_background_df(Path(cfg["backgrounds"]))
        if df.empty:
            raise RuntimeError(f"no background crops found in {cfg['backgrounds']}")
    else:
        from manga_ocr_dev.synthetic_data_generator.renderer import get_default_background_df

        df = get_default_background_df()
        if df.empty:
            raise RuntimeError("no default backgrounds found; pass --backgrounds")

    # Some AnimeText manifests list the same crop more than once (the source
    # extraction can revisit a rectangle).  Duplicates would silently bias
    # background sampling towards those images.
    duplicate_paths = int(df.duplicated("path").sum())
    if duplicate_paths:
        print(f"Dropping {duplicate_paths} duplicate background manifest row(s).")
        df = df.drop_duplicates("path").reset_index(drop=True)
    return df


def _write_shard(rows: list[dict], path: Path) -> None:
    from datasets import Dataset, Features, Image, Value

    features = Features(
        {
            "image": Image(),
            "transcription": Value("string"),
            "language": Value("string"),
            "source": Value("string"),
            "image_id": Value("string"),
            "vertical": Value("bool"),
        }
    )
    data = {
        "image": [row["image"] for row in rows],
        "transcription": [row["transcription"] for row in rows],
        "language": [row["language"] for row in rows],
        "source": [row["source"] for row in rows],
        "image_id": [row["image_id"] for row in rows],
        "vertical": [bool(row["vertical"]) for row in rows],
    }
    ds = Dataset.from_dict(data, features=features)
    tmp = path.with_name(path.name + ".tmp")
    ds.to_parquet(str(tmp))
    os.replace(tmp, path)


def _make_pool(cfg: dict):
    if int(cfg["workers"]) <= 1:
        return None
    import multiprocessing as mp

    method = cfg["start_method"]
    available = mp.get_all_start_methods()
    if method == "auto":
        method = "fork" if "fork" in available else "spawn"
    if method not in available:
        raise RuntimeError(f"start method {method!r} not available; choose from {available}")
    ctx = mp.get_context(method)
    return ProcessPoolExecutor(
        max_workers=int(cfg["workers"]),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(cfg,),
    )


def _run_jobs(cfg: dict, jobs: Iterable[tuple], pool) -> Iterator[dict]:
    if pool is None:
        _init_worker(cfg)
        for job in jobs:
            yield _worker_generate(job)
        return
    yield from pool.map(_worker_generate, jobs, chunksize=max(1, min(16, int(cfg["workers"]))))


def _next_index(existing: Sequence[Path]) -> int:
    """Return the source index after the last row already written.

    Parsing the ``_<idx:07d>`` suffix (instead of ``shards * rows_per_shard``)
    keeps ``--resume`` correct when the last shard was partial or some samples
    failed.
    """
    import pyarrow.parquet as pq

    next_index = 0
    for path in existing:
        for image_id in pq.read_table(path, columns=["image_id"]).column("image_id").to_pylist():
            try:
                idx = int(str(image_id).rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            next_index = max(next_index, idx + 1)
    return next_index


def cmd_generate(args: argparse.Namespace) -> int:
    cfg = _cfg_from_args(args)
    if cfg["n_images"] < 0 and str(cfg["lines"]).lower() in {"random", "sfx"}:
        raise SystemExit("--n-images -1 needs a finite line source; random/sfx never end")
    cfg = _prepare_cfg(cfg)
    out_dir = Path(cfg["out_dir"])
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    total = cfg["n_images"]
    if total == 0:
        _eprint("Nothing to do: --n-images is 0")
        return 2
    if total < 0:
        _eprint("Note: --n-images -1 means 'until the line source is exhausted'.")

    existing = sorted(data_dir.glob(f"{cfg['split']}-append-{cfg['run_id']}-*.parquet"))
    if existing and not args.resume:
        _eprint(
            f"{len(existing)} shard(s) already exist for run {cfg['run_id']!r}. "
            "Pass --resume to continue or use a new --run-id."
        )
        return 2
    start_shard = len(existing)
    start_index = _next_index(existing) if existing else 0
    if start_shard:
        print(f"Resume: {start_shard} shard(s) already written, skipping {start_index} samples.")

    print(
        f"split={cfg['split']} preset={cfg['preset']} lines={cfg['lines']} "
        f"n_images={total} rows_per_shard={cfg['rows_per_shard']} workers={cfg['workers']} "
        f"backgrounds={len(cfg['background_df'])} handwritten_fonts={len(cfg['hand_fonts'])}"
    )
    print(f"run_id={cfg['run_id']} out_dir={out_dir}")

    jobs = _iter_jobs(cfg, start_index=start_index)
    pool = _make_pool(cfg)
    shard_idx = start_shard
    written = 0
    failed = 0
    started = time.time()
    try:
        while True:
            chunk = list(islice(jobs, cfg["rows_per_shard"]))
            if not chunk:
                break
            rows: list[dict] = []
            for result in _run_jobs(cfg, chunk, pool):
                if "error" in result:
                    failed += 1
                    _eprint(f"FAILED {result.get('image_id')}:\n{result['error']}")
                else:
                    rows.append(result)
            if not rows:
                _eprint(f"Shard {shard_idx:05d} produced no rows; stopping.")
                break
            shard_path = data_dir / f"{cfg['split']}-append-{cfg['run_id']}-{shard_idx:05d}.parquet"
            _write_shard(rows, shard_path)
            written += len(rows)
            elapsed = max(1e-6, time.time() - started)
            print(
                f"wrote {shard_path.name} rows={len(rows)} total={written} "
                f"failed={failed} rate={written / elapsed:.1f} img/s"
            )
            shard_idx += 1
            if total >= 0 and written + start_index >= total:
                break
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    manifest = {
        "repo_id": args.repo_id,
        "split": cfg["split"],
        "preset": cfg["preset"],
        "run_id": cfg["run_id"],
        "rows_per_shard": cfg["rows_per_shard"],
        "requested_images": total,
        "written_rows": written,
        "failed_rows": failed,
        "shards": [p.name for p in sorted(data_dir.glob(f"{cfg['split']}-append-{cfg['run_id']}-*.parquet"))],
        "args": {key: value for key, value in cfg.items() if key not in {"background_df", "env"}},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / f"manifest-{cfg['run_id']}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Done: {written} rows, {failed} failed, {shard_idx - start_shard} new shard(s) in {data_dir}")
    return 0 if written else 1


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------


def cmd_preview(args: argparse.Namespace) -> int:
    cfg = _prepare_cfg(_cfg_from_args(args))
    cfg["n_images"] = int(args.preview)
    jobs = list(islice(_iter_jobs(cfg, start_index=0), int(args.preview)))
    results = []
    _init_worker(cfg)
    for job in jobs:
        result = _worker_generate(job)
        if "error" in result:
            _eprint(f"FAILED {result.get('image_id')}:\n{result['error']}")
        else:
            results.append(result)
    if not results:
        _eprint("preview produced no images")
        return 1

    from io import BytesIO

    from PIL import Image, ImageDraw, ImageFont

    cols = 4
    cell_w, cell_h = 380, 300
    img_w, img_h = 360, 230
    rows_n = math.ceil(len(results) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows_n * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    annotation_font = None
    if cfg["hand_fonts"]:
        try:
            annotation_font = ImageFont.truetype(str(cfg["hand_fonts"][0]), 16)
        except Exception:  # noqa: BLE001
            annotation_font = None
    label_font = ImageFont.load_default()
    for n, row in enumerate(results):
        col, line = n % cols, n // cols
        x0, y0 = col * cell_w, line * cell_h
        image = Image.open(BytesIO(row["image"]["bytes"])).convert("RGB")
        image.thumbnail((img_w, img_h), Image.Resampling.LANCZOS)
        sheet.paste(image, (x0 + (cell_w - image.width) // 2, y0 + 8))
        text = row["transcription"].replace("\n", " / ")
        if len(text) > 30:
            text = text[:29] + "…"
        meta = f"{n}: {row['preset']} {'V' if row['vertical'] else 'H'} {row['source']}"
        draw.text((x0 + 8, y0 + img_h + 12), meta, fill=(0, 0, 0), font=label_font)
        draw.text(
            (x0 + 8, y0 + img_h + 32),
            text,
            fill=(0, 0, 0),
            font=annotation_font or label_font,
        )
        print(
            f"[{n}] preset={row['preset']} vertical={row['vertical']} "
            f"real_kana={row.get('real_kana_attached', 0)} source={row['source']} :: {row['transcription']!r}"
        )

    out = Path(args.preview_out).expanduser() if args.preview_out else Path(cfg["out_dir"]) / "preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"Preview written to {out}")
    return 0


# ---------------------------------------------------------------------------
# validation / upload / card
# ---------------------------------------------------------------------------


def _shard_files(args: argparse.Namespace) -> list[Path]:
    if getattr(args, "shards", None):
        return _resolve_files(args.shards)
    if not getattr(args, "out_dir", None) or not getattr(args, "split", None):
        raise SystemExit("pass --shards GLOB or both --out-dir and --split")
    pattern = f"{args.split}-append-{args.run_id}-*.parquet" if args.run_id else f"{args.split}-append-*.parquet"
    files = sorted(Path(args.out_dir).expanduser().glob(f"data/{pattern}"))
    if not files:
        raise SystemExit(f"no shards matched {pattern} under {args.out_dir}/data")
    return files


def validate_shards(files: Sequence[Path], expect_rows: int | None = None) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    expected = pa.schema(
        [
            pa.field("image", pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])),
            pa.field("transcription", pa.string()),
            pa.field("language", pa.string()),
            pa.field("source", pa.string()),
            pa.field("image_id", pa.string()),
            pa.field("vertical", pa.bool_()),
        ]
    )
    total = 0
    seen_ids: set[str] = set()
    for path in files:
        pf = pq.ParquetFile(path)
        if not pf.schema_arrow.equals(expected):
            raise ValueError(f"{path}: schema mismatch\n{pf.schema_arrow}\n!=\n{expected}")
        table = pf.read()
        rows = table.to_pylist()
        if not rows:
            raise ValueError(f"{path}: empty shard")
        for row in rows:
            image = row["image"]
            if not isinstance(image, dict) or not image.get("bytes"):
                raise ValueError(f"{path}: missing image bytes in {row.get('image_id')}")
            if not image["bytes"].startswith(b"\xff\xd8"):
                raise ValueError(f"{path}: image is not JPEG ({row.get('image_id')})")
            if image.get("path") != f"{row['image_id']}.jpg":
                raise ValueError(f"{path}: image.path mismatch in {row.get('image_id')}")
            if not str(row.get("transcription") or "").strip():
                raise ValueError(f"{path}: empty transcription in {row.get('image_id')}")
            if row.get("language") != "ja":
                raise ValueError(f"{path}: language={row.get('language')!r} in {row.get('image_id')}")
            if not isinstance(row.get("vertical"), bool):
                raise TypeError(f"{path}: vertical is not bool in {row.get('image_id')}")
            image_id = row.get("image_id")
            if not image_id or image_id in seen_ids:
                raise ValueError(f"{path}: duplicate/empty image_id {image_id!r}")
            seen_ids.add(image_id)
        total += len(rows)
        print(f"ok {path.name} rows={len(rows)}")
    if expect_rows is not None and total != expect_rows:
        raise ValueError(f"expected {expect_rows} rows, found {total}")
    print(f"validated {len(files)} shard(s), {total} row(s)")
    return {"files": len(files), "rows": total}


def cmd_validate(args: argparse.Namespace) -> int:
    files = _shard_files(args)
    validate_shards(files, expect_rows=args.expect_rows)
    return 0


def _token(value: str | None) -> str | None:
    return value or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def cmd_upload(args: argparse.Namespace) -> int:
    from huggingface_hub import HfApi

    files = _shard_files(args)
    validate_shards(files)
    api = HfApi(token=_token(args.token))
    existing = set(api.list_repo_files(args.repo_id, repo_type="dataset"))
    new_files = [path for path in files if f"data/{path.name}" not in existing]
    skipped = len(files) - len(new_files)
    if skipped:
        print(f"Skipping {skipped} shard(s) that already exist in {args.repo_id}.")
    if not new_files:
        print("Nothing to upload.")
        return 0

    commit = args.commit_message or f"Append {len(new_files)} {args.split or 'synthetic'} shard(s)"
    if hasattr(api, "upload_folder"):
        api.upload_folder(
            folder_path=str(new_files[0].parent),
            path_in_repo="data",
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=[path.name for path in new_files],
            commit_message=commit,
            token=_token(args.token),
        )
    else:  # pragma: no cover - very old huggingface_hub
        for path in new_files:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"data/{path.name}",
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message=commit,
                token=_token(args.token),
            )
    print(f"Uploaded {len(new_files)} shard(s) to https://huggingface.co/datasets/{args.repo_id}")
    return 0


def _readme_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in SPLITS:
        match = re.search(rf"^- `{re.escape(split)}` \(([\d,]+) rows\)", text, flags=re.MULTILINE)
        if match:
            counts[split] = int(match.group(1).replace(",", ""))
    return counts


def _server_counts(repo_id: str) -> dict[str, int]:
    try:
        import requests

        response = requests.get(
            "https://datasets-server.huggingface.co/size",
            params={"dataset": repo_id},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        return {
            item["split"]: int(item["num_rows"])
            for item in payload.get("size", {}).get("splits", [])
            if "split" in item and "num_rows" in item
        }
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not read dataset-server counts: {exc}")
        return {}


def cmd_update_card(args: argparse.Namespace) -> int:
    from huggingface_hub import HfApi, hf_hub_download

    token = _token(args.token)
    path = hf_hub_download(args.repo_id, "README.md", repo_type="dataset", token=token)
    readme = Path(path).read_text(encoding="utf-8")
    counts = _readme_counts(readme)
    if not counts:
        raise SystemExit("could not find split rows in README; update the card manually")

    added: dict[str, int] = {}
    for spec in args.add or []:
        name, raw = spec.split("=", 1)
        name = name.strip()
        if name not in SPLITS:
            raise SystemExit(f"unknown split in --add: {name}")
        counts[name] = counts.get(name, 0) + int(raw)
        added[name] = int(raw)
    for spec in args.set or []:
        name, raw = spec.split("=", 1)
        name = name.strip()
        if name not in SPLITS:
            raise SystemExit(f"unknown split in --set: {name}")
        counts[name] = int(raw)

    if args.wait and added:
        deadline = time.time() + float(args.wait)
        while time.time() < deadline:
            server = _server_counts(args.repo_id)
            if server and all(server.get(name, 0) >= counts.get(name, 0) for name in SPLITS):
                print("dataset-server has indexed the new rows")
                counts.update({name: server[name] for name in SPLITS if name in server})
                break
            time.sleep(15)

    for split, count in counts.items():
        pattern = re.compile(rf"^(- `{re.escape(split)}` \()([\d,]+)( rows\):)", flags=re.MULTILINE)
        replacement = rf"\g<1>{count:,}\g<3>"
        readme, n = pattern.subn(replacement, readme)
        if n == 0:
            print(f"warning: no README bullet found for split {split!r}")

    total = sum(counts.values())
    if total < 100_000:
        bucket = "10K<n<100K"
    elif total < 1_000_000:
        bucket = "100K<n<1M"
    else:
        bucket = "1M<n<10M"
    readme = re.sub(r"(size_categories:\n- )\S+", rf"\g<1>{bucket}", readme)

    print("Updated counts:", ", ".join(f"{k}={v:,}" for k, v in sorted(counts.items())))
    if args.dry_run:
        print(readme[:2000])
        return 0
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.commit_message or "Update split row counts after append",
        token=token,
    )
    print(f"Updated https://huggingface.co/datasets/{args.repo_id}/README.md")
    return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _add_generate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", required=True, choices=SPLITS, help="target dataset split / shard prefix")
    parser.add_argument("--preset", default="auto", choices=PRESETS, help="crop kind (default: auto by split)")
    parser.add_argument("--n-images", type=int, default=5000, help="number of crops (use -1 to drain the line source)")
    parser.add_argument("--rows-per-shard", type=int, default=5000, help="rows per parquet shard (existing shards use 5000)")
    parser.add_argument("--out-dir", default="hayai_append", help="working directory for data/*.parquet")
    parser.add_argument("--run-id", default=None, help="unique run id used in shard/file names")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run with the same --run-id")

    parser.add_argument("--lines", default="random", help="random | sfx | path/glob | hf:dialogue[:splits] | hf:novel[:configs]")
    parser.add_argument("--source-label", default=None, help="override the source column (e.g. novel_Narou_NSFW)")
    parser.add_argument("--text-column", default=None, help="text column in local csv/parquet files")
    parser.add_argument("--hf-dataset", default=None, help="override the HF dataset id for hf:dialogue / hf:novel")
    parser.add_argument("--hf-splits", default=None, help="comma-separated dialogue splits (hf:dialogue)")
    parser.add_argument("--hf-configs", default=None, help="comma-separated novel configs (hf:novel)")
    parser.add_argument("--max-line-len", type=int, default=200, help="skip longer source lines")
    parser.add_argument("--dedup-window", type=int, default=50000, help="rolling duplicate window (0 disables)")

    parser.add_argument("--backgrounds", default=None, help="AnimeText background dir with manifest.csv")
    parser.add_argument("--fonts-root", default=None, help="fonts root matching assets/fonts.csv relative paths")
    parser.add_argument("--assets-dir", default=None, help="assets dir (fonts.csv, vocab.csv, ...)")
    parser.add_argument("--kana-bank", default=None, help="real handwritten kana bank (manifest.jsonl)")
    parser.add_argument("--real-kana", default="auto", help="auto | true | false; handwritten preset defaults to true when a bank is supplied")
    parser.add_argument("--real-kana-prob", type=float, default=None, help="share of kana replaced by real glyphs")

    parser.add_argument("--preset-mix", default=None, help="weights for --preset mixed, e.g. dialogue=0.6,handwritten=0.3,sfx=0.1")
    parser.add_argument("--vertical", default="auto", help="auto | true | false")
    parser.add_argument("--bubble", default="auto", choices=("auto", "always", "never"), help="speech bubble control")
    parser.add_argument("--bubble-prob", type=float, default=None, help="probability of drawing a bubble when --bubble auto")
    parser.add_argument("--tight", action="store_true", help="tight crop around the text (no bubble)")
    parser.add_argument("--irregular", default="auto", help="auto | true | false irregular hand-lettered layout")
    parser.add_argument("--slant-deg", type=float, default=None, help="force whole-block slant in degrees")
    parser.add_argument("--font-size-min", type=int, default=None)
    parser.add_argument("--font-size-max", type=int, default=None)
    parser.add_argument("--stroke-size", type=int, default=None)
    parser.add_argument("--double-outline", default="auto", help="auto | true | false")
    parser.add_argument("--hard-ink", default="auto", choices=("auto", "off"), help="disable distress/faint/extrude/hollow")
    parser.add_argument("--color", default="auto", choices=("auto", "mono", "color"), help="ink colour control")
    parser.add_argument("--color-mode", default=None, choices=(None, "solid", "per_line", "per_char"))
    parser.add_argument("--text-color", default=None, help="named colour or #rrggbb")
    parser.add_argument("--override-json", default=None, help="extra override dict (JSON string or file); wins last")

    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--language", default="ja")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--start-method", default="auto", choices=("auto", "fork", "spawn", "forkserver"))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="render crops and write parquet shards")
    _add_generate_args(gen)
    gen.set_defaults(func=cmd_generate)

    prev = sub.add_parser("preview", help="render a small contact sheet without writing shards")
    _add_generate_args(prev)
    prev.add_argument("--preview", type=int, default=8, help="number of preview crops")
    prev.add_argument("--preview-out", default=None, help="output PNG path")
    prev.set_defaults(func=cmd_preview)

    val = sub.add_parser("validate", help="validate shard schema, JPEG bytes and labels")
    val.add_argument("--shards", default=None, help="file, dir or glob of parquet shards")
    val.add_argument("--out-dir", default=None)
    val.add_argument("--split", default=None, choices=SPLITS)
    val.add_argument("--run-id", default=None)
    val.add_argument("--expect-rows", type=int, default=None)
    val.set_defaults(func=cmd_validate)

    up = sub.add_parser("upload", help="validate and upload shards to the dataset repo")
    up.add_argument("--shards", default=None)
    up.add_argument("--out-dir", default=None)
    up.add_argument("--split", default=None, choices=SPLITS)
    up.add_argument("--run-id", default=None)
    up.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    up.add_argument("--token", default=None, help="HF token (default: $HF_TOKEN)")
    up.add_argument("--commit-message", default=None)
    up.set_defaults(func=cmd_upload)

    card = sub.add_parser("update-card", help="add new rows to the README split counts")
    card.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    card.add_argument("--token", default=None)
    card.add_argument("--add", action="append", default=None, metavar="SPLIT=N", help="add N rows to a split count")
    card.add_argument("--set", action="append", default=None, metavar="SPLIT=N", help="set a split count explicitly")
    card.add_argument("--wait", type=float, default=0, help="seconds to wait for datasets-server to index new rows")
    card.add_argument("--dry-run", action="store_true")
    card.add_argument("--commit-message", default=None)
    card.set_defaults(func=cmd_update_card)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        _eprint("Interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
