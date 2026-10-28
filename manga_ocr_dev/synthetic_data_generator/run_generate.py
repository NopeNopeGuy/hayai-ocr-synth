"""Generate synthetic image-text pairs in parallel batches for manga OCR training."""

from __future__ import annotations

import csv
import gc
import multiprocessing as mp
import os

# Prevent OpenBLAS / MKL / OpenMP thread oversubscription across workers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import random
import traceback
import zlib

import cv2
import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

cv2.setNumThreads(0)

from manga_ocr_dev.env import DATA_SYNTHETIC_ROOT, FONTS_ROOT
from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
from manga_ocr_dev.synthetic_data_generator.lang import normalize_lang

_generator: SyntheticDataGenerator | None = None
_WORKER_KANA_BANK: str | None = None
_WORKER_REAL_KANA = False
_WORKER_REAL_KANA_PROB: float | None = None


def _init_worker(kana_bank=None, real_kana=False, real_kana_prob=None):
    """Disable OpenCV multithreading inside each worker and ensure generator is ready."""
    global _WORKER_KANA_BANK, _WORKER_REAL_KANA, _WORKER_REAL_KANA_PROB
    _WORKER_KANA_BANK = kana_bank
    _WORKER_REAL_KANA = bool(real_kana)
    _WORKER_REAL_KANA_PROB = real_kana_prob
    cv2.setNumThreads(0)
    get_generator()


def get_generator() -> SyntheticDataGenerator:
    """Lazily initialize generator in worker process."""
    global _generator
    if _generator is None:
        _generator = SyntheticDataGenerator(
            kana_bank=_WORKER_KANA_BANK,
            real_kana=_WORKER_REAL_KANA,
            real_kana_prob=_WORKER_REAL_KANA_PROB,
        )
    return _generator


def _random_frame(prefix: str, lang: str, package_str: str, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": prefix,
            "id": [f"{prefix}_{lang}_{package_str}_{i}" for i in range(n)],
            "line": None,
        }
    )


def _package_paths(split: str, lang: str, package_str: str):
    suffix = "" if lang == "ja" else f"_{lang}"
    if split == "sfx":
        out_dir = DATA_SYNTHETIC_ROOT / f"img_sfx{suffix}" / package_str
        meta_path = DATA_SYNTHETIC_ROOT / f"meta_sfx{suffix}" / f"{package_str}.csv"
    else:
        out_dir = DATA_SYNTHETIC_ROOT / f"img{suffix}" / package_str
        meta_path = DATA_SYNTHETIC_ROOT / f"meta{suffix}" / f"{package_str}.csv"
    return out_dir, meta_path


def _load_completed(out_dir: Path, meta_path: Path) -> dict[str, tuple]:
    """Fast O(N) resume check using csv.reader and os.scandir (no per-file stat storm)."""
    if not meta_path.is_file() or not out_dir.is_dir():
        return {}

    existing: dict[str, tuple] = {}
    try:
        with open(meta_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header or "id" not in header:
                return {}
            for row in reader:
                if len(row) >= 5:
                    existing[row[1]] = (row[0], row[1], row[2], row[3], row[4])
    except Exception:  # noqa: BLE001
        return {}

    if not existing:
        return {}

    imaged: set[str] = set()
    try:
        with os.scandir(out_dir) as it:
            for entry in it:
                name = entry.name
                if name.endswith(".jpg") and entry.is_file():
                    stem = name[:-4]
                    if stem in existing:
                        imaged.add(stem)
    except OSError:
        return {}

    return {id_: row for id_, row in existing.items() if id_ in imaged}


def _worker_process_sample(args):
    """Worker task for multiprocessing generation."""
    try:
        _i, out_dir, source, id_, text, sfx, lang = args
        seed = zlib.crc32(str(id_).encode("utf-8")) % (2**32)
        np.random.seed(seed)
        random.seed(seed)

        generator = get_generator()
        filename = f"{id_}.jpg"
        text = None if (text is None or (isinstance(text, float) and pd.isna(text))) else text
        sample_lang = lang if text is None else None
        img, text_gt, params = generator.process(text, sfx=bool(sfx), lang=sample_lang)

        cv2.imwrite(str(Path(out_dir) / filename), img)

        try:
            font_path = str(Path(params["font_path"]).relative_to(FONTS_ROOT))
        except ValueError:
            font_path = str(params["font_path"])

        return source, id_, text_gt, params["vertical"], font_path

    except Exception:  # noqa: BLE001
        print(traceback.format_exc())
        return None


def run(
    package: int = 0,
    n_random: int = 1000,
    n_limit: int | None = None,
    max_workers: int = 16,
    split: str = "dialogue",
    lang: str = "ja",
    resume: bool = False,
    checkpoint_every: int = 1000,
    real_kana: bool = False,
    kana_bank: str | None = None,
    real_kana_prob: float | None = None,
):
    lang = normalize_lang(lang)
    package_str = f"{package:04d}"
    suffix = "" if lang == "ja" else f"_{lang}"
    out_dir, meta_path = _package_paths(split, lang, package_str)

    worker_kana_bank = str(Path(kana_bank).expanduser()) if kana_bank else None
    if real_kana and lang != "ja":
        print(f"Real kana only applies to Japanese (lang={lang}); fonts will be used.")
    elif real_kana:
        if worker_kana_bank is not None and not (Path(worker_kana_bank) / "manifest.jsonl").is_file():
            raise FileNotFoundError(f"kana bank manifest not found: {Path(worker_kana_bank) / 'manifest.jsonl'}")
        where = worker_kana_bank or "$REAL_KANA_ROOT / assets/handwriting / /mnt/real-kana-final"
        print(f"Real kana enabled: rendering kana from {where} (fonts for the rest).")

    if split == "sfx":
        lines = _random_frame("sfx", lang, package_str, n_random)
        sfx_flags = [True] * len(lines)
    else:
        lines_csv = DATA_SYNTHETIC_ROOT / f"lines{suffix}" / f"{package_str}.csv"
        if lines_csv.is_file():
            lines = pd.read_csv(lines_csv)
        else:
            lines = pd.DataFrame(columns=["source", "id", "line"])

        if n_random > 0:
            lines = pd.concat([lines, _random_frame("random", lang, package_str, n_random)], ignore_index=True)

        sfx_flags = [False] * len(lines)

    if n_limit and n_limit < len(lines):
        seed = zlib.crc32(f"{split}/{lang}/{package_str}".encode()) % (2**32)
        idx = lines.sample(n_limit, random_state=seed).index
        lines = lines.loc[idx]
        sfx_flags = [sfx_flags[i] for i in idx]

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    if not max_workers or max_workers < 1:
        max_workers = os.cpu_count() or 1
    checkpoint_every = max(1, int(checkpoint_every))

    cols = ["source", "id", "text", "vertical", "font_path"]
    completed: dict[str, tuple] = {}
    expected_order: list[str] = []
    if resume:
        completed = _load_completed(out_dir, meta_path)
        expected_order = lines["id"].astype(str).tolist() if "id" in lines.columns else []
        expected_set = set(expected_order)
        completed = {k: v for k, v in completed.items() if k in expected_set}
        if completed:
            keep_mask = ~lines["id"].astype(str).isin(completed)
            lines = lines.loc[keep_mask].reset_index(drop=True)
            sfx_flags = [f for f, k in zip(sfx_flags, keep_mask) if k]
            print(f"Resume: skipping {len(completed)} already completed samples, {len(lines)} remaining.")
            tmp_meta = meta_path.with_suffix(".tmp.csv")
            with open(tmp_meta, "w", newline="", encoding="utf-8") as mf:
                mw = csv.writer(mf)
                mw.writerow(cols)
                mw.writerows(completed.values())
            os.replace(tmp_meta, meta_path)
        else:
            print("Resume: no completed samples found, generating from scratch.")
        if len(lines) == 0 and expected_order:
            print(f"Nothing to do: all samples for package {package_str} already generated.")
            return

    total = len(lines)
    out_dir_str = str(out_dir)
    append_mode = resume and bool(completed)
    f = open(meta_path, "a" if append_mode else "w", newline="", encoding="utf-8")  # noqa: SIM115
    writer = csv.writer(f)
    if not append_mode:
        writer.writerow(cols)

    n_failed = 0
    n_new = 0
    desc = f"Processing package {package_str} ({split}/{lang})"
    if total == 0:
        f.close()
        print(f"Successfully generated 0 samples for package {package_str}.")
        return

    # Use 'spawn' or 'forkserver' context when max_tasks_per_child is specified,
    # as Python forbids max_tasks_per_child with the default 'fork' method.
    available_methods = mp.get_all_start_methods()
    if "spawn" in available_methods:
        mp_context = mp.get_context("spawn")
    elif "forkserver" in available_methods:
        mp_context = mp.get_context("forkserver")
    else:
        mp_context = None

    ex = ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_context,
        initializer=_init_worker,
        initargs=(worker_kana_bank, bool(real_kana), real_kana_prob),
        max_tasks_per_child=1500,
    )
    try:
        with tqdm(total=total, desc=desc, unit="img") as pbar:
            for start in range(0, total, checkpoint_every):
                end = min(start + checkpoint_every, total)
                chunk = lines.iloc[start:end]
                chunk_flags = sfx_flags[start:end]
                chunk_args = [
                    (start + j, out_dir_str, *row, sfx, lang)
                    for j, (row, sfx) in enumerate(zip(chunk.values, chunk_flags))
                ]
                map_chunksize = max(1, min(16, len(chunk_args) // (max_workers * 2 or 1)))
                for res in ex.map(_worker_process_sample, chunk_args, chunksize=map_chunksize):
                    if res is None:
                        n_failed += 1
                    else:
                        writer.writerow(res)
                        n_new += 1
                        if n_new % 200 == 0:
                            f.flush()
                    pbar.update(1)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
    except KeyboardInterrupt:
        ex.shutdown(wait=False, cancel_futures=True)
        try:
            f.flush()
            f.close()
        except Exception:  # noqa: BLE001, S110
            pass
        print(f"\nInterrupted: partial progress saved to {meta_path} ({n_new} new rows this run). Re-run with --resume.")
        raise
    except Exception:
        ex.shutdown(wait=False, cancel_futures=True)
        try:
            f.flush()
            f.close()
        except Exception:  # noqa: BLE001, S110
            pass
        raise
    else:
        ex.shutdown(wait=True)
        f.close()

    if n_failed:
        print(f"Warning: {n_failed}/{total} samples failed and were skipped.")

    total_done = len(completed) + n_new
    print(f"Successfully generated {total_done} samples for package {package_str}.")


if __name__ == "__main__":
    fire.Fire(run)
