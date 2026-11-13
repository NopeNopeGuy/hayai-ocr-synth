"""Generate clean (text-free) background crops from deepghs/AnimeText.

Extracts manga/anime illustration background crops without text overlay for synthetic data rendering.

Source: https://huggingface.co/datasets/deepghs/AnimeText (train split).
Boxes are YOLO normalized xywh (center-x, center-y, w, h in 0..1); category
0 = text, 1 = hard_negative. Both are masked so crops are text-free.

Usage:
    python -m manga_ocr_dev.data.generate_backgrounds_animetext --n_images 500
    python -m manga_ocr_dev.data.generate_backgrounds_animetext --n_images 10000 --num_workers 8 --out_dir ~/data/manga/AnimeText_background
"""

import io
import multiprocessing as mp
import os
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import cv2
import fire
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from tqdm import tqdm

from manga_ocr_dev.data.generate_backgrounds import find_rectangle
from manga_ocr_dev.env import BACKGROUND_ANIMETEXT_DIR

_WANTED_COLUMNS = ("image_id", "image", "objects")


def yolo_boxes_to_mask(h, w, bboxes, dilation=3, _kernel=None):
    """Build a boolean mask (True = masked / contains text) from YOLO normalized xywh boxes."""
    if not bboxes:
        return np.zeros((h, w), dtype=bool)

    mask = np.zeros((h, w), dtype=np.uint8)
    has_box = False
    for box in bboxes:
        xc, yc, bw, bh = (float(v) for v in box)
        xmin = max(0, int((xc - bw / 2) * w))
        xmax = min(w, int((xc + bw / 2) * w))
        ymin = max(0, int((yc - bh / 2) * h))
        ymax = min(h, int((yc + bh / 2) * h))
        if xmax > xmin and ymax > ymin:
            mask[ymin:ymax, xmin:xmax] = 1
            has_box = True

    if dilation and has_box:
        kernel = _kernel if _kernel is not None else np.ones((dilation, dilation), np.uint8)
        mask = cv2.dilate(mask, kernel)
    return mask.astype(bool)


def _decode_image_bgr(image_val):
    """Decode HF image column (raw bytes dict or PIL Image) into a contiguous BGR uint8 array."""
    if isinstance(image_val, dict):
        img_bytes = image_val.get("bytes")
        if img_bytes is not None:
            buf = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                return img
            pil_img = PILImage.open(io.BytesIO(img_bytes))
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        path = image_val.get("path")
        if path:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                return img

    if image_val.mode != "RGB":
        image_val = image_val.convert("RGB")
    return cv2.cvtColor(np.asarray(image_val), cv2.COLOR_RGB2BGR)


def _sample_unmasked_point(mask, h, w, rng, ys_xs_cache):
    """Sample a uniform unmasked (y, x) coordinate using fast rejection sampling before np.where."""
    if ys_xs_cache is None:
        for _ in range(16):
            y = int(rng.randint(0, h))
            x = int(rng.randint(0, w))
            if not mask[y, x]:
                return y, x, None
        ys_xs_cache = np.where(~mask)

    ys, xs = ys_xs_cache
    if ys.size == 0:
        return None, None, ys_xs_cache
    idx = int(rng.randint(0, ys.size))
    return int(ys[idx]), int(xs[idx]), ys_xs_cache


def _process_row(
    row,
    fallback_id,
    out_dir_str,
    split,
    crops_per_image,
    min_size,
    dilation,
    kernel,
    rng,
):
    """Mask, crop, and save clean background patches for a single dataset row."""
    image_id = row.get("image_id")
    if image_id is None:
        image_id = fallback_id

    img = _decode_image_bgr(row["image"])
    h, w = img.shape[:2]
    objs = row.get("objects") or {}
    bboxes = objs.get("bbox") or []
    mask = yolo_boxes_to_mask(h, w, bboxes, dilation=dilation, _kernel=kernel)

    manifest_rows = []
    n_kept = 0
    n_attempts = 0
    ys_xs_cache = None

    for _ in range(crops_per_image):
        y, x, ys_xs_cache = _sample_unmasked_point(mask, h, w, rng, ys_xs_cache)
        if y is None:
            if n_attempts == 0:
                return manifest_rows, 0, 0, True
            break

        n_attempts += 1
        ymin, ymax, xmin, xmax = find_rectangle(mask, y, x)
        if (ymax - ymin) >= min_size and (xmax - xmin) >= min_size:
            crop = img[ymin:ymax, xmin:xmax]
            out_filename = f"{image_id}_{ymin}_{ymax}_{xmin}_{xmax}.png"
            cv2.imwrite(os.path.join(out_dir_str, out_filename), crop)
            n_kept += 1
            manifest_rows.append(
                {
                    "source": f"animetext_{split}",
                    "image_id": image_id,
                    "ymin": ymin,
                    "ymax": ymax,
                    "xmin": xmin,
                    "xmax": xmax,
                    "h": ymax - ymin,
                    "w": xmax - xmin,
                    "n_boxes": len(bboxes),
                    "path": out_filename,
                }
            )

    return manifest_rows, n_kept, n_attempts, False


def _get_num_shards(ds):
    return getattr(ds, "num_shards", None) or getattr(ds, "n_shards", 1) or 1


def _open_streaming_dataset(split, do_reshard=False):
    from datasets import Image, load_dataset

    ds = load_dataset("deepghs/AnimeText", split=split, streaming=True)
    if do_reshard and hasattr(ds, "reshard"):
        ds = ds.reshard()

    col_names = getattr(ds, "column_names", None)
    if col_names:
        keep_cols = [c for c in _WANTED_COLUMNS if c in col_names]
        if keep_cols and len(keep_cols) < len(col_names):
            ds = ds.select_columns(keep_cols)

    # Disable internal PIL decode threadpool so workers decode raw bytes directly with cv2
    if hasattr(ds, "decode"):
        try:
            ds = ds.decode(False)
        except Exception:
            pass
    elif not ds.features or "image" in ds.features:
        ds = ds.cast_column("image", Image(decode=False))

    return ds


def _shard_dataset(ds, num_shards, index):
    if num_shards <= 1:
        return ds
    if hasattr(ds, "shard"):
        return ds.shard(num_shards=num_shards, index=index)
    from datasets.distributed import split_dataset_by_node

    return split_dataset_by_node(ds, rank=index, world_size=num_shards)


def _worker_shard_loop(
    worker_id,
    effective_workers,
    do_reshard,
    quota,
    split,
    out_dir_str,
    crops_per_image,
    min_size,
    dilation,
    seed,
    progress_queue,
):
    """Worker entrypoint: streams its assigned shards and processes rows independently."""
    try:
        ds = _open_streaming_dataset(split, do_reshard=do_reshard)
        ds_shard = _shard_dataset(ds, effective_workers, worker_id)

        rng = np.random.RandomState(seed + worker_id)
        kernel = np.ones((dilation, dilation), np.uint8) if dilation else None

        n_seen = n_kept = n_attempts = n_skipped_full = 0
        seen_since_flush = kept_since_flush = 0
        manifest_chunk = []

        for local_idx, row in enumerate(ds_shard, start=1):
            if quota is not None and n_seen >= quota:
                break

            n_seen += 1
            seen_since_flush += 1
            rows, kept, attempts, skipped_full = _process_row(
                row=row,
                fallback_id=f"w{worker_id}_{local_idx}",
                out_dir_str=out_dir_str,
                split=split,
                crops_per_image=crops_per_image,
                min_size=min_size,
                dilation=dilation,
                kernel=kernel,
                rng=rng,
            )
            if skipped_full:
                n_skipped_full += 1
            else:
                n_kept += kept
                n_attempts += attempts
                kept_since_flush += kept
                if rows:
                    manifest_chunk.extend(rows)

            if seen_since_flush >= 16:
                progress_queue.put(("progress", seen_since_flush, kept_since_flush))
                seen_since_flush = kept_since_flush = 0
                if len(manifest_chunk) >= 2000:
                    progress_queue.put(("manifest", manifest_chunk))
                    manifest_chunk = []

        if seen_since_flush or kept_since_flush:
            progress_queue.put(("progress", seen_since_flush, kept_since_flush))
        if manifest_chunk:
            progress_queue.put(("manifest", manifest_chunk))

        progress_queue.put(("done", worker_id, n_seen, n_kept, n_attempts, n_skipped_full))
    except Exception:
        progress_queue.put(("error", worker_id, traceback.format_exc()))


def _process_row_batch(args):
    """Fallback batch processor used when dataset only has 1 shard."""
    batch, out_dir_str, split, crops_per_image, min_size, dilation, seed = args
    rng = np.random.RandomState(seed)
    kernel = np.ones((dilation, dilation), np.uint8) if dilation else None

    manifest_rows = []
    n_seen = len(batch)
    n_kept = n_attempts = n_skipped_full = 0
    for fallback_id, row in batch:
        rows, kept, attempts, skipped_full = _process_row(
            row=row,
            fallback_id=fallback_id,
            out_dir_str=out_dir_str,
            split=split,
            crops_per_image=crops_per_image,
            min_size=min_size,
            dilation=dilation,
            kernel=kernel,
            rng=rng,
        )
        if skipped_full:
            n_skipped_full += 1
        else:
            n_kept += kept
            n_attempts += attempts
            manifest_rows.extend(rows)
    return manifest_rows, n_seen, n_kept, n_attempts, n_skipped_full


def _run_single_shard_parallel(
    ds,
    n_images,
    out_dir_str,
    split,
    crops_per_image,
    min_size,
    dilation,
    seed,
    num_workers,
):
    """Parallelize CPU decoding/cropping when the split cannot be sharded across workers."""
    ctx = mp.get_context("spawn")
    max_in_flight = num_workers * 2
    batch_size = 8

    manifest = []
    n_seen = n_kept = n_attempts = n_skipped_full = 0
    it = ds.take(n_images) if n_images else ds

    with (
        ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool,
        tqdm(total=n_images or None, desc=f"AnimeText processing ({split})") as pbar,
    ):
        pending = set()
        batch = []
        batch_idx = 0

        def drain_done(wait_all=False):
            nonlocal n_seen, n_kept, n_attempts, n_skipped_full
            if not pending:
                return
            done, _ = wait(pending, return_when=FIRST_COMPLETED if not wait_all else None)
            for fut in done:
                pending.remove(fut)
                rows, seen, kept, attempts, skipped = fut.result()
                manifest.extend(rows)
                n_seen += seen
                n_kept += kept
                n_attempts += attempts
                n_skipped_full += skipped
                pbar.update(seen)
                pbar.set_postfix(kept=n_kept, refresh=False)

        for idx, row in enumerate(it, start=1):
            batch.append((idx, row))
            if len(batch) >= batch_size:
                if len(pending) >= max_in_flight:
                    drain_done(wait_all=False)
                pending.add(
                    pool.submit(
                        _process_row_batch,
                        (batch, out_dir_str, split, crops_per_image, min_size, dilation, seed + batch_idx),
                    )
                )
                batch = []
                batch_idx += 1

        if batch:
            pending.add(
                pool.submit(
                    _process_row_batch,
                    (batch, out_dir_str, split, crops_per_image, min_size, dilation, seed + batch_idx),
                )
            )
        while pending:
            drain_done(wait_all=False)

    return manifest, n_seen, n_kept, n_attempts, n_skipped_full


def generate_backgrounds_animetext(
    n_images=500,
    crops_per_image=5,
    min_size=40,
    out_dir=None,
    split="train",
    seed=0,
    dilation=3,
    num_workers=None,
):
    """Stream anime images in parallel, mask detected text regions, and extract clean background crops."""
    out_dir = Path(out_dir).expanduser() if out_dir else BACKGROUND_ANIMETEXT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_str = str(out_dir)

    if num_workers is None:
        num_workers = min(8, max(4, os.cpu_count() or 4))
    num_workers = max(1, int(num_workers))

    ds = _open_streaming_dataset(split, do_reshard=False)
    num_shards = _get_num_shards(ds)
    do_reshard = False
    if num_workers > 1 and num_shards < num_workers and hasattr(ds, "reshard"):
        ds = ds.reshard()
        num_shards = _get_num_shards(ds)
        do_reshard = True

    manifest = []
    n_seen = n_kept = n_attempts = n_skipped_full = 0

    if num_workers == 1 or (num_shards == 1 and n_images and n_images <= 16):
        rng = np.random.RandomState(seed)
        kernel = np.ones((dilation, dilation), np.uint8) if dilation else None
        it = ds.take(n_images) if n_images else ds
        with tqdm(total=n_images or None, desc=f"AnimeText processing ({split})") as pbar:
            for idx, row in enumerate(it, start=1):
                n_seen += 1
                rows, kept, attempts, skipped_full = _process_row(
                    row=row,
                    fallback_id=idx,
                    out_dir_str=out_dir_str,
                    split=split,
                    crops_per_image=crops_per_image,
                    min_size=min_size,
                    dilation=dilation,
                    kernel=kernel,
                    rng=rng,
                )
                if skipped_full:
                    n_skipped_full += 1
                else:
                    n_kept += kept
                    n_attempts += attempts
                    manifest.extend(rows)
                pbar.update(1)
                if idx % 16 == 0:
                    pbar.set_postfix(kept=n_kept, refresh=False)

    elif num_shards == 1:
        manifest, n_seen, n_kept, n_attempts, n_skipped_full = _run_single_shard_parallel(
            ds=ds,
            n_images=n_images,
            out_dir_str=out_dir_str,
            split=split,
            crops_per_image=crops_per_image,
            min_size=min_size,
            dilation=dilation,
            seed=seed,
            num_workers=num_workers,
        )

    else:
        effective_workers = min(num_workers, num_shards, n_images) if n_images else min(num_workers, num_shards)
        ctx = mp.get_context("spawn")
        progress_queue = ctx.Queue()

        workers = []
        base_quota, rem = divmod(n_images, effective_workers) if n_images else (None, 0)
        for worker_id in range(effective_workers):
            quota = (base_quota + (1 if worker_id < rem else 0)) if base_quota is not None else None
            p = ctx.Process(
                target=_worker_shard_loop,
                args=(
                    worker_id,
                    effective_workers,
                    do_reshard,
                    quota,
                    split,
                    out_dir_str,
                    crops_per_image,
                    min_size,
                    dilation,
                    seed,
                    progress_queue,
                ),
            )
            p.start()
            workers.append(p)

        finished_workers = 0
        with tqdm(
            total=n_images or None,
            desc=f"AnimeText processing ({split}, {effective_workers} workers)",
        ) as pbar:
            while finished_workers < effective_workers:
                msg = progress_queue.get()
                kind = msg[0]
                if kind == "progress":
                    _, seen_delta, kept_delta = msg
                    n_kept += kept_delta
                    pbar.update(seen_delta)
                    pbar.set_postfix(kept=n_kept, refresh=False)
                elif kind == "manifest":
                    manifest.extend(msg[1])
                elif kind == "done":
                    _, _, w_seen, _, w_attempts, w_skipped = msg
                    n_seen += w_seen
                    n_attempts += w_attempts
                    n_skipped_full += w_skipped
                    finished_workers += 1
                elif kind == "error":
                    _, worker_id, err_tb = msg
                    for p in workers:
                        if p.is_alive():
                            p.terminate()
                    raise RuntimeError(f"Worker {worker_id} failed:\n{err_tb}")

        for p in workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)
    keep_rate = n_kept / n_attempts if n_attempts else 0.0
    print(
        f"images_seen={n_seen} kept_crops={n_kept} attempts={n_attempts} "
        f"keep_rate={keep_rate:.3f} skipped_full_mask={n_skipped_full} "
        f"crops_per_image={n_kept / n_seen if n_seen else 0:.2f} out={out_dir}"
    )
    return {"images_seen": n_seen, "kept": n_kept, "keep_rate": keep_rate}


if __name__ == "__main__":
    fire.Fire(generate_backgrounds_animetext)
    # Terminate process directly to prevent hung threads during HuggingFace dataset streaming cleanup
    os._exit(0)
