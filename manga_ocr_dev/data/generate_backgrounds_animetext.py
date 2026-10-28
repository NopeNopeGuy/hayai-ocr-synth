"""Generate clean (text-free) background crops from deepghs/AnimeText.

Extracts manga/anime illustration background crops without text overlay for synthetic data rendering.

Source: https://huggingface.co/datasets/deepghs/AnimeText (train split).
Boxes are YOLO normalized xywh (center-x, center-y, w, h in 0..1); category
0 = text, 1 = hard_negative. Both are masked so crops are text-free.

Usage:
    python -m manga_ocr_dev.data.generate_backgrounds_animetext --n_images 500
    python -m manga_ocr_dev.data.generate_backgrounds_animetext --n_images 10000 --out_dir ~/data/manga/AnimeText_background
"""

import os
from pathlib import Path

import cv2
import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

from manga_ocr_dev.data.generate_backgrounds import find_rectangle
from manga_ocr_dev.env import BACKGROUND_ANIMETEXT_DIR


def yolo_boxes_to_mask(h, w, bboxes, dilation=3):
    """Build a boolean mask (True = masked / contains text) from YOLO normalized xywh boxes."""
    mask = np.zeros((h, w), dtype=bool)
    for box in bboxes or []:
        xc, yc, bw, bh = (float(v) for v in box)
        xmin = int((xc - bw / 2) * w)
        xmax = int((xc + bw / 2) * w)
        ymin = int((yc - bh / 2) * h)
        ymax = int((yc + bh / 2) * h)
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)
        if xmax > xmin and ymax > ymin:
            mask[ymin:ymax, xmin:xmax] = True
    if dilation and mask.any():
        kernel = np.ones((dilation, dilation), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    return mask


def generate_backgrounds_animetext(
    n_images=500,
    crops_per_image=5,
    min_size=40,
    out_dir=None,
    split="train",
    seed=0,
    dilation=3,
):
    """Stream anime images, mask detected text regions, and extract clean background crops."""
    from datasets import load_dataset

    out_dir = Path(out_dir).expanduser() if out_dir else BACKGROUND_ANIMETEXT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    ds = load_dataset("deepghs/AnimeText", split=split, streaming=True)

    manifest = []
    n_seen = n_kept = n_attempts = n_skipped_full = 0

    it = ds.take(n_images) if n_images else ds
    for row in tqdm(it, total=n_images or None, desc=f"AnimeText processing ({split})"):
        n_seen += 1
        image_id = row.get("image_id", n_seen)
        img = np.array(row["image"].convert("RGB"))[:, :, ::-1]  # RGB -> BGR for cv2
        h, w = img.shape[:2]
        objs = row.get("objects") or {}
        mask = yolo_boxes_to_mask(h, w, objs.get("bbox"), dilation=dilation)

        if mask.all():
            n_skipped_full += 1
            continue

        unmasked_points = np.stack(np.where(~mask), axis=1)
        for _ in range(crops_per_image):
            n_attempts += 1
            y, x = unmasked_points[rng.randint(0, unmasked_points.shape[0])]
            ymin, ymax, xmin, xmax = find_rectangle(mask, int(y), int(x))
            crop = img[ymin:ymax, xmin:xmax]
            if crop.shape[0] >= min_size and crop.shape[1] >= min_size:
                out_filename = f"{image_id}_{ymin}_{ymax}_{xmin}_{xmax}.png"
                cv2.imwrite(str(out_dir / out_filename), crop)
                n_kept += 1
                manifest.append(
                    {
                        "source": f"animetext_{split}",
                        "image_id": image_id,
                        "ymin": ymin,
                        "ymax": ymax,
                        "xmin": xmin,
                        "xmax": xmax,
                        "h": ymax - ymin,
                        "w": xmax - xmin,
                        "n_boxes": len(objs.get("bbox") or []),
                        "path": out_filename,
                    }
                )

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
