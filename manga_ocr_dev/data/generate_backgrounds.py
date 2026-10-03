from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from manga_ocr_dev.env import MANGA109_ROOT, BACKGROUND_DIR

def find_rectangle(mask, y, x, aspect_ratio_range=(0.33, 3.0)):
    h_max, w_max = mask.shape[:2]
    ymin = y
    ymax = y + 1
    xmin = x
    xmax = x + 1

    can_up = True
    can_down = True
    can_left = True
    can_right = True

    while can_up or can_down or can_left or can_right:
        if can_up:
            if ymin > 0 and not mask[ymin - 1, xmin:xmax].any():
                ymin -= 1
            else:
                can_up = False

        if can_down:
            if ymax < h_max and not mask[ymax, xmin:xmax].any():
                ymax += 1
            else:
                can_down = False

        if can_left:
            if xmin > 0 and not mask[ymin:ymax, xmin - 1].any():
                xmin -= 1
            else:
                can_left = False

        if can_right:
            if xmax < w_max and not mask[ymin:ymax, xmax].any():
                xmax += 1
            else:
                can_right = False

        h = ymax - ymin
        w = xmax - xmin
        if h > 1 and w > 1:
            ratio = w / h
            if ratio < aspect_ratio_range[0] or ratio > aspect_ratio_range[1]:
                break

    return ymin, ymax, xmin, xmax

def generate_backgrounds(crops_per_page=5, min_size=40):
    data = pd.read_csv(MANGA109_ROOT / "data.csv")
    frames_df = pd.read_csv(MANGA109_ROOT / "frames.csv")

    BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)

    page_paths = data.page_path.unique()
    for page_path in tqdm(page_paths):
        page = cv2.imread(str(MANGA109_ROOT / page_path))
        mask = np.zeros((page.shape[0], page.shape[1]), dtype=bool)
        for row in data[data.page_path == page_path].itertuples():
            mask[row.ymin : row.ymax, row.xmin : row.xmax] = True

        frames_mask = np.zeros((page.shape[0], page.shape[1]), dtype=bool)
        for row in frames_df[frames_df.page_path == page_path].itertuples():
            frames_mask[row.ymin : row.ymax, row.xmin : row.xmax] = True

        mask = mask | ~frames_mask

        if mask.all():
            continue

        unmasked_points = np.stack(np.where(~mask), axis=1)
        for i in range(crops_per_page):
            p = unmasked_points[np.random.randint(0, unmasked_points.shape[0])]
            y, x = p
            ymin, ymax, xmin, xmax = find_rectangle(mask, y, x)
            crop = page[ymin:ymax, xmin:xmax]

            if crop.shape[0] >= min_size and crop.shape[1] >= min_size:
                out_filename = (
                    "_".join(Path(page_path).with_suffix("").parts[-2:]) + f"_{ymin}_{ymax}_{xmin}_{xmax}.png"
                )
                cv2.imwrite(str(BACKGROUND_DIR / out_filename), crop)


if __name__ == "__main__":
    generate_backgrounds()
