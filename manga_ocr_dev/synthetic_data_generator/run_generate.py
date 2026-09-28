import traceback
from pathlib import Path

import cv2
import fire
import pandas as pd

# NOTE: processes, not threads. Text rasterization in Pillow holds the GIL,
# so threads no longer parallelize (unlike the old browser subprocesses).
# process_map scales near-linearly with core count. OUT_DIR is passed
# explicitly (not via globals) so spawn-based platforms work too.
from tqdm.contrib.concurrent import process_map

from manga_ocr_dev.env import FONTS_ROOT, DATA_SYNTHETIC_ROOT
from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator

generator = SyntheticDataGenerator()


def f(args):
    try:
        i, out_dir, source, id_, text = args
        filename = f"{id_}.jpg"
        img, text_gt, params = generator.process(text)

        cv2.imwrite(str(Path(out_dir) / filename), img)

        font_path = Path(params["font_path"]).relative_to(FONTS_ROOT)
        ret = source, id_, text_gt, params["vertical"], str(font_path)
        return ret

    except Exception:
        print(traceback.format_exc())


def run(package=0, n_random=1000, n_limit=None, max_workers=16):
    """
    :param package: number of data package to generate
    :param n_random: how many samples with random text to generate
    :param n_limit: limit number of generated samples (for debugging)
    :param max_workers: max number of workers
    """

    package = f"{package:04d}"
    lines = pd.read_csv(DATA_SYNTHETIC_ROOT / f"lines/{package}.csv")
    random_lines = pd.DataFrame(
        {
            "source": "random",
            "id": [f"random_{package}_{i}" for i in range(n_random)],
            "line": None,
        }
    )
    lines = pd.concat([lines, random_lines], ignore_index=True)
    if n_limit:
        lines = lines.sample(n_limit)
    global OUT_DIR
    OUT_DIR = DATA_SYNTHETIC_ROOT / "img" / package
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    args = [(i, str(OUT_DIR), *values) for i, values in enumerate(lines.values)]

    data = process_map(f, args, max_workers=max_workers, desc=f"Processing package {package}")

    data = pd.DataFrame(data, columns=["source", "id", "text", "vertical", "font_path"])
    meta_path = DATA_SYNTHETIC_ROOT / f"meta/{package}.csv"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(meta_path, index=False)


if __name__ == "__main__":
    fire.Fire(run)
