import random
import traceback
import zlib
from pathlib import Path

import cv2
import fire
import numpy as np
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
        i, out_dir, source, id_, text, sfx = args
        # Forked workers all inherit the same global RNG state, which would make
        # different workers emit identical samples. Seed from the sample id:
        # unique per sample and reproducible across runs.
        seed = zlib.crc32(str(id_).encode()) % (2**32)
        np.random.seed(seed)
        random.seed(seed)
        filename = f"{id_}.jpg"
        img, text_gt, params = generator.process(text, sfx=bool(sfx))

        cv2.imwrite(str(Path(out_dir) / filename), img)

        font_path = Path(params["font_path"]).relative_to(FONTS_ROOT)
        ret = source, id_, text_gt, params["vertical"], str(font_path)
        return ret

    except Exception:
        print(traceback.format_exc())


def run(package=0, n_random=1000, n_limit=None, max_workers=16, split="dialogue"):
    """
    :param package: number of data package to generate
    :param n_random: how many samples with random text to generate (dialogue split),
        or how many SFX samples to generate (sfx split)
    :param n_limit: limit number of generated samples (for debugging)
    :param max_workers: max number of workers
    :param split: "dialogue" (default, unchanged behavior) or "sfx" (separate
        SFX dataset split: random onomatopoeia with curved/perspective styling,
        no bubbles, written to img_sfx/meta_sfx)
    """

    package = f"{package:04d}"
    if split == "sfx":
        lines = pd.DataFrame(
            {
                "source": "sfx",
                "id": [f"sfx_{package}_{i}" for i in range(n_random)],
                "line": None,
            }
        )
        sfx_flags = [True] * len(lines)
        OUT_DIR = DATA_SYNTHETIC_ROOT / "img_sfx" / package
        meta_path = DATA_SYNTHETIC_ROOT / "meta_sfx" / f"{package}.csv"
    else:
        lines = pd.read_csv(DATA_SYNTHETIC_ROOT / f"lines/{package}.csv")
        random_lines = pd.DataFrame(
            {
                "source": "random",
                "id": [f"random_{package}_{i}" for i in range(n_random)],
                "line": None,
            }
        )
        lines = pd.concat([lines, random_lines], ignore_index=True)
        sfx_flags = [False] * len(lines)
        OUT_DIR = DATA_SYNTHETIC_ROOT / "img" / package
        meta_path = DATA_SYNTHETIC_ROOT / "meta" / f"{package}.csv"
    if n_limit:
        idx = lines.sample(n_limit).index
        lines = lines.loc[idx]
        sfx_flags = [sfx_flags[i] for i in idx]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    args = [(i, str(OUT_DIR), *values, sfx) for i, (values, sfx) in enumerate(zip(lines.values, sfx_flags))]

    data = process_map(f, args, max_workers=max_workers, desc=f"Processing package {package} ({split})")

    n_failed = sum(d is None for d in data)
    if n_failed:
        print(f"WARNING: {n_failed}/{len(data)} samples failed and were skipped (see tracebacks above)")
    data = [d for d in data if d is not None]
    data = pd.DataFrame(data, columns=["source", "id", "text", "vertical", "font_path"])
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(meta_path, index=False)


if __name__ == "__main__":
    fire.Fire(run)
