"""Process CC-100 multilingual raw text into package CSV files for synthetic data generation.

CC-100 ships one monolingual file per language (e.g. ``ja.txt``, ``ko.txt``).
Pick the file with ``lang`` (``ja``/``ko``/``zh``/``en``); Chinese accepts
``zh``, ``zh-Hans`` and ``zh-Hant`` interchangeably.
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from manga_ocr_dev.env import DATA_SYNTHETIC_ROOT

# CC-100 file stems per synth language (tried in order; first hit wins).
CC100_FILES = {
    "ja": ["ja.txt"],
    "ko": ["ko.txt"],
    "zh": ["zh-Hans.txt", "zh_Hans.txt", "zh.txt", "zh-Hant.txt", "zh_Hant.txt"],
    "en": ["en.txt"],
}


def export_lines(
    num_lines_in_each_package: int = 100_000,
    num_packages: int = 10,
    lang: str = "ja",
    out_dir=None,
):
    """Split a CC-100 monolingual corpus file into packaged CSV files.

    :param num_lines_in_each_package: Number of text lines per package CSV.
    :param num_packages: Total number of packages to generate.
    :param lang: Corpus language: 'ja', 'ko', 'zh' or 'en'. Japanese keeps the
        original ``lines/`` output dir; other languages write ``lines_<lang>/``
        consumed by ``run_generate --lang <lang>``.
    :param out_dir: Optional explicit output directory (overrides the default).
    """
    from pathlib import Path

    from manga_ocr_dev.synthetic_data_generator.lang import normalize_lang

    lang = normalize_lang(lang)
    candidates = CC100_FILES[lang]
    cc100_text_file = next((DATA_SYNTHETIC_ROOT / c for c in candidates if (DATA_SYNTHETIC_ROOT / c).is_file()), None)
    if cc100_text_file is None:
        raise FileNotFoundError(
            f"CC-100 corpus file not found for lang={lang!r}: tried {[str(DATA_SYNTHETIC_ROOT / c) for c in candidates]}"
        )
    if out_dir is None:
        out_dir = DATA_SYNTHETIC_ROOT / ("lines" if lang == "ja" else f"lines_{lang}")
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    id_count = 0
    with open(cc100_text_file, "r", encoding="utf-8") as file:
        for package_count in range(num_packages):
            line_count = 0
            data = []
            for line in tqdm(file, desc=f"Creating package {package_count:04d} of {num_packages} ({lang})"):
                id_count += 1
                stripped_line = line.strip()
                # Skip lines that are too short to form meaningful text blocks
                if len(stripped_line) <= 2:
                    continue

                row = {
                    "source": f"cc-100-{lang}",
                    "id": f"cc-100-{lang}_{id_count}",
                    "line": stripped_line,
                }
                data.append(row)

                line_count += 1
                if line_count >= num_lines_in_each_package:
                    break

            df = pd.DataFrame(data)
            df.to_csv(out_dir / f"{package_count:04d}.csv", index=False)


if __name__ == "__main__":
    import fire

    fire.Fire(export_lines)
