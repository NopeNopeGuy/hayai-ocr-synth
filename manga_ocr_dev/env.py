"""Global environment configuration and default paths for manga_ocr_dev."""

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

# Assets directory: checks package-local assets first, then repository root assets.
ASSETS_PATH = Path(
    os.environ.get(
        "MANGA_OCR_ASSETS_PATH",
        (_ROOT / "assets") if (_ROOT / "assets").is_dir() else (_ROOT.parent / "assets"),
    )
).expanduser()

# Root directory containing Japanese TrueType / OpenType font collections.
_DEFAULT_FONTS = Path("/mnt/data/jp_fonts") if Path("/mnt/data/jp_fonts").is_dir() else (Path.home() / "data" / "jp_fonts")
FONTS_ROOT = Path(os.environ.get("FONTS_ROOT", _DEFAULT_FONTS)).expanduser()

# Destination directory for generated synthetic dataset packages (img/, lines/, meta/).
DATA_SYNTHETIC_ROOT = Path(
    os.environ.get(
        "DATA_SYNTHETIC_ROOT",
        Path("/mnt/Synth_Images") if Path("/mnt/Synth_Images").is_dir() else (Path.home() / "data" / "synthetic"),
    )
).expanduser()

# Background image directories.
BACKGROUND_DIR = Path(
    os.environ.get("BACKGROUND_DIR", Path.home() / "data" / "manga" / "Manga109s" / "background")
).expanduser()

_DEFAULT_BG_ANIMETEXT = (
    Path("/mnt/data/manga/AnimeText_background")
    if Path("/mnt/data/manga/AnimeText_background").is_dir()
    else (Path.home() / "data" / "manga" / "AnimeText_background")
)
BACKGROUND_ANIMETEXT_DIR = Path(
    os.environ.get("BACKGROUND_ANIMETEXT_DIR", _DEFAULT_BG_ANIMETEXT)
).expanduser()

# Manga109-s dataset root directory.
MANGA109_ROOT = Path(os.environ.get("MANGA109_ROOT", Path.home() / "data" / "manga" / "Manga109s")).expanduser()
