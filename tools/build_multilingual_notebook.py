#!/usr/bin/env python3
"""Build the multilingual synthetic manga OCR generator notebook for Kaggle & Colab."""

import json
from pathlib import Path

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_cell(cell_type, source):
        # Format as list of strings with newlines
        if isinstance(source, str):
            lines = [line + "\n" for line in source.split("\n")]
            if lines and lines[-1] == "\n":
                lines[-1] = ""
        else:
            lines = source
        nb["cells"].append({
            "cell_type": cell_type,
            "metadata": {},
            "source": lines,
            **({"outputs": [], "execution_count": None} if cell_type == "code" else {})
        })

    # Cell 0: Header Markdown
    add_cell("markdown", """# Manga OCR Multilingual Synthetic Data Generator (Colab / Kaggle)

Generate high-quality synthetic manga/comic OCR training crops for **Chinese (`zh`)**, **Korean (`ko`)**, **English (`en`)**, and **Japanese (`ja`)**, and append them as parquet shards to a Hugging Face dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NopeNopeGuy/hayai-ocr-synth/blob/master/kaggle/multilingual_synth.ipynb)

### Pipeline Overview:
1. **Fonts**:
   - **Local `/mnt/data/jp_fonts`**: Auto-detected if present (e.g. local workstation).
   - **Kaggle / Colab streaming & download**: Streams/downloads fonts from the private Hugging Face dataset [`JustANormalTinkerer/manga-ocr-multilingual-fonts`](https://huggingface.co/datasets/JustANormalTinkerer/manga-ocr-multilingual-fonts). Supports selective download per language:
     - `zh` (Chinese): `Chinese_Additions/`, `精简包/简体/`, `精简包/繁体/`
     - `ko` (Korean): `Korean/`
     - `en` (English): `Latin_Comic/`
     - `ja` (Japanese): `精简包/日文/`
2. **Backgrounds**:
   - **Local `/mnt/data/manga/AnimeText_background`**: Auto-detected if present.
   - **Streaming AnimeText**: Streams illustration backgrounds directly from [`deepghs/AnimeText`](https://huggingface.co/datasets/deepghs/AnimeText) on the fly, masks text regions, and extracts clean crops.
3. **Parquet Shards & Hub Upload**:
   - Generates Parquet shards with schema: `(image, transcription, language, source, image_id, vertical)`.
   - Supports configurable split names (`chinese`, `korean`, `english`, `sfx`, etc.).
   - Append-only upload with Hugging Face dataset card update.""")

    # Cell 1: Dependencies
    add_cell("code", """# ---------------------------------------------------------------------------
# 1. Dependencies & System Setup
# ---------------------------------------------------------------------------
import os
import sys
import subprocess

IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = os.path.exists("/kaggle")
print(f"Environment: Colab={IN_COLAB}, Kaggle={IN_KAGGLE}, Local={not (IN_COLAB or IN_KAGGLE)}")

REQUIRED_PKGS = [
    "datasets",
    "huggingface_hub",
    "pyarrow",
    "fire",
    "fonttools",
    "budoux",
    "scipy",
    "opencv-python-headless",
    "tqdm",
    "pillow",
]

def ensure_pkgs():
    missing = []
    for pkg in REQUIRED_PKGS:
        mod = pkg.replace("-", "_").split("=")[0]
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
        print("Packages installed successfully.")

ensure_pkgs()
print("All dependencies satisfied.")""")

    # Cell 2: Configuration
    add_cell("code", """# ---------------------------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------------------------
# Target language for synthetic rendering:
#   zh = Chinese (Hanzi, Simplified + Traditional, punctuation)
#   ko = Korean (Hangul dialogue & sound effects)
#   en = English (Comic lettering & sound effects)
#   ja = Japanese (Kanji + Kana dialogue, furigana & SFX)
LANGUAGE = "zh"  # @param ["zh", "ko", "en", "ja"]

# Target dataset split name. Leave "auto" to use standard naming:
#   zh -> chinese
#   ko -> korean
#   en -> english
#   ja -> japanese
# Or specify any custom split (e.g. chinese_dialogue, korean_webtoon, sfx)
SPLIT = "auto"  # @param {type:"string"}

# Destination Hugging Face dataset repository (private or public)
HF_DATASET = "JustANormalTinkerer/hayai-synth-multilingual"  # @param {type:"string"}

# Crop preset:
#   auto             = dialogue for normal splits, sfx for sfx splits
#   dialogue         = normal dialogue, random speech bubble / no-bubble
#   dialogue_bubble  = force a speech bubble
#   dialogue_flat    = text directly on background without bubble
#   dialogue_tight   = tight bounding crop around text
#   handwritten      = irregular hand-lettered layout + handwritten fonts
#   sfx              = comic onomatopoeia with outlines, curves, effects
#   mixed            = balanced mix of dialogue, flat, handwritten, and sfx
#   custom           = use OVERRIDE_JSON
CROP_PRESET = "auto"  # @param ["auto", "dialogue", "dialogue_bubble", "dialogue_flat", "dialogue_tight", "handwritten", "sfx", "mixed", "custom"]

# Generation volume & shard sizing
N_IMAGES = 5000  # @param {type:"integer"}
ROWS_PER_SHARD = 5000  # @param {type:"integer"}
WORKERS = 4  # @param {type:"integer"}
SEED = 0  # @param {type:"integer"}
JPEG_QUALITY = 95  # @param {type:"integer"}
OUT_DIR = ""  # @param {type:"string"}
RUN_ID = ""  # @param {type:"string"}
RESUME = False  # @param {type:"boolean"}

# Text line source:
#   auto                = automatically selects lines matching LANGUAGE:
#                         zh -> hf:chinese_webnovel (or /mnt/data/lines/chinese_webnovel.parquet if present)
#                         en -> hf:english_discord (or /mnt/data/lines/english_discord.parquet if present)
#                         ko -> hf:korean_wiki (or /mnt/data/lines/korean_wiki_dialogue.parquet if present)
#                         ja -> hf:dialogue
#   hf_chinese_webnovel = stream wdndev/webnovel-chinese
#   hf_english_discord  = stream mookiezi/Discord-Dialogues
#   hf_korean_wiki      = stream heegyu/kowiki-sentences & conversation
#   random              = procedural multilingual dictionary sampling (zero setup)
#   sfx                 = procedural multilingual onomatopoeia sampling
#   local               = read custom text file from LINES_PATH (.txt/.csv/.parquet/.jsonl)
#   hf_dialogue         = stream dialogue dataset from Hugging Face (Japanese)
#   hf_novel            = stream novel corpus from Hugging Face (Japanese)
LINES_MODE = "auto"  # @param ["auto", "random", "sfx", "hf_chinese_webnovel", "hf_english_discord", "hf_korean_wiki", "hf_dialogue", "hf_novel", "local"]
LINES_PATH = ""  # @param {type:"string"}
TEXT_COLUMN = ""  # @param {type:"string"}
SOURCE_LABEL = ""  # @param {type:"string"}

# --- Fonts Configuration ---
# HF repository containing the curated CJK & Latin comic fonts
FONTS_HF_DATASET = "JustANormalTinkerer/manga-ocr-multilingual-fonts"  # @param {type:"string"}
# Fetch mode:
#   auto                = check /mnt/data/jp_fonts or inputs; if absent, download only the folders needed for LANGUAGE
#   selected_lang_only  = always fetch only the fonts needed for LANGUAGE from Hub
#   full                = fetch all multilingual font folders
#   local_only          = strictly use local font path (fail if not found)
FONTS_FETCH_MODE = "auto"  # @param ["auto", "selected_lang_only", "full", "local_only"]
FONTS_ROOT_OVERRIDE = ""  # @param {type:"string"}

# --- Backgrounds Configuration ---
# Background source:
#   auto              = check /mnt/data/manga/AnimeText_background or inputs; if absent, stream from AnimeText
#   stream_animetext  = stream deepghs/AnimeText directly and extract clean background crops
#   local_only        = strictly use local background directory
BACKGROUNDS_MODE = "auto"  # @param ["auto", "stream_animetext", "local_only"]
STREAM_BG_IMAGES = 200  # @param {type:"integer"}
BACKGROUNDS_DIR_OVERRIDE = ""  # @param {type:"string"}

# --- Crop & Styling Controls ---
VERTICAL = "auto"  # @param ["auto", "true", "false"]
BUBBLE = "auto"  # @param ["auto", "always", "never"]
BUBBLE_PROB = -1.0  # @param {type:"number"}
TIGHT = False  # @param {type:"boolean"}
IRREGULAR = "auto"  # @param ["auto", "true", "false"]
COLOR = "auto"  # @param ["auto", "mono", "color"]
COLOR_MODE = ""  # @param ["", "solid", "per_line", "per_char"]
TEXT_COLOR = ""  # @param {type:"string"}
FONT_SIZE_MIN = 0  # @param {type:"integer"}
FONT_SIZE_MAX = 0  # @param {type:"integer"}
STROKE_SIZE = -1  # @param {type:"integer"}
SLANT_DEG = -999.0  # @param {type:"number"}
DOUBLE_OUTLINE = "auto"  # @param ["auto", "true", "false"]
HARD_INK = "auto"  # @param ["auto", "off"]
PRESET_MIX = "dialogue=0.6,handwritten=0.2,sfx=0.1,dialogue_flat=0.1"  # @param {type:"string"}
OVERRIDE_JSON = ""  # @param {type:"string"}

# --- Upload & Hub Controls ---
PREVIEW_N = 8  # @param {type:"integer"}
PUSH_TO_HUB = False  # @param {type:"boolean"}
UPDATE_CARD = True  # @param {type:"boolean"}
DRY_RUN_CARD = True  # @param {type:"boolean"}

# Auto-resolve SPLIT if "auto"
if SPLIT == "auto":
    DEFAULT_SPLITS = {"zh": "chinese", "ko": "korean", "en": "english", "ja": "japanese"}
    SPLIT = DEFAULT_SPLITS.get(LANGUAGE, "multilingual")
    if CROP_PRESET == "sfx" or LINES_MODE == "sfx":
        SPLIT = f"{SPLIT}_sfx"
    elif CROP_PRESET == "handwritten":
        SPLIT = f"{SPLIT}_handwritten"

print(f"Configuration loaded: language={LANGUAGE}, split={SPLIT}, preset={CROP_PRESET}, lines={LINES_MODE}")""")

    # Cell 3: Repo & HF Auth
    add_cell("code", """# ---------------------------------------------------------------------------
# 3. Locate Repository & Hugging Face Authentication
# ---------------------------------------------------------------------------
from pathlib import Path
import os
import sys

REPO_URL = "https://github.com/NopeNopeGuy/hayai-ocr-synth.git"

def locate_repo() -> Path:
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path("/kaggle/working/hayai-ocr-synth"),
        Path("/content/hayai-ocr-synth"),
        Path.home() / "hayai-ocr-synth",
        Path("/home/nopenope/Github/manga-ocr-better-synth"),
    ]
    for c in candidates:
        if (c / "manga_ocr_dev").is_dir() and (c / "tools" / "append_hayai_synth.py").is_file():
            return c.resolve()

    target = Path("/content/hayai-ocr-synth" if IN_COLAB else "/kaggle/working/hayai-ocr-synth")
    import subprocess
    if not (target / "manga_ocr_dev").is_dir():
        print(f"Cloning generator repository into {target}...")
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(target)])
    else:
        print(f"Updating existing repository at {target}...")
        try:
            subprocess.run(["git", "-C", str(target), "pull", "--ff-only"], check=False)
        except Exception:
            pass

    # Ensure Git LFS pointers are resolved
    try:
        subprocess.run(["git", "-C", str(target), "lfs", "pull"], check=False)
    except Exception:
        pass
    return target.resolve()

REPO_DIR = locate_repo()
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
print(f"Generator repository located at: {REPO_DIR}")

# Resolve Hugging Face Token
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN and IN_COLAB:
    try:
        from google.colab import userdata
        HF_TOKEN = userdata.get("HF_TOKEN")
    except Exception:
        pass
if not HF_TOKEN and IN_KAGGLE:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=True)
    print("Logged in to Hugging Face successfully.")
else:
    print("HF_TOKEN not found. (Required for fetching private font repos and Hub uploads).")""")

    # Cell 4: Font Discovery & Streaming/Fetching
    add_cell("code", """# ---------------------------------------------------------------------------
# 4. Fetch / Stream Chinese/Korean/English Fonts
# ---------------------------------------------------------------------------
from pathlib import Path
import os

def resolve_fonts() -> Path:
    # 1. Check explicit override
    if FONTS_ROOT_OVERRIDE and Path(FONTS_ROOT_OVERRIDE).is_dir():
        print(f"Using FONTS_ROOT_OVERRIDE: {FONTS_ROOT_OVERRIDE}")
        return Path(FONTS_ROOT_OVERRIDE).resolve()

    # 2. Check local workstation paths (/mnt/data/jp_fonts first)
    local_candidates = [
        Path("/mnt/data/jp_fonts"),
        Path.home() / "data" / "jp_fonts",
        Path("/kaggle/input/jp-fonts"),
        Path("/kaggle/input/manga-fonts"),
    ]
    if FONTS_FETCH_MODE in ("auto", "local_only"):
        for cand in local_candidates:
            if cand.is_dir():
                # Verify it has at least one of the multilingual subfolders
                if any((cand / sub).is_dir() for sub in ["Chinese_Additions", "Korean", "Latin_Comic", "精简包"]):
                    print(f"Found local font root: {cand}")
                    return cand.resolve()

    if FONTS_FETCH_MODE == "local_only":
        raise RuntimeError("FONTS_FETCH_MODE is local_only but no local font directory was found.")

    # 3. Fetch/Stream from Hugging Face dataset
    dest = Path("/content/fonts" if IN_COLAB else "/kaggle/working/fonts" if IN_KAGGLE else REPO_DIR / "downloaded_fonts")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Fetching fonts from Hub repo '{FONTS_HF_DATASET}' into {dest}...")

    # Pattern selection for targeted downloading:
    patterns = None
    if FONTS_FETCH_MODE == "selected_lang_only":
        lang_patterns = {
            "zh": ["Chinese_Additions/**", "精简包/简体/**", "精简包/繁体/**", "精简包/简繁/**"],
            "ko": ["Korean/**"],
            "en": ["Latin_Comic/**"],
            "ja": ["精简包/日文/**", "Chinese_Additions/**"],
        }
        patterns = lang_patterns.get(LANGUAGE)

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=FONTS_HF_DATASET,
        repo_type="dataset",
        token=HF_TOKEN,
        local_dir=str(dest),
        allow_patterns=patterns,
        max_workers=8,
    )
    print(f"Fonts ready at: {dest}")
    return dest.resolve()

RESOLVED_FONTS_ROOT = resolve_fonts()
os.environ["FONTS_ROOT"] = str(RESOLVED_FONTS_ROOT)

# Quick verification of font folder contents
print("Font directory contents:")
for item in sorted(RESOLVED_FONTS_ROOT.iterdir()):
    if item.is_dir():
        font_files = list(item.glob("**/*.[oOtT][tT][fFcC]"))
        print(f"  - {item.name}: {len(font_files)} font file(s)")""")

    # Cell 5: Background Discovery & Streaming
    add_cell("code", """# ---------------------------------------------------------------------------
# 5. Fetch / Stream Backgrounds from AnimeText
# ---------------------------------------------------------------------------
from pathlib import Path
import os

def resolve_backgrounds() -> Path:
    # 1. Check explicit override
    if BACKGROUNDS_DIR_OVERRIDE and Path(BACKGROUNDS_DIR_OVERRIDE).is_dir():
        print(f"Using BACKGROUNDS_DIR_OVERRIDE: {BACKGROUNDS_DIR_OVERRIDE}")
        return Path(BACKGROUNDS_DIR_OVERRIDE).resolve()

    # 2. Check local workstation paths (/mnt/data first)
    local_candidates = [
        Path("/mnt/data/manga/AnimeText_background"),
        Path.home() / "data" / "manga" / "AnimeText_background",
        Path("/kaggle/input/animetext-background"),
        Path("/kaggle/input/manga-backgrounds"),
    ]
    if BACKGROUNDS_MODE in ("auto", "local_only"):
        for cand in local_candidates:
            if cand.is_dir() and (cand / "manifest.csv").is_file():
                print(f"Found existing background directory with manifest: {cand}")
                return cand.resolve()

    if BACKGROUNDS_MODE == "local_only":
        raise RuntimeError("BACKGROUNDS_MODE is local_only but no background directory was found.")

    # 3. Stream from deepghs/AnimeText on the fly
    dest = Path("/content/animetext_bg" if IN_COLAB else "/kaggle/working/animetext_bg" if IN_KAGGLE else REPO_DIR / "animetext_bg")
    dest.mkdir(parents=True, exist_ok=True)
    manifest_file = dest / "manifest.csv"

    if manifest_file.is_file():
        import pandas as pd
        df = pd.read_csv(manifest_file)
        if len(df) >= 200:
            print(f"Reusing existing streamed backgrounds at {dest} ({len(df)} crops).")
            return dest.resolve()

    print(f"Streaming {STREAM_BG_IMAGES} images from deepghs/AnimeText to generate background crops...")
    from manga_ocr_dev.data.generate_backgrounds_animetext import generate_backgrounds_animetext
    generate_backgrounds_animetext(
        n_images=STREAM_BG_IMAGES,
        crops_per_image=5,
        min_size=40,
        out_dir=str(dest),
        split="train",
        seed=SEED,
    )
    print(f"Background extraction complete: {dest}")
    return dest.resolve()

RESOLVED_BG_DIR = resolve_backgrounds()
os.environ["BACKGROUND_ANIMETEXT_DIR"] = str(RESOLVED_BG_DIR)""")

    # Cell 6: Font Sanity Check
    add_cell("code", """# ---------------------------------------------------------------------------
# 6. Sanity Check Font Support for Target Language
# ---------------------------------------------------------------------------
from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
from manga_ocr_dev.synthetic_data_generator.utils import get_background_df

print(f"Initializing generator with FONTS_ROOT={os.environ['FONTS_ROOT']} and lang={LANGUAGE}...")
bg_df = get_background_df(RESOLVED_BG_DIR)
gen = SyntheticDataGenerator(background_df=bg_df)

sample_font = gen.get_random_font(lang=LANGUAGE)
print(f"Successfully sampled font for '{LANGUAGE}': {Path(sample_font).name}")

# Test dialogue sample
img_d, text_d, p_d = gen.process(text=None, sfx=False, lang=LANGUAGE)
print(f"Sample {LANGUAGE} dialogue: text={text_d[:30]!r}, shape={img_d.shape}, font={Path(p_d.get('font_path','')).name}")

# Test SFX sample
img_s, text_s, p_s = gen.process(text=None, sfx=True, lang=LANGUAGE)
print(f"Sample {LANGUAGE} SFX: text={text_s[:30]!r}, shape={img_s.shape}, font={Path(p_s.get('font_path','')).name}")
print("Generator sanity check passed!")""")

    # Cell 7: Build Generator Command Line
    add_cell("code", """# ---------------------------------------------------------------------------
# 7. Build Generation Command Line & Manifest
# ---------------------------------------------------------------------------
import shlex
import uuid
import json
from datetime import datetime, timezone

if not RUN_ID:
    RUN_ID = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"

if not OUT_DIR:
    OUT_DIR = Path("/content/synth_out" if IN_COLAB else "/kaggle/working/synth_out" if IN_KAGGLE else REPO_DIR / "synth_out")
else:
    OUT_DIR = Path(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT = REPO_DIR / "tools" / "append_hayai_synth.py"

# Build lines argument
if LINES_MODE == "auto":
    local_map = {
        "zh": Path("/mnt/data/lines/chinese_webnovel.parquet"),
        "en": Path("/mnt/data/lines/english_discord.parquet"),
        "ko": Path("/mnt/data/lines/korean_wiki_dialogue.parquet"),
    }
    cand = local_map.get(LANGUAGE)
    if cand and cand.is_file():
        LINES_SPEC = str(cand)
        print(f"Auto-selected local lines file: {LINES_SPEC}")
    else:
        stream_map = {
            "zh": "hf:chinese_webnovel",
            "en": "hf:english_discord",
            "ko": "hf:korean_wiki",
            "ja": "hf:dialogue",
        }
        LINES_SPEC = stream_map.get(LANGUAGE, "random")
        print(f"Auto-selected streaming lines: {LINES_SPEC}")
elif LINES_MODE == "hf_chinese_webnovel":
    LINES_SPEC = "hf:chinese_webnovel"
elif LINES_MODE == "hf_english_discord":
    LINES_SPEC = "hf:english_discord"
elif LINES_MODE == "hf_korean_wiki":
    LINES_SPEC = "hf:korean_wiki"
elif LINES_MODE == "random":
    LINES_SPEC = "random"
elif LINES_MODE == "sfx":
    LINES_SPEC = "sfx"
elif LINES_MODE == "local":
    if not LINES_PATH:
        # Check /mnt/data/lines fallback
        default_local = Path(f"/mnt/data/lines/{'chinese_webnovel' if LANGUAGE=='zh' else 'english_discord' if LANGUAGE=='en' else 'korean_wiki_dialogue'}.parquet")
        if default_local.is_file():
            LINES_SPEC = str(default_local)
        else:
            raise FileNotFoundError(f"LINES_MODE is local but LINES_PATH not specified and {default_local} does not exist.")
    else:
        LINES_SPEC = LINES_PATH
elif LINES_MODE == "hf_dialogue":
    LINES_SPEC = "hf:dialogue"
elif LINES_MODE == "hf_novel":
    LINES_SPEC = "hf:novel"
else:
    LINES_SPEC = LINES_MODE

COMMON = [
    "--split", SPLIT,
    "--language", LANGUAGE,
    "--preset", CROP_PRESET,
    "--lines", LINES_SPEC,
    "--out-dir", str(OUT_DIR),
    "--run-id", RUN_ID,
    "--backgrounds", str(RESOLVED_BG_DIR),
    "--fonts-root", str(RESOLVED_FONTS_ROOT),
    "--vertical", VERTICAL,
    "--bubble", BUBBLE,
    "--color", COLOR,
    "--irregular", IRREGULAR,
    "--hard-ink", HARD_INK,
    "--jpeg-quality", str(JPEG_QUALITY),
    "--seed", str(SEED),
]
if BUBBLE_PROB >= 0:
    COMMON += ["--bubble-prob", str(BUBBLE_PROB)]
if TIGHT:
    COMMON += ["--tight"]
if COLOR_MODE:
    COMMON += ["--color-mode", COLOR_MODE]
if TEXT_COLOR:
    COMMON += ["--text-color", TEXT_COLOR]
if FONT_SIZE_MIN > 0:
    COMMON += ["--font-size-min", str(FONT_SIZE_MIN)]
if FONT_SIZE_MAX > 0:
    COMMON += ["--font-size-max", str(FONT_SIZE_MAX)]
if STROKE_SIZE >= 0:
    COMMON += ["--stroke-size", str(STROKE_SIZE)]
if SLANT_DEG > -900:
    COMMON += ["--slant-deg", str(SLANT_DEG)]
if PRESET_MIX and CROP_PRESET == "mixed":
    COMMON += ["--preset-mix", PRESET_MIX]
if OVERRIDE_JSON:
    COMMON += ["--override-json", OVERRIDE_JSON]
if TEXT_COLUMN:
    COMMON += ["--text-column", TEXT_COLUMN]
if SOURCE_LABEL:
    COMMON += ["--source-label", SOURCE_LABEL]

RUN_INFO = {
    "repo": str(REPO_DIR),
    "dataset": HF_DATASET,
    "split": SPLIT,
    "language": LANGUAGE,
    "template": CROP_PRESET,
    "lines": LINES_SPEC,
    "run_id": RUN_ID,
    "n_images": N_IMAGES,
    "rows_per_shard": ROWS_PER_SHARD,
    "seed": SEED,
}
(OUT_DIR / f"run_config-{RUN_ID}.json").write_text(json.dumps(RUN_INFO, indent=2), encoding="utf-8")
print(f"Run ID:    {RUN_ID}")
print(f"Output:    {OUT_DIR}")
print(f"Split:     {SPLIT} (Language: {LANGUAGE})")
print(f"Lines:     {LINES_SPEC}")

def run_script(args: list[str]) -> int:
    import subprocess
    cmd = [sys.executable, str(SCRIPT), *args]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}: {' '.join(shlex.quote(c) for c in cmd)}")
    return res.returncode""")

    # Cell 8: Preview Sample Crops
    add_cell("code", """# ---------------------------------------------------------------------------
# 8. Preview Sample Crops
# ---------------------------------------------------------------------------
preview_png = OUT_DIR / f"preview-{RUN_ID}.png"
run_script([
    "preview", *COMMON,
    "--preview", str(max(1, PREVIEW_N)),
    "--preview-out", str(preview_png),
])

from IPython.display import Image, display
if preview_png.is_file():
    display(Image(filename=str(preview_png)))""")

    # Cell 9: Generate Shards
    add_cell("code", """# ---------------------------------------------------------------------------
# 9. Generate Parquet Shards
# ---------------------------------------------------------------------------
cmd = [
    "generate", *COMMON,
    "--n-images", str(N_IMAGES),
    "--rows-per-shard", str(ROWS_PER_SHARD),
    "--workers", str(max(1, WORKERS)),
    "--start-method", "fork" if sys.platform.startswith("linux") else "spawn",
]
if RESUME:
    cmd.append("--resume")
run_script(cmd)

import pyarrow.parquet as pq

shards = sorted((OUT_DIR / "data").glob(f"{SPLIT}-append-{RUN_ID}-*.parquet"))
total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in shards)
total_size_mb = sum(path.stat().st_size for path in shards) / 1e6
print(f"\\nGenerated {len(shards)} shard(s): {total_rows:,} rows, {total_size_mb:.1f} MB total.")
for p in shards:
    print(f"  - {p.name}: {pq.ParquetFile(p).metadata.num_rows:,} rows ({p.stat().st_size / 1e6:.1f} MB)")""")

    # Cell 10: Validate Local Shards
    add_cell("code", """# ---------------------------------------------------------------------------
# 10. Validate Local Shards (Schema, JPEG Magic, Labels, Language)
# ---------------------------------------------------------------------------
run_script(["validate", "--out-dir", str(OUT_DIR), "--split", SPLIT, "--run-id", RUN_ID])""")

    # Cell 11: Upload to Hub
    add_cell("code", """# ---------------------------------------------------------------------------
# 11. Upload Only New Shards to Hugging Face
# ---------------------------------------------------------------------------
ROWS_TO_ADD = 0
NEW_SHARDS = []

if not PUSH_TO_HUB:
    print("PUSH_TO_HUB is False. Shards were NOT uploaded.")
    print(f"Local shards are stored at: {OUT_DIR / 'data'}")
else:
    if not HF_TOKEN:
        raise RuntimeError("PUSH_TO_HUB is True but HF_TOKEN is missing. Provide HF_TOKEN to upload.")
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)

    # Ensure dataset repository exists (created as private by default if new)
    api.create_repo(repo_id=HF_DATASET, repo_type="dataset", private=True, exist_ok=True)

    remote_files = set(api.list_repo_files(HF_DATASET, repo_type="dataset"))
    NEW_SHARDS = [p for p in shards if f"data/{p.name}" not in remote_files]
    ROWS_TO_ADD = sum(pq.ParquetFile(p).metadata.num_rows for p in NEW_SHARDS)
    print(f"New shards to upload: {len(NEW_SHARDS)} ({ROWS_TO_ADD:,} rows); already on Hub: {len(shards) - len(NEW_SHARDS)}")

    if NEW_SHARDS:
        run_script([
            "upload",
            "--out-dir", str(OUT_DIR),
            "--split", SPLIT,
            "--run-id", RUN_ID,
            "--repo-id", HF_DATASET,
            "--token", HF_TOKEN,
        ])
        print(f"Uploaded {len(NEW_SHARDS)} shard(s) to https://huggingface.co/datasets/{HF_DATASET}")
    else:
        print("All shards already exist in remote repository. Nothing to upload.")""")

    # Cell 12: Update Card
    add_cell("code", """# ---------------------------------------------------------------------------
# 12. Update Dataset Card (README.md) Split Counts
# ---------------------------------------------------------------------------
if PUSH_TO_HUB and UPDATE_CARD and ROWS_TO_ADD:
    card_cmd = [
        "update-card",
        "--repo-id", HF_DATASET,
        "--add", f"{SPLIT}={ROWS_TO_ADD}",
        "--token", HF_TOKEN,
    ]
    if DRY_RUN_CARD:
        card_cmd.append("--dry-run")
    try:
        run_script(card_cmd)
    except Exception as e:
        print(f"Dataset card update notice: {e}")
elif PUSH_TO_HUB and UPDATE_CARD:
    print("No new rows uploaded; dataset card not updated.")
else:
    print("Dataset card update skipped.")""")

    # Cell 13: Spot Check
    add_cell("code", """# ---------------------------------------------------------------------------
# 13. Local / Remote Shard Spot Check
# ---------------------------------------------------------------------------
if shards:
    import io
    from PIL import Image as PILImage
    table = pq.read_table(shards[0])
    sample_row = table.to_pylist()[0]
    print("Sample Shard Record:")
    print(f"  image_id:      {sample_row['image_id']}")
    print(f"  transcription: {sample_row['transcription']!r}")
    print(f"  language:      {sample_row['language']}")
    print(f"  vertical:      {sample_row['vertical']}")
    print(f"  source:        {sample_row['source']}")

    img_bytes = sample_row["image"]["bytes"]
    pil_img = PILImage.open(io.BytesIO(img_bytes))
    display(pil_img)""")

    # Cell 14: Notes Markdown
    add_cell("markdown", """## Notes & Configuration Guide

### Language & Split Mapping
| Target Language | `LANGUAGE` | Default `SPLIT` | Typical Preset | Notes |
|---|---|---|---|---|
| Chinese | `zh` | `chinese` | `dialogue` / `handwritten` | Mixed vertical/horizontal, Hanzi dictionaries & punctuation |
| Korean | `ko` | `korean` | `dialogue` / `handwritten` | Hangul text, syllable weighting, space preservation |
| English | `en` | `english` | `dialogue` / `sfx` | Latin comic fonts, uppercase comic book lettering |
| Japanese | `ja` | `japanese` | `dialogue_bubble` / `sfx` | Vertical dominant, furigana ruby, tcy |

### Font Streaming & Local `/mnt/data/jp_fonts`
- When running locally, `/mnt/data/jp_fonts` is detected first.
- In Kaggle / Colab, setting `FONTS_FETCH_MODE = "selected_lang_only"` downloads only the ~50-180 MB folder corresponding to the chosen language from `JustANormalTinkerer/manga-ocr-multilingual-fonts`, avoiding unnecessary multi-GB downloads.

### Background Streaming & AnimeText
- When running locally, `/mnt/data/manga/AnimeText_background` is detected first.
- In Kaggle / Colab, `deepghs/AnimeText` is streamed live without downloading the full dataset. Clean unmasked illustration rectangles are extracted on the fly.
""")

    out_file = Path("kaggle/multilingual_synth.ipynb")
    out_file.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Successfully generated notebook: {out_file} ({len(nb['cells'])} cells)")

if __name__ == "__main__":
    create_notebook()

