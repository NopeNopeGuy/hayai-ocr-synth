"""Scan new fonts from /mnt/data/jp_fonts/More_JP, append to assets/fonts.csv and assets/font_profiles.json."""

import json
import sys
from pathlib import Path

# Add repo root to sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
from tqdm import tqdm

from manga_ocr_dev.env import ASSETS_PATH, FONTS_ROOT
from manga_ocr_dev.synthetic_data_generator.scan_fonts import process
from manga_ocr_dev.synthetic_data_generator.scan_font_profiles import profile_single_font

FONTS_ROOT = Path("/mnt/data/jp_fonts")
MORE_JP_DIR = FONTS_ROOT / "More_JP"
CSV_PATH = ASSETS_PATH / "fonts.csv"
PROFILES_PATH = ASSETS_PATH / "font_profiles.json"
CACHE_PATH = ASSETS_PATH / "fonts.cache.pkl"

# 1. Discover all .ttf and .otf font files
font_files = sorted(list(MORE_JP_DIR.rglob("*.ttf")) + list(MORE_JP_DIR.rglob("*.otf")) + list(MORE_JP_DIR.rglob("*.ttc")))
print(f"Discovered {len(font_files)} font files in {MORE_JP_DIR}")

# 2. Load existing fonts.csv
existing_df = pd.read_csv(CSV_PATH) if CSV_PATH.is_file() else pd.DataFrame(columns=["font_path", "supported_chars", "num_chars", "label"])
existing_paths = set(existing_df["font_path"].str.replace("\\", "/").tolist())
print(f"Existing rows in fonts.csv: {len(existing_df)}")

# 3. Load existing font_profiles.json
if PROFILES_PATH.is_file():
    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        profiles = json.load(f)
else:
    profiles = {}
print(f"Existing profiles in font_profiles.json: {len(profiles)}")

new_rows = []
new_profiles = {}

for f in tqdm(font_files, desc="Scanning More_JP fonts"):
    rel = f.relative_to(FONTS_ROOT).as_posix()
    if rel in existing_paths:
        continue

    # Extract supported chars
    chars = process(f)
    if not chars:
        print(f"Warning: No supported chars for {rel}")
        continue

    # Profile font
    prof = profile_single_font(rel)
    label = prof.get("label", "manga")
    if prof.get("is_omit", False):
        label = "omit"

    new_rows.append({
        "font_path": rel,
        "supported_chars": chars,
        "num_chars": len(chars),
        "label": label,
    })
    new_profiles[rel] = prof

print(f"Successfully processed {len(new_rows)} new fonts!")

# 4. Append to fonts.csv
if new_rows:
    new_df = pd.DataFrame(new_rows)
    updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    updated_df.to_csv(CSV_PATH, index=False)
    print(f"Updated fonts.csv: {len(updated_df)} total rows (+{len(new_rows)} new).")

# 5. Update font_profiles.json
if new_profiles:
    profiles.update(new_profiles)
    with open(PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    print(f"Updated font_profiles.json: {len(profiles)} total entries (+{len(new_profiles)} new).")

# 6. Delete stale pickle cache
if CACHE_PATH.is_file():
    CACHE_PATH.unlink()
    print("Deleted stale fonts.cache.pkl")

print("Done scanning and registering new JP fonts!")
