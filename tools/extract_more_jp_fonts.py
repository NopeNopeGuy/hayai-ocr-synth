"""Extract fonts from /mnt/more jp fonts into /mnt/data/jp_fonts/More_JP."""

import os
import shutil
import subprocess
from pathlib import Path

SRC_DIR = Path("/mnt/more jp fonts")
DEST_DIR = Path("/mnt/data/jp_fonts/More_JP")
DEST_DIR.mkdir(parents=True, exist_ok=True)

print(f"Source: {SRC_DIR}")
print(f"Destination: {DEST_DIR}")

# 1. Copy loose font files
for f in SRC_DIR.glob("*.*"):
    if f.suffix.lower() in {".ttf", ".otf", ".ttc", ".woff", ".woff2"}:
        target = DEST_DIR / f.name
        print(f"Copying loose font: {f.name}")
        shutil.copy2(f, target)

# 2. Extract archives using 7z
for archive in sorted(SRC_DIR.glob("*.*")):
    if archive.suffix.lower() in {".zip", ".lzh", ".rar", ".7z", ".tar", ".gz"}:
        folder_name = archive.stem
        extract_to = DEST_DIR / folder_name
        extract_to.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {archive.name} -> {folder_name}/")
        try:
            subprocess.run(
                ["7z", "x", "-y", f"-o{extract_to}", str(archive)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception as e:
            print(f"Error extracting {archive.name}: {e}")

# 3. Clean up junk (__MACOSX, .DS_Store)
for p in DEST_DIR.glob("**/__MACOSX"):
    if p.is_dir():
        shutil.rmtree(p)
for p in DEST_DIR.glob("**/.DS_Store"):
    p.unlink()

# 4. Count extracted font files
font_files = []
for p in DEST_DIR.rglob("*"):
    if p.is_file() and p.suffix.lower() in {".ttf", ".otf", ".ttc"}:
        font_files.append(p)

print(f"\nExtraction complete! Found {len(font_files)} valid font files in {DEST_DIR}")
for f in font_files[:15]:
    print(f"  - {f.relative_to(DEST_DIR)}")
if len(font_files) > 15:
    print(f"  ... and {len(font_files) - 15} more.")

