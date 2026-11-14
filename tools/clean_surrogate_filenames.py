"""Decode Shift-JIS surrogate filenames in /mnt/data/jp_fonts/More_JP into clean UTF-8."""

import os
from pathlib import Path

ROOT = Path("/mnt/data/jp_fonts/More_JP")

def fix_name(s: str) -> str:
    try:
        s.encode("utf-8")
        return s
    except UnicodeEncodeError:
        try:
            raw = s.encode("utf-8", "surrogateescape")
            return raw.decode("cp932")
        except Exception:
            # fallback: strip surrogates or safe replacement
            return s.encode("ascii", "replace").decode("ascii")

# Traverse bottom-up so children are renamed before parents
for root, dirs, files in os.walk(ROOT, topdown=False):
    for f in files:
        new_f = fix_name(f)
        if new_f != f:
            old_path = os.path.join(root, f)
            new_path = os.path.join(root, new_f)
            os.rename(old_path, new_path)
            print(f"Renamed file: {repr(f)} -> {new_f}")
    for d in dirs:
        new_d = fix_name(d)
        if new_d != d:
            old_path = os.path.join(root, d)
            new_path = os.path.join(root, new_d)
            os.rename(old_path, new_path)
            print(f"Renamed dir: {repr(d)} -> {new_d}")

# Verify no surrogates remain
surrogates = []
for p in ROOT.rglob("*"):
    try:
        str(p).encode("utf-8")
    except UnicodeEncodeError:
        surrogates.append(str(p))

if surrogates:
    print(f"Warning: {len(surrogates)} files still have surrogates!")
else:
    print("All filenames in /mnt/data/jp_fonts/More_JP are now valid UTF-8!")

