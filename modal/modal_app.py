import json
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "hayai-manga-ocr-synth"
REPO_URL = "https://github.com/NopeNopeGuy/hayai-ocr-synth.git"
CACHE_MOUNT = Path("/cache")
REPO_DIR = Path("/opt/hayai-ocr-synth")

app = modal.App(APP_NAME)
cache_vol = modal.Volume.from_name("hayai-synth-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("git", "git-lfs", "libraqm-dev")
    .pip_install(
        "datasets>2.7.0",
        "huggingface_hub",
        "pyarrow",
        "fire",
        "fonttools",
        "budoux",
        "scipy",
        "opencv-python-headless",
        "tqdm",
        "pillow",
        "pandas",
        "albumentations",
    )
    .run_commands("git lfs install")
)

LANG_FONT_FOLDERS = {
    "zh": ["Chinese_Additions", "精简包/简体", "精简包/繁体", "精简包/简繁"],
    "ko": ["Korean"],
    "en": ["Latin_Comic"],
    "ja": ["精简包/日文", "Chinese_Additions"],
}


def _run_script(script_path: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script_path), *args]
    print(f"Running: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {res.returncode}: {' '.join(shlex.quote(c) for c in cmd)}"
        )


def _ensure_repo() -> Path:
    if not (REPO_DIR / "manga_ocr_dev").is_dir():
        print(f"Cloning generator repository into {REPO_DIR}...")
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
    else:
        print(f"Updating existing repository at {REPO_DIR}...")
        subprocess.run(["git", "-C", str(REPO_DIR), "checkout", "--", "."], check=False)
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)

    subprocess.run(["git", "-C", str(REPO_DIR), "lfs", "pull"], check=False)
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    return REPO_DIR


def _apply_perf_patches(repo_dir: Path) -> None:
    """Fix thrashing LRU caches and float64 blending in the ephemeral container clone."""
    patches = {
        repo_dir / "manga_ocr_dev" / "synthetic_data_generator" / "renderer.py": [
            ("_FONT_CACHE_MAX = 16", "_FONT_CACHE_MAX = 256"),
            ("_LARGE_FONT_BYTES = 25 * (1 << 20)", "_LARGE_FONT_BYTES = 256 * (1 << 20)"),
            ("_BG_IMAGE_CACHE_MAX = 128", "_BG_IMAGE_CACHE_MAX = 2048"),
            (
                "        if _RENDER_COUNT % 250 == 0:\n"
                "            with _FONT_CACHE_LOCK:\n"
                "                _FONT_CACHE.clear()\n"
                "            _trim_native_memory()",
                "        if _RENDER_COUNT % 2000 == 0:\n"
                "            _trim_native_memory()",
            ),
            (
                "def blend(img, background):\n"
                "    alpha = (img[:, :, 3] / 255)[:, :, np.newaxis]\n"
                "    img = img[:, :, :3]\n"
                "    img = (background * (1 - alpha) + img * alpha).astype(np.uint8)\n"
                "    return img",
                "def blend(img, background):\n"
                "    alpha = img[:, :, 3:4].astype(np.float32) * np.float32(1.0 / 255.0)\n"
                "    img = img[:, :, :3].astype(np.float32)\n"
                "    out = background.astype(np.float32) * (np.float32(1.0) - alpha) + img * alpha\n"
                "    return np.clip(out, 0, 255).astype(np.uint8)",
            ),
        ],
        repo_dir / "manga_ocr_dev" / "synthetic_data_generator" / "utils.py": [
            ("def __init__(self, paths, bits, extra, lru_size=16):", "def __init__(self, paths, bits, extra, lru_size=128):"),
            ("@lru_cache(maxsize=8)\ndef _probe_for(font_path):", "@lru_cache(maxsize=128)\ndef _probe_for(font_path):"),
        ],
        repo_dir / "manga_ocr_dev" / "synthetic_data_generator" / "generator.py": [
            ("while len(self._pool_cache) > 16:", "while len(self._pool_cache) > 128:"),
        ],
    }
    for path, replacements in patches.items():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        changed = False
        for old, new in replacements:
            if old in text:
                text = text.replace(old, new, 1)
                changed = True
        if changed:
            path.write_text(text, encoding="utf-8")
            print(f"Applied perf patch to {path.relative_to(repo_dir)}")


def _resolve_fonts(
    fonts_hf_dataset: str,
    fonts_fetch_mode: str,
    fonts_root_override: str,
    language: str,
    hf_token: str | None,
) -> Path:
    if fonts_root_override and Path(fonts_root_override).is_dir():
        print(f"Using fonts_root_override: {fonts_root_override}")
        return Path(fonts_root_override).resolve()

    if fonts_fetch_mode == "local_only":
        raise RuntimeError("fonts_fetch_mode is local_only but no local font directory was provided.")

    dest = CACHE_MOUNT / "fonts"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Fetching/verifying fonts from '{fonts_hf_dataset}' into {dest}...")

    patterns = None
    if fonts_fetch_mode == "selected_lang_only":
        patterns = [f"{folder}/**" for folder in LANG_FONT_FOLDERS.get(language, [])]

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=fonts_hf_dataset,
        repo_type="dataset",
        token=hf_token,
        local_dir=str(dest),
        allow_patterns=patterns,
        max_workers=8,
    )
    cache_vol.commit()
    return dest.resolve()


def _stage_fonts_to_tmp(cache_fonts_root: Path, language: str, fonts_fetch_mode: str) -> Path:
    """Copy fonts from /cache (network volume) to /tmp (local NVMe SSD)."""
    tmp_fonts = Path("/tmp/fonts")
    marker = tmp_fonts / f".staged_{language}_{fonts_fetch_mode}"
    if marker.is_file():
        print(f"Reusing already-staged local fonts at {tmp_fonts}")
        return tmp_fonts

    tmp_fonts.mkdir(parents=True, exist_ok=True)
    folders = LANG_FONT_FOLDERS.get(language, []) if fonts_fetch_mode == "selected_lang_only" else []
    copied = 0
    if folders:
        for rel in folders:
            src = cache_fonts_root / rel
            dst = tmp_fonts / rel
            if src.is_dir() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                subprocess.check_call(["cp", "-a", str(src), str(dst)])
                copied += 1
    if not copied:
        subprocess.check_call(["cp", "-an", f"{cache_fonts_root}/.", str(tmp_fonts)])

    marker.write_text("ok", encoding="utf-8")
    print(f"Staged fonts to local SSD: {tmp_fonts}")
    return tmp_fonts


def _resolve_backgrounds(
    backgrounds_mode: str,
    backgrounds_dir_override: str,
    stream_bg_images: int,
    seed: int,
) -> Path:
    if backgrounds_dir_override and Path(backgrounds_dir_override).is_dir():
        print(f"Using backgrounds_dir_override: {backgrounds_dir_override}")
        return Path(backgrounds_dir_override).resolve()

    if backgrounds_mode == "local_only":
        raise RuntimeError("backgrounds_mode is local_only but no background directory was provided.")

    dest = CACHE_MOUNT / "animetext_bg"
    dest.mkdir(parents=True, exist_ok=True)
    manifest_file = dest / "manifest.csv"

    if manifest_file.is_file():
        import pandas as pd

        df = pd.read_csv(manifest_file)
        if len(df) >= 200:
            print(f"Reusing cached streamed backgrounds at {dest} ({len(df)} crops).")
            return dest.resolve()

    print(
        f"Streaming {stream_bg_images} images from deepghs/AnimeText to generate background crops..."
    )
    from manga_ocr_dev.data.generate_backgrounds_animetext import (
        generate_backgrounds_animetext,
    )

    generate_backgrounds_animetext(
        n_images=stream_bg_images,
        crops_per_image=35,
        min_size=40,
        out_dir=str(dest),
        split="train",
        seed=seed,
        num_workers=32,
    )
    cache_vol.commit()
    print(f"Background extraction complete: {dest}")
    return dest.resolve()


def _stage_backgrounds_to_tmp(cache_bg_dir: Path) -> Path:
    """Stage backgrounds to /tmp fast via tar cache or 32-thread parallel copy."""
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    tmp_bg = Path("/tmp/animetext_bg")
    marker = tmp_bg / ".staged"
    if marker.is_file() and (tmp_bg / "manifest.csv").is_file():
        print(f"Reusing already-staged local backgrounds at {tmp_bg}")
        return tmp_bg

    tmp_bg.mkdir(parents=True, exist_ok=True)
    cache_tar = CACHE_MOUNT / "animetext_bg.tar"

    if cache_tar.is_file():
        print(f"Extracting {cache_tar} -> {tmp_bg}...")
        subprocess.check_call(["tar", "-xf", str(cache_tar), "-C", str(tmp_bg)])
        marker.write_text("ok", encoding="utf-8")
        print(f"Staged backgrounds to local SSD: {tmp_bg}")
        return tmp_bg

    # First run: parallel copy all files from /cache/animetext_bg -> /tmp/animetext_bg
    src_files = [p for p in cache_bg_dir.rglob("*") if p.is_file()]
    print(f"Staging {len(src_files)} background files to {tmp_bg} (32 threads)...")

    def _copy_one(src: Path) -> None:
        rel = src.relative_to(cache_bg_dir)
        dst = tmp_bg / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    with ThreadPoolExecutor(max_workers=32) as ex:
        list(tqdm(ex.map(_copy_one, src_files), total=len(src_files), unit="file"))

    manifest = tmp_bg / "manifest.csv"
    if manifest.is_file():
        raw = manifest.read_text(encoding="utf-8")
        cache_prefix = str(cache_bg_dir).rstrip("/") + "/"
        tmp_prefix = str(tmp_bg).rstrip("/") + "/"
        if cache_prefix in raw:
            manifest.write_text(raw.replace(cache_prefix, tmp_prefix), encoding="utf-8")

    marker.write_text("ok", encoding="utf-8")

    # Pack once so all future runs extract in ~1-2 seconds
    tmp_tar = Path("/tmp/animetext_bg.tar")
    print(f"Packing {cache_tar} for instant future startups...")
    subprocess.check_call(["tar", "-cf", str(tmp_tar), "-C", str(tmp_bg), "."])
    shutil.copyfile(tmp_tar, cache_tar)
    tmp_tar.unlink(missing_ok=True)
    cache_vol.commit()

    print(f"Staged backgrounds to local SSD: {tmp_bg}")
    return tmp_bg


def _stage_lines_if_streamable(lines_spec: str, n_images: int, seed: int = 42) -> str:
    """Cache Wikipedia / OpenSubtitles streams to /cache/lines and /tmp."""
    streamers = {
        "hf:ko": ("ko_opensub_wiki_v2", "stream_korean_lines", "ko_opensubtitles_wiki", {}),
        "hf:zh": ("zh_opensub_wiki_v3", "stream_chinese_lines", "zh_opensubtitles_wiki", {}),
        "hf:en": ("en_opensub_wiki_v2", "stream_english_lines", "en_opensubtitles_wiki", {}),
        "hf:wikipedia:ko": ("ko_wiki_v2", "stream_wikipedia_lines", "wikipedia_ko", {"lang": "ko"}),
        "hf:wikipedia:zh": ("zh_wiki_v3", "stream_wikipedia_lines", "wikipedia_zh", {"lang": "zh"}),
        "hf:wikipedia:en": ("en_wiki_v2", "stream_wikipedia_lines", "wikipedia_en", {"lang": "en"}),
        "hf:opensubtitles:ko": ("ko_opensub_v2", "stream_opensubtitles_lines", "opensubtitles_ko", {"lang": "ko"}),
        "hf:opensubtitles:zh": ("zh_opensub_v3", "stream_opensubtitles_lines", "opensubtitles_zh", {"lang": "zh"}),
        "hf:opensubtitles:en": ("en_opensub_v2", "stream_opensubtitles_lines", "opensubtitles_en", {"lang": "en"}),
    }
    if lines_spec not in streamers:
        return lines_spec

    import pyarrow as pa
    import pyarrow.parquet as pq

    cache_name, fn_name, source_name, extra_kwargs = streamers[lines_spec]
    cache_dir = CACHE_MOUNT / "lines"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_name}.parquet"
    tmp_file = Path(f"/tmp/{cache_name}.parquet")

    target_rows = 1_000_000 if n_images < 0 else min(1_000_000, max(int(n_images * 1.5), 10_000))
    need_fetch = True
    if cache_file.is_file():
        existing_rows = pq.ParquetFile(cache_file).metadata.num_rows
        if existing_rows >= min(target_rows, max(n_images, 1)):
            print(f"Reusing cached lines file {cache_file} ({existing_rows:,} rows)")
            need_fetch = False

    if need_fetch:
        print(f"Caching up to {target_rows:,} lines for {lines_spec} -> {cache_file}...")
        mod = __import__("tools.fetch_multilingual_lines", fromlist=[fn_name])
        stream_fn = getattr(mod, fn_name)
        lines, sources, ids = [], [], []
        for idx, line in enumerate(stream_fn(limit=target_rows, seed=seed, **extra_kwargs)):
            lines.append(line)
            sources.append(source_name)
            ids.append(f"{cache_name}_{idx}")
        table = pa.table({"line": lines, "source": sources, "id": ids})
        tmp_write = cache_file.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp_write)
        os.replace(tmp_write, cache_file)
        cache_vol.commit()
        print(f"Cached {len(lines):,} lines to {cache_file}")

    shutil.copyfile(cache_file, tmp_file)
    return str(tmp_file)


@app.function(
    image=image,
    volumes={str(CACHE_MOUNT): cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    cpu=16.0,
    memory=16384,
    timeout=86400,
)
def run_notebook_pipeline(
    language: str = "ko",
    split: str = "auto",
    hf_dataset: str = "JustANormalTinkerer/hayai-synth-multilingual",
    crop_preset: str = "mixed",
    n_images: int = 350_000,
    rows_per_shard: int = 20_000,
    workers: int = 8,
    seed: int = 0,
    jpeg_quality: int = 95,
    out_dir: str = "",
    run_id: str = "",
    resume: bool = True,
    override_previous: bool = False,
    override_scope: str = "split",
    lines_mode: str = "auto",
    lines_path: str = "",
    text_column: str = "",
    source_label: str = "",
    min_line_len: int = 0,
    fonts_hf_dataset: str = "JustANormalTinkerer/manga-ocr-multilingual-fonts",
    fonts_fetch_mode: str = "selected_lang_only",
    fonts_root_override: str = "",
    backgrounds_mode: str = "auto",
    stream_bg_images: int = 3_000,
    backgrounds_dir_override: str = "",
    vertical: str = "auto",
    bubble: str = "auto",
    bubble_prob: float = 0.80,
    tight: bool = False,
    irregular: str = "auto",
    color: str = "auto",
    color_mode: str = "",
    text_color: str = "",
    font_size_min: int = 0,
    font_size_max: int = 0,
    stroke_size: int = -1,
    slant_deg: float = -999.0,
    hard_ink: str = "auto",
    preset_mix: str = "dialogue_bubble=0.65,dialogue_flat=0.15,handwritten=0.12,sfx=0.08",
    override_json: str = "",
    preview_n: int = 8,
    skip_preview: bool = False,
    push_to_hub: bool = True,
    update_card: bool = True,
    dry_run_card: bool = False,
    perf_patches: bool = True,
) -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, login

    # Prevent OpenBLAS / OpenMP / MKL inside forked workers from oversubscribing cores
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    cache_vol.reload()

    if split == "auto":
        default_splits = {"zh": "chinese", "ko": "korean", "en": "english", "ja": "japanese"}
        split = default_splits.get(language, "multilingual")
        if crop_preset == "sfx" or lines_mode == "sfx":
            split = f"{split}_sfx"
        elif crop_preset == "handwritten":
            split = f"{split}_handwritten"

    print(f"Configuration: language={language}, split={split}, preset={crop_preset}, lines={lines_mode}, workers={workers}")

    repo_dir = _ensure_repo()
    if perf_patches:
        _apply_perf_patches(repo_dir)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=True)
        print("Logged in to Hugging Face successfully.")
    else:
        print("WARNING: HF_TOKEN not found.")

    # Resolve & stage fonts + backgrounds onto local /tmp SSD
    cache_fonts_root = _resolve_fonts(
        fonts_hf_dataset=fonts_hf_dataset,
        fonts_fetch_mode=fonts_fetch_mode,
        fonts_root_override=fonts_root_override,
        language=language,
        hf_token=hf_token,
    )
    resolved_fonts_root = _stage_fonts_to_tmp(cache_fonts_root, language, fonts_fetch_mode)
    os.environ["FONTS_ROOT"] = str(resolved_fonts_root)

    cache_bg_dir = _resolve_backgrounds(
        backgrounds_mode=backgrounds_mode,
        backgrounds_dir_override=backgrounds_dir_override,
        stream_bg_images=stream_bg_images,
        seed=seed,
    )
    resolved_bg_dir = _stage_backgrounds_to_tmp(cache_bg_dir)
    os.environ["BACKGROUND_ANIMETEXT_DIR"] = str(resolved_bg_dir)

    # Sanity check generator
    from manga_ocr_dev.synthetic_data_generator.generator import SyntheticDataGenerator
    from manga_ocr_dev.synthetic_data_generator.utils import get_background_df

    bg_df = get_background_df(resolved_bg_dir)
    gen = SyntheticDataGenerator(background_df=bg_df)
    sample_font = gen.get_random_font(lang=language)
    print(f"Sampled font for '{language}': {Path(sample_font).name}")
    img_d, text_d, _ = gen.process(text=None, sfx=False, lang=language)
    print(f"Sample {language} dialogue: text={text_d[:30]!r}, shape={img_d.shape}")

    if override_previous and override_scope not in ("split", "run"):
        raise ValueError(f"override_scope must be 'split' or 'run', got {override_scope!r}")
    if override_previous and override_scope == "run" and not run_id:
        raise ValueError("override_scope='run' requires an explicit --run-id to override.")

    if not run_id:
        run_id = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"

    out_path = Path(out_dir) if out_dir else (CACHE_MOUNT / "synth_out")
    out_path.mkdir(parents=True, exist_ok=True)
    data_dir = out_path / "data"
    if override_previous and data_dir.is_dir():
        if override_scope == "split":
            removed = 0
            for old_shard in data_dir.glob(f"{split}-append-*.parquet"):
                old_shard.unlink(missing_ok=True)
                removed += 1
            print(f"override_previous (split): removed {removed} local shard(s) for split={split!r}.")
        else:
            removed = 0
            for old_shard in data_dir.glob(f"{split}-append-{run_id}-*.parquet"):
                old_shard.unlink(missing_ok=True)
                removed += 1
            (out_path / f"manifest-{run_id}.json").unlink(missing_ok=True)
            (out_path / f"run_config-{run_id}.json").unlink(missing_ok=True)
            print(f"override_previous (run): removed {removed} local shard(s) for run_id={run_id!r}.")
        cache_vol.commit()
        resume = False

    script = repo_dir / "tools" / "append_hayai_synth.py"

    if lines_mode == "auto":
        stream_map = {
            "zh": "hf:zh",
            "en": "hf:en",
            "ko": "hf:ko",
            "ja": "hf:dialogue",
        }
        lines_spec = stream_map.get(language, "random")
    elif lines_mode == "hf_wikipedia":
        lines_spec = f"hf:wikipedia:{language}"
    elif lines_mode == "hf_opensubtitles":
        lines_spec = f"hf:opensubtitles:{language}"
    elif lines_mode == "hf_dialogue":
        lines_spec = "hf:dialogue"
    elif lines_mode == "hf_novel":
        lines_spec = "hf:novel"
    elif lines_mode in ("random", "sfx"):
        lines_spec = lines_mode
    elif lines_mode == "local":
        if not lines_path:
            raise ValueError("lines_mode is 'local' but lines_path was not provided.")
        lines_spec = lines_path
    else:
        lines_spec = lines_mode

    lines_spec = _stage_lines_if_streamable(lines_spec, n_images, seed=seed)

    resolved_vertical = vertical
    if resolved_vertical == "auto" and language in ("en", "ko"):
        resolved_vertical = "false"

    resolved_min_len = min_line_len if min_line_len > 0 else {"en": 22, "zh": 6, "ko": 10}.get(language, 2)

    common = [
        "--split", split,
        "--language", language,
        "--preset", crop_preset,
        "--lines", lines_spec,
        "--min-line-len", str(resolved_min_len),
        "--out-dir", str(out_path),
        "--run-id", run_id,
        "--backgrounds", str(resolved_bg_dir),
        "--fonts-root", str(resolved_fonts_root),
        "--vertical", resolved_vertical,
        "--bubble", bubble,
        "--color", color,
        "--irregular", irregular,
        "--hard-ink", hard_ink,
        "--jpeg-quality", str(jpeg_quality),
        "--seed", str(seed),
    ]
    if bubble_prob >= 0:
        common += ["--bubble-prob", str(bubble_prob)]
    if tight:
        common += ["--tight"]
    if color_mode:
        common += ["--color-mode", color_mode]
    if text_color:
        common += ["--text-color", text_color]
    if font_size_min > 0:
        common += ["--font-size-min", str(font_size_min)]
    if font_size_max > 0:
        common += ["--font-size-max", str(font_size_max)]
    if stroke_size >= 0:
        common += ["--stroke-size", str(stroke_size)]
    if slant_deg > -900:
        common += ["--slant-deg", str(slant_deg)]
    if preset_mix and crop_preset == "mixed":
        common += ["--preset-mix", preset_mix]
    if override_json:
        common += ["--override-json", override_json]
    if text_column:
        common += ["--text-column", text_column]
    if source_label:
        common += ["--source-label", source_label]

    run_info = {
        "repo": str(repo_dir),
        "dataset": hf_dataset,
        "split": split,
        "language": language,
        "template": crop_preset,
        "lines": lines_spec,
        "run_id": run_id,
        "n_images": n_images,
        "rows_per_shard": rows_per_shard,
        "seed": seed,
    }
    (out_path / f"run_config-{run_id}.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )

    if not skip_preview:
        preview_png = out_path / f"preview-{run_id}.png"
        _run_script(
            script,
            ["preview", *common, "--preview", str(max(1, preview_n)), "--preview-out", str(preview_png)],
        )

    gen_cmd = [
        "generate",
        *common,
        "--n-images", str(n_images),
        "--rows-per-shard", str(rows_per_shard),
        "--workers", str(max(1, workers)),
        "--start-method", "fork" if sys.platform.startswith("linux") else "spawn",
    ]
    if resume:
        gen_cmd.append("--resume")
    _run_script(script, gen_cmd)
    cache_vol.commit()

    shards = sorted((out_path / "data").glob(f"{split}-append-{run_id}-*.parquet"))
    total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in shards)
    total_size_mb = sum(p.stat().st_size for p in shards) / 1e6
    print(f"Generated {len(shards)} shard(s): {total_rows:,} rows, {total_size_mb:.1f} MB total.")

    _run_script(
        script,
        ["validate", "--out-dir", str(out_path), "--split", split, "--run-id", run_id],
    )

    rows_to_add = 0
    delete_pattern = None
    if not push_to_hub:
        print(f"push_to_hub is False. Local shards kept at {out_path / 'data'}")
    else:
        if not hf_token:
            raise RuntimeError("push_to_hub is True but HF_TOKEN is missing.")
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=hf_dataset, repo_type="dataset", private=True, exist_ok=True)
        remote_files = set(api.list_repo_files(hf_dataset, repo_type="dataset"))
        if override_previous:
            delete_pattern = (
                f"{split}-append-*.parquet" if override_scope == "split" else f"{split}-append-{run_id}-*.parquet"
            )
            new_shards = shards
            rows_to_add = total_rows
            print(f"Overriding remote shards matching data/{delete_pattern} with {len(new_shards)} shard(s) ({rows_to_add:,} rows)")
        else:
            new_shards = [p for p in shards if f"data/{p.name}" not in remote_files]
            rows_to_add = sum(pq.ParquetFile(p).metadata.num_rows for p in new_shards)
            print(f"New shards to upload: {len(new_shards)} ({rows_to_add:,} rows)")

        if new_shards:
            upload_cmd = [
                "upload",
                "--out-dir", str(out_path),
                "--split", split,
                "--run-id", run_id,
                "--repo-id", hf_dataset,
                "--token", hf_token,
            ]
            if delete_pattern:
                upload_cmd += ["--delete-patterns", delete_pattern]
            _run_script(script, upload_cmd)

    if push_to_hub and update_card and (rows_to_add or (override_previous and override_scope == "split")):
        card_flag = "--set" if (override_previous and override_scope == "split") else "--add"
        card_val = total_rows if (override_previous and override_scope == "split") else rows_to_add
        card_cmd = [
            "update-card",
            "--repo-id", hf_dataset,
            card_flag, f"{split}={card_val}",
            "--token", hf_token,
        ]
        if dry_run_card:
            card_cmd.append("--dry-run")
        try:
            _run_script(script, card_cmd)
        except Exception as e:
            print(f"Dataset card update notice: {e}")

    if shards:
        table = pq.read_table(shards[0])
        sample_row = table.to_pylist()[0]
        print("Sample Shard Record:")
        print(f"  image_id:      {sample_row['image_id']}")
        print(f"  transcription: {sample_row['transcription']!r}")
        print(f"  language:      {sample_row['language']}")
        print(f"  vertical:      {sample_row['vertical']}")
        print(f"  source:        {sample_row['source']}")
        print(f"  image bytes:   {len(sample_row['image']['bytes'])} bytes")

    return {
        "run_id": run_id,
        "split": split,
        "language": language,
        "shards_generated": len(shards),
        "total_rows": total_rows,
        "rows_uploaded": rows_to_add,
        "total_size_mb": round(total_size_mb, 2),
    }


@app.local_entrypoint()
def main(
    language: str = "ko",
    split: str = "auto",
    hf_dataset: str = "JustANormalTinkerer/hayai-synth-multilingual",
    crop_preset: str = "mixed",
    n_images: int = 350_000,
    rows_per_shard: int = 20_000,
    workers: int = 8,
    seed: int = 0,
    lines_mode: str = "auto",
    fonts_fetch_mode: str = "selected_lang_only",
    stream_bg_images: int = 3_000,
    run_id: str = "",
    resume: bool = True,
    override_previous: bool = False,
    override_scope: str = "split",
    min_line_len: int = 0,
    vertical: str = "auto",
    preset_mix: str = "dialogue_bubble=0.65,dialogue_flat=0.15,handwritten=0.12,sfx=0.08",
    push_to_hub: bool = True,
    update_card: bool = True,
    skip_preview: bool = False,
    perf_patches: bool = True,
):
    result = run_notebook_pipeline.remote(
        language=language,
        split=split,
        hf_dataset=hf_dataset,
        crop_preset=crop_preset,
        n_images=n_images,
        rows_per_shard=rows_per_shard,
        workers=workers,
        seed=seed,
        lines_mode=lines_mode,
        fonts_fetch_mode=fonts_fetch_mode,
        stream_bg_images=stream_bg_images,
        run_id=run_id,
        resume=resume,
        override_previous=override_previous,
        override_scope=override_scope,
        min_line_len=min_line_len,
        vertical=vertical,
        preset_mix=preset_mix,
        push_to_hub=push_to_hub,
        update_card=update_card,
        skip_preview=skip_preview,
        perf_patches=perf_patches,
    )
    print("Pipeline finished:", json.dumps(result, indent=2))
