"""Configuration for the real handwritten-kana pipeline.

Everything here is deliberately a plain dataclass so it can be serialised next
to a bank and reproduced later.  Environment variables are only used to pick
sensible defaults; explicit arguments always win.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from manga_ocr_dev.env import ASSETS_PATH

# The fixed revision used for the candidate pool.  Keeping the revision in the
# bank manifest makes a rebuild byte-comparable even if the dataset moves.
ANIMETEXT_OCR_REPO = "JustANormalTinkerer/animetext-ocr"
ANIMETEXT_OCR_REVISION = "9c70c6b5609c4378d46ac0df4927c4119ebfd83d"

# Backwards/forwards compatible default.  The article used a Gemini "Sol"
# session; this implementation defaults to the cheaper Flash-Lite model and
# accepts any model that supports JSON schemas and image input.
DEFAULT_VISION_MODEL = os.environ.get("REAL_KANA_VISION_MODEL", "gemini-3.5-flash-lite")

DEFAULT_ROOT = Path(os.environ.get("REAL_KANA_ROOT", ASSETS_PATH / "handwriting")).expanduser()


@dataclass
class KanaConfig:
    """All tunables for harvesting, segmentation and use in rendering."""

    root: Path = DEFAULT_ROOT

    # ---- candidate filtering (article section 1.3, step 1) -----------------
    min_chars: int = 3
    max_chars: int = 40
    min_kana: int = 2
    min_short_side: int = 18
    min_ocr_confidence: float = 0.90
    handwriting_threshold: float = 0.95
    exclude_likely_sfx: bool = True
    # A crop with only katakana and at most this many characters is treated as
    # a likely sound effect.  Dialogue in katakana is usually longer.
    sfx_max_katakana_chars: int = 6

    # ---- local detector / recognizer (manga-ocr-charspot-small) -----------
    # ``None`` means "find the checkout next to the project root or import the
    # installed package".  Boxes from this model are already in source pixels.
    local_model_dir: str | None = os.environ.get("MANGA_OCR_CHARSPOT_SMALL_DIR")
    local_backend: str = os.environ.get("REAL_KANA_LOCAL_BACKEND", "auto")
    # Charspot scores are deliberately conservative; genuine kana can score as
    # low as ~0.3, so this is much lower than the vision-model threshold.
    local_min_confidence: float = 0.20
    # ONNX thread count for the charspot session.  ``None`` keeps the runtime
    # default; set it to 1 when running several shards side by side.
    local_threads: int | None = None

    # ---- optional Mercury handwriting classifier (OpenRouter) -------------
    mercury_model: str = os.environ.get("REAL_KANA_MERCURY_MODEL", "inception/mercury-decide:free")
    mercury_url: str = os.environ.get(
        "REAL_KANA_MERCURY_URL", "https://openrouter.ai/api/alpha/decisions"
    )
    mercury_timeout_s: float = 120.0
    mercury_retries: int = 4
    mercury_max_side: int = 768

    # ---- vision verification (final pass only) ----------------------------
    vision_model: str = DEFAULT_VISION_MODEL
    vision_provider: str = os.environ.get("REAL_KANA_VISION_PROVIDER", "gemini")
    vision_temperature: float = 0.0
    vision_max_retries: int = 5
    vision_retry_base_s: float = 2.0
    # Images are enlarged for the model only; the saved pixels are untouched.
    vision_upscale_max_side: int = 1400
    min_glyph_box_side: int = 4
    min_glyph_confidence: float = 0.5
    # Verify only crops the local model was unsure about, plus crops flagged by
    # cleaning/upscaling.  Set ``verify_all`` to audit every glyph.
    verify_confidence: float = 0.65
    verify_all: bool = False
    verify_clean_flags: tuple[str, ...] = ("border_ink", "too_much_ink", "too_little_ink")

    # ---- cleaning / upscaling / uprighting ---------------------------------
    upscale: float = 2.0
    upscale_min_side: int = 48
    max_native_enlargement: float = 4.0
    min_alpha_coverage: float = 0.02
    max_alpha_coverage: float = 0.85
    max_components: int = 8
    remove_border_dirt: bool = True

    # ---- bank matching (article section 1.3, step 7) -----------------------
    stroke_tolerance: float = 1.35
    image_stroke_ratio: float = 1.65
    small_kana_scale: float = 0.50
    max_candidates_per_label: int = 4
    # Never pick a glyph enlarged more than this factor beyond its stored size.
    # ``upscale`` is already accounted for, so the article's "4x native" is
    # enforced as 4 / upscale in :class:`KanaGlyph.max_scale`.

    # ---- renderer hook -----------------------------------------------------
    # Fraction of eligible kana replaced when the opt-in ``real_kana`` flag is
    # enabled; 1.0 means "use the bank instead of fonts wherever possible".
    real_kana_prob: float = float(os.environ.get("REAL_KANA_PROB", "1.0"))
    pen_stroke_frac_min: float = 0.070
    pen_stroke_frac_max: float = 0.100
    pen_pressure_min: float = 0.75
    pen_pressure_max: float = 1.35
    kana_height_frac: float = 0.88
    min_font_size_for_bank: int = 14

    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.min_chars < 1 or self.max_chars < self.min_chars:
            raise ValueError("invalid character length range")
        if self.stroke_tolerance <= 1.0:
            raise ValueError("stroke_tolerance must be > 1")
        if self.image_stroke_ratio <= 1.0:
            raise ValueError("image_stroke_ratio must be > 1")
        if self.upscale <= 0:
            raise ValueError("upscale must be positive")
        if not 0.0 <= self.local_min_confidence <= 1.0:
            raise ValueError("local_min_confidence must be between 0 and 1")
        if not 0.0 <= self.verify_confidence <= 1.0:
            raise ValueError("verify_confidence must be between 0 and 1")
        if self.mercury_timeout_s <= 0:
            raise ValueError("mercury_timeout_s must be positive")
        if self.mercury_retries < 0:
            raise ValueError("mercury_retries must be >= 0")
        if self.mercury_max_side < 32:
            raise ValueError("mercury_max_side must be >= 32")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["root"] = str(self.root)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KanaConfig":
        values = dict(data)
        values["root"] = Path(values.get("root", DEFAULT_ROOT)).expanduser()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in values.items() if k in known})


# Directory names inside a bank root.
CANDIDATE_DIR = "candidates"
RAW_DIR = "raw"
CLEANED_DIR = "cleaned"
UPSCALED_DIR = "upscaled"
GLYPH_DIR = "glyphs"
QUARANTINE_DIR = "quarantine"
REVIEW_DIR = "review"
CACHE_DIR = "vision_cache"


def ensure_bank_dirs(root: Path) -> dict[str, Path]:
    """Create the standard bank directory tree and return the paths."""

    root = Path(root).expanduser()
    paths = {
        "root": root,
        "candidates": root / CANDIDATE_DIR,
        "raw": root / RAW_DIR,
        "cleaned": root / CLEANED_DIR,
        "upscaled": root / UPSCALED_DIR,
        "glyphs": root / GLYPH_DIR,
        "quarantine": root / QUARANTINE_DIR,
        "review": root / REVIEW_DIR,
        "cache": root / CACHE_DIR,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
