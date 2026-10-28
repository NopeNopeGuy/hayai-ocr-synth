"""Small, dependency-light helpers shared by the real-kana pipeline."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import cv2
import numpy as np
from PIL import Image

# Kana that are written smaller in normal text.  The renderer already applies
# its own small-kana scale; the bank only uses this map when a native small
# kana is missing and a full-size glyph has to be shrunk.
SMALL_TO_LARGE = {
    "ぁ": "あ",
    "ぃ": "い",
    "ぅ": "う",
    "ぇ": "え",
    "ぉ": "お",
    "っ": "つ",
    "ゃ": "や",
    "ゅ": "ゆ",
    "ょ": "よ",
    "ゎ": "わ",
    "ゕ": "か",
    "ゖ": "け",
    "ァ": "ア",
    "ィ": "イ",
    "ゥ": "ウ",
    "ェ": "エ",
    "ォ": "オ",
    "ッ": "ツ",
    "ャ": "ヤ",
    "ュ": "ユ",
    "ョ": "ヨ",
    "ヮ": "ワ",
    "ヵ": "カ",
    "ヶ": "ケ",
}

_SMALL_KANA = set(SMALL_TO_LARGE)


def is_hiragana(ch: str) -> bool:
    cp = ord(ch)
    return 0x3041 <= cp <= 0x3096 or 0x309D <= cp <= 0x309F


def is_katakana(ch: str) -> bool:
    cp = ord(ch)
    return 0x30A1 <= cp <= 0x30FA or 0x30FD <= cp <= 0x30FF


def is_kana(ch: str) -> bool:
    """True for a single, full-width hiragana or katakana character.

    The prolonged sound mark (ー), middle dot (・) and iteration marks are
    intentionally excluded: they are punctuation-like symbols, not kana that
    can be replaced by a handwritten glyph.
    """

    return len(ch) == 1 and (is_hiragana(ch) or is_katakana(ch))


def is_kanji(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2EBEF
        or 0x30000 <= cp <= 0x323AF
    )


def normalize_kana_label(value: str) -> str | None:
    """Normalise a vision-model label, returning ``None`` if it is not kana."""

    if not isinstance(value, str):
        return None
    text = unicodedata.normalize("NFKC", value).strip()
    # Models occasionally answer "「あ」" or "あ (a)".
    text = text.strip("「」『』\"'`.,、。 ")
    if len(text) != 1 or not is_kana(text):
        return None
    if text in {"・", "ー", "〜", "～"}:
        return None
    return text


def kana_sequence(text: str, *, normalise: bool = True) -> list[str]:
    """Return the kana of *text* in reading order."""

    if normalise:
        text = unicodedata.normalize("NFKC", text)
    return [ch for ch in text if is_kana(ch)]


def has_letters(text: str) -> bool:
    return any(is_kana(ch) or is_kanji(ch) or ch.isalpha() for ch in text)


def is_likely_sfx(text: str, max_katakana_chars: int = 6) -> bool:
    """Conservative sound-effect heuristic used only during harvesting."""

    seq = kana_sequence(text)
    if not seq:
        return False
    kata = sum(is_katakana(ch) for ch in seq)
    hira = sum(is_hiragana(ch) for ch in seq)
    if kata == len(seq) and len(seq) <= max_katakana_chars:
        return True
    # Repeated katakana syllables ("ドドド", "ゴゴゴ") are almost always SFX.
    if len(seq) >= 3 and len(set(seq)) <= 2 and hira == 0:
        return True
    return False


# ---------------------------------------------------------------------------
# File / JSON helpers


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | os.PathLike[str], obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def iter_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    try:
        fh = open(path, "r", encoding="utf-8")
    except FileNotFoundError:
        return
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> int:
    """Write rows atomically and return the number of rows."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return count


def append_jsonl(path: str | os.PathLike[str], row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def safe_id(value: str, max_len: int = 120) -> str:
    """Filesystem-safe identifier that still resembles the source id."""

    out = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))
    return out[:max_len].strip("._") or sha256_bytes(str(value).encode())[:16]


def relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Image helpers


def pil_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def load_image(path: str | os.PathLike[str], mode: str = "RGB") -> Image.Image:
    """Open a file, convert it and close the file handle immediately."""

    with Image.open(path) as image:
        return image.convert(mode)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def trim_alpha(image: Image.Image, threshold: int = 4, pad: int = 1) -> Image.Image:
    """Trim fully transparent margins, keeping *pad* pixels."""

    if image.mode != "RGBA":
        image = image.convert("RGBA")
    alpha = np.asarray(image.getchannel("A"))
    ys, xs = np.nonzero(alpha > threshold)
    if len(xs) == 0:
        return image
    x0, x1 = max(0, int(xs.min()) - pad), min(image.width, int(xs.max()) + 1 + pad)
    y0, y1 = max(0, int(ys.min()) - pad), min(image.height, int(ys.max()) + 1 + pad)
    if (x0, y0, x1, y1) == (0, 0, image.width, image.height):
        return image
    return image.crop((x0, y0, x1, y1))


def alpha_bbox(alpha: np.ndarray, threshold: int = 4) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(alpha > threshold)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def estimate_stroke_width(alpha: np.ndarray) -> float:
    """Estimate pen-stroke width in pixels from an alpha mask.

    The distance transform peaks along the medial axis.  Twice the median
    ridge distance is a stable estimate for clean and anti-aliased strokes,
    unlike a global ink-area/perimeter ratio, which is dominated by corners.
    """

    arr = np.asarray(alpha, dtype=np.uint8)
    ink = (arr >= 127).astype(np.uint8)
    if int(ink.sum()) < 4:
        return 0.0
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    dilated = cv2.dilate(dist, np.ones((3, 3), np.uint8))
    ridge = ink.astype(bool) & (dist >= dilated - 1e-4) & (dist > 0.5)
    values = dist[ridge]
    if values.size < 3:
        values = dist[ink.astype(bool)]
    if values.size == 0:
        return 0.0
    return float(max(0.5, 2.0 * np.median(values)))


def estimate_ink_rgb(image: Image.Image, alpha: np.ndarray | None = None) -> tuple[int, int, int]:
    """Median RGB of the ink pixels (or of the darkest 10% as a fallback)."""

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    if alpha is None:
        alpha = np.asarray(image.convert("RGBA").getchannel("A"))
    mask = np.asarray(alpha) > 127
    if mask.sum() < 4:
        flat = rgb.reshape(-1, 3)
        luma = flat @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        mask = luma <= np.percentile(luma, 10)
        values = flat[mask]
    else:
        values = rgb[mask]
    if values.size == 0:
        return (0, 0, 0)
    med = np.median(values, axis=0)
    return tuple(int(round(v)) for v in np.clip(med, 0, 255))


def colour_separated_alpha(rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract a soft alpha mask by projecting colours onto the ink axis.

    ``rgb`` is HxWx3 uint8.  The background is estimated from a border ring,
    the ink colour from the pixels farthest from it.  A pixel is considered
    ink when it lies close to the background->ink colour line; pixels of a
    different hue (for example a neighbouring character in another colour)
    are rejected.  This is the "colour-separated alpha mask" from the article.
    """

    arr = np.asarray(rgb, dtype=np.float32)
    h, w = arr.shape[:2]
    if h < 2 or w < 2:
        return np.zeros((h, w), dtype=np.uint8), {"reason": "tiny"}

    ring = max(1, min(h, w) // 8)
    border = np.concatenate(
        [
            arr[:ring].reshape(-1, 3),
            arr[-ring:].reshape(-1, 3),
            arr[:, :ring].reshape(-1, 3),
            arr[:, -ring:].reshape(-1, 3),
        ]
    )
    bg = np.median(border, axis=0)
    dist_bg = np.linalg.norm(arr - bg, axis=2)
    threshold = max(18.0, float(np.percentile(dist_bg, 92)))
    far = arr[dist_bg >= threshold]
    if far.shape[0] < 4:
        far = arr.reshape(-1, 3)
    # Use a robust "far" colour, not a single darkest pixel.
    fg = np.median(far, axis=0)

    direction = fg - bg
    norm2 = float(np.dot(direction, direction))
    if norm2 < 25.0:
        # Nearly monochrome: fall back to luminance polarity.
        luma = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        bg_luma = float(bg @ np.array([0.299, 0.587, 0.114], dtype=np.float32))
        if bg_luma >= 128:
            alpha = (bg_luma - luma) / max(1.0, bg_luma)
        else:
            alpha = (luma - bg_luma) / max(1.0, 255.0 - bg_luma)
        alpha = np.clip(alpha, 0.0, 1.0)
        return (alpha * 255.0 + 0.5).astype(np.uint8), {
            "background_rgb": [float(v) for v in bg],
            "ink_rgb": [float(v) for v in fg],
            "method": "luma",
        }

    t = ((arr - bg) @ direction) / norm2
    residual = np.linalg.norm(arr - (bg + t[..., None] * direction), axis=2)
    residual_limit = max(30.0, 0.42 * math.sqrt(norm2))
    hue_ok = (residual <= residual_limit) | (t >= 0.80)
    alpha = np.clip(t, 0.0, 1.0) * hue_ok
    alpha = np.where(alpha < 0.06, 0.0, alpha)
    # A light blur removes one-pixel halos without eating the stroke.
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (3, 3), 0.55)
    return (np.clip(alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), {
        "background_rgb": [float(v) for v in bg],
        "ink_rgb": [float(v) for v in fg],
        "method": "colour_axis",
        "residual_limit": residual_limit,
    }


def composite_on_white(rgba: Image.Image) -> Image.Image:
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))
    return bg


def checkerboard(width: int, height: int, cell: int = 12, light: int = 255, dark: int = 210) -> Image.Image:
    yy, xx = np.mgrid[0:height, 0:width]
    board = np.where(((yy // cell + xx // cell) % 2) == 0, light, dark).astype(np.uint8)
    return Image.fromarray(np.repeat(board[:, :, None], 3, axis=2), mode="RGB")


def rotated(image: Image.Image, angle_deg: float, *, expand: bool = True) -> Image.Image:
    """Rotate with a stable convention: positive = counter-clockwise."""

    if abs(angle_deg) < 1e-6:
        return image
    return image.rotate(
        float(angle_deg),
        resample=Image.BICUBIC,
        expand=expand,
        fillcolor=(0, 0, 0, 0) if image.mode == "RGBA" else (255, 255, 255),
    )


def resize_rgba(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    size = (max(1, int(size[0])), max(1, int(size[1])))
    arr = np.asarray(image.convert("RGBA"), dtype=np.float32)
    # Resize in premultiplied space to avoid dark/white fringes around alpha.
    alpha = arr[..., 3:4] / 255.0
    premult = np.concatenate([arr[..., :3] * alpha, arr[..., 3:4]], axis=2)
    out = cv2.resize(premult, size, interpolation=cv2.INTER_LANCZOS4)
    out[..., :3] = np.clip(out[..., :3], 0, 255 * np.maximum(out[..., 3:4] / 255.0, 1e-6))
    out[..., 3] = np.clip(out[..., 3], 0, 255)
    with np.errstate(divide="ignore", invalid="ignore"):
        rgb = np.where(out[..., 3:4] > 1e-6, out[..., :3] / (out[..., 3:4] / 255.0), 0)
    rgba = np.concatenate([np.clip(rgb, 0, 255), out[..., 3:4]], axis=2)
    return Image.fromarray(rgba.astype(np.uint8), mode="RGBA")


def mask_to_rgba(
    alpha: np.ndarray,
    ink_rgb: Sequence[int],
    *,
    colour: bool = False,
    rgb: np.ndarray | None = None,
) -> Image.Image:
    """Build an RGBA glyph from an alpha mask and an ink colour."""

    alpha = np.asarray(alpha, dtype=np.uint8)
    if colour and rgb is not None and rgb.shape[:2] == alpha.shape:
        out_rgb = np.asarray(rgb, dtype=np.uint8).copy()
        out_rgb[alpha == 0] = 0
    else:
        out_rgb = np.zeros((*alpha.shape, 3), dtype=np.uint8)
        out_rgb[...] = np.asarray(ink_rgb, dtype=np.uint8)
    return Image.fromarray(np.dstack([out_rgb, alpha]), mode="RGBA")


def seed_from(*parts: Any) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little", signed=False) & 0x7FFFFFFF
