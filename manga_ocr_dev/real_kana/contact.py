"""Contact sheets used by the vision review stage and by humans."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from manga_ocr_dev.real_kana.utils import checkerboard

_CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _font(size: int) -> ImageFont.ImageFont:
    for path in _CJK_FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _fit(image: Image.Image, box: int, *, pad: int = 6) -> Image.Image:
    image = image.convert("RGBA")
    target = max(1, box - 2 * pad)
    scale = min(target / max(1, image.width), target / max(1, image.height), 4.0)
    if abs(scale - 1.0) > 1e-3:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
    return image


def contact_sheet(
    items: Sequence[tuple[Image.Image, str, str | None]],
    *,
    cols: int = 8,
    cell: int = 112,
    title: str | None = None,
    header: int = 34,
    footer: int = 22,
) -> Image.Image:
    """Build a numbered grid.

    ``items`` are ``(image, index_label, caption)`` triples.  The index is drawn
    at the top-left of each cell and the caption (if any) at the bottom.  The
    checkerboard makes white ink and transparent glyphs visible.
    """

    n = len(items)
    rows = max(1, int(np.ceil(n / cols)))
    width = cols * cell
    height = header + rows * (cell + footer)
    sheet = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    font = _font(14)
    small = _font(12)
    if title:
        draw.text((8, 8), title[:200], fill=(20, 20, 20), font=font)
    for i, (image, index_label, caption) in enumerate(items):
        r, c = divmod(i, cols)
        x, y = c * cell, header + r * (cell + footer)
        board = checkerboard(cell, cell, cell=10, light=255, dark=225)
        sheet.paste(board, (x, y))
        glyph = _fit(image, cell)
        gx = x + (cell - glyph.width) // 2
        gy = y + (cell - glyph.height) // 2
        sheet.paste(glyph.convert("RGB"), (gx, gy), glyph.getchannel("A"))
        draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline=(180, 180, 180))
        draw.text((x + 4, y + 3), str(index_label), fill=(180, 20, 20), font=small)
        if caption:
            draw.text((x + 3, y + cell + 3), caption[:18], fill=(20, 20, 20), font=small)
    return sheet


def overlay_boxes(
    image: Image.Image,
    boxes: Sequence[Sequence[int]],
    *,
    labels: Sequence[str] | None = None,
    scale: int = 3,
) -> Image.Image:
    """Draw boxes on an enlarged copy of *image* (for provenance sheets)."""

    base = image.convert("RGB")
    if scale > 1:
        base = base.resize((base.width * scale, base.height * scale), Image.NEAREST)
    draw = ImageDraw.Draw(base)
    font = _font(13)
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = [int(v) * scale for v in box]
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(220, 0, 0), width=2)
        text = labels[i] if labels and i < len(labels) else str(i)
        draw.text((x0 + 2, max(0, y0 - 14)), text, fill=(220, 0, 0), font=font)
    return base


def label_sheet(glyphs: Sequence[tuple[str, Image.Image, str]], *, cols: int = 8) -> Image.Image:
    """Grid for human review, grouped by label and annotated with ids."""

    items = [(image, label, glyph_id) for label, image, glyph_id in glyphs]
    return contact_sheet(items, cols=cols, cell=120, title="Real kana review")
