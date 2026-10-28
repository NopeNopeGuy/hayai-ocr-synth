"""Preprocess Manga109-s annotations and extract panel frames and dialogue text crops."""

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from manga_ocr_dev.env import MANGA109_ROOT


def get_books() -> pd.DataFrame:
    """Load metadata for all books in the Manga109-s release."""
    root = MANGA109_ROOT / "Manga109s_released_2021_02_28"
    books = (root / "books.txt").read_text(encoding="utf-8").splitlines()
    return pd.DataFrame(
        {
            "book": books,
            "annotations": [str(root / "annotations" / f"{book}.xml") for book in books],
            "images": [str(root / "images" / book) for book in books],
        }
    )


def export_frames():
    """Extract comic frame bounding boxes from Manga109-s annotations and save to frames.csv."""
    books = get_books()

    data = []
    for book in tqdm(books.itertuples(), total=len(books), desc="Exporting frames"):
        tree = ET.parse(book.annotations)
        root = tree.getroot()
        for page in root.findall("./pages/page"):
            for frame in page.findall("./frame"):
                row = {
                    "book": book.book,
                    "page_index": int(page.attrib["index"]),
                    "page_path": str(Path(book.images) / f"{int(page.attrib['index']):03d}.jpg"),
                    "page_width": int(page.attrib["width"]),
                    "page_height": int(page.attrib["height"]),
                    "id": frame.attrib["id"],
                    "xmin": int(frame.attrib["xmin"]),
                    "ymin": int(frame.attrib["ymin"]),
                    "xmax": int(frame.attrib["xmax"]),
                    "ymax": int(frame.attrib["ymax"]),
                }
                data.append(row)
    df = pd.DataFrame(data)
    df.page_path = df.page_path.apply(lambda x: "/".join(Path(x).parts[-4:]))
    df.to_csv(MANGA109_ROOT / "frames.csv", index=False)


def export_crops():
    """Extract individual text crops from Manga109-s pages and save metadata to data.csv."""
    crops_root = MANGA109_ROOT / "crops"
    crops_root.mkdir(parents=True, exist_ok=True)
    margin = 10

    books = get_books()

    data = []
    for book in tqdm(books.itertuples(), total=len(books), desc="Parsing text annotations"):
        tree = ET.parse(book.annotations)
        root = tree.getroot()
        for page in root.findall("./pages/page"):
            for text in page.findall("./text"):
                row = {
                    "book": book.book,
                    "page_index": int(page.attrib["index"]),
                    "page_path": str(Path(book.images) / f"{int(page.attrib['index']):03d}.jpg"),
                    "page_width": int(page.attrib["width"]),
                    "page_height": int(page.attrib["height"]),
                    "id": text.attrib["id"],
                    "text": text.text,
                    "xmin": int(text.attrib["xmin"]),
                    "ymin": int(text.attrib["ymin"]),
                    "xmax": int(text.attrib["xmax"]),
                    "ymax": int(text.attrib["ymax"]),
                }
                data.append(row)
    df = pd.DataFrame(data)

    n_test = int(0.1 * len(df))
    df["split"] = "train"
    df.loc[df.sample(len(df)).iloc[:n_test].index, "split"] = "test"

    df["crop_path"] = str(crops_root) + "/" + df.id + ".png"
    df.page_path = df.page_path.apply(lambda x: "/".join(Path(x).parts[-4:]))
    df.crop_path = df.crop_path.apply(lambda x: "/".join(Path(x).parts[-2:]))
    df.to_csv(MANGA109_ROOT / "data.csv", index=False)

    for page_path, boxes in tqdm(df.groupby("page_path"), total=df.page_path.nunique(), desc="Saving crops"):
        img = cv2.imread(str(MANGA109_ROOT / page_path))
        if img is None:
            continue

        for box in boxes.itertuples():
            xmin = max(box.xmin - margin, 0)
            xmax = min(box.xmax + margin, img.shape[1])
            ymin = max(box.ymin - margin, 0)
            ymax = min(box.ymax + margin, img.shape[0])
            crop = img[ymin:ymax, xmin:xmax]
            out_path = (crops_root / box.id).with_suffix(".png")
            cv2.imwrite(str(out_path), crop)


if __name__ == "__main__":
    export_frames()
    export_crops()
