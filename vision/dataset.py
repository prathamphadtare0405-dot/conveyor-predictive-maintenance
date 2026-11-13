"""Turn the downloaded crack-seg dataset into a small detection dataset for CPU training.

    python -m vision.dataset

What it does:

* The source labels are segmentation polygons. The API has to return boxes, so each
  polygon is replaced by its bounding box (YOLO detection format: class, centre x,
  centre y, width, height, all as fractions of the image size).
* Only a subset of the training images is used (default 800 of 3,717), chosen with a
  fixed seed, so that training fits a stated CPU budget. All 200 validation images are
  kept, so the validation metrics are measured on the published validation split.
* Every image is written through the same preprocessing used at inference.

Output: data/datasets/crack-det/ with images/, labels/ and data.yaml.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

from backend.config import settings
from vision.preprocessing import enhance

SOURCE = settings.data_dir / "datasets" / "crack-seg"
TARGET = settings.data_dir / "datasets" / "crack-det"
CLASS_NAMES = {0: "crack"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def polygon_line_to_box(line: str) -> str | None:
    """Convert one label line (class x1 y1 x2 y2 ...) to a detection line (class xc yc w h).

    A line that already has exactly four numbers after the class is taken to be a box.
    Returns None for a line that cannot be used.
    """
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        numbers = [float(v) for v in parts[1:]]
    except ValueError:
        return None
    if len(numbers) == 4:
        xc, yc, w, h = numbers
    else:
        if len(numbers) < 6 or len(numbers) % 2:
            return None
        xs = [min(1.0, max(0.0, v)) for v in numbers[0::2]]
        ys = [min(1.0, max(0.0, v)) for v in numbers[1::2]]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        xc, yc = min(xs) + w / 2.0, min(ys) + h / 2.0
    if w <= 0.0 or h <= 0.0:
        return None
    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def convert_labels(text: str) -> str:
    boxes = [box for box in (polygon_line_to_box(line) for line in text.splitlines() if line.strip()) if box]
    return "\n".join(boxes) + ("\n" if boxes else "")


def build(source: Path = SOURCE, target: Path = TARGET, train_count: int = 800, seed: int = 0, preprocess: bool = True) -> Path:
    """Create the detection dataset and return the path of its data.yaml."""
    if not (source / "images" / "train").is_dir():
        raise FileNotFoundError(f"No dataset at {source}. Run:  python scripts/download_dataset.py")

    summary = {}
    for split in ("train", "val"):
        images = sorted(p for p in (source / "images" / split).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if split == "train" and train_count < len(images):
            images = sorted(random.Random(seed).sample(images, train_count))
        image_dir, label_dir = target / "images" / split, target / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for old in list(image_dir.iterdir()) + list(label_dir.iterdir()):
            old.unlink()
        boxes = empty = 0
        for path in images:
            image = cv2.imread(str(path))
            if image is None:
                continue  # unreadable source file: skip it rather than train on it
            cv2.imwrite(str(image_dir / f"{path.stem}.jpg"), enhance(image) if preprocess else image)
            label_file = source / "labels" / split / f"{path.stem}.txt"
            converted = convert_labels(label_file.read_text(encoding="utf-8")) if label_file.exists() else ""
            (label_dir / f"{path.stem}.txt").write_text(converted, encoding="utf-8")
            count = converted.count("\n")
            boxes += count
            empty += count == 0
        summary[split] = (len(images), boxes, empty)

    yaml = target / "data.yaml"
    yaml.write_text(
        f"# Detection subset of crack-seg, built by vision/dataset.py (seed {seed}, preprocess={preprocess})\n"
        f"path: {target.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {k}: {v}\n" for k, v in CLASS_NAMES.items()),
        encoding="utf-8",
    )
    for split, (n, boxes, empty) in summary.items():
        print(f"{split}: {n} images, {boxes} boxes, {empty} images without a crack")
    print(f"Wrote {yaml}")
    return yaml


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m vision.dataset", description=__doc__.split("\n\n")[0])
    parser.add_argument("--train-count", type=int, default=800, help="training images to use (default %(default)s)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-preprocess", action="store_true", help="write the images unchanged")
    args = parser.parse_args()
    try:
        build(train_count=args.train_count, seed=args.seed, preprocess=not args.no_preprocess)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
