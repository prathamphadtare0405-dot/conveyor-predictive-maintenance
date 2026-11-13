"""Download the vision dataset. No account and no API key are needed.

    python scripts/download_dataset.py

Dataset: "crack-seg", published by Ultralytics (documentation:
https://docs.ultralytics.com/datasets/segment/crack-seg/). 4,029 photographs of cracks in
roads and walls with instance-segmentation masks, one class ("crack"); 3,717 train, 200
validation, 112 test. The documentation page marks it Public Domain and cites the source
as the "crack" dataset on Roboflow Universe (university-bswxt/crack-bphdr, 2022).

It is NOT conveyor-belt imagery. It is used here as a stand-in surface-defect dataset so
the detection pipeline can be built and measured; see the README for why and for what
that means for the results.

The archive (about 92 MB) is fetched from the public release URL given in the dataset's
own configuration file and unpacked into data/datasets/crack-seg/.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings  # noqa: E402

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/crack-seg.zip"
EXPECTED = {"train": 3717, "val": 200, "test": 112}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def dataset_dir() -> Path:
    return settings.data_dir / "datasets" / "crack-seg"


def count_images(root: Path) -> dict[str, int]:
    return {
        split: sum(1 for p in (root / "images" / split).glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        for split in EXPECTED
        if (root / "images" / split).is_dir()
    }


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        next_report = 0
        with partial.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    print(f"  {done / 1e6:6.1f} MB" + (f" of {total / 1e6:.1f} MB" if total else ""), flush=True)
                    next_report += 10 << 20
    partial.replace(target)


def safe_extract(archive: Path, destination: Path) -> None:
    """Unpack, refusing any member that would land outside the destination folder."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            resolved = (destination / member.filename).resolve()
            if destination not in resolved.parents and resolved != destination:
                raise RuntimeError(f"Archive member {member.filename!r} points outside the dataset folder; not extracting.")
        zf.extractall(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="download again even if the dataset is already present")
    args = parser.parse_args()

    root = dataset_dir()
    if not args.force and count_images(root) == EXPECTED:
        print(f"Dataset already present at {root}: {EXPECTED}")
        return 0

    archive = root.parent / "crack-seg.zip"
    print(f"Downloading {URL}")
    try:
        download(URL, archive)
    except requests.RequestException as exc:
        print(f"Download failed: {exc}\nCheck the internet connection and try again.", file=sys.stderr)
        return 1

    # The archive holds images/ and labels/ at its top level, with no enclosing folder.
    print(f"Unpacking into {root}")
    root.mkdir(parents=True, exist_ok=True)
    try:
        safe_extract(archive, root)
    except (zipfile.BadZipFile, RuntimeError) as exc:
        print(f"Could not unpack the archive: {exc}", file=sys.stderr)
        return 1

    found = count_images(root)
    print(f"Images found: {found}")
    if found != EXPECTED:
        print(f"Warning: expected {EXPECTED}. The published dataset may have changed; check before training.", file=sys.stderr)
        return 1
    archive.unlink()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
