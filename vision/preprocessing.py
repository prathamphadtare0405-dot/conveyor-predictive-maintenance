"""Image preprocessing for belt-surface images.

Belt images are hard for a detector for specific reasons, and each step answers one:

1. **Denoising (bilateral filter).** Cameras near a conveyor run at short exposure to
   freeze a belt moving at several metres per second, which means high gain and sensor
   noise; airborne dust adds speckle. A bilateral filter smooths that out while keeping
   edges, and edges are what a crack or tear is. (Non-local-means denoising is better
   but takes seconds per frame on a CPU.)
2. **Local contrast (CLAHE on the lightness channel).** Black rubber under dust has very
   little contrast, and lighting across the belt is uneven. CLAHE stretches contrast
   tile by tile, so a dark crack in a dark region becomes visible without washing out
   the bright regions. It is applied to lightness only, so colours are not distorted,
   and its clip limit stops it from amplifying noise in flat areas (which is also why
   denoising comes first).

The same function is applied to the training images and to every image at inference,
so the model never sees a kind of image it was not trained on.
"""

from __future__ import annotations

import cv2
import numpy as np

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_SIDE = 1600  # larger frames are scaled down before processing


class InvalidImage(ValueError):
    """The uploaded bytes are not a usable image. The message says why."""


def decode_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes to a BGR image, rejecting anything that is not one."""
    if not data:
        raise InvalidImage("The file is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise InvalidImage(f"The file is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImage("The file is not a readable image (unsupported format, or the file is corrupt).")
    height, width = image.shape[:2]
    if height < 16 or width < 16:
        raise InvalidImage(f"The image is too small to analyse ({width}x{height} pixels).")
    if max(height, width) > MAX_SIDE:
        scale = MAX_SIDE / max(height, width)
        image = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return image


def enhance(image: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Denoise, then equalise local contrast. Input and output are BGR uint8 of the same size."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    smooth = cv2.bilateralFilter(image, d=5, sigmaColor=40, sigmaSpace=40)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)


def difference_hash(image: np.ndarray, size: int = 16) -> np.ndarray:
    """Perceptual hash of an image region as a flat array of `size * size` bits.

    Each bit says whether a pixel is brighter than its right-hand neighbour after the
    region is shrunk to (size+1) x size grey pixels. It keeps the coarse structure of
    the region and ignores overall brightness, small blur and noise, which is what is
    wanted when the same damage is photographed again a lap later.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(grey, (size + 1, size), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (small[:, 1:] > small[:, :-1]).flatten()


def hash_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Share of equal bits, 0..1. Unrelated regions score about 0.5."""
    if a.shape != b.shape:
        raise ValueError("hashes have different sizes")
    return float(np.mean(a == b))


def hash_to_hex(bits: np.ndarray) -> str:
    return np.packbits(bits.astype(np.uint8)).tobytes().hex()


def hash_from_hex(text: str, size: int = 16) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes.fromhex(text), dtype=np.uint8))[: size * size].astype(bool)
