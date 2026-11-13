"""Appearance check for the damage registry: is this the damage we saw before?

First attempt, and why it was replaced. The first version compared a perceptual hash of
the detection box crop from one lap with the hash from the next. In the loop replay it
split 7 damages into 19 registry entries: the detector does not draw the same box around
the same crack twice. Sometimes the box is larger, sometimes it covers half the crack,
sometimes there are two boxes. A hash of the crop changes when the crop does, even though
the belt has not changed.

What is done now. Each registry entry keeps a small grey "context" image: the detection
box plus a margin, with the position of the box inside it. To compare a new detection
with an entry, the question is asked both ways round:

* is the entry's stored box content found somewhere in the new detection's context?
* is the new detection's box content found somewhere in the entry's stored context?

"Found" is normalised cross-correlation (template matching) over a few scales, so it does
not depend on brightness or contrast, tolerates the box moving and changing size, and
tolerates moderate growth. The higher of the two scores is used: one direction covers
"the new box is bigger than the old one", the other covers the opposite.

Limits: it compares grey-level structure, so heavy dirt, a wet belt or strong shadows
would lower the score, and two different cracks that look alike and lie within the
position tolerance of each other can still be confused. It has only been exercised on
replayed still images (see scripts/seed_demo.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WORK_WIDTH = 320  # frames are compared at this width, whatever the camera resolution
CONTEXT_MARGIN = 0.6  # context = box grown by this share of its larger side, on every side
SCALES = (0.8, 0.9, 1.0, 1.1, 1.25, 1.4)
MIN_TEMPLATE_SIDE = 12
MIN_TEMPLATE_STD = 4.0  # a flat patch correlates with anything


@dataclass
class Appearance:
    """A grey context image around a detection, and where the detection box sits inside it."""

    context: np.ndarray  # uint8, grey
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def patch(self) -> np.ndarray:
        return self.context[self.y1 : self.y2, self.x1 : self.x2]

    def encode(self) -> bytes:
        ok, buffer = cv2.imencode(".png", self.context)
        if not ok:
            raise ValueError("could not encode the context image")
        return buffer.tobytes()

    @classmethod
    def decode(cls, data: bytes, x1: int, y1: int, x2: int, y2: int) -> "Appearance":
        context = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if context is None:
            raise ValueError("stored context image is unreadable")
        return cls(context, x1, y1, x2, y2)


def work_frame(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Grey copy of the frame at the working width, and the scale factor applied."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = WORK_WIDTH / grey.shape[1]
    if abs(scale - 1.0) > 1e-3:
        grey = cv2.resize(grey, (WORK_WIDTH, max(1, int(round(grey.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(grey, (3, 3), 0), scale


def extract(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> Appearance:
    """Cut the context of a detection box (pixel coordinates in `image`) out of a frame."""
    frame, scale = work_frame(image)
    h, w = frame.shape
    bx1, by1 = int(np.floor(x1 * scale)), int(np.floor(y1 * scale))
    bx2, by2 = int(np.ceil(x2 * scale)), int(np.ceil(y2 * scale))
    bx1, by1 = max(0, min(bx1, w - 2)), max(0, min(by1, h - 2))
    bx2, by2 = max(bx1 + 2, min(bx2, w)), max(by1 + 2, min(by2, h))
    margin = int(round(CONTEXT_MARGIN * max(bx2 - bx1, by2 - by1)))
    cx1, cy1 = max(0, bx1 - margin), max(0, by1 - margin)
    cx2, cy2 = min(w, bx2 + margin), min(h, by2 + margin)
    return Appearance(frame[cy1:cy2, cx1:cx2].copy(), bx1 - cx1, by1 - cy1, bx2 - cx1, by2 - cy1)


def _find(template: np.ndarray, search: np.ndarray) -> float:
    """Best normalised correlation of `template` inside `search` over a few scales, 0..1."""
    if min(template.shape) < MIN_TEMPLATE_SIDE or float(template.std()) < MIN_TEMPLATE_STD:
        return 0.0
    best = 0.0
    for s in SCALES:
        tw, th = int(round(template.shape[1] * s)), int(round(template.shape[0] * s))
        if tw < MIN_TEMPLATE_SIDE or th < MIN_TEMPLATE_SIDE or tw > search.shape[1] or th > search.shape[0]:
            continue
        scaled = template if s == 1.0 else cv2.resize(template, (tw, th), interpolation=cv2.INTER_LINEAR)
        result = cv2.matchTemplate(search, scaled, cv2.TM_CCOEFF_NORMED)
        value = float(np.nanmax(result)) if result.size else 0.0
        if np.isfinite(value):
            best = max(best, value)
    return max(0.0, min(1.0, best))


def similarity(a: Appearance, b: Appearance) -> float:
    """How alike two sightings are, 0..1: the better of "a's box found in b's context" and the reverse."""
    return max(_find(a.patch, b.context), _find(b.patch, a.context))
