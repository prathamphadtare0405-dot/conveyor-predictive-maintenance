"""Damage detection on one image with the fine-tuned YOLO nano model.

The model is loaded the first time an image is analysed, not when the backend starts,
so the backend runs (and the sensor side works) on a machine that has no weights file.
A missing weights file raises `VisionModelMissing` with the commands that create it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.config import settings
from vision.preprocessing import enhance

INFERENCE_SIZE = 320


class VisionModelMissing(RuntimeError):
    """No trained weights. The message says how to produce them."""


@dataclass
class Detection:
    damage_class: str
    confidence: float
    x1: float  # pixels, in the analysed image
    y1: float
    x2: float
    y2: float
    image_width: int
    image_height: int

    @property
    def area_fraction(self) -> float:
        """Box area as a share of the image area: the size measure used for growth."""
        return max(0.0, (self.x2 - self.x1) * (self.y2 - self.y1)) / float(self.image_width * self.image_height)

    @property
    def lateral(self) -> float:
        """Horizontal centre of the box, 0 (left edge of the frame) to 1 (right edge)."""
        return ((self.x1 + self.x2) / 2.0) / float(self.image_width)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(area_fraction=round(self.area_fraction, 5), lateral=round(self.lateral, 4))
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in data.items()}


def _load_model(weights: Path):
    """Load the YOLO model. Kept as a separate function so tests can replace it."""
    from ultralytics import YOLO  # imported here: importing torch takes seconds

    return YOLO(str(weights))


class DamageDetector:
    def __init__(self, weights: Path | None = None, confidence: float | None = None) -> None:
        self.weights = Path(weights) if weights else settings.vision_weights
        self.confidence = confidence if confidence is not None else settings.vision_conf_threshold
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def model(self):
        if self._model is None:
            if not self.weights.exists():
                raise VisionModelMissing(
                    f"No vision model at {self.weights}. Create it with:  python scripts/download_dataset.py  "
                    "then  python -m vision.dataset  then  python -m vision.train"
                )
            self._model = _load_model(self.weights)
        return self._model

    def detect(self, image: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        """Preprocess the image and run the detector. Returns detections and the image analysed."""
        prepared = enhance(image)
        result = self.model().predict(prepared, imgsz=INFERENCE_SIZE, conf=self.confidence, device="cpu", verbose=False)[0]
        height, width = prepared.shape[:2]
        names = result.names
        detections = []
        for (x1, y1, x2, y2), conf, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
            x1, x2 = sorted((min(max(x1, 0.0), width), min(max(x2, 0.0), width)))
            y1, y2 = sorted((min(max(y1, 0.0), height), min(max(y2, 0.0), height)))
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue
            detections.append(Detection(str(names[int(cls)]), float(conf), x1, y1, x2, y2, width, height))
        detections.sort(key=lambda d: -d.confidence)
        return detections, prepared


def annotate(image: np.ndarray, detections: list[Detection], labels: list[str] | None = None) -> np.ndarray:
    """Draw boxes with class, confidence and (optionally) a label such as the damage id."""
    out = image.copy()
    scale = max(0.4, min(out.shape[:2]) / 700.0)
    thickness = max(1, int(round(scale * 2)))
    for i, det in enumerate(detections):
        p1, p2 = (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2))
        cv2.rectangle(out, p1, p2, (0, 0, 255), thickness)
        text = f"{det.damage_class} {det.confidence:.2f}"
        if labels and labels[i]:
            text = f"{labels[i]} {text}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        top = max(0, p1[1] - th - 6)
        cv2.rectangle(out, (p1[0], top), (p1[0] + tw + 4, top + th + 6), (0, 0, 255), -1)
        cv2.putText(out, text, (p1[0] + 2, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def crop(image: np.ndarray, det: Detection) -> np.ndarray:
    return image[int(det.y1) : max(int(det.y1) + 1, int(det.y2)), int(det.x1) : max(int(det.x1) + 1, int(det.x2))]


_detector: DamageDetector | None = None


def get_detector() -> DamageDetector:
    """The shared detector. Creating it is cheap; the model loads on first use."""
    global _detector
    if _detector is None or _detector.weights != settings.vision_weights:
        _detector = DamageDetector()
    return _detector
