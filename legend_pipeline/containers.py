"""Data containers — Detection / OcrText / VisualSignature dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np


@dataclass
class Detection:
    """A single object detection plus everything we derive from it."""

    class_name: str
    confidence: float
    # Bounding box in absolute pixel coords: (x1, y1, x2, y2).
    bbox: Tuple[int, int, int, int]
    # Optional polygon/mask points [(x, y), ...] if the model is segmentation.
    polygon: Optional[List[Tuple[float, float]]] = None
    crop: Optional[np.ndarray] = field(default=None, repr=False)
    signature: Optional["VisualSignature"] = field(default=None, repr=False)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


@dataclass
class OcrText:
    """One OCR text token with its (cleaned) string and bounding box."""

    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class VisualSignature:
    """Bundle of visual features used for classical template + ORB matching.

    ``glyph`` is the background-free, canonically-sized grayscale symbol — the
    primary object we match on.  ``keypoints``/``orb_descriptors`` describe that
    same glyph.  ``phash`` is retained only for the human-readable Hamming
    reports, not for the rename decision.
    """

    glyph: Optional[np.ndarray] = field(default=None, repr=False)   # HxW gray.
    keypoints: Optional[Any] = field(default=None, repr=False)      # ORB kps.
    orb_descriptors: Optional[np.ndarray] = field(default=None, repr=False)
    aspect_ratio: float = 1.0
    phash: Optional[Any] = None                 # imagehash.ImageHash (reporting)
    phash_hex: Optional[str] = None
