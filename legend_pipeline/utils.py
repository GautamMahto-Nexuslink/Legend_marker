"""Utility helpers — logging setup, image IO, cropping, filename hygiene."""
from __future__ import annotations

import logging
import os
from typing import Sequence

import numpy as np

from .deps import LOGGER, _require, cv2


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging once, with a compact, timestamped format."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_image(path: str) -> np.ndarray:
    """Read an image from disk as BGR uint8, raising a clear error on failure."""
    _require(cv2, "opencv-python", "pip install opencv-python")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image (unsupported/corrupt?): {path}")
    LOGGER.debug("Loaded image %s with shape %s", path, img.shape)
    return img


def safe_crop(image: np.ndarray, bbox: Sequence[int]) -> np.ndarray:
    """Crop an image by bbox, clamping to bounds and preserving pixels exactly.

    Returns a *copy* so the original image is never mutated and the crop is
    contiguous (important for deterministic hashing / encoding).
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    # Clamp and normalise ordering so a bad box never crashes the pipeline.
    x1, x2 = sorted((int(round(x1)), int(round(x2))))
    y1, y2 = sorted((int(round(y1)), int(round(y2))))
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    return image[y1:y2, x1:x2].copy()


def ensure_dir(path: str) -> str:
    """Create a directory (and parents) if needed; return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def sanitize_filename(name: str) -> str:
    """Make an arbitrary label safe to use as a filename component."""
    keep = "-_.() "
    cleaned = "".join(c if (c.isalnum() or c in keep) else "_" for c in name)
    return cleaned.strip().replace(" ", "_") or "unnamed"
