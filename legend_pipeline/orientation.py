"""Auto-orientation — correct legends / maps rotated by a multiple of 90 deg."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from .config import PipelineConfig
from .containers import OcrText
from .deps import LOGGER, _require, cv2

if TYPE_CHECKING:  # avoid an import cycle; only needed for type hints.
    from .ocr import OcrEngine


# Map a clockwise correction angle to the matching lossless OpenCV rotation.
_ROTATE_OPS: Dict[int, Optional[int]] = {}


def _rotate_ops() -> Dict[int, Optional[int]]:
    """Build the {clockwise-degrees: cv2 rotate flag} table (cv2 may be absent)."""
    global _ROTATE_OPS
    if not _ROTATE_OPS and cv2 is not None:
        _ROTATE_OPS = {
            0: None,
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,   # 270 CW == 90 CCW.
        }
    return _ROTATE_OPS


def rotate_image(image: np.ndarray, angle_cw: int) -> np.ndarray:
    """Rotate ``image`` clockwise by ``angle_cw`` (one of 0/90/180/270) losslessly.

    Only right-angle rotations are supported — they need no interpolation and
    never crop, so the pixel data is preserved exactly.  ``angle_cw`` is
    normalised into [0, 360), so passing e.g. -90 or 450 also works.
    """
    _require(cv2, "opencv-python", "pip install opencv-python")
    angle = int(angle_cw) % 360
    op = _rotate_ops().get(angle)
    if op is None:
        if angle != 0:
            raise ValueError(f"Only 0/90/180/270 deg rotations are supported, got {angle_cw}.")
        return image
    return cv2.rotate(image, op)


def _resize_long_side(image: np.ndarray, target: int) -> np.ndarray:
    """Downscale so the long side is at most ``target`` px (never upscales)."""
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= target:
        return image
    scale = target / float(long_side)
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def _text_orientation_score(texts: List["OcrText"]) -> float:
    """Score how upright text reads: sum of confidence x alphabetic-char count.

    Upright text OCRs into many high-confidence, letter-rich tokens; the same
    text rotated 90/180 deg OCRs into little or nothing, so this score peaks at
    the correct orientation.  Weighting by letters (not raw token count) keeps
    a few solid words from being outvoted by many one-character misreads.
    """
    total = 0.0
    for t in texts:
        letters = sum(ch.isalpha() for ch in t.text)
        if letters >= 2:
            total += t.confidence * letters
    return total


def detect_upright_rotation(
    image: np.ndarray,
    ocr: "OcrEngine",
    config: PipelineConfig,
) -> Tuple[int, Dict[int, float]]:
    """Find the clockwise rotation that makes ``image`` read upright.

    Each candidate right-angle rotation is OCR'd (on a size-capped copy for
    speed) and scored by :func:`_text_orientation_score`.  The best-scoring
    orientation is returned as the clockwise correction to apply, together with
    the per-angle scores (for logging).  Upright (0 deg) is only abandoned when
    another orientation beats it by ``config.rotate_min_gain`` — a safeguard so
    an already-upright image is never rotated on a near-tie.
    """
    if not config.auto_rotate:
        return 0, {}
    probe = _resize_long_side(image, config.rotate_probe_long_side)
    scores: Dict[int, float] = {}
    for angle in config.rotate_candidate_angles:
        try:
            rotated = rotate_image(probe, angle)
        except ValueError:
            LOGGER.warning("Skipping unsupported rotation angle %s.", angle)
            continue
        scores[angle] = _text_orientation_score(ocr.read(rotated))
        LOGGER.debug("Orientation probe %3d deg CW -> score %.2f", angle, scores[angle])
    if not scores:
        return 0, {}
    best = max(scores, key=lambda a: scores[a])
    upright = scores.get(0, 0.0)
    # Keep upright unless a rotation reads clearly more text.
    if best != 0 and scores[best] < config.rotate_min_gain * max(upright, 1.0):
        LOGGER.debug("Best angle %d only scored %.2f vs upright %.2f — keeping upright.",
                     best, scores[best], upright)
        best = 0
    return best, scores
