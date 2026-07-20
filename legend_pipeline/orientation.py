"""Auto-orientation — correct legends / maps rotated by a multiple of 90 deg."""
from __future__ import annotations

import re
import subprocess
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from .config import PipelineConfig
from .containers import OcrText
from .deps import LOGGER, _require, cv2

if TYPE_CHECKING:  # avoid an import cycle; only needed for type hints.
    from .ocr import OcrEngine


# Parse the two fields we need out of a Tesseract OSD (``--psm 0``) report.
_OSD_ROTATE_RE = re.compile(r"Rotate:\s*(\d+)")
_OSD_CONF_RE = re.compile(r"Orientation confidence:\s*([\d.]+)")


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


def _osd_rotation(image: np.ndarray, config: PipelineConfig) -> Optional[int]:
    """Detect the upright correction angle with Tesseract OSD in a single pass.

    Runs the tesseract binary in orientation-detection mode (``--psm 0``) on a
    size-capped copy of the image, feeding the PNG bytes via stdin so no temp
    file is needed.  The OSD "Rotate" field is exactly the clockwise angle that
    makes the page upright, so it maps directly onto :func:`rotate_image`.

    Returns the clockwise correction (0/90/180/270), or ``None`` when OSD is
    unavailable (binary missing), errors/times out, yields a non-right angle, or
    reports a confidence below ``config.osd_min_confidence`` — in every such case
    the caller falls back to the OCR probe.  Because it shells out to tesseract
    directly, it works no matter which OCR engine the pipeline is configured for.
    """
    if cv2 is None:
        return None
    probe = _resize_long_side(image, config.rotate_probe_long_side)
    ok, buf = cv2.imencode(".png", probe)
    if not ok:
        return None
    try:
        proc = subprocess.run(
            [config.tesseract_cmd, "-", "stdout", "--psm", "0"],
            input=buf.tobytes(),
            capture_output=True,
            timeout=config.osd_timeout,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        LOGGER.debug("OSD unavailable (%s) — falling back to OCR probe.", exc)
        return None

    out = proc.stdout.decode("utf-8", errors="replace")
    m = _OSD_ROTATE_RE.search(out)
    if not m:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        LOGGER.debug("OSD returned no angle (%s) — falling back to OCR probe.",
                     err[:120] or "no output")
        return None
    angle = int(m.group(1)) % 360
    if angle not in (0, 90, 180, 270):
        LOGGER.debug("OSD angle %d not a right angle — falling back.", angle)
        return None
    conf_match = _OSD_CONF_RE.search(out)
    conf = float(conf_match.group(1)) if conf_match else 0.0
    if conf < config.osd_min_confidence:
        LOGGER.debug("OSD confidence %.2f < %.2f — falling back to OCR probe.",
                     conf, config.osd_min_confidence)
        return None
    LOGGER.debug("OSD: rotate %d deg CW (confidence %.2f).", angle, conf)
    return angle


def detect_upright_rotation(
    image: np.ndarray,
    ocr: "OcrEngine",
    config: PipelineConfig,
) -> Tuple[int, Dict[int, float]]:
    """Find the clockwise rotation that makes ``image`` read upright.

    Two strategies, selected by ``config.orientation_method``:

    * **OSD** (fast) — one Tesseract OSD pass reads the angle directly
      (:func:`_osd_rotation`).  Used first for "osd" and "auto".
    * **OCR probe** (fallback) — each candidate right-angle rotation is OCR'd (on
      a size-capped copy) and scored by :func:`_text_orientation_score`; the
      best-scoring orientation wins.  Upright (0 deg) is only abandoned when
      another orientation beats it by ``config.rotate_min_gain`` — a safeguard so
      an already-upright image is never rotated on a near-tie.

    Returns ``(clockwise_correction, per_angle_scores)``; the scores dict is
    populated only when the OCR probe runs (OSD reports no per-angle scores).
    """
    if not config.auto_rotate:
        return 0, {}

    method = getattr(config, "orientation_method", "auto")
    if method in ("osd", "auto"):
        angle = _osd_rotation(image, config)
        if angle is not None:
            LOGGER.info("Orientation via Tesseract OSD: rotate %d deg CW.", angle)
            return angle, {}
        if method == "osd":
            LOGGER.info("OSD gave no confident angle — keeping upright.")
            return 0, {}
        LOGGER.info("OSD inconclusive — falling back to the 4-way OCR probe.")

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
