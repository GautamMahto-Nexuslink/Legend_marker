"""Visualization — annotated debug images for legend / map / raw stages."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .containers import Detection, OcrText
from .deps import cv2


def _scaled_font(image: np.ndarray) -> Tuple[float, int]:
    """Pick a font scale / thickness proportional to the image size.

    Keeps annotations legible on both small legend crops and large maps.
    """
    diag = float(np.hypot(image.shape[0], image.shape[1]))
    scale = max(0.4, min(1.4, diag / 1600.0))
    thickness = max(1, int(round(scale * 2)))
    return scale, thickness


def draw_label(
    image: np.ndarray,
    bbox: Sequence[int],
    label: str,
    color: Tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    """Draw one box + a filled label chip (with contrasting text) in-place."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if not label:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Place the chip above the box, or below it if there is no room at the top.
    ty = y1 - baseline - 3
    if ty - th < 0:
        ty = y2 + th + baseline + 3
    chip_top = ty - th - baseline
    chip_bottom = ty + baseline
    x_right = min(x1 + tw + 6, image.shape[1] - 1)
    cv2.rectangle(image, (x1, chip_top), (x_right, chip_bottom), color, -1)

    # White text on dark chips, black on light chips (luminance heuristic).
    lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    text_color = (0, 0, 0) if lum > 140 else (255, 255, 255)
    cv2.putText(image, label, (x1 + 3, ty - 2), font, font_scale,
                text_color, thickness, cv2.LINE_AA)


def visualize_legend(
    image: np.ndarray,
    icons: List["Detection"],
    icon_text: Dict[int, Optional["OcrText"]],
    names: Dict[int, str],
) -> np.ndarray:
    """Annotate the legend: icon boxes (blue), matched text boxes (green),
    and a connector line between an icon and the text assigned to it."""
    canvas = image.copy()
    font_scale, thickness = _scaled_font(canvas)
    icon_color = (255, 128, 0)   # BGR: blue-ish for icons.
    text_color = (0, 180, 0)     # green for matched text.

    for idx, icon in enumerate(icons):
        draw_label(canvas, icon.bbox, names.get(idx, ""), icon_color,
                   font_scale, thickness)
        matched = icon_text.get(idx)
        if matched is not None:
            cv2.rectangle(canvas,
                          (matched.bbox[0], matched.bbox[1]),
                          (matched.bbox[2], matched.bbox[3]),
                          text_color, max(1, thickness - 1))
            # Connector from icon centre to text centre.
            ic = tuple(int(v) for v in icon.center)
            tc = tuple(int(v) for v in matched.center)
            cv2.line(canvas, ic, tc, text_color, 1, cv2.LINE_AA)
    return canvas


def visualize_map(
    image: np.ndarray,
    detections: List["Detection"],
    results: List[Dict[str, Any]],
) -> np.ndarray:
    """Annotate the map with a colour per match method:

    * blue   — renamed via the known-icon pHash database (JSON)
    * green  — renamed via legend matching
    * orange — kept original Roboflow class (no confident match)

    The label shows the final class plus the match score when renamed via the
    legend.
    """
    canvas = image.copy()
    font_scale, thickness = _scaled_font(canvas)
    phash_db_color = (255, 0, 0)    # blue (BGR) — matched via the pHash DB (JSON).
    renamed_color = (0, 170, 0)     # green — successfully matched to legend.
    kept_color = (0, 140, 255)      # orange — kept original Roboflow class.

    for det, res in zip(detections, results):
        renamed = res.get("renamed", False)
        method = res.get("match_method")
        if method == "phash_db":
            color = phash_db_color
        elif renamed:
            color = renamed_color
        else:
            color = kept_color

        label = res.get("class", det.class_name)
        score = res.get("match_score")
        # For legend matches show the score; pHash-DB matches are exact-ish so
        # the score is not meaningful for them.
        if method == "legend" and score is not None:
            label = f"{label} ({score:.2f})"
        draw_label(canvas, det.bbox, label, color, font_scale, thickness)
    return canvas


def visualize_detections(
    image: np.ndarray,
    detections: List["Detection"],
    color: Tuple[int, int, int] = (0, 128, 255),
) -> np.ndarray:
    """Draw the RAW Roboflow detections exactly as returned by the model.

    Every box is labelled with its detected class and confidence — no
    filtering, matching or renaming.  Useful for sanity-checking what the
    detector actually saw on the legend and the original map.
    """
    canvas = image.copy()
    font_scale, thickness = _scaled_font(canvas)
    for det in detections:
        label = f"{det.class_name} {det.confidence:.2f}"
        draw_label(canvas, det.bbox, label, color, font_scale, thickness)
    return canvas


def visualize_ocr_text(
    image: np.ndarray,
    texts: List["OcrText"],
    color: Tuple[int, int, int] = (200, 0, 160),
) -> np.ndarray:
    """Draw ONLY the OCR text boxes + their recognised strings.

    Lets you verify what the OCR engine read (and where) independently of the
    icon detections — the other half of the icon<->text matching input.
    """
    canvas = image.copy()
    font_scale, thickness = _scaled_font(canvas)
    for t in texts:
        label = f"{t.text} {t.confidence:.2f}"
        draw_label(canvas, t.bbox, label, color, font_scale, thickness)
    return canvas
