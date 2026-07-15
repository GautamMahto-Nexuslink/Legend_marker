#!/usr/bin/env python3
"""
legend_marker.py
================

End-to-end pipeline that replaces generic icon class names detected on a full
map image with their *real* names, taken from the map's legend.

High-level flow
---------------
1.  Run a Roboflow model on the cropped **legend** image  -> legend icon detections.
2.  OCR the legend image                                  -> text + boxes.
3.  Spatially match each legend icon to its nearest text  -> icon -> name.
4.  Build a background-free "glyph" signature (segmented   -> signature DB
    symbol + ORB descriptors; pHash kept for reporting) for
    every legend icon and remember the OCR-derived name.
5.  Run the same Roboflow model on the full **map** image -> map icon detections.
6.  Match every map glyph against the legend glyphs with a  -> renamed detections.
    multi-scale template correlation + ORB score, renaming
    only when the best beats an absolute floor AND the runner-up (margin gate).

The script is intentionally modular: every stage is a small, independently
testable function, thresholds/weights live in a single dataclass, and there are
no hard-coded paths.

Author: (generated) — production-ready reference implementation.


cli eg:

python legend_marker.py --map 'Input_image_path' --legend 'legend_sub_part' --api-key xx-xx-xx --project xxx-xxx --version 1 --output-dir 'Output_of_folder' -v(verbose)

python3 legend_marker.py --map /home/nls34/Work/OuterMap/Main_Dataset/Icons_Dataset/map.coco/train/AhjumawiLavaSpringsStatePark_page-0004_jpgrfSmY3DOtP6Zazv8jbmCdK.jpg  --legend /home/nls34/Documents/POCs/legend_marker/legend/AhjumawiLavaSpringsStatePark_page-0004_jpgSmY3DOtP6Zazv8jbmCdK.jpg  --api-key K06rVQD1zQ46eOFObJvi --project plotmymap-icon-lqf56 --version 1 --output-dir output/ahujawani_easyocr_new_6_updated -v
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional third-party dependencies.
#
# We import them lazily/defensively so that the module can at least be imported
# (and `--help` shown) even if an optional engine is missing.  Each import error
# is turned into a clear, actionable message at the point of use.
# ---------------------------------------------------------------------------
try:
    import cv2  # OpenCV — image IO, colour, ORB, resizing.
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from PIL import Image  # Pillow — required by imagehash.
except ImportError:  # pragma: no cover
    Image = None  # type: ignore

try:
    import imagehash  # Perceptual hashing (pHash / dHash / …) — reporting only.
except ImportError:  # pragma: no cover
    imagehash = None  # type: ignore


LOGGER = logging.getLogger("legend_marker")


# ===========================================================================
# Configuration
# ===========================================================================
@dataclass
class PipelineConfig:
    """All tunable knobs of the pipeline live here (nothing hard-coded).

    Distances/weights are documented inline.  A single object is threaded
    through the whole pipeline so behaviour is reproducible and easy to log.
    """

    # ---- Roboflow -------------------------------------------------------
    api_key: str = ""
    workspace: str = ""
    project: str = ""
    version: int = 1
    # Roboflow serverless inference endpoint (overridable for self-hosted).
    api_url: str = "https://detect.roboflow.com"
    # Minimum detection confidence (percent, Roboflow convention: 0-100).
    # rf_confidence: float = 40.0
    rf_confidence: float = 25.0
    # Non-max-suppression overlap (percent).
    rf_overlap: float = 30.0

    # ---- OCR ------------------------------------------------------------
    ocr_engine: str = "tesseract"        # "tesseract" | "easyocr" | "paddleocr"
    ocr_languages: Tuple[str, ...] = ("en",)
    ocr_gpu: bool = False
    # Tesseract page-segmentation config.  --psm 6 = "uniform block of text",
    # which works well for the tidy rows/columns of a map legend.
    tesseract_config: str = "--oem 3 --psm 6"
    tesseract_lang: str = "eng"
    ocr_min_confidence: float = 0.30      # Drop very low-confidence text.
    # Keep only tokens that look like real words: at least this many alphabetic
    # characters.  Drops pure numbers ("4", "0.91") and symbols ("=", "@", "#").
    text_min_letters: int = 2
    # Legend crops are often tiny (e.g. 105x388), where the text is only a few
    # pixels tall and OCR fails.  Upscale small legends before OCR so the text
    # is large enough to read; OCR boxes are mapped back to original coords.
    ocr_upscale: bool = True
    ocr_target_long_side: int = 1600      # upscale until the long side hits this
    ocr_max_upscale: float = 6.0          # never enlarge more than this factor
    # Paint the detected icon boxes out of the image BEFORE OCR so the icon's
    # own glyph (e.g. the "H<tent>B" symbol misread as "HAE") can never be read
    # as text and glued onto the neighbouring label.  Legends put the label in a
    # column to the RIGHT of the icon, so blanking the icon box leaves the real
    # label untouched.  A small inward margin avoids clipping any label pixels
    # that abut the icon box.
    mask_icons_for_ocr: bool = True
    icon_mask_shrink: int = 1             # px eroded from each icon box edge

    # ---- Icon <-> text spatial matching --------------------------------
    # A text box is only considered as a label for an icon if its centre lies
    # within these gates (expressed as multiples of the icon's own size).
    text_max_horizontal_gap_factor: float = 4.0   # x-gap <= factor * icon_w
    text_max_vertical_offset_factor: float = 1.2   # |y-align| <= factor * icon_h
    # A text token is on the icon's row when their vertical spans overlap by at
    # least this fraction of the shorter box.  Overlap is far more robust than
    # comparing centres when the icon and text boxes differ in height.
    row_vertical_overlap: float = 0.30
    # When merging the tokens of one label, only join tokens whose horizontal
    # gap is at most this multiple of the text height — i.e. words that belong
    # together.  A larger gap means a separate label (e.g. the next column).
    max_word_gap_factor: float = 2.0
    # We prefer text to the RIGHT of the icon (typical legend layout) but also
    # allow left / below as fall-backs with a penalty.
    prefer_right_of_icon: bool = True
    # If this fraction of a detection's area lies inside an OCR text box, the
    # detection is treated as text (not an icon) and dropped from the legend.
    text_containment_threshold: float = 0.6
    # If this fraction of an OCR text box lies inside an icon detection, the
    # text is treated as the icon's glyph misread as text (e.g. "P", "=", "#")
    # and dropped, so it can't be mistaken for the icon's real label.
    text_on_icon_threshold: float = 0.5

    # ---- Foreground / glyph normalisation ------------------------------
    # Icons are rendered symbols on a solid or textured background.  Matching
    # the *glyph* (not the background) is the key to reliable results, so every
    # crop is segmented to its foreground and resized to a canonical square.
    glyph_size: int = 64                   # canonical template size (px).
    seg_bg_tolerance: int = 28             # colour distance from bg => foreground.
    seg_min_fg_ratio: float = 0.02         # too little fg -> use the whole crop.

    # ---- Perceptual hashing (kept for reporting / Hamming .txt) --------
    hash_algorithm: str = "phash"          # "phash" | "dhash" | "ahash" | "whash"
    hash_size: int = 16                    # 16 -> 256-bit hash (max Hamming 256).

    # ---- Classical matching (multi-scale template + ORB) ---------------
    # Decision score = weighted mix of a scale-swept normalised template
    # correlation and an ORB ratio-test inlier fraction — both computed on the
    # background-free glyph, so terrain/badge colour cannot dominate.
    template_scales: Tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.25)
    template_search_pad: int = 6           # px slack so the template can slide.
    orb_ratio: float = 0.75                # Lowe ratio-test threshold.
    w_template: float = 0.75               # weight of template correlation.
    w_orb: float = 0.25                    # weight of ORB inlier fraction.

    # ---- Rename decision -----------------------------------------------
    # A map icon is renamed to a legend name ONLY when the best match clears an
    # absolute floor AND clearly beats the runner-up (the margin gate).  This
    # stops "least-bad" wrong replacements when nothing really matches.
    match_score_threshold: float = 0.60    # absolute floor.
    match_margin: float = 0.08             # best must exceed 2nd-best by this.

    # ---- Output ---------------------------------------------------------
    output_dir: str = "output"
    save_crops: bool = True
    save_debug_json: bool = True
    save_visualization: bool = True   # Draw annotated legend/map images.


# ===========================================================================
# Data containers
# ===========================================================================
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


# ===========================================================================
# Utility helpers
# ===========================================================================
def setup_logging(verbose: bool = False) -> None:
    """Configure root logging once, with a compact, timestamped format."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _require(module: Any, name: str, install_hint: str) -> None:
    """Raise a clear error if an optional dependency is missing."""
    if module is None:
        raise RuntimeError(
            f"The '{name}' library is required for this step but is not "
            f"installed. Install it with: {install_hint}"
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


# ===========================================================================
# Visualization
# ===========================================================================
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
    """Annotate the map: green box for renamed icons, orange for kept-as-is.

    The label shows the final class plus the match score when renamed.
    """
    canvas = image.copy()
    font_scale, thickness = _scaled_font(canvas)
    renamed_color = (0, 170, 0)    # green — successfully matched to legend.
    kept_color = (0, 140, 255)     # orange — kept original Roboflow class.

    for det, res in zip(detections, results):
        renamed = res.get("renamed", False)
        color = renamed_color if renamed else kept_color
        label = res.get("class", det.class_name)
        score = res.get("match_score")
        if renamed and score is not None:
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


# ===========================================================================
# Per-crop Hamming-distance report (.txt)
# ===========================================================================
def write_hamming_info(
    txt_path: str,
    *,
    title: str,
    bbox: Sequence[int],
    confidence: float,
    phash_hex: Optional[str],
    hash_size: int,
    rows: List[Dict[str, Any]],
    footer_lines: Sequence[str] = (),
) -> None:
    """Write a human-readable .txt describing one crop's Hamming distances.

    ``rows`` is a ranked list (nearest first) of dicts with keys
    ``name``, ``hamming``, ``phash_similarity`` and ``score`` — i.e. the output
    of :meth:`SignatureMatcher.rank`.  ``footer_lines`` carries the final
    decision (best match / rename verdict) appended verbatim at the bottom.
    """
    n_bits = hash_size * hash_size
    lines: List[str] = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append(f"Bounding box (x1,y1,x2,y2): {list(bbox)}")
    lines.append(f"Detection confidence      : {confidence:.4f}")
    lines.append(f"pHash (hex)               : {phash_hex}")
    lines.append(
        f"Hash size                 : {hash_size}x{hash_size} = {n_bits} bits "
        f"(max possible Hamming distance = {n_bits})"
    )
    lines.append("")
    lines.append("Hamming distance to each legend icon (nearest first):")
    header = f"  {'legend name':<34}{'hamming':>9}{'hash_sim':>10}{'weighted':>10}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for r in rows:
        h = r.get("hamming")
        hs = r.get("phash_similarity")
        sc = r.get("score", 0.0)
        h_str = str(h) if h is not None else "n/a"
        hs_str = f"{hs:.3f}" if hs is not None else "n/a"
        name = str(r.get("name", ""))[:34]
        lines.append(f"  {name:<34}{h_str:>9}{hs_str:>10}{sc:>10.3f}")
    if footer_lines:
        lines.append("")
        lines.extend(footer_lines)
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ===========================================================================
# Step 1 & 5: Roboflow inference
# ===========================================================================
class RoboflowDetector:
    """Wrapper around local Roboflow inference via the ``inference`` package.

    Uses ``inference.get_model`` to load the model in-process (no HTTP API), then
    calls ``model.infer(...)``.  The response is normalised into a plain dict
    with a ``predictions`` list so the rest of the pipeline is model-agnostic.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._model = self._load_model()

    def _load_model(self) -> Any:
        """Load the model with ``inference.get_model`` (raises if unavailable)."""
        try:
            from inference import get_model  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "The 'inference' package is required. Install it with: "
                "pip install inference"
            ) from exc

        LOGGER.info("Loading Roboflow model '%s' via inference.get_model.",
                    self._model_id)
        # get_model resolves & caches weights locally; api_key authorises the pull.
        return get_model(model_id=self._model_id, api_key=self.config.api_key)

    @property
    def _model_id(self) -> str:
        """Roboflow model id in the form 'project/version'."""
        return f"{self.config.project}/{self.config.version}"

    def infer(self, image_path: str) -> Dict[str, Any]:
        """Run local inference and return a normalised ``{'predictions': [...]}`` dict."""
        LOGGER.info("Running Roboflow inference (local get_model) on %s", image_path)
        # inference models accept a path/np.ndarray/PIL/URL. Confidence is a
        # 0-1 fraction here (unlike the 0-100 percent HTTP API convention).
        raw = self._model.infer(
            image_path,
            confidence=self.config.rf_confidence / 100.0,
        )
        return self._normalize_response(raw)

    @staticmethod
    def _normalize_response(raw: Any) -> Dict[str, Any]:
        """Coerce an inference-package response into a plain predictions dict.

        ``model.infer`` returns either a single response object or a list of
        them (one per input image).  Each may be a pydantic model or a dict; we
        handle all shapes and always emit ``{"predictions": [ {...}, ... ]}``.
        """
        # Unwrap a per-image list to its first element (we infer one image).
        if isinstance(raw, (list, tuple)):
            raw = raw[0] if raw else {}

        # Pydantic model -> dict (support both v1 .dict() and v2 .model_dump()).
        if hasattr(raw, "model_dump"):
            data = raw.model_dump()
        elif hasattr(raw, "dict"):
            data = raw.dict()
        elif isinstance(raw, dict):
            data = raw
        else:
            data = {"predictions": getattr(raw, "predictions", []) or []}

        preds = data.get("predictions", []) or []
        normalized: List[Dict[str, Any]] = []
        for p in preds:
            # Each prediction may itself be a model or a dict.
            if hasattr(p, "model_dump"):
                p = p.model_dump()
            elif hasattr(p, "dict"):
                p = p.dict()
            elif not isinstance(p, dict):
                p = {
                    "x": getattr(p, "x", None),
                    "y": getattr(p, "y", None),
                    "width": getattr(p, "width", None),
                    "height": getattr(p, "height", None),
                    "confidence": getattr(p, "confidence", 0.0),
                    "class": getattr(p, "class_name", getattr(p, "class", "icon")),
                    "points": getattr(p, "points", None),
                }
            # The inference package names the label 'class_name'; the rest of the
            # pipeline expects 'class'. Bridge the two without losing either.
            if "class" not in p and "class_name" in p:
                p["class"] = p["class_name"]
            normalized.append(p)

        data["predictions"] = normalized
        return data

    def detect(
        self,
        image_path: str,
        image: np.ndarray,
        raw_dump_path: Optional[str] = None,
    ) -> List[Detection]:
        """Run inference and convert the response into Detection objects (+crops).

        If ``raw_dump_path`` is given, the *unmodified* Roboflow JSON response is
        written there verbatim, so the raw inference output is always auditable.
        """
        raw = self.infer(image_path)

        if raw_dump_path is not None:
            try:
                with open(raw_dump_path, "w", encoding="utf-8") as fh:
                    json.dump(raw, fh, indent=2, ensure_ascii=False, default=str)
                LOGGER.info("Saved raw Roboflow response -> %s", raw_dump_path)
            except Exception as exc:  # Saving must never break the pipeline.
                LOGGER.warning("Could not save raw response to %s: %s",
                               raw_dump_path, exc)

        predictions = raw.get("predictions", []) or []
        LOGGER.info("Roboflow returned %d prediction(s).", len(predictions))

        detections: List[Detection] = []
        for pred in predictions:
            try:
                det = self._parse_prediction(pred, image)
            except Exception as exc:  # Never let one bad box kill the batch.
                LOGGER.warning("Skipping malformed prediction: %s (%s)", pred, exc)
                continue
            detections.append(det)
        return detections

    @staticmethod
    def _parse_prediction(pred: Dict[str, Any], image: np.ndarray) -> Detection:
        """Roboflow uses centre-x/centre-y/width/height in pixels."""
        cx = float(pred["x"])
        cy = float(pred["y"])
        w = float(pred["width"])
        h = float(pred["height"])
        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2 = int(round(cx + w / 2.0))
        y2 = int(round(cy + h / 2.0))

        polygon = None
        if "points" in pred and pred["points"]:
            polygon = [(float(p["x"]), float(p["y"])) for p in pred["points"]]

        bbox = (x1, y1, x2, y2)
        crop = safe_crop(image, bbox)
        return Detection(
            class_name=str(pred.get("class", "icon")),
            confidence=float(pred.get("confidence", 0.0)),
            bbox=bbox,
            polygon=polygon,
            crop=crop,
        )


# ===========================================================================
# Step 2: OCR
# ===========================================================================
class OcrEngine:
    """OCR wrapper (Tesseract / EasyOCR / PaddleOCR) yielding cleaned OcrText.

    Tesseract is the default: on the small, printed text of map legends it
    consistently reads far more real labels than EasyOCR (which tends to return
    icon-glyph garbage on low-resolution crops).
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._reader = self._build_reader()

    def _build_reader(self) -> Any:
        engine = self.config.ocr_engine.lower()
        if engine == "tesseract":
            return self._build_tesseract()
        if engine == "easyocr":
            return self._build_easyocr()
        if engine == "paddleocr":
            return self._build_paddleocr()
        raise ValueError(f"Unknown OCR engine: {self.config.ocr_engine!r}")

    def _build_tesseract(self) -> Any:
        try:
            import pytesseract  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pytesseract not installed. Install with: pip install pytesseract "
                "(and the tesseract binary: apt-get install tesseract-ocr)"
            ) from exc
        # Fail early with a clear message if the binary itself is missing.
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:
            raise RuntimeError(
                "The tesseract binary was not found. Install it, e.g. "
                "`sudo apt-get install tesseract-ocr`."
            ) from exc
        LOGGER.info("Using Tesseract OCR (lang=%s, config=%r).",
                    self.config.tesseract_lang, self.config.tesseract_config)
        return pytesseract

    def _build_easyocr(self) -> Any:
        try:
            import easyocr  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "easyocr not installed. Install with: pip install easyocr"
            ) from exc
        LOGGER.info("Initialising EasyOCR (langs=%s, gpu=%s)",
                    self.config.ocr_languages, self.config.ocr_gpu)
        return easyocr.Reader(list(self.config.ocr_languages), gpu=self.config.ocr_gpu)

    def _build_paddleocr(self) -> Any:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "paddleocr not installed. Install with: pip install paddleocr"
            ) from exc
        LOGGER.info("Initialising PaddleOCR.")
        lang = self.config.ocr_languages[0] if self.config.ocr_languages else "en"
        return PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    # -- Text cleaning ----------------------------------------------------
    @staticmethod
    def _clean_text(text: str) -> str:
        """Collapse whitespace and strip stray control characters/noise."""
        # Normalise all whitespace runs to a single space.
        cleaned = " ".join(str(text).split())
        # Remove characters that are almost always OCR noise at token edges.
        return cleaned.strip(" .:;|_-[](){}<>*=+~^`\"'").strip()

    @staticmethod
    def _is_real_word(text: str, min_letters: int) -> bool:
        """True if the token is genuine text (enough letters), not a number/symbol.

        Rejects pure numbers ("4", "0.91"), punctuation/symbols ("=", "@", "#")
        and icon-glyph noise, while keeping any real label (which always has
        several letters, even things like "Boat Docks 8 a.m.").
        """
        return sum(ch.isalpha() for ch in text) >= min_letters

    # -- Inference --------------------------------------------------------
    def _upscale_for_ocr(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Enlarge a small legend so its text is big enough for OCR.

        Returns the (possibly upscaled) image and the scale factor applied, so
        detected boxes can be divided back to the original coordinate system.
        """
        if not self.config.ocr_upscale:
            return image, 1.0
        h, w = image.shape[:2]
        long_side = max(h, w)
        if long_side >= self.config.ocr_target_long_side:
            return image, 1.0
        scale = min(self.config.ocr_max_upscale,
                    self.config.ocr_target_long_side / float(long_side))
        if scale <= 1.01:
            return image, 1.0
        up = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))),
                        interpolation=cv2.INTER_CUBIC)
        LOGGER.info("Upscaled legend %dx%d -> %dx%d (x%.2f) for OCR.",
                    w, h, up.shape[1], up.shape[0], scale)
        return up, scale

    def read(self, image: np.ndarray) -> List[OcrText]:
        """Run OCR and return cleaned, spatially-aware text tokens."""
        proc, scale = self._upscale_for_ocr(image)
        engine = self.config.ocr_engine.lower()
        if engine == "tesseract":
            raw = self._read_tesseract(proc)
        elif engine == "easyocr":
            raw = self._read_easyocr(proc)
        else:
            raw = self._read_paddleocr(proc)

        inv = 1.0 / scale if scale else 1.0
        results: List[OcrText] = []
        for text, conf, box in raw:
            # Map the box from the upscaled image back to original coordinates.
            if scale != 1.0:
                box = (int(round(box[0] * inv)), int(round(box[1] * inv)),
                       int(round(box[2] * inv)), int(round(box[3] * inv)))
            cleaned = self._clean_text(text)
            if not cleaned:
                continue
            if conf < self.config.ocr_min_confidence:
                LOGGER.debug("Dropping low-conf OCR %r (%.2f)", cleaned, conf)
                continue
            # Keep only real words — drop pure numbers / special characters.
            if not self._is_real_word(cleaned, self.config.text_min_letters):
                LOGGER.debug("Dropping non-word OCR %r (number/symbol).", cleaned)
                continue
            results.append(OcrText(text=cleaned, confidence=float(conf), bbox=box))
        LOGGER.info("OCR produced %d cleaned text token(s).", len(results))
        return results

    def _read_tesseract(
        self, image: np.ndarray
    ) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Phrase-level tokens from Tesseract (best for small printed legends).

        Tesseract returns words; we group them back into phrases per text line,
        splitting a line wherever there is a big horizontal gap (a gap marks a
        column boundary or the jump from an icon glyph to its label).  So a
        label like "Picnic Area" stays one token, while single-character junk
        Tesseract reads over an icon becomes its own token and is filtered out.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        data = self._reader.image_to_data(
            gray,
            lang=self.config.tesseract_lang,
            config=self.config.tesseract_config,
            output_type=self._reader.Output.DICT,
        )

        # Collect words grouped by Tesseract's (block, paragraph, line) index.
        lines: Dict[Tuple[int, int, int], List[Tuple[str, float, int, int, int, int]]] = {}
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if not text or conf < 0:
                continue
            # Drop pure symbol/number words (icon-glyph noise like "$", "|", "=")
            # BEFORE grouping, so they never get merged into a real label phrase.
            if not any(ch.isalpha() for ch in text):
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            x, y = int(data["left"][i]), int(data["top"][i])
            w, h = int(data["width"][i]), int(data["height"][i])
            lines.setdefault(key, []).append((text, conf, x, y, x + w, y + h))

        out: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
        for words in lines.values():
            words.sort(key=lambda t: t[2])            # left-to-right.
            med_h = float(np.median([w[5] - w[3] for w in words])) or 12.0
            gap_thresh = 1.5 * med_h                  # bigger gap => new phrase.
            run: List[Tuple[str, float, int, int, int, int]] = [words[0]]
            for wd in words[1:]:
                if wd[2] - run[-1][4] <= gap_thresh:
                    run.append(wd)
                else:
                    out.append(self._merge_word_run(run))
                    run = [wd]
            out.append(self._merge_word_run(run))
        return out

    @staticmethod
    def _merge_word_run(
        run: List[Tuple[str, float, int, int, int, int]]
    ) -> Tuple[str, float, Tuple[int, int, int, int]]:
        """Join a run of words into one phrase token (union box, mean conf)."""
        text = " ".join(w[0] for w in run)
        conf = float(np.mean([w[1] for w in run])) / 100.0   # 0-100 -> 0-1.
        x1 = min(w[2] for w in run)
        y1 = min(w[3] for w in run)
        x2 = max(w[4] for w in run)
        y2 = max(w[5] for w in run)
        return text, conf, (x1, y1, x2, y2)

    def _read_easyocr(
        self, image: np.ndarray
    ) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        # EasyOCR returns [ [box(4 pts)], text, confidence ].
        # Keep grouping tight: EasyOCR's default width_ths (0.5) merges
        # horizontally-close boxes, which glues an icon's glyph (e.g. the
        # "H<tent>B" symbol misread as "HAE") onto its neighbouring label
        # ("Hike & Bike Campground"). Once merged, filter_text_on_icons can't
        # strip the glyph because the combined box sits mostly OFF the icon.
        # Emitting finer boxes lets the glyph become its own token that the
        # on-icon filter drops; a label's real words are re-joined later by
        # _contiguous_tokens / _merge_texts, so tighter splitting is safe.
        detections = self._reader.readtext(image, width_ths=0.1, paragraph=False)
        out = []
        for box, text, conf in detections:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            out.append((text, float(conf), bbox))
        return out

    def _read_paddleocr(
        self, image: np.ndarray
    ) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        # PaddleOCR returns [[ [box(4pts)], (text, conf) ], ...] per image.
        result = self._reader.ocr(image, cls=True)
        out = []
        # result is a list (one entry per image); guard for None/empty.
        for page in result or []:
            for line in page or []:
                box, (text, conf) = line
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                out.append((text, float(conf), bbox))
        return out


# ===========================================================================
# Step 3: Spatially match icons to OCR text
# ===========================================================================
def _fraction_inside(inner: Sequence[int], outer: Sequence[int]) -> float:
    """Fraction of the ``inner`` box's area that lies within the ``outer`` box."""
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inter_w = max(0, min(ix2, ox2) - max(ix1, ox1))
    inter_h = max(0, min(iy2, oy2) - max(iy1, oy1))
    inter = inter_w * inter_h
    inner_area = max(1, (ix2 - ix1) * (iy2 - iy1))
    return inter / float(inner_area)


def mask_icons_in_image(
    image: np.ndarray,
    icons: List[Detection],
    shrink: int = 1,
) -> np.ndarray:
    """Return a copy of ``image`` with every icon box painted over.

    Legends lay the label out in a column to the RIGHT of the icon, so blanking
    the icon's own box removes the glyph (which OCR would otherwise misread as
    text, e.g. "HAE", and glue onto the label) while leaving the label intact.
    The box is shrunk a couple of pixels so a label glyph touching the icon's
    edge is not clipped.  The fill is the image's median border colour so the
    patch blends into the legend background (usually white) instead of adding a
    hard-edged rectangle that the detector could latch onto.
    """
    if not icons:
        return image
    masked = image.copy()
    h, w = masked.shape[:2]
    # Median colour of the image border ~ the legend background.
    if masked.ndim == 3:
        border = np.concatenate([
            masked[0, :, :], masked[-1, :, :], masked[:, 0, :], masked[:, -1, :]
        ])
        fill = tuple(int(c) for c in np.median(border, axis=0))
    else:
        border = np.concatenate([
            masked[0, :], masked[-1, :], masked[:, 0], masked[:, -1]
        ])
        fill = int(np.median(border))
    for icon in icons:
        x1, y1, x2, y2 = icon.bbox
        x1 = max(0, int(x1) + shrink)
        y1 = max(0, int(y1) + shrink)
        x2 = min(w, int(x2) - shrink)
        y2 = min(h, int(y2) - shrink)
        if x2 > x1 and y2 > y1:
            masked[y1:y2, x1:x2] = fill
    return masked


def detection_inside_text(
    detection: Detection,
    texts: List[OcrText],
    config: PipelineConfig,
) -> bool:
    """True if the detection sits (mostly) inside any OCR text box.

    Roboflow occasionally boxes a text label itself (e.g. the "Legend" title)
    as an icon.  Such a detection is largely contained by a text box, so we
    drop it rather than treat it as a real legend icon.
    """
    for text in texts:
        if _fraction_inside(detection.bbox, text.bbox) >= config.text_containment_threshold:
            return True
    return False


def filter_text_on_icons(
    texts: List[OcrText],
    icons: List[Detection],
    config: PipelineConfig,
) -> List[OcrText]:
    """Drop OCR tokens that sit on top of an icon (the glyph read as text).

    OCR frequently "reads" an icon's symbol as spurious characters ("P", "=",
    "#", "4", ...).  Those boxes overlap the icon almost entirely, so a
    nearest-neighbour matcher would grab them instead of the real label to the
    right.  We remove any text token whose box is largely inside an icon box.
    """
    kept: List[OcrText] = []
    for text in texts:
        on_icon = False
        for icon in icons:
            # Two overlap directions catch both shapes of glyph-as-text:
            #  - a small glyph box mostly INSIDE the icon, and
            #  - a tall/wide glyph box that CONTAINS the icon.
            text_in_icon = _fraction_inside(text.bbox, icon.bbox)
            icon_in_text = _fraction_inside(icon.bbox, text.bbox)
            if (text_in_icon >= config.text_on_icon_threshold
                    or icon_in_text >= config.text_on_icon_threshold):
                on_icon = True
                break
        if on_icon:
            LOGGER.debug("Dropping icon-glyph text %r (sits on an icon).", text.text)
        else:
            kept.append(text)
    dropped = len(texts) - len(kept)
    if dropped:
        LOGGER.info("Dropped %d OCR token(s) sitting on icons (glyph-as-text).",
                    dropped)
    return kept


def _vertical_overlap_ratio(a: Sequence[int], b: Sequence[int]) -> float:
    """Vertical overlap of two boxes as a fraction of the shorter box's height."""
    ay1, ay2 = a[1], a[3]
    by1, by2 = b[1], b[3]
    overlap = max(0, min(ay2, by2) - max(ay1, by1))
    min_h = max(1, min(ay2 - ay1, by2 - by1))
    return overlap / float(min_h)


def _same_line_tokens(
    candidates: List[OcrText],
    icon_center_y: float,
) -> List[OcrText]:
    """Return only the tokens on the icon's OWN line (a single row of text).

    We anchor on the token whose row centre is closest to the icon, then keep
    just the tokens that share that same line — their vertical centre lies
    within half a text-height of the anchor's centre.  Lines above or below
    (other legend rows, the "LEGEND" header) are deliberately excluded, so the
    label is always the single line beside the icon.
    """
    anchor = min(candidates, key=lambda t: abs(t.center[1] - icon_center_y))
    anchor_cy = anchor.center[1]
    anchor_h = max(anchor.bbox[3] - anchor.bbox[1], 1)
    # A token is "on the same line" if its centre is within half a line height.
    tol = 0.5 * anchor_h
    return [t for t in candidates if abs(t.center[1] - anchor_cy) <= tol]


def _contiguous_tokens(tokens: List[OcrText], max_gap: float) -> List[OcrText]:
    """Keep only the run of horizontally-adjacent tokens (one label's words).

    Starting from the leftmost token, walk right and stop at the first big
    horizontal gap — that gap marks the start of a *different* label (e.g. the
    next column), so words far from each other are never merged together.
    """
    if not tokens:
        return tokens
    ordered = sorted(tokens, key=lambda t: t.bbox[0])
    run = [ordered[0]]
    for t in ordered[1:]:
        gap = t.bbox[0] - run[-1].bbox[2]     # negative when boxes overlap.
        if gap <= max_gap:
            run.append(t)
        else:
            break                              # big gap -> separate label.
    return run


def _merge_texts(tokens: List[OcrText]) -> OcrText:
    """Combine same-line OCR tokens into one label (reading order + union bbox).

    Tokens are ordered left-to-right (top-to-bottom tie-break) so a label split
    into several tokens on one line (e.g. "Reservation Headquarters") reads
    naturally.  The merged bbox is the union of the boxes; confidence is mean.
    """
    ordered = sorted(tokens, key=lambda t: (t.bbox[1], t.bbox[0]))
    text = " ".join(t.text for t in ordered).strip()
    x1 = min(t.bbox[0] for t in ordered)
    y1 = min(t.bbox[1] for t in ordered)
    x2 = max(t.bbox[2] for t in ordered)
    y2 = max(t.bbox[3] for t in ordered)
    conf = float(np.mean([t.confidence for t in ordered]))
    return OcrText(text=text, confidence=conf, bbox=(x1, y1, x2, y2))


def match_icons_to_text(
    icons: List[Detection],
    texts: List[OcrText],
    config: PipelineConfig,
) -> Dict[int, Optional[OcrText]]:
    """Map each icon to its legend label by row assignment (offset-robust).

    Legends put one icon per row with its label to the right (often in several
    columns).  Real detection and OCR boxes rarely share the exact same centre,
    so we choose the icon's row by **maximum vertical overlap** (tie-break by
    nearest centre) rather than a strict alignment gate.

    For each icon:
      1. Gather text tokens to the RIGHT (within the horizontal gap) and BEFORE
         the next icon on the same row (column gate — no reaching into the next
         column's label).
      2. Pick the token whose vertical span overlaps the icon most; ties and the
         no-overlap case fall back to the nearest centre.
      3. Drop the icon only if the best candidate neither overlaps nor lies
         within ~one row spacing — i.e. it has no label of its own.
      4. Merge the tokens on the chosen row into the final label.

    (OCR tokens sitting on top of icons should already have been removed by
    ``filter_text_on_icons`` before this call.)

    Returns icon-index -> merged OcrText | None.
    """
    matches: Dict[int, Optional[OcrText]] = {}
    if not icons:
        return matches

    min_ov = config.row_vertical_overlap
    # Estimate the legend's row spacing from the distinct text-row centres, so
    # "same row" is judged relative to the actual layout — not a box-size guess.
    centers = sorted({round(t.center[1]) for t in texts})
    gaps = [b - a for a, b in zip(centers, centers[1:]) if b - a > 3]
    row_gap = float(np.median(gaps)) if gaps else 40.0
    # The label must be HORIZONTALLY ALIGNED with the icon (same row): accept a
    # non-overlapping candidate only if it is well within half a row of the
    # icon's centre.  This stops "slanted" matches to a row above/below.
    v_cap = 0.45 * row_gap

    for idx, icon in enumerate(icons):
        ix1, _iy1, ix2, _iy2 = icon.bbox
        icy = icon.center[1]
        max_h_gap = config.text_max_horizontal_gap_factor * max(icon.width, 1)

        # Column boundary: left edge of the nearest OTHER icon sharing this row
        # and lying to the right.  Text at/after this x is the next column's.
        x_limit = float("inf")
        for j, other in enumerate(icons):
            if j == idx:
                continue
            ojx1 = other.bbox[0]
            if ojx1 >= ix2 and _vertical_overlap_ratio(icon.bbox, other.bbox) >= min_ov:
                x_limit = min(x_limit, float(ojx1))

        # Collect right-side, in-column text tokens (no per-token distance gate:
        # a long label's later words are naturally far from the icon; we bound
        # only where the label STARTS, after grouping, below).
        right: List[OcrText] = []
        for text in texts:
            tx1, _ty1, tx2, _ty2 = text.bbox
            if tx1 >= ix2:                      # entirely to the right.
                pass
            elif tx2 > ix1 and tx1 < ix2:       # overlaps icon horizontally.
                pass
            elif not config.prefer_right_of_icon and tx2 <= ix1:  # left allowed.
                pass
            else:
                continue                         # to the left but right-only mode.
            if tx1 >= x_limit:                   # next column — excluded.
                continue
            right.append(text)

        if not right:
            matches[idx] = None
            LOGGER.debug("Icon %d had no text to its right.", idx)
            continue

        # Choose the row the icon actually belongs to: strongest vertical
        # overlap wins; ties (and the no-overlap case) fall back to the nearest
        # centre.  This is robust to the icon box and text box being offset.
        def _row_key(t: OcrText) -> Tuple[float, float]:
            overlap = _vertical_overlap_ratio(icon.bbox, t.bbox)
            return (-overlap, abs(t.center[1] - icy))

        best = min(right, key=_row_key)
        v_dist = abs(best.center[1] - icy)
        overlaps = _vertical_overlap_ratio(icon.bbox, best.bbox) >= min_ov

        # Drop only when the best candidate is neither overlapping nor within a
        # row of the icon — i.e. this icon has no label of its own.
        if not overlaps and v_dist > v_cap:
            matches[idx] = None
            LOGGER.debug("Icon %d: best text %r too far (v_dist=%.0f > cap=%.0f).",
                         idx, best.text, v_dist, v_cap)
            continue

        # Tokens on the chosen row, then keep only the ones near each other
        # (one label's words) — never merge across a wide gap.
        line = _same_line_tokens(right, best.center[1])
        line_heights = [t.bbox[3] - t.bbox[1] for t in line if t.bbox[3] > t.bbox[1]]
        line_h = float(np.median(line_heights)) if line_heights else 20.0
        max_word_gap = config.max_word_gap_factor * line_h
        group = _contiguous_tokens(line, max_word_gap)

        # The label must START near the icon; if even its first word is far to
        # the right, this text belongs to something else (too far), so skip.
        anchor_gap = min(t.bbox[0] for t in group) - ix2
        if anchor_gap > max_h_gap:
            matches[idx] = None
            LOGGER.debug("Icon %d: label %r starts too far (gap=%.0f > %.0f).",
                         idx, _merge_texts(group).text, anchor_gap, max_h_gap)
            continue

        merged = _merge_texts(group)
        matches[idx] = merged
        LOGGER.debug("Icon %d -> %r (%d/%d token(s), gap=%.0f, v_dist=%.0f)",
                     idx, merged.text, len(group), len(line), anchor_gap, v_dist)

    matched = sum(1 for v in matches.values() if v is not None)
    LOGGER.info("Matched %d/%d legend icon(s) to text.", matched, len(icons))
    return matches


# ===========================================================================
# Step 4: Visual signatures (background-free glyph + ORB + pHash-for-reporting)
# ===========================================================================
class SignatureBuilder:
    """Turns an icon crop into a background-free, canonically-sized *glyph*.

    Rendered map symbols are consistent up to scale/compression, but their
    surrounding pixels are not (flat badge in the legend vs. terrain on the
    map).  We therefore estimate the background from the crop border, keep only
    the pixels that differ from it (the symbol), tight-crop to that, and resize
    to a fixed square.  Every downstream feature is computed on this glyph.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        _require(cv2, "opencv-python", "pip install opencv-python")
        # ORB on a fixed glyph is deterministic (same pixels -> same output).
        self._orb = cv2.ORB_create(nfeatures=300, fastThreshold=5)

    # -- Foreground segmentation -----------------------------------------
    def _segment_glyph(self, crop: np.ndarray) -> np.ndarray:
        """Return the tight, background-free grayscale glyph at canonical size."""
        size = self.config.glyph_size
        if crop is None or crop.size == 0:
            return np.zeros((size, size), dtype=np.uint8)

        bgr = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        h, w = bgr.shape[:2]

        # 1) Estimate the background colour from a thin border ring (robust
        #    median) — the frame around a symbol is almost always background.
        b = max(1, min(h, w) // 10)
        border = np.concatenate([
            bgr[:b, :, :].reshape(-1, 3), bgr[-b:, :, :].reshape(-1, 3),
            bgr[:, :b, :].reshape(-1, 3), bgr[:, -b:, :].reshape(-1, 3),
        ], axis=0)
        bg = np.median(border, axis=0)

        # 2) Foreground = pixels whose colour distance from bg exceeds tolerance.
        dist = np.linalg.norm(bgr.astype(np.float32) - bg[None, None, :], axis=2)
        mask = (dist > self.config.seg_bg_tolerance).astype(np.uint8)

        # 3) Clean tiny speckles with a morphological opening.
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
        )

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(mask > 0)
        # If segmentation found too little, fall back to the whole crop so we
        # never return an empty glyph (better a noisy glyph than nothing).
        if xs.size < self.config.seg_min_fg_ratio * h * w:
            glyph_region = gray
        else:
            x1, x2 = xs.min(), xs.max() + 1
            y1, y2 = ys.min(), ys.max() + 1
            glyph_region = gray[y1:y2, x1:x2]
            # Blank the background inside the box to neutral grey so correlation
            # keys on the symbol, not leftover background texture.
            region_mask = mask[y1:y2, x1:x2]
            glyph_region = np.where(region_mask > 0, glyph_region, 127).astype(np.uint8)

        return self._pad_resize(glyph_region, size)

    @staticmethod
    def _pad_resize(gray: np.ndarray, size: int) -> np.ndarray:
        """Pad to square (neutral grey) then resize to size×size, keeping aspect."""
        if gray.size == 0:
            return np.zeros((size, size), dtype=np.uint8)
        gh, gw = gray.shape[:2]
        side = max(gh, gw)
        canvas = np.full((side, side), 127, dtype=np.uint8)
        y0, x0 = (side - gh) // 2, (side - gw) // 2
        canvas[y0:y0 + gh, x0:x0 + gw] = gray
        interp = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
        return cv2.resize(canvas, (size, size), interpolation=interp)

    # -- pHash (reporting only) ------------------------------------------
    def _compute_phash(self, glyph_gray: np.ndarray) -> Tuple[Any, str]:
        _require(imagehash, "imagehash", "pip install imagehash")
        _require(Image, "Pillow", "pip install pillow")
        pil = Image.fromarray(glyph_gray)
        algo = {
            "phash": imagehash.phash,
            "dhash": imagehash.dhash,
            "ahash": imagehash.average_hash,
            "whash": imagehash.whash,
        }.get(self.config.hash_algorithm, imagehash.phash)
        h = algo(pil, hash_size=self.config.hash_size)
        return h, str(h)

    # -- Public API -------------------------------------------------------
    def build(self, crop: np.ndarray) -> VisualSignature:
        """Compute the glyph and its features for a crop, tolerating failures."""
        sig = VisualSignature()

        # Aspect ratio from the ORIGINAL crop (segmentation removes it).
        if crop is not None and crop.size > 0:
            h, w = crop.shape[:2]
            sig.aspect_ratio = float(w) / float(h) if h else 1.0

        sig.glyph = self._segment_glyph(crop)

        try:
            kp, des = self._orb.detectAndCompute(sig.glyph, None)
            sig.keypoints, sig.orb_descriptors = kp, des
        except Exception as exc:
            LOGGER.warning("ORB failed: %s", exc)

        try:
            sig.phash, sig.phash_hex = self._compute_phash(sig.glyph)
        except Exception as exc:
            LOGGER.warning("pHash failed: %s", exc)

        return sig


# ===========================================================================
# Step 6: Weighted similarity + matching
# ===========================================================================
class SignatureMatcher:
    """Scores glyph similarity with multi-scale template correlation + ORB.

    Both sub-scores live in [0, 1] and are computed on the background-free
    glyph.  The decision score is their (renormalised) weighted mean, so a
    missing sub-score (e.g. ORB found no keypoints) does not drag the total.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        # knnMatch needs crossCheck OFF so we can apply Lowe's ratio test.
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # -- Template correlation (scale-swept, alignment-tolerant) ----------
    def _score_template(self, a: VisualSignature, b: VisualSignature) -> Optional[float]:
        """Best normalised cross-correlation of glyph *b* against glyph *a*.

        The query glyph is padded so the (rescaled) legend glyph can slide a few
        pixels, absorbing minor misalignment; several scales absorb size drift.
        TM_CCOEFF_NORMED is mean-subtracted, so uniform brightness shifts don't
        matter — only the symbol's structure does.
        """
        if a.glyph is None or b.glyph is None:
            return None
        pad = self.config.template_search_pad
        # Neutral-grey border keeps the correlation focused on the glyph.
        search = cv2.copyMakeBorder(a.glyph, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=127)
        sh, sw = search.shape[:2]

        best = -1.0
        for s in self.config.template_scales:
            side = max(4, int(round(self.config.glyph_size * s)))
            if side > sh or side > sw:      # template must fit inside search.
                continue
            templ = cv2.resize(b.glyph, (side, side),
                               interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            try:
                res = cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                continue
            best = max(best, float(res.max()))

        if best < 0:
            return 0.0
        return max(0.0, min(1.0, best))     # clamp; negatives => no similarity.

    # -- ORB ratio-test inlier fraction ----------------------------------
    def _score_orb(self, a: VisualSignature, b: VisualSignature) -> Optional[float]:
        da, db = a.orb_descriptors, b.orb_descriptors
        if da is None or db is None or len(da) < 2 or len(db) < 2:
            return None
        try:
            knn = self._bf.knnMatch(da, db, k=2)
        except cv2.error:
            return None
        good = 0
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            # Lowe's ratio test keeps only confidently-unique matches.
            if m.distance < self.config.orb_ratio * n.distance:
                good += 1
        denom = min(len(da), len(db))
        return good / float(denom) if denom else 0.0

    def _score_phash(self, a: VisualSignature, b: VisualSignature) -> Optional[float]:
        """pHash similarity — reporting only, not part of the decision."""
        if a.phash is None or b.phash is None:
            return None
        distance = a.phash - b.phash
        n_bits = self.config.hash_size * self.config.hash_size
        return 1.0 - min(distance, n_bits) / float(n_bits)

    # -- Combined ---------------------------------------------------------
    def score(self, a: VisualSignature, b: VisualSignature) -> Tuple[float, Dict[str, float]]:
        """Renormalised weighted mean of the template + ORB sub-scores."""
        cfg = self.config
        parts: List[Tuple[float, Optional[float]]] = [
            (cfg.w_template, self._score_template(a, b)),
            (cfg.w_orb, self._score_orb(a, b)),
        ]
        names = ["template", "orb"]

        total, weight_sum = 0.0, 0.0
        breakdown: Dict[str, float] = {}
        for (weight, value), name in zip(parts, names):
            if weight <= 0 or value is None:
                continue
            total += weight * value
            weight_sum += weight
            breakdown[name] = round(value, 4)

        combined = total / weight_sum if weight_sum > 0 else 0.0
        return combined, breakdown

    @staticmethod
    def phash_hamming(a: VisualSignature, b: VisualSignature) -> Optional[int]:
        """Raw Hamming distance between two pHashes (None if either missing)."""
        if a.phash is None or b.phash is None:
            return None
        return int(a.phash - b.phash)

    def rank(
        self,
        query: VisualSignature,
        legend_sigs: List[Tuple[str, VisualSignature]],
    ) -> List[Dict[str, Any]]:
        """Score the query against every legend entry, sorted best-first.

        Each row carries both the raw pHash Hamming distance and the combined
        weighted similarity, so callers can report/inspect either.
        """
        rows: List[Dict[str, Any]] = []
        for name, sig in legend_sigs:
            score, breakdown = self.score(query, sig)
            hamming = self.phash_hamming(query, sig)
            rows.append(
                {
                    "name": name,
                    "hamming": hamming,
                    "phash_hex": sig.phash_hex,
                    "phash_similarity": self._score_phash(query, sig),
                    "score": score,
                    "breakdown": breakdown,
                }
            )
        # Sort by weighted score desc; break ties by smaller Hamming distance.
        rows.sort(
            key=lambda r: (-r["score"],
                           r["hamming"] if r["hamming"] is not None else 10**9)
        )
        return rows

    def best_match(
        self,
        query: VisualSignature,
        legend_sigs: List[Tuple[str, VisualSignature]],
    ) -> Tuple[Optional[str], float, Dict[str, float]]:
        """Return (best_name, best_score, breakdown) over all legend entries."""
        rows = self.rank(query, legend_sigs)
        if not rows:
            return None, -1.0, {}
        top = rows[0]
        return top["name"], top["score"], top["breakdown"]


# ===========================================================================
# Orchestration
# ===========================================================================
class LegendMarkerPipeline:
    """Ties every stage together into one `run()` call."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.detector = RoboflowDetector(config)
        self.sig_builder = SignatureBuilder(config)
        self.matcher = SignatureMatcher(config)
        # OCR is heavy to init; build lazily only when the legend stage runs.
        self._ocr: Optional[OcrEngine] = None

    @property
    def ocr(self) -> OcrEngine:
        if self._ocr is None:
            self._ocr = OcrEngine(self.config)
        return self._ocr

    # -- Legend side ------------------------------------------------------
    def build_legend_database(
        self, legend_path: str
    ) -> List[Tuple[str, VisualSignature]]:
        """Steps 1-4: detect legend icons, OCR, match, sign -> name<->signature."""
        legend_img = load_image(legend_path)

        # Step 1: detect legend icons (raw Roboflow JSON saved alongside).
        raw_path = (
            os.path.join(self.config.output_dir, "legend_roboflow_raw.json")
            if self.config.save_debug_json else None
        )
        icons = self.detector.detect(legend_path, legend_img, raw_dump_path=raw_path)
        if not icons:
            LOGGER.warning("No icons detected in the legend image.")
            return []

        # Visualization: raw legend detections exactly as the model returned
        # them (before any filtering), for sanity-checking the detector.
        if self.config.save_visualization:
            raw_viz = visualize_detections(legend_img, icons)
            out_path = os.path.join(self.config.output_dir, "legend_detections_raw.png")
            cv2.imwrite(out_path, raw_viz)
            LOGGER.info("Saved raw legend detections -> %s", out_path)

        # Step 2: OCR the whole legend.  Paint the icon boxes out first so an
        # icon's glyph can never be read as text and merged into its label
        # (e.g. the "H<tent>B" symbol misread as "HAE Hike & Bike Campground").
        ocr_img = legend_img
        if self.config.mask_icons_for_ocr:
            ocr_img = mask_icons_in_image(legend_img, icons, self.config.icon_mask_shrink)
        texts = self.ocr.read(ocr_img)

        # Visualization: OCR text boxes only (what the OCR engine read + where).
        if self.config.save_visualization:
            ocr_viz = visualize_ocr_text(legend_img, texts)
            out_path = os.path.join(self.config.output_dir, "legend_ocr_text.png")
            cv2.imwrite(out_path, ocr_viz)
            LOGGER.info("Saved OCR text visualization -> %s", out_path)

        # Filter 1: drop OCR tokens sitting on an icon (the glyph read as text,
        # e.g. "P"/"="/"#"/"4").  This MUST run first: OCR often boxes the glyph
        # in a tall box that fully contains the icon, and Filter 2 below would
        # otherwise delete the icon as "inside a text box".
        texts = filter_text_on_icons(texts, icons, self.config)

        # Filter 2: drop detections that lie inside a (remaining, real) text box
        # — text regions the model mistook for icons (e.g. the "Legend" title).
        kept = [ic for ic in icons if not detection_inside_text(ic, texts, self.config)]
        dropped_inside = len(icons) - len(kept)
        if dropped_inside:
            LOGGER.info("Dropped %d detection(s) contained within text boxes.",
                        dropped_inside)
        icons = kept

        # Step 3: spatially match icon -> text.
        icon_text = match_icons_to_text(icons, texts, self.config)

        # Step 4: signatures keyed by OCR name.  We keep ONLY icons that have a
        # nearby text label; those without one are skipped entirely (no hash).
        legend_db: List[Tuple[str, VisualSignature]] = []
        crop_dir = ensure_dir(os.path.join(self.config.output_dir, "legend_crops"))

        legend_hash_dict: Dict[str, str] = {}       # spec's {hash: name} artefact.
        legend_results: List[Dict[str, Any]] = []    # structured legend final results.
        # Re-indexed containers holding only the icons we keep (for plotting).
        viz_icons: List[Detection] = []
        viz_icon_text: Dict[int, Optional[OcrText]] = {}
        names: Dict[int, str] = {}

        skipped_no_text = 0
        for idx, icon in enumerate(icons):
            matched_text = icon_text.get(idx)

            # Filter 2: no nearby text -> not a real legend entry; do NOT hash it.
            if matched_text is None:
                skipped_no_text += 1
                LOGGER.info("Skipping legend icon %d: no nearby text (no hash).", idx)
                continue

            name = matched_text.text
            icon.signature = self.sig_builder.build(icon.crop)

            kidx = len(viz_icons)         # compact index into the kept set.
            viz_icons.append(icon)
            viz_icon_text[kidx] = matched_text
            names[kidx] = name
            legend_db.append((name, icon.signature))

            crop_file = None
            if self.config.save_crops and icon.crop is not None and icon.crop.size:
                crop_file = f"{kidx:03d}_{sanitize_filename(name)}.png"
                cv2.imwrite(os.path.join(crop_dir, crop_file), icon.crop)

            if icon.signature.phash_hex:
                legend_hash_dict[icon.signature.phash_hex] = name

            # One record per kept legend icon: detection + matched OCR name + hash.
            legend_results.append(
                {
                    "index": kidx,
                    "name": name,                       # OCR-derived legend label
                    "detected_class": icon.class_name,  # raw Roboflow class
                    "confidence": round(icon.confidence, 4),
                    "bbox": list(icon.bbox),
                    "polygon": icon.polygon,
                    "hash": icon.signature.phash_hex,
                    "matched_text": matched_text.text,
                    "matched_text_bbox": list(matched_text.bbox),
                    "matched_text_confidence": round(matched_text.confidence, 4),
                    "crop_file": (
                        os.path.join("legend_crops", crop_file) if crop_file else None
                    ),
                }
            )

        LOGGER.info(
            "Built legend database with %d entries (dropped %d text-boxes, "
            "%d without nearby text).",
            len(legend_db), dropped_inside, skipped_no_text,
        )

        # Per-legend-crop info .txt: each icon's Hamming distance to the OTHER
        # legend icons (a self-distance is always 0, so it is excluded). This
        # shows how visually distinct the legend entries are from one another.
        if self.config.save_crops:
            for k, icon in enumerate(viz_icons):
                others = [(nm, sig) for j, (nm, sig) in enumerate(legend_db) if j != k]
                rows = self.matcher.rank(icon.signature, others)
                info_path = os.path.join(
                    crop_dir, f"{k:03d}_{sanitize_filename(names[k])}.txt"
                )
                footer = [
                    f"This legend icon : '{names[k]}'",
                    "Note: distances are to OTHER legend icons "
                    "(a nearest-neighbour of 0 would mean a duplicate icon).",
                ]
                write_hamming_info(
                    info_path,
                    title=f"Legend icon {k}: {names[k]}",
                    bbox=icon.bbox,
                    confidence=icon.confidence,
                    phash_hex=icon.signature.phash_hex,
                    hash_size=self.config.hash_size,
                    rows=rows,
                    footer_lines=footer,
                )

        if self.config.save_debug_json:
            self._dump_json("legend_hash_dict.json", legend_hash_dict)
            self._dump_json("legend_results.json", legend_results)

        # Visualization: only the kept icons + their matched text + links.
        if self.config.save_visualization:
            annotated = visualize_legend(legend_img, viz_icons, viz_icon_text, names)
            out_path = os.path.join(self.config.output_dir, "legend_annotated.png")
            cv2.imwrite(out_path, annotated)
            LOGGER.info("Saved legend visualization -> %s", out_path)
        return legend_db

    # -- Map side ---------------------------------------------------------
    def process_map(
        self,
        map_path: str,
        legend_db: List[Tuple[str, VisualSignature]],
    ) -> List[Dict[str, Any]]:
        """Steps 5-6: detect map icons, sign, match against legend, rename."""
        map_img = load_image(map_path)

        # Step 5: detect icons on the full map (raw Roboflow JSON saved too).
        raw_path = (
            os.path.join(self.config.output_dir, "map_roboflow_raw.json")
            if self.config.save_debug_json else None
        )
        detections = self.detector.detect(map_path, map_img, raw_dump_path=raw_path)
        if not detections:
            LOGGER.warning("No icons detected on the map image.")
            return []

        # Visualization: raw map detections exactly as the model returned them
        # (original class + confidence), before any hash matching / renaming.
        if self.config.save_visualization:
            raw_viz = visualize_detections(map_img, detections)
            out_path = os.path.join(self.config.output_dir, "map_detections_raw.png")
            cv2.imwrite(out_path, raw_viz)
            LOGGER.info("Saved raw map detections -> %s", out_path)

        crop_dir = ensure_dir(os.path.join(self.config.output_dir, "map_crops"))
        results: List[Dict[str, Any]] = []

        for idx, det in enumerate(detections):
            det.signature = self.sig_builder.build(det.crop)

            # Step 6: rank the detection against every legend signature.
            rows = self.matcher.rank(det.signature, legend_db)
            top = rows[0] if rows else None
            name = top["name"] if top else None
            score = top["score"] if top else -1.0
            breakdown = top["breakdown"] if top else {}
            best_hamming = top["hamming"] if top else None
            second_score = rows[1]["score"] if len(rows) > 1 else 0.0
            margin = score - second_score

            # Decision: rename only when the best match clears the absolute
            # floor AND clearly beats the runner-up (margin gate).  Otherwise
            # keep the original class rather than force a "least-bad" match.
            final_class = det.class_name
            renamed = False
            passes_floor = name is not None and score >= self.config.match_score_threshold
            passes_margin = (len(rows) < 2) or (margin >= self.config.match_margin)
            if passes_floor and passes_margin:
                final_class = name
                renamed = True

            crop_file = f"{idx:03d}_{sanitize_filename(final_class)}.png"
            if self.config.save_crops and det.crop is not None and det.crop.size:
                cv2.imwrite(os.path.join(crop_dir, crop_file), det.crop)

                # Per-crop report .txt right beside the image.  Note the score
                # is the template+ORB match score; the hamming column is pHash
                # (informational only).
                footer = [
                    f"Best match  : {name}  (match score={score:.3f}, "
                    f"pHash hamming={best_hamming})",
                    f"Floor gate  : score {score:.3f} >= "
                    f"{self.config.match_score_threshold} -> "
                    f"{'PASS' if passes_floor else 'FAIL'}",
                    f"Margin gate : best-2nd = {margin:.3f} >= "
                    f"{self.config.match_margin} -> "
                    f"{'PASS' if passes_margin else 'FAIL'}",
                    f"Decision    : {'RENAMED' if renamed else 'KEPT'}  "
                    f"'{det.class_name}' -> '{final_class}'",
                ]
                info_path = os.path.join(
                    crop_dir, os.path.splitext(crop_file)[0] + ".txt"
                )
                write_hamming_info(
                    info_path,
                    title=f"Map icon {idx}: {crop_file}",
                    bbox=det.bbox,
                    confidence=det.confidence,
                    phash_hex=det.signature.phash_hex,
                    hash_size=self.config.hash_size,
                    rows=rows,
                    footer_lines=footer,
                )

            results.append(
                {
                    "class": final_class,
                    "original_class": det.class_name,
                    "confidence": round(det.confidence, 4),
                    "bbox": list(det.bbox),
                    "polygon": det.polygon,
                    "hash": det.signature.phash_hex,
                    "match_score": round(score, 4) if name else None,
                    "best_hamming": best_hamming,
                    "match_breakdown": breakdown,
                    "renamed": renamed,
                }
            )
            LOGGER.info(
                "Map icon %d: '%s' -> '%s' (score=%.3f, renamed=%s)",
                idx, det.class_name, final_class, score, renamed,
            )

        renamed_count = sum(1 for r in results if r["renamed"])
        LOGGER.info("Renamed %d/%d map detections.", renamed_count, len(results))

        # Visualization: annotated map with final class labels drawn on-image.
        if self.config.save_visualization:
            annotated = visualize_map(map_img, detections, results)
            out_path = os.path.join(self.config.output_dir, "map_annotated.png")
            cv2.imwrite(out_path, annotated)
            LOGGER.info("Saved map visualization -> %s", out_path)
        return results

    # -- Full run ---------------------------------------------------------
    def run(self, map_path: str, legend_path: str) -> List[Dict[str, Any]]:
        ensure_dir(self.config.output_dir)
        LOGGER.info("=== Legend Marker pipeline started ===")
        legend_db = self.build_legend_database(legend_path)
        results = self.process_map(map_path, legend_db)
        if self.config.save_debug_json:
            self._dump_json("map_results.json", results)
        LOGGER.info("=== Pipeline finished: %d detection(s) ===", len(results))
        return results

    # -- Helpers ----------------------------------------------------------
    def _dump_json(self, filename: str, data: Any) -> None:
        path = os.path.join(self.config.output_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        LOGGER.debug("Wrote %s", path)


# ===========================================================================
# CLI
# ===========================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replace generic map-icon classes with real names from the legend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Inputs
    p.add_argument("--map", dest="map_path", required=True,
                   help="Path to the original full map image.")
    p.add_argument("--legend", dest="legend_path", required=True,
                   help="Path to the cropped legend image.")

    # Roboflow (env-var fallbacks so keys need not be on the command line).
    p.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""),
                   help="Roboflow API key (or set ROBOFLOW_API_KEY).")
    p.add_argument("--workspace", default=os.environ.get("ROBOFLOW_WORKSPACE", ""),
                   help="Roboflow workspace id (or set ROBOFLOW_WORKSPACE).")
    p.add_argument("--project", required=False,
                   default=os.environ.get("ROBOFLOW_PROJECT", ""),
                   help="Roboflow project id (or set ROBOFLOW_PROJECT).")
    p.add_argument("--version", type=int,
                   default=int(os.environ.get("ROBOFLOW_VERSION", "1")),
                   help="Roboflow model version.")
    p.add_argument("--api-url", default="https://detect.roboflow.com",
                   help="Roboflow inference endpoint.")

    # OCR
    p.add_argument("--ocr-engine", choices=["tesseract", "easyocr", "paddleocr"],
                   default="tesseract")
    p.add_argument("--ocr-gpu", action="store_true", help="Use GPU for OCR.")

    # Thresholds
    p.add_argument("--match-threshold", type=float, default=0.60,
                   help="Absolute floor: min template+ORB score to rename.")
    p.add_argument("--match-margin", type=float, default=0.08,
                   help="Best match must beat the 2nd-best by this margin.")
    p.add_argument("--hash-algorithm",
                   choices=["phash", "dhash", "ahash", "whash"], default="phash",
                   help="pHash variant used for the Hamming .txt reports.")

    # Output / misc
    p.add_argument("--output-dir", default="output",
                   help="Where crops/JSON artefacts are written.")
    p.add_argument("--no-crops", action="store_true", help="Do not save crops.")
    p.add_argument("--no-viz", action="store_true",
                   help="Do not save annotated visualization images.")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return p


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        api_key=args.api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        api_url=args.api_url,
        ocr_engine=args.ocr_engine,
        ocr_gpu=args.ocr_gpu,
        hash_algorithm=args.hash_algorithm,
        match_score_threshold=args.match_threshold,
        match_margin=args.match_margin,
        output_dir=args.output_dir,
        save_crops=not args.no_crops,
        save_visualization=not args.no_viz,
    )


def validate(args: argparse.Namespace) -> None:
    """Fail fast with actionable messages before doing any heavy work."""
    problems: List[str] = []
    if not os.path.isfile(args.map_path):
        problems.append(f"--map not found: {args.map_path}")
    if not os.path.isfile(args.legend_path):
        problems.append(f"--legend not found: {args.legend_path}")
    if not args.api_key:
        problems.append("Roboflow API key missing (--api-key / ROBOFLOW_API_KEY).")
    if not args.project:
        problems.append("Roboflow project missing (--project / ROBOFLOW_PROJECT).")
    if problems:
        for prob in problems:
            LOGGER.error(prob)
        raise SystemExit(2)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(verbose=args.verbose)
    validate(args)

    config = config_from_args(args)
    pipeline = LegendMarkerPipeline(config)

    try:
        results = pipeline.run(args.map_path, args.legend_path)
    except Exception as exc:
        LOGGER.exception("Pipeline failed: %s", exc)
        return 1

    # Print a concise summary to stdout (the full artefact is in results.json).
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())


#kaptams-workspace/plotmymap-icon-lqf56-1-yolo11x-seg-t1