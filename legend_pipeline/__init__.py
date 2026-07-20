"""legend_pipeline
================

Modular implementation of the legend-marker pipeline (formerly the single
``legend_marker.py`` script).  The code is unchanged — it is only split across
focused modules for readability and testability:

    deps            optional third-party imports (cv2/Pillow/imagehash) + LOGGER
    config          PipelineConfig (all tunable knobs)
    containers      Detection / OcrText / VisualSignature dataclasses
    utils           logging, image IO, cropping, filename helpers
    orientation     auto-rotation of sideways legends/maps
    visualization   annotated debug images
    reporting       per-crop Hamming .txt reports
    detector        RoboflowDetector (Step 1 & 5)
    ocr             OcrEngine (Step 2)
    matching        icon <-> text spatial matching (Step 3)
    signatures      SignatureBuilder + SignatureMatcher (Step 4 & 6)
    pipeline        LegendMarkerPipeline orchestration
    cli             argparse + main()

The public names re-exported here match what the old single-file module exposed,
so ``import legend_marker as lm`` keeps working unchanged.
"""
from __future__ import annotations

from .config import PipelineConfig
from .containers import Detection, OcrText, VisualSignature
from .deps import LOGGER, cv2, imagehash, Image
from .utils import (
    setup_logging,
    load_image,
    safe_crop,
    ensure_dir,
    sanitize_filename,
)
from .orientation import (
    rotate_image,
    detect_upright_rotation,
)
from .visualization import (
    draw_label,
    visualize_legend,
    visualize_map,
    visualize_detections,
    visualize_ocr_text,
)
from .reporting import write_hamming_info
from .detector import RoboflowDetector
from .ocr import OcrEngine
from .matching import (
    filter_text_zone_false_positives,
    mask_icons_in_image,
    detection_inside_text,
    filter_text_on_icons,
    match_icons_to_text,
)
from .signatures import SignatureBuilder, SignatureMatcher
from .pipeline import LegendMarkerPipeline
from .cli import build_arg_parser, config_from_args, validate, main

__all__ = [
    "PipelineConfig",
    "Detection",
    "OcrText",
    "VisualSignature",
    "LOGGER",
    "cv2",
    "imagehash",
    "Image",
    "setup_logging",
    "load_image",
    "safe_crop",
    "ensure_dir",
    "sanitize_filename",
    "rotate_image",
    "detect_upright_rotation",
    "draw_label",
    "visualize_legend",
    "visualize_map",
    "visualize_detections",
    "visualize_ocr_text",
    "write_hamming_info",
    "RoboflowDetector",
    "OcrEngine",
    "filter_text_zone_false_positives",
    "mask_icons_in_image",
    "detection_inside_text",
    "filter_text_on_icons",
    "match_icons_to_text",
    "SignatureBuilder",
    "SignatureMatcher",
    "LegendMarkerPipeline",
    "build_arg_parser",
    "config_from_args",
    "validate",
    "main",
]
