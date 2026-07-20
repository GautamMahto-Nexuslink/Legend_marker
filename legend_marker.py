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

The implementation is intentionally modular: every stage lives in its own module
under the :mod:`legend_pipeline` package (config, containers, utils, orientation,
visualization, reporting, detector, ocr, matching, signatures, pipeline, cli).
This file is a thin compatibility shim that re-exports the whole public API and
runs the CLI, so both ``python legend_marker.py ...`` and
``import legend_marker as lm`` keep working exactly as before.

Author: (generated) — production-ready reference implementation.


cli eg:

python legend_marker.py --map 'Input_image_path' --legend 'legend_sub_part' --api-key xx-xx-xx --project xxx-xxx --version 1 --output-dir 'Output_of_folder' -v(verbose)

python3 legend_marker.py --map /home/nls34/Work/OuterMap/Main_Dataset/Icons_Dataset/map.coco/train/AhjumawiLavaSpringsStatePark_page-0004_jpgrfSmY3DOtP6Zazv8jbmCdK.jpg  --legend /home/nls34/Documents/POCs/legend_marker/legend/AhjumawiLavaSpringsStatePark_page-0004_jpgSmY3DOtP6Zazv8jbmCdK.jpg  --api-key K06rVQD1zQ46eOFObJvi --project plotmymap-icon-lqf56 --version 1 --output-dir output/ahujawani_easyocr_new_6_updated -v
"""

from __future__ import annotations

import sys

# Re-export the full public API from the modular package so any existing
# ``import legend_marker as lm; lm.<name>`` keeps resolving unchanged.
from legend_pipeline import *  # noqa: F401,F403
from legend_pipeline import (  # noqa: F401  (explicit for star-import-averse tools)
    LOGGER,
    PipelineConfig,
    Detection,
    OcrText,
    VisualSignature,
    RoboflowDetector,
    OcrEngine,
    SignatureBuilder,
    SignatureMatcher,
    LegendMarkerPipeline,
    setup_logging,
    load_image,
    safe_crop,
    ensure_dir,
    sanitize_filename,
    rotate_image,
    detect_upright_rotation,
    write_hamming_info,
    filter_text_zone_false_positives,
    mask_icons_in_image,
    detection_inside_text,
    filter_text_on_icons,
    match_icons_to_text,
    build_arg_parser,
    config_from_args,
    validate,
    main,
)


if __name__ == "__main__":
    sys.exit(main())


#kaptams-workspace/plotmymap-icon-lqf56-1-yolo11x-seg-t1
