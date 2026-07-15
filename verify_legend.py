#!/usr/bin/env python3
"""Replay a saved Roboflow legend response through the CURRENT legend pipeline
(no network / no `inference` package needed) so you can confirm the code in this
folder actually produces the fixed mapping.

Usage:
    python3 verify_legend.py <legend_image> <legend_roboflow_raw.json>

Example (your last run):
    python3 verify_legend.py \
        legend/CrystalCoveMoroCampMap2011Rev_page-0002_jpg.jpg \
        output/temp_test_3/legend_roboflow_raw.json
"""
import json
import sys

import legend_marker as lm


def main() -> None:
    legend_img_path, raw_json_path = sys.argv[1], sys.argv[2]
    lm.setup_logging(verbose=False)

    cfg = lm.PipelineConfig(
        api_key="x", project="p", version=1,
        ocr_engine="easyocr", ocr_gpu=False,
        save_crops=False, save_visualization=False, save_debug_json=False,
    )

    raw = json.load(open(raw_json_path))
    # Bypass the network model entirely: feed the saved response.
    lm.RoboflowDetector._load_model = lambda self: None
    lm.RoboflowDetector.infer = lambda self, image_path: raw

    db = lm.LegendMarkerPipeline(cfg).build_legend_database(legend_img_path)

    print("\n===== LEGEND LABELS (current code) =====")
    for name, _sig in db:
        print("  ", name)
    print(f"\nTotal: {len(db)} entries")


if __name__ == "__main__":
    main()
