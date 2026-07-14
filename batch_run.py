#!/usr/bin/env python3
"""
batch_run.py
============

Batch driver for the legend-marker pipeline.  Edit the CONFIG block below with
your folders and Roboflow details, then simply run:

    python3 batch_run.py

For every map image in ``INPUT_FOLDER`` it finds the legend of the *same name*
in ``LEGEND_FOLDER`` and writes that map's results to a per-map sub-folder under
``OUTPUT_FOLDER``.  Each map is processed independently, so one failure never
stops the batch; a summary (and ``batch_summary.json``) is printed at the end.

The heavy lifting is reused from ``legend_marker.py`` — this file only handles
folder iteration, name matching and per-item error isolation.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# Reuse the whole pipeline from the sibling module.
import legend_marker as lm


# ===========================================================================
# CONFIG — edit these values, then run `python3 batch_run.py`
# ===========================================================================
CONFIG = {
    # ---- Folders (absolute paths recommended) -----------------------------
    "INPUT_FOLDER":  "/home/nls34/Documents/POCs/legend_marker/inputs",
    "LEGEND_FOLDER": "/home/nls34/Documents/POCs/legend_marker/legend",
    "OUTPUT_FOLDER": "/home/nls34/Documents/POCs/legend_marker/output/batch_3",

    # ---- Roboflow ---------------------------------------------------------
    "API_KEY":   "K06rVQD1zQ46eOFObJvi",
    "WORKSPACE": "",                       # optional
    "PROJECT":   "plotmymap-icon-lqf56",
    "VERSION":   1,

    # ---- OCR --------------------------------------------------------------
    "OCR_ENGINE": "easyocr",               # "easyocr" | "paddleocr"
    "OCR_GPU":    False,

    # ---- Matching thresholds ---------------------------------------------
    "MATCH_THRESHOLD": 0.60,               # absolute floor to rename a map icon
    "MATCH_MARGIN":    0.08,               # best must beat 2nd-best by this
    "HASH_ALGORITHM":  "phash",

    # ---- Batch behaviour --------------------------------------------------
    "SAVE_CROPS":         True,
    "SAVE_VISUALIZATION": True,
    "SAVE_DEBUG_JSON":    True,
    "SKIP_EXISTING":      True,            # skip maps whose output dir exists
    "LIMIT":              0,               # 0 = all; else process at most N maps
    "VERBOSE":            False,
}

# Image file extensions we treat as maps / legends.
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

LOGGER = logging.getLogger("batch_run")


# ===========================================================================
# Filename matching
# ===========================================================================
def _normalize_stem(name: str) -> str:
    """Lower-cased stem with spaces / dots / underscores / hyphens removed.

    Used as a *fallback* key so slightly-differently-named files still match
    (e.g. "Alamo lake_jpg.rf.abc" vs "Alamolake_jpgrfabc").
    """
    stem = os.path.splitext(os.path.basename(name))[0]
    return "".join(ch for ch in stem.lower() if ch.isalnum())


def list_images(folder: str) -> List[str]:
    """Return sorted absolute paths of image files directly in ``folder``."""
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Folder not found: {folder}")
    out = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(IMAGE_EXTS)
    ]
    return out


def build_legend_index(legend_folder: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Index legends by exact stem and by normalized stem for fuzzy fallback."""
    by_stem: Dict[str, str] = {}
    by_norm: Dict[str, str] = {}
    for path in list_images(legend_folder):
        stem = os.path.splitext(os.path.basename(path))[0]
        by_stem.setdefault(stem, path)
        by_norm.setdefault(_normalize_stem(path), path)
    return by_stem, by_norm


def find_legend(
    map_path: str,
    by_stem: Dict[str, str],
    by_norm: Dict[str, str],
) -> Optional[str]:
    """Locate the legend for a map: exact stem first, then normalized fallback."""
    stem = os.path.splitext(os.path.basename(map_path))[0]
    if stem in by_stem:
        return by_stem[stem]
    return by_norm.get(_normalize_stem(map_path))


# ===========================================================================
# Config -> PipelineConfig
# ===========================================================================
def make_pipeline_config(output_dir: str) -> lm.PipelineConfig:
    """Translate the CONFIG dict into a per-map PipelineConfig."""
    return lm.PipelineConfig(
        api_key=CONFIG["API_KEY"],
        workspace=CONFIG["WORKSPACE"],
        project=CONFIG["PROJECT"],
        version=int(CONFIG["VERSION"]),
        ocr_engine=CONFIG["OCR_ENGINE"],
        ocr_gpu=bool(CONFIG["OCR_GPU"]),
        hash_algorithm=CONFIG["HASH_ALGORITHM"],
        match_score_threshold=float(CONFIG["MATCH_THRESHOLD"]),
        match_margin=float(CONFIG["MATCH_MARGIN"]),
        output_dir=output_dir,
        save_crops=bool(CONFIG["SAVE_CROPS"]),
        save_visualization=bool(CONFIG["SAVE_VISUALIZATION"]),
        save_debug_json=bool(CONFIG["SAVE_DEBUG_JSON"]),
    )


# ===========================================================================
# Batch driver
# ===========================================================================
def run_batch() -> int:
    lm.setup_logging(verbose=bool(CONFIG["VERBOSE"]))

    input_folder = CONFIG["INPUT_FOLDER"]
    legend_folder = CONFIG["LEGEND_FOLDER"]
    output_root = CONFIG["OUTPUT_FOLDER"]

    # Fail fast on obvious misconfiguration.
    for key in ("API_KEY", "PROJECT"):
        if not CONFIG.get(key):
            LOGGER.error("CONFIG['%s'] is empty — set it before running.", key)
            return 2

    maps = list_images(input_folder)
    by_stem, by_norm = build_legend_index(legend_folder)
    LOGGER.info("Found %d map(s) in %s and %d legend(s) in %s.",
                len(maps), input_folder, len(by_stem), legend_folder)

    lm.ensure_dir(output_root)

    # Build the pipeline ONCE and reuse it — the Roboflow model and the OCR
    # engine are expensive to initialise, and they are identical for every map.
    shared = lm.LegendMarkerPipeline(make_pipeline_config(output_root))

    limit = int(CONFIG["LIMIT"]) or len(maps)
    summary: List[Dict] = []
    t0 = time.time()

    for i, map_path in enumerate(maps[:limit], start=1):
        stem = os.path.splitext(os.path.basename(map_path))[0]
        out_dir = os.path.join(output_root, lm.sanitize_filename(stem))
        record: Dict = {"map": os.path.basename(map_path), "output_dir": out_dir}

        legend_path = find_legend(map_path, by_stem, by_norm)
        if legend_path is None:
            LOGGER.warning("[%d/%d] %s -> NO matching legend, skipping.",
                           i, limit, os.path.basename(map_path))
            record.update(status="no_legend", detections=0, renamed=0)
            summary.append(record)
            continue

        if CONFIG["SKIP_EXISTING"] and os.path.isfile(
                os.path.join(out_dir, "map_results.json")):
            LOGGER.info("[%d/%d] %s -> already done, skipping.",
                        i, limit, os.path.basename(map_path))
            record.update(status="skipped", legend=os.path.basename(legend_path))
            summary.append(record)
            continue

        LOGGER.info("[%d/%d] %s  <=>  %s", i, limit,
                    os.path.basename(map_path), os.path.basename(legend_path))
        try:
            # Point the shared pipeline at this map's output directory.
            shared.config.output_dir = lm.ensure_dir(out_dir)
            results = shared.run(map_path, legend_path)
            renamed = sum(1 for r in results if r.get("renamed"))
            record.update(
                status="ok",
                legend=os.path.basename(legend_path),
                detections=len(results),
                renamed=renamed,
            )
        except Exception as exc:  # Isolate failures — never stop the batch.
            LOGGER.exception("[%d/%d] FAILED on %s: %s",
                             i, limit, os.path.basename(map_path), exc)
            record.update(status="error", error=str(exc), detections=0, renamed=0)

        summary.append(record)

    # Persist and print a batch summary.
    summary_path = os.path.join(output_root, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    ok = sum(1 for r in summary if r["status"] == "ok")
    skipped = sum(1 for r in summary if r["status"] == "skipped")
    no_leg = sum(1 for r in summary if r["status"] == "no_legend")
    errs = sum(1 for r in summary if r["status"] == "error")
    total_renamed = sum(r.get("renamed", 0) for r in summary)
    LOGGER.info("=" * 60)
    LOGGER.info("Batch done in %.1fs: %d ok, %d skipped, %d no-legend, %d errors.",
                time.time() - t0, ok, skipped, no_leg, errs)
    LOGGER.info("Total icons renamed across batch: %d", total_renamed)
    LOGGER.info("Summary written to %s", summary_path)
    return 0 if errs == 0 else 1


if __name__ == "__main__":
    sys.exit(run_batch())
