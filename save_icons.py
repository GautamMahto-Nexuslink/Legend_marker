#!/usr/bin/env python3
"""
save_icons.py
=============

Batch driver that runs the legend-marker pipeline over a folder of maps and
saves *only the final-output icons* — nothing else.

Same inputs as ``batch_run.py`` (INPUT_FOLDER, LEGEND_FOLDER, OUTPUT_FOLDER),
but instead of writing a per-map sub-folder full of crops / JSON / visualisations,
this collates the finished icons into ``OUTPUT_FOLDER`` like so::

    OUTPUT_FOLDER/
        <ClassName>/                 # one folder per final class name
            <mapstem>_1.png          # icons named after their parent map,
            <mapstem>_2.png          # numbered 1, 2, ... within this class
        <OtherClass>/
            <mapstem>_1.png
            <othermap>_1.png

There is a single flat set of class folders — NO separate folder per map.
The numbering restarts at 1 for each class within a given map.  Every final
detection is saved (icons renamed from the legend AND those that kept their
original Roboflow class).

Usage::

    python3 save_icons.py
    python3 save_icons.py <input_folder> <legend_folder> <output_folder>

The heavy lifting (detection, OCR, matching, cropping) is reused from
``legend_marker.py``; this file only handles folder iteration, name matching
and gathering the final crops into class folders.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import cv2

# Reuse the whole pipeline from the sibling module.
import legend_marker as lm


# ===========================================================================
# CONFIG — edit these values, then run `python3 save_icons.py`
# (or pass the three folders on the command line to override them).
# ===========================================================================
CONFIG = {
    # ---- Folders (absolute paths recommended) -----------------------------
    "INPUT_FOLDER":  "/home/nls34/Documents/POCs/legend_marker/temp",
    "LEGEND_FOLDER": "/home/nls34/Documents/POCs/legend_marker/legend",
    "OUTPUT_FOLDER": "/home/nls34/Documents/POCs/legend_marker/output/icons_only",

    # ---- Roboflow ---------------------------------------------------------
    "API_KEY":   "K06rVQD1zQ46eOFObJvi",
    "WORKSPACE": "",                       # optional
    "PROJECT":   "plotmymap-icon-lqf56",
    "VERSION":   1,

    # ---- OCR --------------------------------------------------------------
    "OCR_ENGINE": "easyocr",             # "tesseract" | "easyocr" | "paddleocr"
    "OCR_GPU":    False,

    # ---- Matching thresholds ---------------------------------------------
    "MATCH_THRESHOLD": 0.60,               # absolute floor to rename a map icon
    "MATCH_MARGIN":    0.08,               # best must beat 2nd-best by this
    "HASH_ALGORITHM":  "phash",

    # ---- Auto-orientation -------------------------------------------------
    "AUTO_ROTATE":        True,            # correct sideways (90/180/270) inputs

    # ---- Batch behaviour --------------------------------------------------
    "LIMIT":              0,               # 0 = all; else process at most N maps
    "VERBOSE":            False,
}

# Image file extensions we treat as maps / legends.
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

LOGGER = logging.getLogger("save_icons")


# ===========================================================================
# Filename matching (identical logic to batch_run.py)
# ===========================================================================
def _normalize_stem(name: str) -> str:
    """Lower-cased stem with all non-alphanumerics removed (fuzzy match key)."""
    stem = os.path.splitext(os.path.basename(name))[0]
    return "".join(ch for ch in stem.lower() if ch.isalnum())


def list_images(folder: str) -> List[str]:
    """Return sorted absolute paths of image files directly in ``folder``."""
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Folder not found: {folder}")
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith(IMAGE_EXTS)
    ]


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
    """PipelineConfig pointed at a scratch dir.

    We need the crops on disk (save_crops=True) but nothing else — no
    visualisations, no debug JSON — since only the final icons are kept.
    """
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
        save_crops=True,            # we harvest these crops
        save_visualization=False,   # not needed — icons only
        save_debug_json=False,      # not needed — icons only
        auto_rotate=bool(CONFIG.get("AUTO_ROTATE", True)),
    )


# ===========================================================================
# Icon collation
# ===========================================================================
def collate_icons(
    results: List[Dict],
    map_crops_dir: str,
    map_stem: str,
    output_root: str,
    applied_angle: int = 0,
) -> int:
    """Copy each final-output crop into ``output_root/<class>/<mapstem>_N.png``.

    The pipeline writes map crops as ``<idx:03d>_<sanitized_class>.png`` in
    ``map_crops_dir`` (see ``process_map``), so we reconstruct each source name
    from its index and final class.  Numbering restarts at 1 per class, per map.

    ``applied_angle`` is the clockwise rotation the pipeline applied to make the
    map upright before cropping.  When it is non-zero the crops are in the
    upright frame, so we rotate each one BACK by ``-applied_angle`` to save the
    icon exactly as it appears in the ORIGINAL (rotated) map.  Right-angle
    rotations are lossless, so this is pixel-exact.

    Returns the number of icons saved for this map.
    """
    per_class_counter: Dict[str, int] = {}
    saved = 0

    for idx, res in enumerate(results):
        final_class = res.get("class")
        if not final_class:
            continue

        safe_class = lm.sanitize_filename(str(final_class))
        src = os.path.join(map_crops_dir, f"{idx:03d}_{safe_class}.png")
        if not os.path.isfile(src):
            # No crop was written for this detection (empty crop); skip it.
            LOGGER.debug("No crop file for map icon %d ('%s') — skipping.",
                         idx, final_class)
            continue

        n = per_class_counter.get(safe_class, 0) + 1
        per_class_counter[safe_class] = n

        class_dir = lm.ensure_dir(os.path.join(output_root, safe_class))
        dst = os.path.join(class_dir, f"{map_stem}_{n}.png")

        if applied_angle % 360 == 0:
            # No rotation was applied — the crop already matches the original.
            shutil.copyfile(src, dst)
        else:
            # Undo the pipeline's upright correction to restore the original
            # (rotated) orientation of the icon.
            crop = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            crop = lm.rotate_image(crop, -applied_angle)
            cv2.imwrite(dst, crop)
        saved += 1

    return saved


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
    # engine are expensive to initialise and identical for every map.  Its
    # output_dir is redirected to a throwaway scratch dir per map.
    scratch_root = tempfile.mkdtemp(prefix="save_icons_")
    shared = lm.LegendMarkerPipeline(make_pipeline_config(scratch_root))

    limit = int(CONFIG["LIMIT"]) or len(maps)
    total_saved = 0
    ok = skipped = errors = 0
    t0 = time.time()

    try:
        for i, map_path in enumerate(maps[:limit], start=1):
            map_stem = lm.sanitize_filename(
                os.path.splitext(os.path.basename(map_path))[0]
            )

            legend_path = find_legend(map_path, by_stem, by_norm)
            if legend_path is None:
                LOGGER.warning("[%d/%d] %s -> NO matching legend, skipping.",
                               i, limit, os.path.basename(map_path))
                skipped += 1
                continue

            LOGGER.info("[%d/%d] %s  <=>  %s", i, limit,
                        os.path.basename(map_path), os.path.basename(legend_path))

            # Fresh scratch dir for this map so crop names never collide.
            map_scratch = tempfile.mkdtemp(prefix=f"{map_stem}_", dir=scratch_root)
            try:
                shared.config.output_dir = lm.ensure_dir(map_scratch)
                results = shared.run(map_path, legend_path)

                # The pipeline rotates a sideways map upright before cropping
                # (auto_rotate) and, by default, the map reuses the legend's
                # angle (share_legend_map_orientation).  That shared angle is
                # left on the pipeline after run(); undo it so icons are saved
                # in the ORIGINAL map orientation.  If orientation sharing is
                # off we cannot recover the map's own angle, so leave crops as-is.
                applied_angle = 0
                if (shared.config.auto_rotate
                        and shared.config.share_legend_map_orientation
                        and shared._legend_angle):
                    applied_angle = int(shared._legend_angle)

                saved = collate_icons(
                    results,
                    os.path.join(map_scratch, "map_crops"),
                    map_stem,
                    output_root,
                    applied_angle=applied_angle,
                )
                total_saved += saved
                ok += 1
                LOGGER.info("[%d/%d] %s -> saved %d icon(s).",
                            i, limit, os.path.basename(map_path), saved)
            except Exception as exc:  # Isolate failures — never stop the batch.
                LOGGER.exception("[%d/%d] FAILED on %s: %s",
                                 i, limit, os.path.basename(map_path), exc)
                errors += 1
            finally:
                shutil.rmtree(map_scratch, ignore_errors=True)
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)

    LOGGER.info("=" * 60)
    LOGGER.info("Done in %.1fs: %d ok, %d skipped, %d errors.",
                time.time() - t0, ok, skipped, errors)
    LOGGER.info("Total icons saved: %d -> %s", total_saved, output_root)
    return 0 if errors == 0 else 1


def _apply_cli_overrides(argv: List[str]) -> None:
    """Optional positional overrides: input_folder legend_folder output_folder."""
    if len(argv) >= 1:
        CONFIG["INPUT_FOLDER"] = argv[0]
    if len(argv) >= 2:
        CONFIG["LEGEND_FOLDER"] = argv[1]
    if len(argv) >= 3:
        CONFIG["OUTPUT_FOLDER"] = argv[2]


if __name__ == "__main__":
    _apply_cli_overrides(sys.argv[1:])
    sys.exit(run_batch())
