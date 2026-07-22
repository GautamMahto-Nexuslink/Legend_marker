"""Orchestration — LegendMarkerPipeline ties every stage into one run()."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import PipelineConfig
from .containers import Detection, OcrText, VisualSignature
from .deps import LOGGER, cv2, imagehash
from .detector import RoboflowDetector
from .matching import (
    detection_inside_text,
    filter_text_on_icons,
    filter_text_zone_false_positives,
    mask_icons_in_image,
    match_icons_to_text,
)
from .ocr import OcrEngine
from .orientation import detect_upright_rotation, rotate_image
from .reporting import write_hamming_info
from .signatures import SignatureBuilder, SignatureMatcher
from .timing import StepTimer
from .utils import ensure_dir, load_image, sanitize_filename
from .visualization import (
    visualize_detections,
    visualize_legend,
    visualize_map,
    visualize_ocr_text,
)


class LegendMarkerPipeline:
    """Ties every stage together into one `run()` call."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.detector = RoboflowDetector(config)
        self.sig_builder = SignatureBuilder(config)
        self.matcher = SignatureMatcher(config)
        # OCR is heavy to init; build lazily only when the legend stage runs.
        self._ocr: Optional[OcrEngine] = None
        # Upright correction found for the legend; reused for the map (same page)
        # so the map never re-runs the slow 4-way OCR orientation probe.
        self._legend_angle: Optional[int] = None
        # Per-step timing for the current run(); (re)created at the top of run().
        self._timer: StepTimer = StepTimer(LOGGER)
        # Known-icon pHash database ({phash_hex: classname}); loaded lazily from
        # config.phash_db_path on first use.  None = "not loaded yet".
        self._phash_db: Optional[List[Tuple[Any, str]]] = None

    # -- Known-icon pHash database ---------------------------------------
    def _load_phash_db(self) -> List[Tuple[Any, str]]:
        """Load & parse the {phash_hex: classname} JSON into (ImageHash, name).

        Cached after the first call.  Returns an empty list when no path is
        configured, the file is missing, or imagehash is unavailable — the
        pipeline then behaves exactly as before (no DB stage).
        """
        if self._phash_db is not None:
            return self._phash_db

        db: List[Tuple[Any, str]] = []
        path = self.config.phash_db_path
        if path and imagehash is not None and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                for phash_hex, class_name in raw.items():
                    try:
                        db.append((imagehash.hex_to_hash(phash_hex), class_name))
                    except Exception as exc:
                        LOGGER.warning("Bad pHash entry '%s' in %s: %s",
                                       phash_hex, path, exc)
                LOGGER.info("Loaded %d known-icon pHash entrie(s) from %s.",
                            len(db), path)
            except Exception as exc:
                LOGGER.warning("Failed to load pHash DB %s: %s", path, exc)
        elif path and imagehash is None:
            LOGGER.warning("phash_db_path set but imagehash is unavailable — "
                           "pHash DB stage disabled.")
        elif path:
            LOGGER.warning("phash_db_path '%s' not found — pHash DB stage "
                           "disabled.", path)

        self._phash_db = db
        return db

    def _match_phash_db(
        self, sig: VisualSignature
    ) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """Look a detection's pHash up in the known-icon DB.

        Compares the detection's ``phash`` (computed by SignatureBuilder, so
        identical to how the DB was generated) against every DB entry.

        Returns ``(matched_name, nearest_name, nearest_dist)``:
          * ``matched_name`` — the class when the nearest entry is within
            ``config.phash_db_max_hamming`` Hamming distance, else ``None``.
          * ``nearest_name`` / ``nearest_dist`` — the closest entry regardless of
            the threshold, so the report can show how close the DB got even on a
            miss (helps calibrate the threshold).  ``(None, None)`` if the DB is
            empty or the detection has no pHash.
        """
        db = self._load_phash_db()
        if not db or sig is None or sig.phash is None:
            return None, None, None

        nearest_name: Optional[str] = None
        nearest_dist: Optional[int] = None
        for db_hash, name in db:
            dist = int(sig.phash - db_hash)      # Hamming distance
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist, nearest_name = dist, name
                if dist == 0:
                    break                        # perfect hit — can't do better

        matched_name = (
            nearest_name
            if nearest_dist is not None
            and nearest_dist <= self.config.phash_db_max_hamming
            else None
        )
        return matched_name, nearest_name, nearest_dist

    @property
    def ocr(self) -> OcrEngine:
        if self._ocr is None:
            self._ocr = OcrEngine(self.config)
        return self._ocr

    # -- Auto-orientation -------------------------------------------------
    def _prepare_oriented_image(
        self, image_path: str, kind: str, reuse_angle: Optional[int] = None
    ) -> Tuple[str, np.ndarray, int]:
        """Load an image and, if rotated, return an upright copy + its new path.

        ``kind`` is "legend" or "map" and only tags the saved filenames.  The
        detector infers from a *path* (Roboflow) while crops are taken from the
        *array*, so a rotated image must be written to disk and its path used —
        otherwise the boxes and the crops would disagree.  When a rotation is
        applied, BOTH the original and the rotated image are saved to the output
        directory for auditing; an already-upright image is passed through
        untouched (no extra files, original path preserved).

        ``reuse_angle`` short-circuits detection with a known clockwise
        correction (e.g. the map reusing the legend's angle — same source page),
        avoiding the slow 4-way OCR orientation probe.

        Returns ``(path_to_use, image_array, angle_applied)``.
        """
        image = load_image(image_path)
        if not self.config.auto_rotate:
            return image_path, image, 0

        if reuse_angle is not None:
            angle = reuse_angle
            LOGGER.info("%s reusing legend orientation: rotate %d deg CW "
                        "(orientation detection skipped).", kind, angle)
        else:
            angle, scores = detect_upright_rotation(image, self.ocr, self.config)
            if scores:
                LOGGER.info("%s orientation scores (deg CW -> text score): %s",
                            kind, {a: round(s, 1) for a, s in scores.items()})
        if angle == 0:
            LOGGER.info("%s already upright — no rotation applied.", kind)
            return image_path, image, 0

        rotated = rotate_image(image, angle)
        out_dir = ensure_dir(self.config.output_dir)
        stem = sanitize_filename(os.path.splitext(os.path.basename(image_path))[0])
        original_out = os.path.join(out_dir, f"{kind}_{stem}_original.png")
        rotated_out = os.path.join(out_dir, f"{kind}_{stem}_rotated_{angle}cw.png")
        cv2.imwrite(original_out, image)
        cv2.imwrite(rotated_out, rotated)
        LOGGER.info(
            "%s was rotated %d deg CW to upright. Saved original -> %s and "
            "rotated -> %s (rotated image is used for detection/OCR).",
            kind, angle, original_out, rotated_out,
        )
        return rotated_out, rotated, angle

    # -- Legend side ------------------------------------------------------
    def build_legend_database(
        self, legend_path: str
    ) -> List[Tuple[str, VisualSignature]]:
        """Steps 1-4: detect legend icons, OCR, match, sign -> name<->signature."""
        # Correct a sideways legend first: rotated labels defeat OCR entirely.
        with self._timer.step("legend: orientation"):
            legend_path, legend_img, legend_angle = self._prepare_oriented_image(
                legend_path, "legend")
        # Remember it so the map (same source page) can reuse this angle instead
        # of re-running the slow orientation probe.
        self._legend_angle = legend_angle

        # Step 1: detect legend icons (raw Roboflow JSON saved alongside).
        raw_path = (
            os.path.join(self.config.output_dir, "legend_roboflow_raw.json")
            if self.config.save_debug_json else None
        )
        with self._timer.step("legend: roboflow detect"):
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

        # Step 2a: OCR the UNMASKED legend to locate the real label text.  We use
        # this pass only to spot false-positive detections (the model firing on a
        # letter of a label) and then discard the detections it flags.  When
        # masking is on, pass 1 is throwaway (only its box positions are used),
        # so run it at a reduced resolution to save time; when masking is off,
        # pass 1 IS the labels and must stay at full resolution.
        fp_scan_target = (
            self.config.ocr_fp_scan_long_side
            if (self.config.mask_icons_for_ocr and self.config.ocr_fp_scan_long_side > 0)
            else None
        )
        with self._timer.step("legend: ocr pass1 (fp scan)"):
            texts_pass1 = self.ocr.read(legend_img, target_long_side=fp_scan_target)

        # Filter 0: drop text-zone false positives (e.g. the model boxing the "O"
        # of "Overlook").  This MUST run before masking so the letter stays
        # visible to OCR and the label reads correctly; only REAL icons remain.
        icons = filter_text_zone_false_positives(icons, texts_pass1, self.config)

        # Step 2b: mask ONLY the real icons and re-read, so an icon's own glyph
        # can't be read as text and merged into its label (e.g. the "H<tent>B"
        # symbol misread as "HAE Hike & Bike Campground").  False-positive
        # letters were already dropped, so they are NOT masked and their labels
        # remain intact.  When masking is off, reuse the first pass.
        if self.config.mask_icons_for_ocr:
            with self._timer.step("legend: ocr pass2 (masked)"):
                ocr_img = mask_icons_in_image(legend_img, icons, self.config.icon_mask_shrink)
                texts = self.ocr.read(ocr_img)
        else:
            texts = texts_pass1

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
        _t_sig = time.perf_counter()
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

        self._timer.add("legend: match + signatures", time.perf_counter() - _t_sig)
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
        # Correct a sideways map the same way the legend is corrected.  The map
        # and legend come from the same source page, so reuse the legend's angle
        # (found fast via OSD) instead of re-probing — orientation is ambiguous
        # on a full map, where the probe otherwise OCRs all four rotations.
        reuse_angle = (
            self._legend_angle
            if (self.config.share_legend_map_orientation
                and self.config.auto_rotate
                and self._legend_angle is not None)
            else None
        )
        with self._timer.step("map: orientation"):
            map_path, map_img, _ = self._prepare_oriented_image(
                map_path, "map", reuse_angle=reuse_angle)

        # Step 5: detect icons on the full map (raw Roboflow JSON saved too).
        raw_path = (
            os.path.join(self.config.output_dir, "map_roboflow_raw.json")
            if self.config.save_debug_json else None
        )
        with self._timer.step("map: roboflow detect"):
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

        _t_match = time.perf_counter()
        for idx, det in enumerate(detections):
            det.signature = self.sig_builder.build(det.crop)

            # Step 5.5: known-icon pHash DB lookup FIRST (priority shortcut).  If
            # this detection's glyph pHash matches a curated entry, rename it
            # straight away — the DB is a cross-map, human-verified source of
            # truth.  We also keep the nearest entry (even on a miss) so the
            # report shows the DB was consulted first and how close it got.
            db_class, db_nearest_name, db_nearest_dist = self._match_phash_db(
                det.signature)

            # Step 6: rank the detection against every legend signature.  We
            # still compute this for the report even when the DB already
            # decided, so the .txt / JSON keep the full match breakdown.
            rows = self.matcher.rank(det.signature, legend_db)
            top = rows[0] if rows else None
            name = top["name"] if top else None
            score = top["score"] if top else -1.0
            breakdown = top["breakdown"] if top else {}
            best_hamming = top["hamming"] if top else None
            second_score = rows[1]["score"] if len(rows) > 1 else 0.0
            margin = score - second_score

            # Decision.  pHash DB wins first; otherwise rename only when the best
            # legend match clears the absolute floor AND clearly beats the
            # runner-up (margin gate) — never force a "least-bad" match.
            final_class = det.class_name
            renamed = False
            match_method: Optional[str] = None
            passes_floor = name is not None and score >= self.config.match_score_threshold
            passes_margin = (len(rows) < 2) or (margin >= self.config.match_margin)
            if db_class is not None:
                final_class = db_class
                renamed = True
                match_method = "phash_db"
            elif passes_floor and passes_margin:
                final_class = name
                renamed = True
                match_method = "legend"

            crop_file = f"{idx:03d}_{sanitize_filename(final_class)}.png"
            if self.config.save_crops and det.crop is not None and det.crop.size:
                cv2.imwrite(os.path.join(crop_dir, crop_file), det.crop)

                # Per-crop report .txt right beside the image.  Note the score
                # is the template+ORB match score; the hamming column is pHash
                # (informational only).
                # Step 5.5 report line — ALWAYS shown first, so it is clear the
                # JSON DB is consulted before the legend.  Describes what the DB
                # lookup found (hit / nearest miss / disabled).
                if not self._load_phash_db():
                    db_line = "pHash DB    : disabled (no --phash-db configured)"
                elif db_class is not None:
                    db_line = (
                        f"pHash DB    : HIT -> '{db_class}' "
                        f"(hamming={db_nearest_dist} <= "
                        f"{self.config.phash_db_max_hamming}) — wins, "
                        f"legend match ignored"
                    )
                else:
                    db_line = (
                        f"pHash DB    : miss (nearest '{db_nearest_name}' "
                        f"hamming={db_nearest_dist} > "
                        f"{self.config.phash_db_max_hamming}) — fell back to legend"
                    )

                footer = [
                    db_line,
                    f"Best match  : {name}  (match score={score:.3f}, "
                    f"pHash hamming={best_hamming})",
                    f"Floor gate  : score {score:.3f} >= "
                    f"{self.config.match_score_threshold} -> "
                    f"{'PASS' if passes_floor else 'FAIL'}",
                    f"Margin gate : best-2nd = {margin:.3f} >= "
                    f"{self.config.match_margin} -> "
                    f"{'PASS' if passes_margin else 'FAIL'}",
                    f"Decision    : {'RENAMED' if renamed else 'KEPT'}  "
                    f"'{det.class_name}' -> '{final_class}'"
                    f"{f' (via {match_method})' if match_method else ''}",
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
                    "match_method": match_method,   # "phash_db" | "legend" | None
                }
            )
            LOGGER.info(
                "Map icon %d: '%s' -> '%s' (score=%.3f, renamed=%s, method=%s)",
                idx, det.class_name, final_class, score, renamed, match_method,
            )

        self._timer.add("map: signatures + match", time.perf_counter() - _t_match)
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
        # Fresh timer per run so a reused pipeline (batch mode) reports per-map.
        self._timer = StepTimer(LOGGER)
        _t_run = time.perf_counter()

        legend_db = self.build_legend_database(legend_path)
        results = self.process_map(map_path, legend_db)
        if self.config.save_debug_json:
            self._dump_json("map_results.json", results)

        # Per-step timing breakdown.  The timed steps sum to ~= wall clock; the
        # remainder is untimed glue (visualization writes, JSON dumps, one-time
        # OCR/model init on the first run).
        wall = time.perf_counter() - _t_run
        LOGGER.info("%s\n  %-32s%8.2fs  (untimed glue: %.2fs)",
                    self._timer.summary(), "WALL CLOCK", wall,
                    max(0.0, wall - self._timer.total))
        if self.config.save_debug_json:
            timings = self._timer.as_dict()
            timings["wall_clock_seconds"] = round(wall, 3)
            self._dump_json("timings.json", timings)

        LOGGER.info("=== Pipeline finished: %d detection(s) ===", len(results))
        return results

    # -- Helpers ----------------------------------------------------------
    def _dump_json(self, filename: str, data: Any) -> None:
        path = os.path.join(self.config.output_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        LOGGER.debug("Wrote %s", path)
