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
from .reporting import stage_report_lines, write_hamming_info
from .signatures import SignatureBuilder, SignatureMatcher
from .synonyms import SynonymMap
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
        # Semantic class grouping ("Observation Area" == "Overlook"); loaded
        # lazily from config.synonyms_path.  None = "not loaded yet".
        self._synonyms: Optional[SynonymMap] = None

    # -- Semantic class reconciliation ------------------------------------
    @property
    def synonyms(self) -> SynonymMap:
        """Canonical-name lookup shared by the whole run (cached, never raises)."""
        if self._synonyms is None:
            self._synonyms = SynonymMap.load(
                self.config.synonyms_path,
                fuzzy_cutoff=self.config.synonym_fuzzy_cutoff,
            )
        return self._synonyms

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

    # -- Final class decision ---------------------------------------------
    def _find_semantic_legend(
        self, class_name: Optional[str], rows: List[Dict[str, Any]]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Find the legend entry that means (or nearly means) ``class_name``.

        Searches EVERY legend name on this map, not just the best-scoring one:
        a pHash class of "Bicycling" should find the legend's "Motor Bike Trail"
        even when some unrelated glyph correlates better.  ``rows`` is already
        sorted by match score, so the first hit at each strength is also the
        visually closest one.

        Returns ``(row, "same")`` for a synonym, ``(row, "related")`` for a
        neighbouring concept, or ``(None, None)``.
        """
        syn = self.synonyms
        if not syn or not class_name or not rows:
            return None, None
        related_hit: Optional[Dict[str, Any]] = None
        for row in rows:
            relation = syn.relation(class_name, row.get("name"))
            if relation == "same":
                return row, "same"
            if relation == "related" and related_hit is None:
                related_hit = row
        if related_hit is not None:
            return related_hit, "related"
        return None, None

    def _decide_class(
        self,
        original_class: str,
        db_class: Optional[str],
        db_nearest_name: Optional[str],
        db_nearest_dist: Optional[int],
        rows: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Reconcile the pHash DB and the legend, then pick the final label.

        ``rows`` are the legend candidates ranked by visual score (best first).

        Precedence:

        1. **pHash DB hit** (Hamming <= ``phash_db_max_hamming``).  The class it
           returns is then looked for in *this map's legend*: a same-meaning
           label ("Observation Area" -> legend "Overlook") or, failing that, a
           neighbouring one ("Bicycling" -> legend "Motor Bike Trail").  When
           one is found the detection is renamed to the legend's wording
           (``config.synonym_naming`` can override whose wording wins) and the
           method becomes ``phash_db+legend`` / ``phash_db+related``.  With no
           semantic legend entry the DB class is kept verbatim (``phash_db``).
        2. **Legend match** clearing the score floor *and* the margin gate
           (``legend``).
        3. **Synonym rescue** (``config.synonym_rescue``) — the DB missed its
           Hamming gate but its nearest class is a synonym of a legend entry.
           Two independent weak signals pointing at one concept are treated as
           one strong signal: the margin gate is waived and the floor drops to
           ``synonym_rescue_min_score`` (``synonym_agree``).
        4. Otherwise the original detector class is kept (no rename).

        Returns every intermediate so the caller can log / report it.
        """
        syn = self.synonyms
        rows = rows or []
        top = rows[0] if rows else None
        legend_name = top["name"] if top else None
        score = top["score"] if top else -1.0
        second_score = rows[1]["score"] if len(rows) > 1 else 0.0
        margin = score - second_score

        db_candidate = db_class or db_nearest_name
        # The semantic search runs over all legend entries, so a DB class can
        # find its legend twin even when that twin is not the top visual match.
        sem_row, relation = self._find_semantic_legend(db_candidate, rows)
        sem_name = sem_row["name"] if sem_row else None
        sem_score = sem_row["score"] if sem_row else None
        synonym_agree = relation is not None

        passes_floor = (legend_name is not None
                        and score >= self.config.match_score_threshold)
        passes_margin = (len(rows) < 2) or (margin >= self.config.match_margin)
        rescued = bool(
            self.config.synonym_rescue
            and db_class is None
            and relation == "same"
            and not (passes_floor and passes_margin)
            and sem_score is not None
            and sem_score >= self.config.synonym_rescue_min_score
            and db_nearest_dist is not None
            and db_nearest_dist <= self.config.synonym_rescue_max_hamming
        )

        final_class = original_class
        renamed = False
        match_method: Optional[str] = None
        if db_class is not None:
            final_class = (
                syn.pick_label(self.config.synonym_naming, sem_name, db_class)
                if relation else db_class
            )
            renamed = True
            match_method = f"phash_db+{relation}" if relation else "phash_db"
        elif passes_floor and passes_margin:
            # The legend already won on its own; only re-word it when the DB's
            # nearest class says the same thing.
            final_class = (
                syn.pick_label(self.config.synonym_naming, legend_name,
                               db_nearest_name)
                if relation == "same" and sem_name == legend_name else legend_name
            )
            renamed = True
            match_method = ("legend+synonym"
                            if relation == "same" and sem_name == legend_name
                            else "legend")
        elif rescued:
            final_class = syn.pick_label(
                self.config.synonym_naming, sem_name, db_nearest_name)
            renamed = True
            match_method = "synonym_agree"

        return {
            "final_class": final_class,
            "renamed": renamed,
            "match_method": match_method,
            "relation": relation,               # "same" | "related" | None
            "synonym_agree": synonym_agree,
            "rescued": rescued,
            "passes_floor": passes_floor,
            "passes_margin": passes_margin,
            "db_candidate": db_candidate,
            "legend_name": legend_name,
            "legend_score": score,
            "legend_margin": margin,
            "semantic_legend": sem_name,
            "semantic_score": sem_score,
            "canonical": (syn.canonical(final_class)
                          or syn.canonical(db_candidate)
                          or syn.canonical(legend_name)),
        }

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
                syn = self.synonyms
                canonical = syn.canonical(names[k])
                footer = [
                    f"This legend icon : '{names[k]}'",
                    # Which semantic group this label lands in decides whether a
                    # pHash class can be renamed to it later, so record it here.
                    f"Semantic group   : {canonical or '(unmapped)'}",
                    f"Nearby groups    : "
                    f"{', '.join(sorted(syn.related(names[k]))) or '-'}",
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

            # Steps 5.6 + 6b: reconcile the two sources and decide the label.
            decision = self._decide_class(
                original_class=det.class_name,
                db_class=db_class,
                db_nearest_name=db_nearest_name,
                db_nearest_dist=db_nearest_dist,
                rows=rows,
            )
            syn = self.synonyms
            final_class = decision["final_class"]
            renamed = decision["renamed"]
            match_method = decision["match_method"]
            relation = decision["relation"]
            synonym_agree = decision["synonym_agree"]
            canonical_hint = decision["canonical"]
            db_candidate = decision["db_candidate"]
            sem_name = decision["semantic_legend"]
            sem_score = decision["semantic_score"]
            passes_floor = decision["passes_floor"]
            passes_margin = decision["passes_margin"]
            rescued = decision["rescued"]

            crop_file = f"{idx:03d}_{sanitize_filename(final_class)}.png"
            if self.config.save_crops and det.crop is not None and det.crop.size:
                cv2.imwrite(os.path.join(crop_dir, crop_file), det.crop)

                # Per-crop report .txt right beside the image: what every stage
                # called this icon, colour-coded exactly like the annotated map,
                # followed by the gate details.  The score is the template+ORB
                # match score; the hamming column is pHash (informational only).
                footer = stage_report_lines(
                    original_class=det.class_name,
                    final_class=final_class,
                    renamed=renamed,
                    match_method=match_method,
                    db_enabled=bool(self._load_phash_db()),
                    db_class=db_class,
                    db_nearest_name=db_nearest_name,
                    db_nearest_dist=db_nearest_dist,
                    db_max_hamming=self.config.phash_db_max_hamming,
                    legend_name=name,
                    legend_score=score,
                    legend_margin=margin,
                    passes_floor=passes_floor,
                    passes_margin=passes_margin,
                    score_threshold=self.config.match_score_threshold,
                    margin_threshold=self.config.match_margin,
                    syn_enabled=bool(syn),
                    relation=relation,
                    semantic_legend=sem_name,
                    semantic_score=sem_score,
                    canonical=canonical_hint,
                    naming=self.config.synonym_naming,
                    rescued=rescued,
                    n_legend_entries=len(rows),
                )
                footer += [
                    "",
                    f"Best legend match : {name}  (score={score:.3f}, "
                    f"pHash hamming={best_hamming})",
                    f"Decision          : {'RENAMED' if renamed else 'KEPT'}  "
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
                    # "phash_db" | "phash_db+same" | "phash_db+related" |
                    # "legend" | "legend+synonym" | "synonym_agree" | None
                    "match_method": match_method,
                    # Semantic group of the final class — stable key for search
                    # and for merging equivalent classes across maps.
                    "canonical_class": canonical_hint,
                    "db_class": db_candidate,
                    "legend_class": name,
                    "semantic_legend_class": sem_name,
                    "semantic_relation": relation,   # "same" | "related" | None
                    "synonym_agree": synonym_agree,
                }
            )
            LOGGER.info(
                "Map icon %d: '%s' -> '%s' (score=%.3f, renamed=%s, method=%s, "
                "canonical=%s)",
                idx, det.class_name, final_class, score, renamed, match_method,
                canonical_hint,
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
