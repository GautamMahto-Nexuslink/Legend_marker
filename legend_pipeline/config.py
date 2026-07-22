"""Configuration — every tunable knob of the pipeline in one dataclass."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


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
    # OCR runs twice on the legend: pass 1 (unmasked) locates label text so
    # false-positive detections can be dropped, then pass 2 (masked) reads the
    # real labels.  Pass 1's recognised TEXT is discarded — only its box
    # positions are used — so it needs no upscaling and can run at this reduced
    # long-side to save time (pass 2 keeps the accurate ocr_target_long_side).
    # Only applied when mask_icons_for_ocr is True (otherwise pass 1 IS the
    # labels).  Set 0 to run pass 1 at full resolution (old behaviour).
    ocr_fp_scan_long_side: int = 1000

    # ---- False-positive detection filtering ----------------------------
    # The model sometimes fires a false-positive "icon" on a LETTER of a label
    # (e.g. the "O" of "Overlook").  We drop a detection only when BOTH:
    #   (a) it is largely contained by a text box clearly wider than itself
    #       (it sits INSIDE a real label, not to the left of one), AND
    #   (b) its left edge is NOT in an icon column.
    # The icon columns are learned from the detections that are clearly real
    # icons (those NOT inside a label, i.e. sitting to the left of their text).
    # This keeps a real icon even when OCR merges its own glyph into the label
    # (e.g. "CH Camp Host"): the icon still sits in the icon column, so it
    # survives and is masked out before the final OCR.  It runs BEFORE masking so
    # a dropped letter stays visible and its label reads correctly.
    filter_text_zone_false_positives: bool = True
    column_x_tolerance_factor: float = 1.2 # icon-column half-width = factor * median icon width

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

    # ---- Known-icon pHash database ({phash_hex: classname}) ------------
    # A curated JSON mapping perceptual hashes to their final class name (see
    # save_phash.py / icons_phash_flat.json).  Before the per-map legend
    # matching, every map detection's crop is hashed and looked up here; a hit
    # renames the detection to the stored class straight away (a high-confidence
    # shortcut), and only the misses fall through to the legend/OCR matching.
    # Empty path disables this stage entirely (original behaviour).
    phash_db_path: str = "/home/nls34/Documents/POCs/legend_marker/icons_phash_flat.json"
    # Max Hamming distance for a DB hit, compared against det.signature.phash_hex.
    # 0 = exact match only; a small value (e.g. 6-10 for a 256-bit hash)
    # tolerates minor rendering differences.  The DB MUST be generated with the
    # same hash_algorithm / hash_size as the pipeline (save_phash.py reuses
    # SignatureBuilder, so its default hash_size=16 already matches).
    phash_db_max_hamming: int = 0

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

    # ---- Auto-orientation (rotated legend / map correction) ------------
    # Some source pages are scanned/exported sideways, so a legend's labels
    # (and sometimes the whole map) end up rotated by a multiple of 90 degrees.
    # OCR and the detector both expect upright text, so before any detection we
    # probe the four right-angle orientations, OCR each, and keep the one that
    # reads the most real text.  The image is rotated to that upright pose; both
    # the original and the rotated image are then saved for auditing.
    auto_rotate: bool = True
    # Candidate clockwise rotations to try (degrees).  Legends are only ever off
    # by a right angle, so these four cover every real case.
    rotate_candidate_angles: Tuple[int, ...] = (0, 90, 180, 270)
    # Only rotate away from upright (0) when another orientation reads clearly
    # more text: its score must beat upright's by at least this factor.  Stops a
    # near-tie (already-upright image) from being needlessly rotated.
    rotate_min_gain: float = 1.5
    # Working long-side for the orientation probe/OSD.  A large map is
    # downscaled to this, and a tiny legend is UPSCALED to it, so OSD has enough
    # resolution to read orientation confidently (a starved OSD bails to the slow
    # OCR probe).  The decision only needs relative scores, so this stays modest.
    rotate_probe_long_side: int = 1600
    # OCR resolution cap for the 4-way OCR probe (the OSD fallback).  The probe
    # only needs which rotation reads the MOST text — a relative comparison — so
    # it OCRs at this reduced long-side instead of ocr_target_long_side.  Four
    # full-res probe passes are the single biggest cost when OSD is inconclusive;
    # lowering this cuts that time proportionally.  0 = full resolution.
    rotate_probe_ocr_long_side: int = 800
    # How the upright rotation is found:
    #   "osd"       — Tesseract OSD only: reads the angle in ONE fast pass.
    #   "ocr_probe" — the original 4-way OCR probe (slow but OCR-engine-agnostic).
    #   "auto"      — try OSD first, fall back to the OCR probe (default).
    # The OCR probe OCRs all four orientations (4 passes); OSD needs a single
    # tesseract pass, so it is dramatically faster — especially with EasyOCR.
    orientation_method: str = "auto"
    # Minimum OSD "orientation confidence" to trust its angle.  Below this we
    # fall back to the OCR probe (in "auto"), or keep upright (in "osd").
    osd_min_confidence: float = 0.5
    # tesseract binary used for OSD (PATH name or absolute path).  OSD is invoked
    # directly on this binary, so it works even when ocr_engine is easyocr/paddle.
    tesseract_cmd: str = "tesseract"
    # OSD's default `min_characters_to_try` is 50 — far more than a map legend's
    # handful of words, so OSD aborts with "Too few characters. Skipping this
    # page" and the pipeline falls back to the slow 4-way OCR probe.  Lowering it
    # lets OSD read sparse legends and return the angle directly.
    osd_min_characters: int = 5
    # Hard cap (seconds) on a single OSD subprocess call, so a stuck tesseract
    # can never hang the pipeline; on timeout we fall back to the OCR probe.
    osd_timeout: float = 30.0
    # The legend is a crop of the SAME source page as the map, so they share the
    # same rotation.  Once the legend's upright angle is found (fast, via OSD on
    # its tidy text), reuse it for the map instead of re-detecting — this skips
    # the slow 4-way OCR probe on the full map (orientation is ambiguous there,
    # so OSD often falls back to OCR'ing all four rotations, ~minutes each).
    # Set False if a dataset's legends and maps can be rotated independently.
    share_legend_map_orientation: bool = True

    # ---- Output ---------------------------------------------------------
    output_dir: str = "output"
    save_crops: bool = True
    save_debug_json: bool = True
    save_visualization: bool = True   # Draw annotated legend/map images.
