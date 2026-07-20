"""Step 3: spatially match icons to OCR text (plus the FP / glyph filters)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import PipelineConfig
from .containers import Detection, OcrText
from .deps import LOGGER


def _fraction_inside(inner: Sequence[int], outer: Sequence[int]) -> float:
    """Fraction of the ``inner`` box's area that lies within the ``outer`` box."""
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inter_w = max(0, min(ix2, ox2) - max(ix1, ox1))
    inter_h = max(0, min(iy2, oy2) - max(iy1, oy1))
    inter = inter_w * inter_h
    inner_area = max(1, (ix2 - ix1) * (iy2 - iy1))
    return inter / float(inner_area)


def filter_text_zone_false_positives(
    icons: List[Detection],
    texts: List[OcrText],
    config: PipelineConfig,
) -> List[Detection]:
    """Drop detections the model fired on the label text itself (text-zone FPs).

    Legend icons line up in a left column and every label is left-aligned in the
    column to their right.  So a *real* icon sits to the LEFT of its label and
    does not overlap any text box, whereas a false-positive (the model boxing the
    "O" of "Overlook", or a stray box on "Picnic Area") sits ON the label text.
    A detection is treated as an FP -- ignored for mapping -- when it BOTH:
      1. overlaps a text box, AND
      2. does not sit in an icon column.

    The icon column(s) are learned from the detections that clearly are real
    icons: the ones that do NOT overlap any text (they lie to the left of it).
    Only columns with at least two members count, so a lone stray box out in the
    text zone can't pretend to be its own icon column.  This still keeps a real
    icon whose own glyph OCR merged into the label (e.g. "CH Camp Host"): the
    icon remains aligned with the icon column, so rule 2 fails and it survives
    (it is then masked out before the final OCR).
    """
    if not config.filter_text_zone_false_positives or not texts or len(icons) < 2:
        return icons

    med_w = float(np.median([max(ic.width, 1) for ic in icons])) or 20.0
    tol = config.column_x_tolerance_factor * med_w

    def overlaps_text(ic: Detection) -> bool:
        return any(
            _fraction_inside(ic.bbox, t.bbox) >= config.text_on_icon_threshold
            for t in texts
        )

    # Learn the icon column(s) from the detections that clearly are real icons:
    # those NOT overlapping any text (they sit to the LEFT of their label).  A big
    # gap between sorted left-edges starts a new column (multi-column legends).
    clean_lefts = sorted(ic.bbox[0] for ic in icons if not overlaps_text(ic))
    columns: List[List[int]] = []
    for x in clean_lefts:
        if columns and x - columns[-1][-1] <= tol:
            columns[-1].append(x)
        else:
            columns.append([x])
    # Only a column backed by >= 2 aligned icons is trusted as a real column.
    col_bounds = [(c[0], c[-1]) for c in columns if len(c) >= 2]

    def in_icon_column(ic: Detection) -> bool:
        x = ic.bbox[0]
        return any(lo - tol <= x <= hi + tol for lo, hi in col_bounds)

    kept: List[Detection] = []
    dropped: List[Detection] = []
    for ic in icons:
        is_fp = overlaps_text(ic) and not in_icon_column(ic)
        (dropped if is_fp else kept).append(ic)

    for ic in dropped:
        LOGGER.info(
            "Dropping text-zone false positive '%s' at bbox=%s (sits on label text, outside icon column).",
            ic.class_name, ic.bbox,
        )
    # Safety net: never let the filter delete every detection.
    return kept if kept else icons


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

    Reading order is by the box's LEFT edge (x) first: these tokens are already
    on one line, and neighbouring words have slightly different box tops (e.g.
    lowercase "on" vs "Leash"), so sorting by y first would scramble the words
    ("Dogs Allowed on Leash" -> "Dogs Allowed Leash on").  The y tie-break only
    orders tokens that start at the same x.
    """
    ordered = sorted(tokens, key=lambda t: (t.bbox[0], t.bbox[1]))
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
