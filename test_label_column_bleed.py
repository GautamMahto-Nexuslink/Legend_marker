"""Checks that a legend label never swallows the next column's text — no OCR needed.

    python3 test_label_column_bleed.py

A label is grown by walking right from the icon and joining the tokens that sit
close enough together.  Two ways that walk used to run past the end of the label:

  1. The next column has no icon of its own (a plain "TRAILS:" list), so the
     next-icon column gate never fires — and on a real map the gap across to
     that column (67px) was SMALLER than the legal gap between a label's own
     words (74px), so no word-gap threshold could separate them either.
     "Reservation Headquarters" came back as
     "Reservation Headquarters Little McCool".        (_column_separators)

  2. repair_truncated_labels rebuilds the row from the UNMASKED OCR pass, which
     still contains every icon glyph read as a word.  It accepted any longer
     reading that *contained* the label, so a glyph tacked onto the end was
     adopted: "Trash/Recycle bin" -> "Trash/Recycle bin HAE".
                                                       (_is_truncation_of)

The coordinates below are the real OCR/detection boxes from the two maps that
first showed each failure.
"""
from __future__ import annotations

import sys

from legend_pipeline import PipelineConfig
from legend_pipeline.containers import Detection, OcrText
from legend_pipeline.matching import (
    _column_separators,
    filter_text_on_icons,
    match_icons_to_text,
    repair_truncated_labels,
)

FAILURES = 0


def check(label, got, expected):
    global FAILURES
    ok = got == expected
    FAILURES += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {label}"
          + (f"\n       got={got!r} want={expected!r}" if not ok else f"  -> {got!r}"))


CFG = PipelineConfig(api_key="x", project="x")


def icon(x1, y1, x2, y2, name="icon"):
    return Detection(class_name=name, confidence=0.9, bbox=(x1, y1, x2, y2))


def text(s, x1, y1, x2, y2):
    return OcrText(text=s, confidence=0.95, bbox=(x1, y1, x2, y2))


def labels(icons, tokens, unmasked=None, cfg=CFG):
    m = match_icons_to_text(icons, tokens, cfg)
    m = repair_truncated_labels(m, icons, unmasked if unmasked is not None else tokens, cfg)
    return [m[i].text if m.get(i) else None for i in range(len(icons))]


# --------------------------------------------------------------------------- #
# 1. a text-only second column (2017_MTOM_Trail_Map_page_1)
# --------------------------------------------------------------------------- #
print("== text-only next column (Mt Tom) ==")
# Left column: the legend proper.  Right column: the "TRAILS: (Color Key)" list,
# which has no icons at all.  Note the 67px gap Headquarters->Little is SMALLER
# than max_word_gap_factor * text height (2.0 * 37 = 74), so only the whitespace
# corridor between x=608 (where the widest label ends) and x=673 (where the
# right column starts) can tell the two columns apart.
MTOM_ICON = icon(48, 844, 109, 895, "forest_headquarter")
MTOM = [
    text("Picnic", 175, 540, 286, 577), text("Area", 290, 541, 367, 578),
    text("Reservation", 177, 854, 376, 891),
    text("Headquarters", 376, 848, 608, 904),
    text("Scenic", 175, 916, 290, 953), text("Vista", 290, 915, 378, 952),
    text("DO'C", 673, 532, 758, 569),            # right column, other rows
    text("Lake", 673, 807, 745, 839),
    text("Little", 675, 842, 754, 874),          # right column, same row
    text("McCool", 675, 876, 791, 909),          # right column, next line
]
MTOM_ICONS = [MTOM_ICON, icon(56, 533, 107, 584, "picnic_area"),
              icon(66, 920, 98, 951, "scenic_vista")]

check("the whitespace corridor between the columns is found",
      _column_separators(MTOM, CFG), [(608, 673)])
check("the label stops at its own column",
      labels(MTOM_ICONS, MTOM),
      ["Reservation Headquarters", "Picnic Area", "Scenic Vista"])
check("with the corridor gate off it bleeds again (the original bug)",
      labels(MTOM_ICONS, MTOM,
             cfg=PipelineConfig(api_key="x", project="x", text_column_gap_factor=0.0))[0],
      "Reservation Headquarters Little")

# --------------------------------------------------------------------------- #
# 2. an icon glyph read as a word (Andrew_Molera_State_Park..._page-0002)
# --------------------------------------------------------------------------- #
print("\n== icon glyph appended by the repair (Andrew Molera) ==")
# The hike-and-bike symbol OCRs as the word "HAE".  Its detection box is a
# little SMALLER than the drawn symbol, so masking does not erase the glyph and
# "HAE" comes back from both OCR passes — filter_text_on_icons is what removes
# it, and the repair was reading pass 1 before that filter had been applied.
TRASH = icon(31, 245, 56, 278, "trash")
HB = icon(309, 253, 340, 272, "hike_and_bike")
MOLERA_ICONS = [TRASH, HB]
LABELS = [text("Trash/Recycle", 77, 247, 220, 279), text("bin", 220, 248, 256, 273),
          text("Hike", 362, 249, 408, 273), text("and", 407, 249, 451, 274),
          text("Bike", 453, 249, 499, 273)]
GLYPH = text("HAE", 305, 250, 343, 273)               # sits on the HB icon
MOLERA = LABELS + [GLYPH]

check("the glyph is recognised as sitting on an icon",
      [t.text for t in filter_text_on_icons(MOLERA, MOLERA_ICONS, CFG)],
      [t.text for t in LABELS])
CLEAN = filter_text_on_icons(MOLERA, MOLERA_ICONS, CFG)
check("neither label picks up the glyph",
      labels(MOLERA_ICONS, CLEAN), ["Trash/Recycle bin", "Hike and Bike"])
check("the repair refuses to append it even unfiltered (the original bug)",
      labels(MOLERA_ICONS, CLEAN, unmasked=MOLERA)[0], "Trash/Recycle bin")

# --------------------------------------------------------------------------- #
# 3. the gate never fires on an ordinary single-column legend
# --------------------------------------------------------------------------- #
print("\n== single-column legend is unaffected ==")
ONE_COL = [text("Barn", 122, 129, 219, 171),
           text("Parking", 117, 479, 275, 534), text("Area", 273, 483, 370, 527),
           text("Public", 122, 554, 247, 598), text("Restroom", 249, 554, 437, 598)]
ONE_COL_ICONS = [icon(33, 129, 87, 175), icon(33, 475, 87, 538), icon(33, 554, 87, 598)]
check("no corridor is invented", _column_separators(ONE_COL, CFG), [])
check("labels read exactly as before", labels(ONE_COL_ICONS, ONE_COL),
      ["Barn", "Parking Area", "Public Restroom"])
check("a truncated label is still repaired",
      labels(ONE_COL_ICONS,
             [ONE_COL[0], text("Area", 273, 483, 370, 527)] + ONE_COL[3:],
             unmasked=ONE_COL)[1],
      "Parking Area")

# --------------------------------------------------------------------------- #
# 4. a corridor BEFORE the label must not cut the label off
# --------------------------------------------------------------------------- #
print("\n== corridor to the left of the label ==")
# Stray text out in the left margin ("Key") opens a corridor between it and the
# legend proper.  That corridor sits between the icon and its own label, so
# gating on the icon's right edge would leave every label unmatched; the label's
# own left edge is the right reference point.
MARGIN = [text("Key", 5, 10, 45, 45)] + ONE_COL
check("the margin corridor is found",
      _column_separators(MARGIN, CFG), [(45, 117)])
check("labels still read in full",
      labels(ONE_COL_ICONS, MARGIN),
      ["Barn", "Parking Area", "Public Restroom"])

print(f"\n{'ALL PASSED' if not FAILURES else str(FAILURES) + ' FAILURE(S)'}")
sys.exit(1 if FAILURES else 0)
