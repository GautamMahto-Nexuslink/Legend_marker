"""Checks that a legend label is never read truncated — no OCR needed.

    python3 test_label_truncation.py

Masking the icon boxes before the second OCR pass is what stops an icon's own
glyph being read as text.  But when a detection box reaches into its label (the
model boxing the "P" of "Parking Area", or a box drawn a few pixels too wide)
that same mask erases the label's first letter, and the label comes back as
"arking Area".  Two defences are tested here:

  1. label pixels are excluded from the mask   (label_text_to_protect)
  2. a still-partial label is rebuilt from the unmasked pass
                                               (repair_truncated_labels)
"""
from __future__ import annotations

import sys

import numpy as np

from legend_pipeline import PipelineConfig
from legend_pipeline.containers import Detection, OcrText
from legend_pipeline.matching import (
    label_text_to_protect,
    mask_icons_in_image,
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


# --------------------------------------------------------------------------- #
# 1. which text is protected from the mask
# --------------------------------------------------------------------------- #
print("== label_text_to_protect ==")
ICON = icon(33, 475, 160, 538, "parking")           # reaches into the label
LABEL = text("Parking", 117, 479, 275, 534)         # 27% inside the icon box
GLYPH = text("P", 40, 480, 80, 530)                 # the icon's own glyph
GLYPH_WORD = text("HAE", 36, 478, 85, 535)          # glyph misread as a word
WIDE_GLYPH = text("HAE", 30, 470, 90, 545)          # box CONTAINING the icon

kept = [t.text for t in label_text_to_protect(
    [LABEL, GLYPH, GLYPH_WORD, WIDE_GLYPH], [ICON], CFG)]
check("protects the label", "Parking" in kept, True)
check("not a 1-letter glyph read ('P')", "P" in kept, False)
check("not a glyph misread as a word ('HAE')", kept.count("HAE"), 0)
check("only the label survives", kept, ["Parking"])

# --------------------------------------------------------------------------- #
# 2. the mask leaves protected pixels alone
# --------------------------------------------------------------------------- #
print("\n== mask_icons_in_image(protect=...) ==")
img = np.random.RandomState(0).randint(0, 255, (600, 400, 3), dtype=np.uint8)
plain = mask_icons_in_image(img, [ICON], CFG.icon_mask_shrink)
guarded = mask_icons_in_image(img, [ICON], CFG.icon_mask_shrink, protect=[LABEL])
lx1, ly1, lx2, ly2 = LABEL.bbox
check("without protection the label pixels are destroyed",
      np.array_equal(plain[ly1:ly2, lx1:lx2], img[ly1:ly2, lx1:lx2]), False)
check("with protection they are intact",
      np.array_equal(guarded[ly1:ly2, lx1:lx2], img[ly1:ly2, lx1:lx2]), True)
gx1, gy1, gx2, gy2 = GLYPH.bbox
check("and the icon glyph is still masked",
      np.array_equal(guarded[gy1 + 2:gy2 - 2, gx1 + 2:gx2 - 2],
                     img[gy1 + 2:gy2 - 2, gx1 + 2:gx2 - 2]), False)
check("a detection that does not touch text masks exactly as before",
      np.array_equal(
          mask_icons_in_image(img, [icon(33, 475, 87, 538)], 1, protect=[LABEL]),
          mask_icons_in_image(img, [icon(33, 475, 87, 538)], 1)), True)

# --------------------------------------------------------------------------- #
# 3. repairing a label the masked pass still read partially
# --------------------------------------------------------------------------- #
print("\n== repair_truncated_labels ==")
ICONS = [icon(33, 129, 87, 175), ICON, icon(33, 554, 87, 598)]
UNMASKED = [text("Barn", 122, 129, 219, 171),
            text("Parking", 117, 479, 275, 534), text("Area", 273, 483, 370, 527),
            text("Public", 122, 554, 247, 598), text("Restroom", 249, 554, 437, 598)]


def repaired(masked_tokens):
    m = match_icons_to_text(ICONS, masked_tokens, CFG)
    m = repair_truncated_labels(m, ICONS, UNMASKED, CFG)
    return [m[i].text if m.get(i) else None for i in range(len(ICONS))]


BARN = text("Barn", 122, 129, 219, 171)
PUBLIC = [text("Public", 122, 554, 247, 598), text("Restroom", 249, 554, 437, 598)]
check("'arking Area' -> 'Parking Area'",
      repaired([BARN, text("arking", 140, 479, 275, 534),
                text("Area", 273, 483, 370, 527)] + PUBLIC)[1], "Parking Area")
check("'Area' alone -> 'Parking Area'",
      repaired([BARN, text("Area", 273, 483, 370, 527)] + PUBLIC)[1],
      "Parking Area")
check("a complete label is left alone",
      repaired([BARN, text("Parking", 117, 479, 275, 534),
                text("Area", 273, 483, 370, 527)] + PUBLIC),
      ["Barn", "Parking Area", "Public Restroom"])
check("a DIFFERENT reading is never substituted",
      repaired([BARN, text("Scenic", 117, 479, 275, 534)] + PUBLIC)[1], "Scenic")
check("other rows are untouched by a neighbour's repair",
      repaired([BARN, text("Area", 273, 483, 370, 527)] + PUBLIC),
      ["Barn", "Parking Area", "Public Restroom"])

print("\n== disabled ==")
off = PipelineConfig(api_key="x", project="x", repair_truncated_labels=False)
m = match_icons_to_text(ICONS, [BARN, text("Area", 273, 483, 370, 527)] + PUBLIC, off)
m = repair_truncated_labels(m, ICONS, UNMASKED, off)
check("repair_truncated_labels=False keeps the partial label", m[1].text, "Area")

print(f"\n{'ALL PASSED' if not FAILURES else str(FAILURES) + ' FAILURE(S)'}")
sys.exit(1 if FAILURES else 0)
