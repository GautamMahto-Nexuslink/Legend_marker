"""Checks for the known-icon ("blue") stage — no Roboflow / OCR needed.

    python3 test_icon_db.py

Queries the real database with real icon crops taken from the dataset, so these
exercise the whole stage: hash prefilter -> glyph re-scoring -> group vote ->
floor + margin decision.
"""
from __future__ import annotations

import os
import random
import sys

import cv2
import numpy as np

import legend_marker as lm
from legend_pipeline.icon_db import IconDatabase, IconDbMatch
from legend_pipeline.synonyms import DEFAULT_SYNONYMS_PATH, SynonymMap

DB_PATH = "icons_glyph_db.npz"
LEGACY_PATH = "icons_phash_flat.json"
CROPS = "/home/nls34/Downloads/Dataset_lengend_marker_viewer/IMP/Merged_Save_Crops"

FAILURES = 0


def check(label, got, expected):
    global FAILURES
    ok = got == expected
    FAILURES += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {label}"
          + (f"\n       got={got!r} want={expected!r}" if not ok else f"  -> {got!r}"))


if not os.path.isfile(DB_PATH):
    print(f"{DB_PATH} missing — run:  python3 build_icon_db.py", file=sys.stderr)
    sys.exit(2)

syn = SynonymMap.load(DEFAULT_SYNONYMS_PATH)
cfg = lm.PipelineConfig()
builder = lm.SignatureBuilder(cfg)
matcher = lm.SignatureMatcher(cfg)

print("== database ==")
db = IconDatabase.load(DB_PATH, synonyms=syn)
check("loads", bool(db), True)
check("has glyphs", db.has_glyphs, True)
check("classes", db.n_classes, 290)
check("groups collapse near-duplicate classes",
      db.group_of("Restrooms") == db.group_of("Public Restroom") ==
      db.group_of("Restroom"), True)

# --------------------------------------------------------------------------- #
print("\n== identifying real crops ==")
rng = random.Random(3)


def query(class_dir: str, file_name: str) -> IconDbMatch:
    crop = cv2.imread(os.path.join(CROPS, class_dir, file_name), cv2.IMREAD_COLOR)
    sig = builder.build(crop)
    # Hide this exact exemplar so the answer must come from OTHER exemplars.
    mask = np.ones(len(db), dtype=bool)
    same = (db.names == class_dir) & (db.files == file_name)
    mask[np.nonzero(same)[0]] = False
    return db.match(sig, matcher, mask=mask)


tested = correct = answered = 0
for class_dir in sorted(os.listdir(CROPS)):
    files = sorted(f for f in os.listdir(os.path.join(CROPS, class_dir))
                   if f.lower().endswith(".png"))
    if len(files) < 3:            # need other exemplars left after masking
        continue
    m = query(class_dir, rng.choice(files))
    tested += 1
    if m.name:
        answered += 1
        correct += (syn.canonical(m.name) or m.name.lower()) == \
                   (syn.canonical(class_dir) or class_dir.lower())

print(f"     {tested} classes probed, {answered} identified, {correct} in the "
      f"right group")
check("identifies most icons (coverage > 60%)", answered / tested > 0.60, True)
check("and is rarely wrong (accuracy > 90%)", correct / answered > 0.90, True)

# --------------------------------------------------------------------------- #
print("\n== decision gates ==")
# A blank glyph resembles nothing in particular: the stage must refuse it
# rather than name the least-bad candidate.
blank = lm.SignatureBuilder(cfg).build(np.full((40, 40, 3), 255, np.uint8))
m = db.match(blank, matcher)
check("refuses a featureless crop", m.name, None)
check("and says why", bool(m.rejected), True)

strict = IconDatabase.load(DB_PATH, synonyms=syn, margin=0.99)
crop_dir = "Marina" if os.path.isdir(os.path.join(CROPS, "Marina")) else None
if crop_dir:
    f = sorted(os.listdir(os.path.join(CROPS, crop_dir)))[0]
    sig = builder.build(cv2.imread(os.path.join(CROPS, crop_dir, f), cv2.IMREAD_COLOR))
    check("an impossible margin refuses everything",
          strict.match(sig, matcher).name, None)
    check("a zero margin always answers",
          IconDatabase.load(DB_PATH, synonyms=syn, margin=0.0, min_score=0.0)
          .match(sig, matcher).name is not None, True)

# --------------------------------------------------------------------------- #
print("\n== group voting matters ==")
ungrouped = IconDatabase.load(DB_PATH)          # no synonym map
check("without synonyms every class is its own group",
      ungrouped.group_of("Restrooms") == ungrouped.group_of("Restroom"), False)

# --------------------------------------------------------------------------- #
print("\n== legacy JSON fallback ==")
if os.path.isfile(LEGACY_PATH):
    legacy = IconDatabase.load(LEGACY_PATH, synonyms=syn)
    check("loads the {phash: class} JSON", bool(legacy), True)
    check("but has no glyphs to re-score", legacy.has_glyphs, False)
    f = sorted(os.listdir(os.path.join(CROPS, "Marina")))[0]
    sig = builder.build(cv2.imread(os.path.join(CROPS, "Marina", f), cv2.IMREAD_COLOR))
    m = legacy.match(sig, matcher)
    check("still identifies via the hash path", m.method, "hash")
    check("finds the icon it was built from",
          (syn.canonical(m.name or "") or "") == "marina", True)

check("a missing DB is an empty DB, not a crash",
      bool(IconDatabase.load("/nonexistent/icons.npz")), False)

print(f"\n{'ALL PASSED' if not FAILURES else str(FAILURES) + ' FAILURE(S)'}")
sys.exit(1 if FAILURES else 0)
