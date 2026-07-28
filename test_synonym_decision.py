"""Checks for the semantic class reconciliation (no Roboflow / OCR needed).

    python test_synonym_decision.py
"""
from __future__ import annotations

import json
import sys

from legend_pipeline import PipelineConfig, LegendMarkerPipeline
from legend_pipeline.synonyms import SynonymMap, DEFAULT_SYNONYMS_PATH

FAILURES = 0


def check(label, got, expected):
    global FAILURES
    ok = got == expected
    FAILURES += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {label}\n       got={got!r} want={expected!r}"
          if not ok else f"ok   {label}  -> {got!r}")


# --------------------------------------------------------------------------- #
# 1. synonym lookup
# --------------------------------------------------------------------------- #
print("== SynonymMap ==")
syn = SynonymMap.load(DEFAULT_SYNONYMS_PATH)
check("groups loaded", len(syn) > 0, True)
check("canonical('Overlook')", syn.canonical("Overlook"), "observation area")
check("canonical('Scenic  Vista')", syn.canonical("Scenic  Vista"), "observation area")
check("canonical('0verlook')  [OCR noise]", syn.canonical("0verlook"), "observation area")
check("canonical('Contact_Station')", syn.canonical("Contact_Station"), "contact station")
check("canonical('Campground Hike & Bike')",
      syn.canonical("Campground Hike & Bike"), "hike and bike campground")
check("same_concept('Observation Area','Overlook ')",
      syn.same_concept("Observation Area", "Overlook "), True)
check("same_concept('Parking','Overlook')",
      syn.same_concept("Parking", "Overlook"), False)
check("same_concept('Overlook Trail','Hiking')  [ambiguous]",
      syn.same_concept("Overlook Trail", "Hiking"), False)

print("\n-- neighbour graph --")
check("relation('Bicycling','Motor Bike Trail')",
      syn.relation("Bicycling", "Motor Bike Trail"), "related")
check("relation('Bicycling','Biking')", syn.relation("Bicycling", "Biking"), "same")
check("relation('Bicycling','Restrooms')",
      syn.relation("Bicycling", "Restrooms"), None)
check("relation('Overlook','Mountain Summit')",
      syn.relation("Overlook", "Mountain Summit"), "related")
check("graph is symmetric",
      sorted(g for g in syn.related_graph
             if any(g not in syn.related_graph[n] for n in syn.related_graph[g])), [])
check("every group has neighbours",
      [g for g in syn.groups if not syn.related_graph.get(g)], [])

with open("class_to_canonical.json") as fh:
    class_to_canonical = json.load(fh)
mismatched = [c for c, k in class_to_canonical.items() if syn.canonical(c) != k]
check(f"all {len(class_to_canonical)} dataset classes resolve", mismatched, [])

# --------------------------------------------------------------------------- #
# 2. pipeline decision — every branch
# --------------------------------------------------------------------------- #
print("\n== _decide_class ==")


def make_pipeline(**cfg_kw) -> LegendMarkerPipeline:
    """A pipeline with only what _decide_class needs.

    ``__init__`` eagerly builds the Roboflow detector (network + API key), which
    this test neither has nor needs — the decision logic depends solely on the
    config and the synonym map.
    """
    cfg_kw.setdefault("synonyms_path", DEFAULT_SYNONYMS_PATH)
    pipe = LegendMarkerPipeline.__new__(LegendMarkerPipeline)
    pipe.config = PipelineConfig(api_key="x", project="x", **cfg_kw)
    pipe._synonyms = None
    return pipe


pipe = make_pipeline()


def legend(*pairs):
    """Ranked legend candidates: legend(("Overlook", 0.71), ("Parking", 0.4))."""
    return [{"name": n, "score": s, "breakdown": {}, "hamming": None}
            for n, s in pairs]


def decide(**kw):
    args = dict(original_class="icon", db_class=None, db_nearest_name=None,
                db_nearest_dist=None, rows=[])
    args.update(kw)
    d = pipe._decide_class(**args)
    return d["final_class"], d["match_method"], d["canonical"]


# the reported case: DB hash says "observation area", legend OCR reads "Overlook"
check("DB hit + synonymous legend -> legend wording",
      decide(db_class="observation area", db_nearest_dist=0,
             rows=legend(("Overlook ", 0.71), ("Parking", 0.30))),
      ("Overlook", "phash_db+same", "observation area"))

# the second reported case: hash says Bicycling, legend has Motor Bike Trail —
# allowed because that legend entry is also the best visual match and passes.
check("DB hit + NEARBY legend class that wins visually -> legend wording",
      decide(db_class="Bicycling", db_nearest_dist=0,
             rows=legend(("Motor Bike Trail", 0.81), ("Restrooms", 0.42))),
      ("Motor Bike Trail", "phash_db+related", "motorbike trail"))

check("same-concept legend beats a nearby one even when it scores lower",
      decide(db_class="Bicycling", db_nearest_dist=0,
             rows=legend(("Motor Bike Trail", 0.66), ("Biking", 0.31))),
      ("Biking", "phash_db+same", "biking"))

print("\n-- related needs visual evidence (regression: 002_Campground) --")
# Exact hash hit for Trash; legend's only nearby word is Campground at 0.477,
# which already FAILED its own floor gate.  Must not rename.
TRASH_ROWS = legend(("Campground", 0.477), ("Ramada", 0.380), ("Parking", 0.352))
check("weak related label cannot override an exact hash hit",
      decide(original_class="litter_receptacle", db_class="Trash",
             db_nearest_dist=0, rows=TRASH_ROWS),
      ("Trash", "phash_db", "trash"))

check("related rejected when it is not the best legend match",
      decide(db_class="Bicycling", db_nearest_dist=0,
             rows=legend(("Restrooms", 0.72), ("Motor Bike Trail", 0.65))),
      ("Bicycling", "phash_db", "biking"))

check("related rejected when the margin gate fails",
      decide(db_class="Bicycling", db_nearest_dist=0,
             rows=legend(("Motor Bike Trail", 0.72), ("Restrooms", 0.70))),
      ("Bicycling", "phash_db", "biking"))

_d = pipe._decide_class(original_class="litter_receptacle", db_class="Trash",
                        db_nearest_name="Trash", db_nearest_dist=0, rows=TRASH_ROWS)
check("rejection reason is recorded",
      (_d["found_relation"], _d["relation"],
       "0.477" in (_d["relation_rejected"] or "")),
      ("related", None, True))

_weak = make_pipeline(synonym_related_requires_legend_win=False)
_dw = _weak._decide_class(original_class="litter_receptacle", db_class="Trash",
                          db_nearest_name="Trash", db_nearest_dist=0,
                          rows=TRASH_ROWS)
check("--weak-related-ok restores the permissive behaviour",
      (_dw["final_class"], _dw["match_method"]),
      ("Campground", "phash_db+related"))

check("DB hit + unrelated legend -> DB class (unchanged behaviour)",
      decide(db_class="observation area", db_nearest_dist=0,
             rows=legend(("Parking", 0.71))),
      ("observation area", "phash_db", "observation area"))

check("DB hit, no legend at all -> DB class",
      decide(db_class="restroom", db_nearest_dist=0),
      ("restroom", "phash_db", "restroom"))

check("no DB, legend clears both gates -> legend name",
      decide(db_nearest_name="picnic area", db_nearest_dist=90,
             rows=legend(("Overlook", 0.80), ("Parking", 0.60))),
      ("Overlook", "legend", "observation area"))

check("DB near-miss agrees with a below-floor legend -> RESCUED",
      decide(db_nearest_name="observation area", db_nearest_dist=9,
             rows=legend(("Scenic View", 0.55), ("Parking", 0.53))),
      ("Scenic View", "synonym_agree", "observation area"))

check("DB near-miss only RELATED to a below-floor legend -> kept (too weak)",
      decide(db_nearest_name="Bicycling", db_nearest_dist=9,
             rows=legend(("Motor Bike Trail", 0.55), ("Parking", 0.53))),
      ("icon", None, "biking"))

check("DB near-miss disagrees, legend below floor -> kept",
      decide(db_nearest_name="parking", db_nearest_dist=9,
             rows=legend(("Scenic View", 0.55), ("Overlook", 0.53))),
      ("icon", None, "parking"))

check("rescue refused when the DB near-miss is far away",
      decide(db_nearest_name="observation area", db_nearest_dist=120,
             rows=legend(("Scenic View", 0.55), ("Parking", 0.53))),
      ("icon", None, "observation area"))

check("rescue refused when the legend score is too low",
      decide(db_nearest_name="observation area", db_nearest_dist=9,
             rows=legend(("Scenic View", 0.20), ("Parking", 0.19))),
      ("icon", None, "observation area"))

# --------------------------------------------------------------------------- #
# 3. naming policies
# --------------------------------------------------------------------------- #
print("\n== synonym_naming ==")
for policy, expected in (("legend", "Overlook"),
                         ("canonical", "observation area"),
                         ("db", "Observation Area")):
    pipe.config.synonym_naming = policy
    check(f"policy={policy}",
          decide(db_class="Observation Area", db_nearest_dist=0,
                 rows=legend(("Overlook", 0.71)))[0],
          expected)
pipe.config.synonym_naming = "legend"

# --------------------------------------------------------------------------- #
# 4. disabled map == previous behaviour
# --------------------------------------------------------------------------- #
print("\n== synonyms disabled ==")
off = make_pipeline(synonyms_path="")
d = off._decide_class(original_class="icon", db_class="observation area",
                      db_nearest_name="observation area", db_nearest_dist=0,
                      rows=legend(("Overlook", 0.71)))
check("DB wins, no reconciliation",
      (d["final_class"], d["match_method"], d["synonym_agree"]),
      ("observation area", "phash_db", False))

# --------------------------------------------------------------------------- #
# 5. visualization colour per method
# --------------------------------------------------------------------------- #
print("\n== visualize_map colours ==")
import numpy as np  # noqa: E402
from legend_pipeline.containers import Detection  # noqa: E402
from legend_pipeline.visualization import visualize_map  # noqa: E402

METHOD_COLOR = {"phash_db": (255, 0, 0), "phash_db+same": (200, 0, 200),
                "phash_db+related": (200, 0, 200), "synonym_agree": (200, 0, 200),
                "legend": (0, 170, 0), None: (0, 140, 255)}
for method, want in METHOD_COLOR.items():
    img = np.zeros((80, 240, 3), dtype=np.uint8)
    det = Detection(class_name="icon", confidence=0.9, bbox=(10, 30, 60, 70))
    out = visualize_map(img, [det],
                        [{"class": "X", "match_method": method,
                          "renamed": method is not None, "match_score": 0.5}])
    drawn = {tuple(int(c) for c in px)
             for px in out.reshape(-1, 3) if tuple(px) != (0, 0, 0)}
    check(f"method={method!r}", want in drawn, True)

# --------------------------------------------------------------------------- #
# 6. per-crop .txt stage block
# --------------------------------------------------------------------------- #
print("\n== stage_report_lines ==")
from legend_pipeline.reporting import stage_report_lines  # noqa: E402
from legend_pipeline.visualization import color_for_method  # noqa: E402


def report(**kw):
    """Run a decision, render its stage block, return (lines, decision)."""
    d = pipe._decide_class(original_class=kw["original_class"],
                           db_class=kw["db_class"],
                           db_nearest_name=kw["db_nearest_name"],
                           db_nearest_dist=kw["db_nearest_dist"],
                           rows=kw["rows"])
    lines = stage_report_lines(
        original_class=kw["original_class"], final_class=d["final_class"],
        renamed=d["renamed"], match_method=d["match_method"], db_enabled=True,
        db_class=kw["db_class"], db_nearest_name=kw["db_nearest_name"],
        db_nearest_dist=kw["db_nearest_dist"], db_max_hamming=10,
        legend_name=d["legend_name"], legend_score=d["legend_score"],
        legend_margin=d["legend_margin"], passes_floor=d["passes_floor"],
        passes_margin=d["passes_margin"], score_threshold=0.60,
        margin_threshold=0.08, syn_enabled=True, relation=d["relation"],
        found_relation=d["found_relation"],
        relation_rejected=d["relation_rejected"],
        semantic_legend=d["semantic_legend"], semantic_score=d["semantic_score"],
        canonical=d["canonical"], naming="legend", rescued=d["rescued"],
        n_legend_entries=len(kw["rows"]))
    return lines, d


def stage_value(lines, colour):
    """The class name a given colour's stage line reports."""
    for line in lines:
        if line.strip().startswith(colour):
            return line.split(":", 1)[1].strip().split("  ")[0].strip()
    return None


SCENARIOS = {
    "blue (hash hit, legend unrelated)": dict(
        original_class="Parking", db_class="Marina", db_nearest_name="Marina",
        db_nearest_dist=3, rows=legend(("Restrooms", 0.42), ("Playground", 0.31))),
    "magenta (hash hit, same class in legend)": dict(
        original_class="Parking", db_class="Marina", db_nearest_name="Marina",
        db_nearest_dist=3, rows=legend(("Harbor", 0.71), ("Restrooms", 0.30))),
    "green (hash miss, legend text wins)": dict(
        original_class="Parking", db_class=None, db_nearest_name="Boat Dock",
        db_nearest_dist=47, rows=legend(("Marina", 0.78), ("Restrooms", 0.40))),
    "orange (nothing matches)": dict(
        original_class="Parking", db_class=None, db_nearest_name="Boat Dock",
        db_nearest_dist=61, rows=legend(("Restrooms", 0.41), ("Playground", 0.39))),
    "blue (weak related rejected: 002_Campground)": dict(
        original_class="litter_receptacle", db_class="Trash",
        db_nearest_name="Trash", db_nearest_dist=0,
        rows=legend(("Campground", 0.477), ("Ramada", 0.380))),
}
EXPECTED = {
    "blue (hash hit, legend unrelated)":
        {"orange": "Parking", "blue": "Marina", "green": "Restrooms",
         "magenta": "-", "final": "Marina", "colour": "blue"},
    "magenta (hash hit, same class in legend)":
        {"orange": "Parking", "blue": "Marina", "green": "Harbor",
         "magenta": "Harbor", "final": "Harbor", "colour": "magenta"},
    "green (hash miss, legend text wins)":
        {"orange": "Parking", "blue": "-", "green": "Marina",
         "magenta": "Marina", "final": "Marina", "colour": "green"},
    "orange (nothing matches)":
        {"orange": "Parking", "blue": "-", "green": "Restrooms",
         "magenta": "-", "final": "Parking", "colour": "orange"},
    "blue (weak related rejected: 002_Campground)":
        {"orange": "litter_receptacle", "blue": "Trash", "green": "Campground",
         "magenta": "-", "final": "Trash", "colour": "blue"},
}
for title, kw in SCENARIOS.items():
    lines, d = report(**kw)
    want = EXPECTED[title]
    got = {c: stage_value(lines, c) for c in ("orange", "blue", "green", "magenta")}
    got["final"] = stage_value(lines, "FINAL")
    got["colour"] = color_for_method(d["match_method"], d["renamed"])[0]
    check(title, got, want)
    # exactly one stage is marked as the winner
    check(f"  one FINAL marker in {title}",
          sum("<== FINAL" in ln for ln in lines), 1)

print(f"\n{'ALL PASSED' if not FAILURES else str(FAILURES) + ' FAILURE(S)'}")
sys.exit(1 if FAILURES else 0)
