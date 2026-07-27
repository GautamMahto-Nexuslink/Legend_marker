"""Build the semantic class-grouping artifacts for the legend icon dataset.

Sources of class names (both are used so nothing is lost):
  * values of icons_phash_flat.json   -> classes that have at least one icon hash
  * sub-folder names of CROPS_DIR     -> classes that exist on disk (incl. empty ones)

Every original class name is kept, no matter if another class is a duplicate,
a misspelling or a case variant of it ("Wi-Fi"/"WiFi", "Youth camping"/"Youth
Camping", "Contact Station"/"Contact_Station", ...).

Outputs (written next to this file):
  icon_class_synonyms.json   canonical name -> deduped list of class names,
                             synonyms, alternate spellings and search terms
  class_to_canonical.json    every original class name -> canonical name
  canonical_to_classes.json  canonical name -> original class names in it
  search_index.json          search term -> canonical names it can resolve to

Usage:
    python build_class_synonyms.py                 # build + validate + write
    python build_class_synonyms.py --check         # validate only, write nothing
    python build_class_synonyms.py --search "wheelchair parking"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

from icon_class_synonyms import ICON_CLASS_SYNONYMS

HERE = os.path.dirname(os.path.abspath(__file__))
PHASH_FLAT = os.path.join(HERE, "icons_phash_flat.json")
CROPS_DIR = "/home/nls34/Downloads/Dataset_lengend_marker_viewer/IMP/Merged_Save_Crops"


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def normalize(name: str) -> str:
    """Lowercase, unify separators/ampersands and squeeze whitespace.

    'Contact_Station' -> 'contact station', 'Campground Hike & Bike' ->
    'campground hike and bike', 'Scenic  Vista' -> 'scenic vista'.
    """
    s = name.strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[_/]+", " ", s)
    s = re.sub(r"\s*:\s*", " : ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# --------------------------------------------------------------------------- #
# class collection
# --------------------------------------------------------------------------- #
def collect_classes(phash_flat: str = PHASH_FLAT, crops_dir: str = CROPS_DIR) -> list[str]:
    """Return every original class name from both sources, duplicates included."""
    names: list[str] = []

    if os.path.exists(phash_flat):
        with open(phash_flat) as fh:
            data = json.load(fh)
        names.extend(data.values() if isinstance(data, dict) else data)
    else:
        print(f"warning: {phash_flat} not found", file=sys.stderr)

    if os.path.isdir(crops_dir):
        names.extend(
            d for d in os.listdir(crops_dir) if os.path.isdir(os.path.join(crops_dir, d))
        )
    else:
        print(f"warning: {crops_dir} not found", file=sys.stderr)

    # keep every distinct spelling/casing, drop only exact repeats
    seen, out = set(), []
    for n in names:
        n = n.strip()
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return sorted(out, key=str.lower)


# --------------------------------------------------------------------------- #
# lookup tables
# --------------------------------------------------------------------------- #
def build_term_index(groups: dict[str, list[str]]) -> dict[str, str]:
    """normalized term -> canonical name (first group wins, canonical keys win)."""
    index: dict[str, str] = {}
    for canonical, terms in groups.items():
        index.setdefault(normalize(canonical), canonical)
    for canonical, terms in groups.items():
        for term in terms:
            index.setdefault(normalize(term), canonical)
    return index


def build(classes: list[str], groups: dict[str, list[str]]):
    term_index = build_term_index(groups)

    class_to_canonical: dict[str, str] = {}
    unmapped: list[str] = []
    for cls in classes:
        canonical = term_index.get(normalize(cls))
        if canonical is None:
            unmapped.append(cls)
        else:
            class_to_canonical[cls] = canonical

    canonical_to_classes: dict[str, list[str]] = {c: [] for c in groups}
    for cls, canonical in class_to_canonical.items():
        canonical_to_classes[canonical].append(cls)
    for v in canonical_to_classes.values():
        v.sort(key=str.lower)

    # original class names first (lowercased), then the curated synonyms
    merged: dict[str, list[str]] = {}
    for canonical, terms in groups.items():
        ordered, seen = [], set()
        for term in [canonical] + [c.lower() for c in canonical_to_classes[canonical]] + terms:
            key = normalize(term)
            if key not in seen:
                seen.add(key)
                ordered.append(term.lower())
        merged[canonical] = ordered

    search_index: dict[str, list[str]] = defaultdict(list)
    for canonical, terms in merged.items():
        for term in terms:
            key = normalize(term)
            if canonical not in search_index[key]:
                search_index[key].append(canonical)

    return {
        "merged": merged,
        "class_to_canonical": class_to_canonical,
        "canonical_to_classes": canonical_to_classes,
        "search_index": dict(sorted(search_index.items())),
        "unmapped": unmapped,
    }


# --------------------------------------------------------------------------- #
# search helper
# --------------------------------------------------------------------------- #
def search(query: str, merged: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Return (canonical, matched_term) for exact then substring matches."""
    q = normalize(query)
    exact, partial = [], []
    for canonical, terms in merged.items():
        for term in terms:
            t = normalize(term)
            if t == q:
                exact.append((canonical, term))
                break
            if q in t or t in q:
                partial.append((canonical, term))
                break
    return exact + partial


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phash", default=PHASH_FLAT, help="path to icons_phash_flat.json")
    ap.add_argument("--crops", default=CROPS_DIR, help="path to the crops directory")
    ap.add_argument("--out", default=HERE, help="output directory")
    ap.add_argument("--check", action="store_true", help="validate only, write nothing")
    ap.add_argument("--search", metavar="QUERY", help="look a term up and exit")
    args = ap.parse_args()

    classes = collect_classes(args.phash, args.crops)
    result = build(classes, ICON_CLASS_SYNONYMS)

    if args.search:
        for canonical, term in search(args.search, result["merged"]):
            folders = result["canonical_to_classes"][canonical]
            print(f"{canonical:<30} (matched '{term}')  folders: {', '.join(folders) or '-'}")
        return 0

    print(f"classes collected : {len(classes)}")
    print(f"canonical groups  : {len(ICON_CLASS_SYNONYMS)}")
    print(f"search terms      : {len(result['search_index'])}")

    empty = [c for c, v in result["canonical_to_classes"].items() if not v]
    if empty:
        print(f"groups with no source class ({len(empty)}): {', '.join(empty)}")

    ambiguous = {t: c for t, c in result["search_index"].items() if len(c) > 1}
    if ambiguous:
        print(f"terms mapping to >1 group ({len(ambiguous)}):")
        for term, canonicals in sorted(ambiguous.items()):
            print(f"  {term!r} -> {canonicals}")

    if result["unmapped"]:
        print(f"UNMAPPED CLASSES ({len(result['unmapped'])}):", file=sys.stderr)
        for cls in result["unmapped"]:
            print(f"  {cls}", file=sys.stderr)
        return 1
    print("all classes mapped ✓")

    if args.check:
        return 0

    files = {
        "icon_class_synonyms.json": result["merged"],
        "class_to_canonical.json": dict(sorted(result["class_to_canonical"].items(), key=lambda kv: kv[0].lower())),
        "canonical_to_classes.json": result["canonical_to_classes"],
        "search_index.json": result["search_index"],
    }
    for name, payload in files.items():
        path = os.path.join(args.out, name)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
