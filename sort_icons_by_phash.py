#!/usr/bin/env python3
"""
sort_icons_by_phash.py
======================

Sort a pile of icon images into per-class folders using the SAME glyph matching
the pipeline uses for its legend mapping — multi-scale template correlation +
ORB (``SignatureMatcher``), NOT raw pHash Hamming distance.

Why not pHash?  pHash is kept in the pipeline for reporting only; the actual
icon<->legend decision is the template+ORB score, which is far more reliable on
these small rendered glyphs.  This script mirrors that decision exactly.

INPUT
-----
  * ``input_folder`` — a parent folder containing sub-folders (any depth) of
    icon images to be sorted.
  * ``--ref-folder`` — the reference set: a parent folder whose sub-folders are
    named after classes and hold example icons for each class (e.g.
    ``Save_icons_modified``).  A glyph signature is built for every reference
    icon and every input icon is matched against them.

WHAT IT DOES
------------
For every image under ``input_folder`` it:
  1. builds a glyph signature (``SignatureBuilder``);
  2. ranks it against every reference signature with ``SignatureMatcher`` and
     picks the best class, applying the pipeline's gates:
       * score >= ``--match-threshold``  (absolute floor), AND
       * best beats 2nd-best by >= ``--match-margin`` (margin gate);
  3. on a match, MOVES (or copies with ``--copy``) the icon into
     ``<out>/<classname>/``;
  4. never overwrites: same-named files get a numeric suffix
     (``icon.png`` -> ``icon_1.png`` -> ...).

Unmatched icons are left in place by default (or moved to ``--unmatched-dir``).

Usage::

    python3 sort_icons_by_phash.py <input_folder> --out matched_icons \\
        --ref-folder Save_icons_modified
    python3 sort_icons_by_phash.py <input_folder> --out matched_icons \\
        --ref-folder Save_icons_modified --match-threshold 0.60 \\
        --match-margin 0.08 --copy -v
    python3 sort_icons_by_phash.py <input_folder> --out matched_icons \
    --ref-folder Save_icons_modified --copy -v

"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

# Reuse the pipeline's signature builder + matcher so the decision is identical
# to the legend mapping.
import legend_marker as lm

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

LOGGER = logging.getLogger("sort_icons_by_phash")


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------
def iter_images(root: str):
    """Yield every image file path under ``root`` (recursively), sorted."""
    for dirpath, _dirnames, filenames in sorted(os.walk(root)):
        for name in sorted(filenames):
            if name.lower().endswith(IMAGE_EXTS):
                yield os.path.join(dirpath, name)


def unique_destination(dest_dir: str, filename: str) -> str:
    """Return a path in ``dest_dir`` for ``filename`` that does not exist yet.

    ``icon.png`` -> ``icon.png``; if taken, ``icon_1.png``, ``icon_2.png``, ...
    so two same-named icons never overwrite each other.
    """
    stem, ext = os.path.splitext(filename)
    candidate = os.path.join(dest_dir, filename)
    n = 1
    while os.path.exists(candidate):
        candidate = os.path.join(dest_dir, f"{stem}_{n}{ext}")
        n += 1
    return candidate


# ---------------------------------------------------------------------------
# Reference signature database
# ---------------------------------------------------------------------------
def build_reference_db(
    ref_folder: str, sig_builder: "lm.SignatureBuilder"
) -> List[Tuple[str, Any]]:
    """Build ``[(classname, VisualSignature), ...]`` from the reference folder.

    Each immediate sub-folder of ``ref_folder`` is a class; every image inside
    contributes one reference signature for that class.
    """
    if not os.path.isdir(ref_folder):
        raise NotADirectoryError(f"Reference folder not found: {ref_folder}")

    db: List[Tuple[str, Any]] = []
    for class_name in sorted(os.listdir(ref_folder)):
        class_dir = os.path.join(ref_folder, class_name)
        if not os.path.isdir(class_dir):
            continue
        n = 0
        for name in sorted(os.listdir(class_dir)):
            if not name.lower().endswith(IMAGE_EXTS):
                continue
            img_path = os.path.join(class_dir, name)
            crop = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if crop is None:
                LOGGER.warning("Could not read reference icon %s — skipping.", img_path)
                continue
            db.append((class_name, sig_builder.build(crop)))
            n += 1
        if n:
            LOGGER.info("%-40s %d reference icon(s).", class_name, n)
    return db


def build_phash_matrix(db: List[Tuple[str, Any]]) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Pack every reference pHash into a (N, nbits) bool matrix for fast pruning.

    Returns ``(bits, valid)``: ``bits[i]`` is the flattened pHash of ``db[i]``
    (a zero row when that entry has no pHash), and ``valid[i]`` marks whether
    it was real.  Returns ``(None, ...)`` if no entry has a pHash.
    """
    nbits = None
    for _, sig in db:
        if sig.phash is not None:
            nbits = int(np.asarray(sig.phash.hash).size)
            break
    if nbits is None:
        return None, np.zeros(len(db), dtype=bool)

    bits = np.zeros((len(db), nbits), dtype=bool)
    valid = np.zeros(len(db), dtype=bool)
    for i, (_, sig) in enumerate(db):
        if sig.phash is not None:
            bits[i] = np.asarray(sig.phash.hash).flatten()
            valid[i] = True
    return bits, valid


def shortlist_indices(
    query_sig: Any, bits: Optional[np.ndarray], valid: np.ndarray, k: int
) -> Optional[np.ndarray]:
    """Indices of the ``k`` pHash-nearest reference entries (fast pre-filter).

    Returns ``None`` when shortlisting can't be applied (no matrix, no query
    pHash, or ``k <= 0``), so the caller falls back to ranking the full DB.
    """
    if bits is None or k <= 0 or query_sig.phash is None:
        return None
    q = np.asarray(query_sig.phash.hash).flatten()
    if q.size != bits.shape[1]:
        return None
    dist = np.count_nonzero(bits ^ q, axis=1)
    dist[~valid] = q.size + 1               # push invalid rows to the back.
    if k >= len(dist):
        return np.argsort(dist)
    # argpartition is O(n): grab the k smallest, then order just those.
    part = np.argpartition(dist, k)[:k]
    return part[np.argsort(dist[part])]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Sort icon images into per-class folders via glyph "
                    "(template+ORB) matching — the same decision as the "
                    "pipeline's legend mapping.")
    parser.add_argument("input_folder",
                        help="Parent folder containing sub-folders of icons to sort.")
    parser.add_argument("--out", required=True,
                        help="Destination root; per-class sub-folders go here.")
    parser.add_argument("--ref-folder", default="Save_icons_modified",
                        help="Reference set: parent folder of class sub-folders "
                             "of example icons (default: Save_icons_modified).")
    parser.add_argument("--match-threshold", type=float, default=0.60,
                        help="Absolute score floor to accept a match "
                             "(default: 0.60, same as the pipeline).")
    parser.add_argument("--match-margin", type=float, default=0.02,
                        help="Best must beat 2nd-best by this margin "
                             "(default: 0.08, same as the pipeline).")
    parser.add_argument("--copy", action="store_true",
                        help="Copy instead of move (leaves the source intact).")
    parser.add_argument("--unmatched-dir", default="",
                        help="If set, unmatched icons are moved/copied here "
                             "(default: leave them in place).")
    parser.add_argument("--shortlist", type=int, default=60,
                        help="Speed knob: template+ORB-rank only the N "
                             "pHash-nearest references per icon instead of all "
                             "of them (0 = rank every reference; default: 60).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    if not os.path.isdir(args.input_folder):
        sys.exit(f"Input folder not found: {args.input_folder}")

    # Config drives glyph segmentation + template/ORB weights; identical to the
    # pipeline defaults so the score means the same thing.
    config = lm.PipelineConfig(
        match_score_threshold=args.match_threshold,
        match_margin=args.match_margin,
    )
    sig_builder = lm.SignatureBuilder(config)
    matcher = lm.SignatureMatcher(config)

    print(f"Building reference signatures from {args.ref_folder} ...")
    db = build_reference_db(args.ref_folder, sig_builder)
    if not db:
        sys.exit(f"No reference signatures built from {args.ref_folder}")
    n_classes = len({c for c, _ in db})
    print(f"Built {len(db)} reference signature(s) across {n_classes} class(es).")

    # Fast pHash pre-filter: rank only the N nearest references per icon.
    ref_bits, ref_valid = build_phash_matrix(db)
    if args.shortlist > 0 and ref_bits is not None:
        print(f"pHash shortlist: ranking top {args.shortlist} of {len(db)} "
              f"references per icon.")

    transfer = shutil.copy2 if args.copy else shutil.move
    verb = "Copied" if args.copy else "Moved"

    matched = unmatched = errors = 0
    for img_path in iter_images(args.input_folder):
        try:
            crop = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if crop is None:
                raise ValueError("cv2 could not read the image")
            sig = sig_builder.build(crop)

            idx = shortlist_indices(sig, ref_bits, ref_valid, args.shortlist)
            candidates = db if idx is None else [db[i] for i in idx]
            rows = matcher.rank(sig, candidates)
            top = rows[0] if rows else None
            name = top["name"] if top else None
            score = top["score"] if top else -1.0
            second = rows[1]["score"] if len(rows) > 1 else 0.0
            margin = score - second
            hamming = top["hamming"] if top else None
        except Exception as exc:
            LOGGER.error("Failed on %s: %s", img_path, exc)
            errors += 1
            continue

        passes_floor = name is not None and score >= args.match_threshold
        passes_margin = (len(rows) < 2) or (margin >= args.match_margin)
        cls = name if (passes_floor and passes_margin) else None

        if cls is not None:
            dest_dir = os.path.join(args.out, cls)
            os.makedirs(dest_dir, exist_ok=True)
            dest = unique_destination(dest_dir, os.path.basename(img_path))
            transfer(img_path, dest)
            matched += 1
            print(f"[score={score:.3f} margin={margin:.3f} hamming={hamming}] "
                  f"{os.path.basename(img_path)} -> {cls}")
        else:
            unmatched += 1
            reason = "score<floor" if not passes_floor else "margin<gate"
            print(f"[score={score:.3f} margin={margin:.3f} hamming={hamming}] "
                  f"{os.path.basename(img_path)} -> NO MATCH "
                  f"(best '{name}', {reason})")
            if args.unmatched_dir:
                os.makedirs(args.unmatched_dir, exist_ok=True)
                dest = unique_destination(args.unmatched_dir,
                                          os.path.basename(img_path))
                transfer(img_path, dest)

    print(f"\n--- Summary ---")
    print(f"{verb} (matched): {matched}")
    print(f"Unmatched       : {unmatched}"
          + (f" -> {args.unmatched_dir}" if args.unmatched_dir else " (left in place)"))
    print(f"Errors          : {errors}")
    print(f"Output root     : {args.out}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
