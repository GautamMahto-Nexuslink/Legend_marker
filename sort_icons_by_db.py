#!/usr/bin/env python3
"""
sort_icons_by_db.py
===================

Sort a pile of icon images into per-class folders using the known-icon
**database** (``icons_glyph_db.npz``) — the same "blue stage" the pipeline uses
to identify a map icon.

Relation to ``sort_icons_by_phash.py``
-------------------------------------
That script needs a *reference folder* of example icons: it re-reads and
re-hashes every one of them on each run (~7500 images, and the cost is paid
again every time), then template-ranks the whole set per input icon.

This one loads the pre-built database instead, so:

  * **startup is instant** — glyphs and hashes were computed once by
    ``build_icon_db.py``, not on every run;
  * **the decision is the pipeline's**, not an approximation of it: three-hash
    prefilter -> template+ORB re-scoring of the shortlist -> per-**semantic
    group** vote -> absolute floor + runner-up margin;
  * **votes count** — a class with several exemplars agreeing wins over one
    lucky outlier, and near-duplicate class names ("Restroom" / "Restrooms" /
    "Public Restroom") stop splitting their votes against each other;
  * a refusal comes with a **reason** and the runners-up, so a mis-sort is
    diagnosable instead of just wrong.

Sorting into the same names the pipeline would assign is the point: this is how
you grow the dataset that the pipeline then reads back.

INPUT
-----
  * ``input_folder`` — a folder (searched recursively) of icon images to sort.
  * ``--db`` — ``icons_glyph_db.npz`` from ``build_icon_db.py``.  A legacy
    ``{phash_hex: classname}`` JSON also works, but without stored glyphs only
    the hash prefilter can run, which is much weaker.

WHAT IT DOES
------------
For every image under ``input_folder`` it:
  1. builds a glyph signature (``SignatureBuilder``, identical to the pipeline);
  2. identifies it against the database, applying the floor + margin gates;
  3. on a hit, MOVES (or copies with ``--copy``) the icon into
     ``<out>/<classname>/``;
  4. never overwrites: same-named files get a numeric suffix
     (``icon.png`` -> ``icon_1.png`` -> ...).

Unmatched icons are left in place by default (or moved to ``--unmatched-dir``).
``--dry-run`` decides everything and moves nothing — always worth doing first.

Usage::

    # see what would happen, nothing is touched
    python3 sort_icons_by_db.py <input_folder> --out sorted_icons --dry-run

    # sort for real, keeping the originals
    python3 sort_icons_by_db.py <input_folder> --out sorted_icons --copy -v

    # be stricter (fewer, safer matches) and quarantine the rest for review
    python3 sort_icons_by_db.py <input_folder> --out sorted_icons \\
        --margin 0.10 --unmatched-dir needs_review --copy

    # write per-class folders named by SEMANTIC GROUP instead of class name
    python3 sort_icons_by_db.py <input_folder> --out sorted_icons --by-group
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from collections import Counter
from typing import Optional

import cv2

# Reuse the pipeline's builder/matcher so a score here means what it means there.
import legend_marker as lm
from legend_pipeline.icon_db import IconDatabase
from legend_pipeline.synonyms import DEFAULT_SYNONYMS_PATH, SynonymMap
from legend_pipeline.utils import sanitize_filename

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

LOGGER = logging.getLogger("sort_icons_by_db")


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
# Main
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sort icon images into per-class folders using the known-icon "
                    "database (icons_glyph_db.npz) — the pipeline's own "
                    "identification stage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_folder",
                   help="Folder of icons to sort (searched recursively).")
    p.add_argument("--out", required=True,
                   help="Destination root; per-class sub-folders go here.")
    p.add_argument("--db", default=lm.PipelineConfig.icon_db_path,
                   help="icons_glyph_db.npz from build_icon_db.py (a legacy "
                        "{phash: class} JSON also works, but far more weakly).")

    # Decision gates — same meaning as the pipeline's icon-DB knobs.
    p.add_argument("--margin", type=float, default=0.06,
                   help="Winner must beat the runner-up group by this. THE dial "
                        "for precision vs coverage: raise it to sort fewer icons "
                        "but misfile fewer of them.")
    p.add_argument("--min-score", type=float, default=0.55,
                   help="Absolute floor on the voted score.")
    p.add_argument("--prefilter-k", type=int, default=24,
                   help="Candidates kept by the hash prefilter and re-scored.")
    p.add_argument("--no-group-vote", action="store_true",
                   help="Vote per raw class instead of per semantic group "
                        "(splits 'Restroom'/'Restrooms' and sorts less).")
    p.add_argument("--synonyms", default=DEFAULT_SYNONYMS_PATH,
                   help="Synonym map used for the group vote.")

    # What to do with the files.
    p.add_argument("--by-group", action="store_true",
                   help="Name the destination folder after the semantic group "
                        "(e.g. 'restroom') instead of the matched class name "
                        "(e.g. 'Public Restroom') — collapses duplicates.")
    p.add_argument("--copy", action="store_true",
                   help="Copy instead of move (leaves the source intact).")
    p.add_argument("--unmatched-dir", default="",
                   help="If set, unmatched icons go here (default: left alone).")
    p.add_argument("--dry-run", action="store_true",
                   help="Decide and report, but move/copy nothing.")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after N icons (0 = all). Handy with --dry-run.")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="Only print the summary, not one line per icon.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    if not os.path.isdir(args.input_folder):
        sys.exit(f"Input folder not found: {args.input_folder}")
    if not os.path.isfile(args.db):
        sys.exit(f"Database not found: {args.db}\n"
                 f"Build it first:  python3 build_icon_db.py <crops folder> -v")

    config = lm.PipelineConfig()
    sig_builder = lm.SignatureBuilder(config)
    matcher = lm.SignatureMatcher(config)

    synonyms = None if args.no_group_vote else SynonymMap.load(args.synonyms)
    db = IconDatabase.load(
        args.db,
        prefilter_k=args.prefilter_k,
        min_score=args.min_score,
        margin=args.margin,
        synonyms=synonyms,
    )
    if not db:
        sys.exit(f"No usable entries in {args.db}")

    print(f"Database : {len(db)} icon(s), {db.n_classes} class(es), "
          f"glyph re-scoring {'ON' if db.has_glyphs else 'OFF (hash only!)'}")
    print(f"Gates    : min_score={args.min_score} margin={args.margin} "
          f"prefilter_k={args.prefilter_k} "
          f"group_vote={'off' if args.no_group_vote else 'on'}")
    if args.dry_run:
        print("DRY RUN  : nothing will be moved or copied.")

    transfer = shutil.copy2 if args.copy else shutil.move
    verb = "copied" if args.copy else "moved"

    matched = unmatched = errors = 0
    per_class: Counter = Counter()
    reasons: Counter = Counter()

    for n, img_path in enumerate(iter_images(args.input_folder)):
        if args.limit and n >= args.limit:
            break
        try:
            crop = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if crop is None:
                raise ValueError("cv2 could not read the image")
            sig = sig_builder.build(crop)
            hit = db.match(sig, matcher)
        except Exception as exc:
            LOGGER.error("Failed on %s: %s", img_path, exc)
            errors += 1
            continue

        name = os.path.basename(img_path)
        if hit.name:
            # --by-group collapses the near-duplicate class names of a group
            # into one folder; sanitize because a group key may contain spaces
            # and a class name may contain "/" or ":".
            folder = sanitize_filename(hit.group or hit.name) if args.by_group \
                else sanitize_filename(hit.name)
            dest_dir = os.path.join(args.out, folder)
            dest = os.path.join(dest_dir, name)
            if not args.dry_run:
                os.makedirs(dest_dir, exist_ok=True)
                dest = unique_destination(dest_dir, name)
                transfer(img_path, dest)
            matched += 1
            per_class[folder] += 1
            if not args.quiet:
                print(f"[score={hit.score:.3f} margin={hit.margin:.3f} "
                      f"votes={hit.votes}] {name} -> {folder}")
        else:
            unmatched += 1
            runner = f"{hit.rows[0]['name']!r}" if hit.rows else "-"
            # Bucket by CAUSE, not by message: the message carries names and
            # scores, so counting those would just re-list every icon.
            if not hit.rows:
                reasons["nothing in the database resembled it"] += 1
            elif hit.score < args.min_score:
                reasons[f"score below the {args.min_score} floor"] += 1
            else:
                reasons[f"two classes too close (margin < {args.margin})"] += 1
            if not args.quiet:
                print(f"[score={hit.score:.3f} margin={hit.margin:.3f}] "
                      f"{name} -> NO MATCH (best {runner}: {hit.rejected})")
            if args.unmatched_dir and not args.dry_run:
                os.makedirs(args.unmatched_dir, exist_ok=True)
                transfer(img_path, unique_destination(args.unmatched_dir, name))

    total = matched + unmatched
    print("\n--- Summary ---")
    print(f"Sorted ({verb}) : {matched}"
          + (f"/{total} ({matched/total:.1%})" if total else ""))
    print(f"Unmatched      : {unmatched}"
          + (f" -> {args.unmatched_dir}" if args.unmatched_dir
             else " (left in place)"))
    print(f"Errors         : {errors}")
    print(f"Output root    : {args.out}"
          + ("  [dry run — nothing written]" if args.dry_run else ""))

    if per_class:
        print(f"\nTop classes ({len(per_class)} in total):")
        for cls, count in per_class.most_common(15):
            print(f"  {count:>5}  {cls}")
    if reasons:
        print("\nWhy icons were not sorted:")
        for reason, count in reasons.most_common():
            print(f"  {count:>5}  {reason}")
        print("  (raise --margin for fewer, safer matches; lower it for more)")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
