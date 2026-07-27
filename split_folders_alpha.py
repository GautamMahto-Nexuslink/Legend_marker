#!/usr/bin/env python3
"""
split_folders_alpha.py
======================

Split the ICONS inside a folder tree into three alphabetical groups by the
first letter of each icon's FILENAME:

    A-H   (a b c d e f g h)
    I-P   (i j k l m n o p)
    Q-Z   (q r s t u v w x y z)

The directory structure is MAINTAINED: an icon keeps its containing sub-folder
path inside whichever group it lands in.

INPUT
-----
    input_folder/
        class1/
            Apple.png
            Zebra.png
        class2/
            Banana.png
            Restrooms.png

OUTPUT
------
    <out>/
        A-H/
            class1/ Apple.png
            class2/ Banana.png
        Q-Z/
            class1/ Zebra.png
            class2/ Restrooms.png

So every icon whose name starts with A-H ends up under ``A-H/<same sub-path>/``,
and likewise for the other groups.  Icons whose name does not start with a
letter go into an ``Other`` group.

Usage::

    python3 split_folders_alpha.py <input_folder>
    python3 split_folders_alpha.py <input_folder> --out split_out --move
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import List, Optional, Tuple

# (label, first_letter, last_letter) — inclusive ranges, lower-cased.
BUCKETS: List[Tuple[str, str, str]] = [
    ("A-H", "a", "h"),
    ("I-P", "i", "p"),
    ("Q-Z", "q", "z"),
]
OTHER_LABEL = "Other"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def bucket_for(filename: str) -> str:
    """Return the group label for an icon, from its filename's first letter."""
    first = next((ch.lower() for ch in filename if ch.isalpha()), None)
    if first is None:
        return OTHER_LABEL
    for label, lo, hi in BUCKETS:
        if lo <= first <= hi:
            return label
    return OTHER_LABEL


def unique_destination(path: str) -> str:
    """Return ``path`` or, if it exists, ``name_1.ext`` / ``name_2.ext`` ..."""
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(f"{root}_{n}{ext}"):
        n += 1
    return f"{root}_{n}{ext}"


def split_icons(input_folder: str, out_folder: str, move: bool) -> int:
    if not os.path.isdir(input_folder):
        sys.exit(f"Input folder not found: {input_folder}")

    input_folder = os.path.abspath(input_folder)
    out_folder = os.path.abspath(out_folder)

    # Guard: never let the output land inside the input (would recurse).
    if out_folder == input_folder or out_folder.startswith(input_folder + os.sep):
        sys.exit("--out must be OUTSIDE the input folder.")

    counts = {label: 0 for label, _, _ in BUCKETS}
    counts[OTHER_LABEL] = 0
    skipped = errors = 0
    transfer = shutil.move if move else shutil.copy2

    for dirpath, _dirnames, filenames in os.walk(input_folder):
        for fname in sorted(filenames):
            if not fname.lower().endswith(IMAGE_EXTS):
                skipped += 1
                continue

            src = os.path.join(dirpath, fname)
            label = bucket_for(fname)
            # Preserve the icon's sub-folder path relative to the input root.
            rel_dir = os.path.relpath(dirpath, input_folder)
            dest_dir = os.path.join(out_folder, label)
            if rel_dir != ".":
                dest_dir = os.path.join(dest_dir, rel_dir)

            try:
                os.makedirs(dest_dir, exist_ok=True)
                dest = unique_destination(os.path.join(dest_dir, fname))
                transfer(src, dest)
                counts[label] += 1
            except Exception as exc:
                errors += 1
                print(f"[error] {src}: {exc}")

    verb = "Moved" if move else "Copied"
    print("--- Summary ---")
    total = 0
    for label in [b[0] for b in BUCKETS] + [OTHER_LABEL]:
        if counts[label] or label != OTHER_LABEL:
            print(f"  {label:<6} {counts[label]} icon(s)")
        total += counts[label]
    print(f"  total  {total} icon(s)")
    if skipped:
        print(f"  (skipped {skipped} non-image file(s))")
    if errors:
        print(f"  errors: {errors}")
    print(f"{verb} into: {out_folder}")
    return 0 if errors == 0 else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Split icons into A-H / I-P / Q-Z groups by filename, "
                    "keeping their sub-folder structure.")
    parser.add_argument("input_folder",
                        help="Folder tree whose icons will be split.")
    parser.add_argument("--out", default="",
                        help="Output folder (default: <input_folder>_split).")
    parser.add_argument("--move", action="store_true",
                        help="Move instead of copy (empties the source icons).")
    args = parser.parse_args(argv)

    out = args.out or (os.path.abspath(args.input_folder.rstrip("/\\")) + "_split")
    return split_icons(args.input_folder, out, args.move)


if __name__ == "__main__":
    sys.exit(main())
