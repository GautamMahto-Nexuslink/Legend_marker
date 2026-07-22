#!/usr/bin/env python3
"""
save_phash.py
=============

Compute and save the perceptual hash (pHash) of icon images.

INPUT
-----
A *parent* folder whose immediate children are class folders, each holding the
icon images for that class::

    <parent_folder>/                 # e.g. Save_icons_modified/
        <ClassName>/                  # one folder per class
            <icon_1>.png
            <icon_2>.png
        <OtherClass>/
            <icon>.png

OUTPUT
------
A dict mapping each class name to the pHash of every icon in it::

    {
        "Beach": {"CAMP_HELEN..._1.png": "d1e2...", "..._2.png": "..."},
        "Amphitheater": { ... },
        ...
    }

The dict is written to a JSON file (``--out``, default ``icons_phash.json``).

The pHash is computed with ``imagehash.phash`` on a grayscale version of the
icon — the same algorithm the legend-marker pipeline uses (see
``legend_pipeline/signatures.py``).

Usage::

    python3 save_phash.py <parent_folder>
    python3 save_phash.py Save_icons_modified --out icons_phash.json --hash-size 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict

import imagehash
from PIL import Image

# Image file extensions we treat as icons.
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

LOGGER = logging.getLogger("save_phash")

# Available imagehash algorithms (phash matches the pipeline default).
HASH_ALGORITHMS = {
    "phash": imagehash.phash,
    "dhash": imagehash.dhash,
    "ahash": imagehash.average_hash,
    "whash": imagehash.whash,
}


def compute_phash(image_path: str, algorithm: str = "phash", hash_size: int = 8) -> str:
    """Return the perceptual hash (hex string) of a single image.

    The image is loaded and converted to grayscale ("L") first, matching the
    pipeline which hashes the grayscale glyph.
    """
    algo = HASH_ALGORITHMS.get(algorithm, imagehash.phash)
    with Image.open(image_path) as img:
        gray = img.convert("L")
        return str(algo(gray, hash_size=hash_size))


def build_phash_dict(
    parent_folder: str,
    algorithm: str = "phash",
    hash_size: int = 8,
) -> Dict[str, Dict[str, str]]:
    """Walk ``parent_folder`` and build ``{classname: {image_name: phash_hex}}``."""
    if not os.path.isdir(parent_folder):
        raise NotADirectoryError(f"Parent folder not found: {parent_folder}")

    result: Dict[str, Dict[str, str]] = {}

    for class_name in sorted(os.listdir(parent_folder)):
        class_dir = os.path.join(parent_folder, class_name)
        if not os.path.isdir(class_dir):
            continue

        images = sorted(
            f for f in os.listdir(class_dir)
            if f.lower().endswith(IMAGE_EXTS)
        )
        if not images:
            LOGGER.warning("Class '%s' has no images — skipping.", class_name)
            continue

        class_hashes: Dict[str, str] = {}
        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            try:
                class_hashes[img_name] = compute_phash(img_path, algorithm, hash_size)
            except Exception as exc:  # Never let one bad image stop the batch.
                LOGGER.error("Failed to hash %s: %s", img_path, exc)

        if class_hashes:
            result[class_name] = class_hashes
            LOGGER.info("%-40s %d icon(s) hashed.", class_name, len(class_hashes))

    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Save pHash of icons per class.")
    parser.add_argument("parent_folder", help="Parent folder of class sub-folders.")
    parser.add_argument("--out", default="icons_phash.json",
                        help="Output JSON path, {classname: {image: phash}} "
                             "(default: icons_phash.json).")
    parser.add_argument("--out-flat", default="icons_phash_flat.json",
                        help="Flat output JSON path, {phash: classname} for "
                             "every icon (default: icons_phash_flat.json).")
    parser.add_argument("--algorithm", default="phash",
                        choices=list(HASH_ALGORITHMS),
                        help="Hash algorithm (default: phash).")
    parser.add_argument("--hash-size", type=int, default=8,
                        help="Hash size passed to imagehash (default: 8).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    phash_dict = build_phash_dict(args.parent_folder, args.algorithm, args.hash_size)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(phash_dict, fh, indent=2, ensure_ascii=False)

    # Flat {phash_hex: classname} — one entry per icon (phash is the key).
    flat_dict: Dict[str, str] = {}
    for cls, hashes in phash_dict.items():
        for phash_hex in hashes.values():
            if phash_hex in flat_dict and flat_dict[phash_hex] != cls:
                LOGGER.warning("pHash collision: %s maps to both '%s' and '%s'.",
                               phash_hex, flat_dict[phash_hex], cls)
            flat_dict[phash_hex] = cls
    with open(args.out_flat, "w", encoding="utf-8") as fh:
        json.dump(flat_dict, fh, indent=2, ensure_ascii=False)

    total_icons = sum(len(v) for v in phash_dict.values())
    print(f"Saved pHash for {total_icons} icon(s) across "
          f"{len(phash_dict)} class(es).")
    print(f"  detailed -> {args.out}")
    print(f"  flat     -> {args.out_flat}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
