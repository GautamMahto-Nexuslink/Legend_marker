#!/usr/bin/env python3
"""
save_phash.py
=============

Compute and save the perceptual hash (pHash) of icon images.

The pHash is computed with the SAME code path the legend-marker pipeline uses
(``legend_pipeline.SignatureBuilder`` -> segmented glyph -> ``imagehash.phash``
at ``hash_size=16``).  This is important: the pipeline hashes the *segmented
glyph*, not the raw crop, so hashing icons any other way would produce values
that never match at run time.  By reusing SignatureBuilder here, a hash saved in
this file is byte-for-byte comparable to ``det.signature.phash_hex`` computed on
a map detection.

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
Two JSON files:
  * ``--out``      detailed  {classname: {image_name: phash_hex}}
  * ``--out-flat`` flat      {phash_hex: classname}   (one entry per icon)

Usage::

    python3 save_phash.py <parent_folder>
    python3 save_phash.py Save_icons_modified --hash-size 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict

import cv2

# Reuse the pipeline's signature builder so hashes match at run time.
import legend_marker as lm

# Image file extensions we treat as icons.
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

LOGGER = logging.getLogger("save_phash")


def build_phash_dict(
    parent_folder: str,
    hash_size: int = 16,
    hash_algorithm: str = "phash",
) -> Dict[str, Dict[str, str]]:
    """Walk ``parent_folder`` and build ``{classname: {image_name: phash_hex}}``.

    Each icon is hashed exactly as the pipeline hashes a map detection crop:
    ``SignatureBuilder.build(crop).phash_hex`` on the segmented glyph.
    """
    if not os.path.isdir(parent_folder):
        raise NotADirectoryError(f"Parent folder not found: {parent_folder}")

    # A minimal config that only affects glyph segmentation + hashing; the
    # detector / OCR are never touched, so no API key or model load is needed.
    config = lm.PipelineConfig(
        hash_algorithm=hash_algorithm,
        hash_size=hash_size,
    )
    sig_builder = lm.SignatureBuilder(config)

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
                crop = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if crop is None:
                    raise ValueError("cv2 could not read the image")
                sig = sig_builder.build(crop)
                if sig.phash_hex:
                    class_hashes[img_name] = sig.phash_hex
                else:
                    LOGGER.warning("No pHash produced for %s — skipping.", img_path)
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
                        help="Detailed output JSON, {classname: {image: phash}} "
                             "(default: icons_phash.json).")
    parser.add_argument("--out-flat", default="icons_phash_flat.json",
                        help="Flat output JSON, {phash: classname} for every "
                             "icon (default: icons_phash_flat.json).")
    parser.add_argument("--algorithm", default="phash",
                        choices=["phash", "dhash", "ahash", "whash"],
                        help="Hash algorithm (default: phash).")
    parser.add_argument("--hash-size", type=int, default=16,
                        help="Hash size — MUST match the pipeline's "
                             "config.hash_size (default: 16).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    phash_dict = build_phash_dict(args.parent_folder, args.hash_size, args.algorithm)

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
    print(f"Saved pHash (hash_size={args.hash_size}) for {total_icons} icon(s) "
          f"across {len(phash_dict)} class(es).")
    print(f"  detailed -> {args.out}")
    print(f"  flat     -> {args.out_flat}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
