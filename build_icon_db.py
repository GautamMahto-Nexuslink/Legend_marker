#!/usr/bin/env python3
"""
build_icon_db.py
================

Build the known-icon database used by the pipeline's "blue" stage.

Why this exists
---------------
``save_phash.py`` stores ONE 256-bit pHash per icon and the pipeline accepts a
hit only below a fixed Hamming distance.  At ``hash_size=16`` a distance of 10
means 96% of the bits must be identical, so a map icon that is rescaled,
re-antialiased or drawn on terrain almost never matches — the stage fires on
very few detections, and loosening the threshold instead starts inventing wrong
hits, because a single hash carries no way to tell a near-miss from a rival.

This builder therefore stores, per icon:

  * the **64x64 segmented glyph** — the same canonical glyph the pipeline builds
    for a map detection, so the run-time matcher can re-score a candidate with
    multi-scale template correlation + ORB instead of only comparing hashes,
  * **three hashes** (pHash, dHash, aHash) used as a cheap prefilter that cuts
    ~5000 candidates down to a few dozen before the expensive re-scoring,
  * the class name and source file of every exemplar, so several exemplars of
    one class can vote.

INPUT
-----
A parent folder of class sub-folders (same layout as save_phash.py)::

    <parent>/<ClassName>/<icon>.png

OUTPUT
------
``--out`` (default ``icons_glyph_db.npz``), containing:

    glyphs   uint8  (N, 64, 64)   segmented glyphs
    phash    uint8  (N, 32)       bit-packed 256-bit pHash
    dhash    uint8  (N, 32)       bit-packed dHash
    ahash    uint8  (N, 32)       bit-packed aHash
    names    <U…    (N,)          class name per icon
    files    <U…    (N,)          source filename per icon
    meta     json string          glyph_size / hash_size / algorithm / counts

Usage::

    python3 build_icon_db.py /path/to/Merged_Save_Crops
    python3 build_icon_db.py /path/to/crops --out icons_glyph_db.npz -v
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import List

import cv2
import numpy as np

import legend_marker as lm
from legend_pipeline.deps import Image, imagehash

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
LOGGER = logging.getLogger("build_icon_db")

DEFAULT_CROPS = "/home/nls34/Downloads/Dataset_lengend_marker_viewer/IMP/Merged_Save_Crops"


def pack_hash(image_hash) -> np.ndarray:
    """imagehash.ImageHash -> bit-packed uint8 vector (32 bytes for 256 bits)."""
    return np.packbits(np.asarray(image_hash.hash, dtype=bool).ravel())


def hashes_for_glyph(glyph: np.ndarray, hash_size: int) -> tuple:
    """(pHash, dHash, aHash) of a glyph, each bit-packed."""
    pil = Image.fromarray(glyph)
    return (
        pack_hash(imagehash.phash(pil, hash_size=hash_size)),
        pack_hash(imagehash.dhash(pil, hash_size=hash_size)),
        pack_hash(imagehash.average_hash(pil, hash_size=hash_size)),
    )


def build(parent: str, glyph_size: int = 64, hash_size: int = 16) -> dict:
    """Walk ``parent`` and return the arrays that make up the database."""
    if not os.path.isdir(parent):
        raise NotADirectoryError(f"Parent folder not found: {parent}")

    config = lm.PipelineConfig(glyph_size=glyph_size, hash_size=hash_size)
    builder = lm.SignatureBuilder(config)

    glyphs: List[np.ndarray] = []
    phash: List[np.ndarray] = []
    dhash: List[np.ndarray] = []
    ahash: List[np.ndarray] = []
    names: List[str] = []
    files: List[str] = []

    classes = sorted(d for d in os.listdir(parent)
                     if os.path.isdir(os.path.join(parent, d)))
    started = time.perf_counter()
    for class_name in classes:
        class_dir = os.path.join(parent, class_name)
        images = sorted(f for f in os.listdir(class_dir)
                        if f.lower().endswith(IMAGE_EXTS))
        if not images:
            LOGGER.warning("Class '%s' has no images — skipped.", class_name)
            continue
        kept = 0
        for img_name in images:
            path = os.path.join(class_dir, img_name)
            try:
                crop = cv2.imread(path, cv2.IMREAD_COLOR)
                if crop is None:
                    raise ValueError("cv2 could not read the image")
                sig = builder.build(crop)
                if sig.glyph is None:
                    raise ValueError("no glyph produced")
                ph, dh, ah = hashes_for_glyph(sig.glyph, hash_size)
            except Exception as exc:          # never let one bad file stop it
                LOGGER.error("Failed on %s: %s", path, exc)
                continue
            glyphs.append(sig.glyph)
            phash.append(ph)
            dhash.append(dh)
            ahash.append(ah)
            names.append(class_name)
            files.append(img_name)
            kept += 1
        LOGGER.info("%-42s %d icon(s)", class_name, kept)

    if not glyphs:
        raise RuntimeError(f"No icons could be read under {parent}")

    elapsed = time.perf_counter() - started
    meta = {
        "glyph_size": glyph_size,
        "hash_size": hash_size,
        "hash_bits": hash_size * hash_size,
        "n_icons": len(glyphs),
        "n_classes": len(set(names)),
        "source": os.path.abspath(parent),
        "build_seconds": round(elapsed, 1),
    }
    return {
        "glyphs": np.stack(glyphs).astype(np.uint8),
        "phash": np.stack(phash),
        "dhash": np.stack(dhash),
        "ahash": np.stack(ahash),
        "names": np.array(names),
        "files": np.array(files),
        "meta": np.array(json.dumps(meta)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parent_folder", nargs="?", default=DEFAULT_CROPS,
                    help="Parent folder of class sub-folders.")
    ap.add_argument("--out", default="icons_glyph_db.npz", help="Output .npz path.")
    ap.add_argument("--glyph-size", type=int, default=64,
                    help="MUST match the pipeline's config.glyph_size.")
    ap.add_argument("--hash-size", type=int, default=16,
                    help="MUST match the pipeline's config.hash_size.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(message)s")

    data = build(args.parent_folder, args.glyph_size, args.hash_size)
    np.savez_compressed(args.out, **data)
    meta = json.loads(str(data["meta"]))
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Wrote {args.out} ({size_mb:.1f} MB)")
    print(f"  {meta['n_icons']} icon(s) across {meta['n_classes']} class(es), "
          f"glyph={meta['glyph_size']}px, hash={meta['hash_size']} "
          f"({meta['hash_bits']} bits), {meta['build_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
