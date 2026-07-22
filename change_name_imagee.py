#!/usr/bin/env python3
"""
Convert a Roboflow-exported image filename back to its original clean
filename, as listed in legend.txt.

Usage:
    python rf_to_legend.py <input_folder> <legend.txt> [--copy] [--outdir DIR]

    Processes every image file inside <input_folder> and prints/copies
    each one's reconstructed clean name.

Logic:
    Roboflow name:  "<Name>_<ext>.rf.<HASH>.<ext>"
    Legend name:    "<Name_with_underscores>.<ext>"

    e.g. "Alamo lake_jpg.rf.HkJN0jEQZrlNqUWwmuM5.jpg" -> "Alamo_lake.jpg"

Steps:
    1. Strip the "_<ext>.rf.<HASH>.<ext>" suffix -> base name.
    2. Replace spaces in the base name with underscores.
    3. Re-attach the real extension.
    4. Verify the result exists in legend.txt. If not, fall back to a
       fuzzy match (difflib) against all legend entries.
"""

import argparse
import difflib
import re
import shutil
import sys
from pathlib import Path


RF_SUFFIX_RE = re.compile(r"_(?P<ext>[A-Za-z0-9]+)\.rf\.[A-Za-z0-9]+\.(?P<ext2>[A-Za-z0-9]+)$")


def rf_name_to_clean_name(filename: str) -> str:
    """Convert a Roboflow-style filename into the reconstructed clean name."""
    m = RF_SUFFIX_RE.search(filename)
    if not m:
        # No roboflow suffix found — just normalize spaces and return as-is.
        stem, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
        stem = stem.replace(" ", "_")
        return f"{stem}.{ext}" if ext else stem

    ext = m.group("ext2")
    base = filename[: m.start()]           # everything before "_<ext>.rf...."
    base = base.replace(" ", "_")
    return f"{base}.{ext}"


def load_legend(legend_path: Path) -> list[str]:
    with open(legend_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def best_match(name: str, legend_entries: list[str]) -> str | None:
    matches = difflib.get_close_matches(name, legend_entries, n=1, cutoff=0.6)
    return matches[0] if matches else None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def process_one(input_path: Path, legend_entries: list[str], copy: bool, outdir: Path | None):
    reconstructed = rf_name_to_clean_name(input_path.name)

    if reconstructed in legend_entries:
        final_name = reconstructed
        status = "exact match"
    else:
        fuzzy = best_match(reconstructed, legend_entries)
        if fuzzy:
            final_name = fuzzy
            status = f"fuzzy match (reconstructed was '{reconstructed}')"
        else:
            final_name = reconstructed
            status = "NO MATCH in legend.txt"

    print(f"{input_path.name}  ->  {final_name}   [{status}]")

    if copy:
        dest_dir = outdir if outdir else input_path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / final_name
        shutil.copy2(input_path, dest)

    return status


def main():
    parser = argparse.ArgumentParser(description="Map Roboflow image filenames in a folder to their legend.txt clean names.")
    parser.add_argument("input_folder", help="Path to the folder containing Roboflow-style images.")
    parser.add_argument("legend", help="Path to legend.txt containing the clean filenames.")
    parser.add_argument("--copy", action="store_true", help="Copy each file to its clean name instead of only printing it.")
    parser.add_argument("--outdir", default=None, help="Directory to place the renamed copies (default: same dir as input).")
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    legend_path = Path(args.legend)

    if not input_folder.is_dir():
        sys.exit(f"Input folder not found or not a directory: {input_folder}")
    if not legend_path.exists():
        sys.exit(f"Legend file not found: {legend_path}")

    legend_entries = load_legend(legend_path)
    outdir = Path(args.outdir) if args.outdir else None

    images = sorted(p for p in input_folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        sys.exit(f"No image files found in: {input_folder}")

    counts = {"exact match": 0, "fuzzy": 0, "no match": 0}
    for img_path in images:
        status = process_one(img_path, legend_entries, args.copy, outdir)
        if status == "exact match":
            counts["exact match"] += 1
        elif status.startswith("fuzzy"):
            counts["fuzzy"] += 1
        else:
            counts["no match"] += 1

    print("\n--- Summary ---")
    print(f"Total images:  {len(images)}")
    print(f"Exact matches: {counts['exact match']}")
    print(f"Fuzzy matches: {counts['fuzzy']}")
    print(f"No matches:    {counts['no match']}")


if __name__ == "__main__":
    main()