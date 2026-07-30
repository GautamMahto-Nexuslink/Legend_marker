#!/usr/bin/env python3
"""
move_common_icons.py
====================

Given TWO folder trees of icons, collect the icons that appear in **both** into a
third folder, keeping each icon's sub-folder path.

    folder1/                    folder2/                    out/
        Barn/                       Barn/                       Barn/
            map_a_1.png  <──────────── map_a_1.png   ────►          map_a_1.png
            map_b_2.png                 (absent)                    (not moved)
        Gate/                       Gate/                       Gate/
            map_c_1.png  <──────────── map_c_1.png   ────►          map_c_1.png

Typical use: two people (or two runs) sorted the same icons independently, and
you want the ones they AGREE on — those are the safe, high-confidence icons to
fold into the dataset — separated from the ones only one side produced.

WHAT COUNTS AS "THE SAME ICON"  (``--match``)
---------------------------------------------
``same-path`` (default)  ``Barn/map_a_1.png`` must exist under both roots at the
                         same relative path.  Use this when the two trees have
                         the same sub-folders — agreement on the CLASS too.
``filename``             the file name alone must appear somewhere under
                         folder2, in any sub-folder.  Use this when the two
                         trees are organised differently and you only care that
                         the same icon file is present.

Add ``--ignore-ext`` to treat ``x.png`` and ``x.jpg`` as the same icon.

WHICH COPY MOVES  (``--from``)
------------------------------
``1`` (default) the folder1 copy is moved, folder2 is left untouched.
``2``           the folder2 copy is moved instead.
``both``        both are moved, into ``out/<root name>/<relative path>`` so they
                stay distinguishable.

Nothing is ever overwritten: a name already taken in the destination gets a
numeric suffix (``icon.png`` -> ``icon_1.png``).

Usage::

    # ALWAYS look first — this moves your files
    python3 move_common_icons.py folder1 folder2 --out common --dry-run

    # then do it
    python3 move_common_icons.py folder1 folder2 --out common

    # keep the originals where they are
    python3 move_common_icons.py folder1 folder2 --out common --copy

    # match on file name alone, regardless of which sub-folder it sits in
    python3 move_common_icons.py folder1 folder2 --out common --match filename

    # take both sides' copies, and tidy up the sub-folders left empty
    python3 move_common_icons.py folder1 folder2 --out common --from both --prune-empty
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

LOGGER = logging.getLogger("move_common_icons")


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------
def iter_images(root: str) -> List[str]:
    """Every image under ``root``, as paths relative to ``root`` (sorted)."""
    out: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith(IMAGE_EXTS):
                full = os.path.join(dirpath, name)
                out.append(os.path.relpath(full, root))
    return sorted(out)


def unique_destination(path: str) -> str:
    """``path`` if free, else ``stem_1.ext``, ``stem_2.ext``, ... — never clobbers."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(f"{stem}_{n}{ext}"):
        n += 1
    return f"{stem}_{n}{ext}"


def match_key(rel_path: str, mode: str, ignore_ext: bool) -> str:
    """The value compared between the two trees.

    ``same-path`` keeps the sub-folder ("Barn/x.png"), ``filename`` reduces to
    the file name ("x.png").  Paths are normalised to forward slashes and
    lower-cased so the comparison does not depend on the OS or on casing.
    """
    key = rel_path.replace(os.sep, "/")
    if mode == "filename":
        key = os.path.basename(key)
    if ignore_ext:
        key = os.path.splitext(key)[0]
    return key.lower()


def prune_empty_dirs(root: str, dry_run: bool) -> int:
    """Remove directories left empty under ``root`` (deepest first)."""
    removed = 0
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if dirpath == root or dirnames or filenames:
            continue
        if not dry_run:
            try:
                os.rmdir(dirpath)
            except OSError as exc:
                LOGGER.warning("Could not remove %s: %s", dirpath, exc)
                continue
        removed += 1
    return removed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Move the icons present in BOTH folder trees into a third "
                    "folder, preserving each icon's sub-folder path.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("folder1", help="First tree (sub-folders of images).")
    p.add_argument("folder2", help="Second tree (same sub-folders).")
    p.add_argument("--out", required=True,
                   help="Destination root; the sub-folder path is recreated here.")
    p.add_argument("--match", choices=["same-path", "filename"], default="same-path",
                   help="'same-path': sub-folder AND name must agree. "
                        "'filename': the name may sit in any sub-folder.")
    p.add_argument("--ignore-ext", action="store_true",
                   help="Treat 'x.png' and 'x.jpg' as the same icon.")
    p.add_argument("--from", dest="take", choices=["1", "2", "both"], default="1",
                   help="Which side's copy to move.")
    p.add_argument("--copy", action="store_true",
                   help="Copy instead of move (leaves both trees intact).")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would move, without touching anything.")
    p.add_argument("--prune-empty", action="store_true",
                   help="Delete sub-folders left empty in the source tree(s).")
    p.add_argument("--limit", type=int, default=0,
                   help="Stop after N icons (0 = all).")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="Only print the summary, not one line per icon.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(message)s")

    for folder in (args.folder1, args.folder2):
        if not os.path.isdir(folder):
            sys.exit(f"Folder not found: {folder}")
    if os.path.abspath(args.folder1) == os.path.abspath(args.folder2):
        sys.exit("folder1 and folder2 are the same directory.")
    # Moving INTO one of the source trees would make the walk eat its own output.
    out_abs = os.path.abspath(args.out)
    for folder in (args.folder1, args.folder2):
        root = os.path.abspath(folder)
        if out_abs == root or out_abs.startswith(root + os.sep):
            sys.exit(f"--out must not live inside {folder}")

    files1 = iter_images(args.folder1)
    files2 = iter_images(args.folder2)
    if not files1 or not files2:
        sys.exit(f"No images found in "
                 f"{args.folder1 if not files1 else args.folder2}")

    # Index folder2 by the comparison key.  One key can have several paths when
    # matching by filename (the same name in two sub-folders), so keep them all
    # and report the ambiguity rather than silently picking one.
    index2: Dict[str, List[str]] = defaultdict(list)
    for rel in files2:
        index2[match_key(rel, args.match, args.ignore_ext)].append(rel)

    name1 = os.path.basename(os.path.normpath(args.folder1)) or "folder1"
    name2 = os.path.basename(os.path.normpath(args.folder2)) or "folder2"

    print(f"folder1 : {len(files1):>6} image(s)  {args.folder1}")
    print(f"folder2 : {len(files2):>6} image(s)  {args.folder2}")
    print(f"match   : {args.match}"
          + ("  (extension ignored)" if args.ignore_ext else ""))
    print(f"action  : {'COPY' if args.copy else 'MOVE'} the "
          + {"1": f"{name1}", "2": f"{name2}", "both": "BOTH"}[args.take]
          + f" copy -> {args.out}")
    if args.dry_run:
        print("DRY RUN : nothing will be moved, copied or deleted.")
    print()

    transfer = shutil.copy2 if args.copy else shutil.move
    moved = 0
    only1 = 0
    ambiguous = 0
    errors = 0
    per_folder: Counter = Counter()
    seen_keys: Set[str] = set()

    for rel in files1:
        key = match_key(rel, args.match, args.ignore_ext)
        partners = index2.get(key)
        if not partners:
            only1 += 1
            continue
        if args.limit and moved >= args.limit:
            break
        if len(partners) > 1:
            ambiguous += 1
            LOGGER.info("%s matches %d paths in folder2 (%s) — using the first.",
                        rel, len(partners), ", ".join(partners[:3]))
        seen_keys.add(key)

        # Which copies to take, and where each lands.  With --from both the two
        # roots get their own top-level folder so the copies stay apart.
        jobs: List[Tuple[str, str]] = []
        if args.take in ("1", "both"):
            sub = os.path.join(name1, rel) if args.take == "both" else rel
            jobs.append((os.path.join(args.folder1, rel),
                         os.path.join(args.out, sub)))
        if args.take in ("2", "both"):
            partner = partners[0]
            sub = os.path.join(name2, partner) if args.take == "both" else partner
            jobs.append((os.path.join(args.folder2, partner),
                         os.path.join(args.out, sub)))

        try:
            for src, dest in jobs:
                if not args.dry_run:
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    transfer(src, unique_destination(dest))
        except Exception as exc:
            LOGGER.error("Failed on %s: %s", rel, exc)
            errors += 1
            continue

        moved += 1
        per_folder[os.path.dirname(rel) or "."] += 1
        if not args.quiet:
            print(f"  {rel}  ->  {os.path.relpath(jobs[0][1], args.out)}"
                  + (f"  (+{len(jobs) - 1} more)" if len(jobs) > 1 else ""))

    only2 = sum(1 for rel in files2
                if match_key(rel, args.match, args.ignore_ext) not in seen_keys)

    pruned = 0
    if args.prune_empty:
        roots = ([args.folder1] if args.take in ("1", "both") else []) + \
                ([args.folder2] if args.take in ("2", "both") else [])
        for root in roots:
            pruned += prune_empty_dirs(root, args.dry_run)

    verb = "copied" if args.copy else "moved"
    print("\n--- Summary ---")
    print(f"In both trees   : {moved}  ({verb} to {args.out})")
    print(f"Only in folder1 : {only1}  (left in place)")
    print(f"Only in folder2 : {only2}  (left in place)")
    if ambiguous:
        print(f"Ambiguous       : {ambiguous}  (name found in several folder2 "
              f"sub-folders; first used — run with -v to see them)")
    if errors:
        print(f"Errors          : {errors}")
    if args.prune_empty:
        print(f"Empty dirs      : {pruned} removed")
    if args.dry_run:
        print("(dry run — nothing was written)")

    if per_folder:
        print(f"\nPer sub-folder ({len(per_folder)} with matches):")
        for folder, count in sorted(per_folder.items()):
            print(f"  {count:>5}  {folder}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
