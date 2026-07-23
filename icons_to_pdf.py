#!/usr/bin/env python3
"""
icons_to_pdf.py
===============

Build a PDF catalogue of icon crops, grouped by SOURCE IMAGE and then by CLASS.

INPUT
-----
A folder whose immediate sub-folders are class names, each holding icon crops
taken from several source maps/images::

    input_folder/
        <ClassName>/
            <ImageName>_1.png       # crop 1 of this class from <ImageName>
            <ImageName>_2.png
            <OtherImage>_1.png
        <OtherClass>/
            <ImageName>_1.png

The source-image name is recovered from each crop's filename by stripping the
trailing ``_<N>`` icon index (and any ``" (k)"`` de-duplication suffix), exactly
how ``save_icons.py`` names them: ``<mapstem>_<n>.png``.

OUTPUT
------
A PDF laid out as::

    <ImageName 1>                (page title)
        <ClassA>
            [icon] [icon] [icon]
        <ClassB>
            [icon] [icon]
    <ImageName 2>
        <ClassA>
            [icon]
    ...

Content flows down each page and starts a new page automatically when it runs
out of room; a new source image always begins on a fresh page.

Usage::

    python3 icons_to_pdf.py <input_folder>
    python3 icons_to_pdf.py <input_folder> --out icons_report.pdf --page letter
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")                       # headless: no display needed.
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Strip a de-dup suffix like " (2)" and then the trailing "_<index>".
_COLLISION_RE = re.compile(r"\s*\(\d+\)$")
_INDEX_RE = re.compile(r"_\d+$")

# Page sizes in inches (w, h), portrait.
PAGE_SIZES = {"a4": (8.27, 11.69), "letter": (8.5, 11.0)}


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------
def image_name_from(filename: str) -> str:
    """Recover the source-image name from a crop filename.

    ``CAMP_HELEN..._page-0001_jpg_1 (2).png`` -> ``CAMP_HELEN..._page-0001_jpg``
    """
    stem = os.path.splitext(filename)[0]
    stem = _COLLISION_RE.sub("", stem)
    stem = _INDEX_RE.sub("", stem)
    return stem or os.path.splitext(filename)[0]


def group_icons(input_folder: str) -> Dict[str, Dict[str, List[str]]]:
    """Return ``{image_name: {class_name: [icon_path, ...]}}`` from the folder."""
    if not os.path.isdir(input_folder):
        raise NotADirectoryError(f"Input folder not found: {input_folder}")

    grouped: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for class_name in sorted(os.listdir(input_folder)):
        class_dir = os.path.join(input_folder, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            img_name = image_name_from(fname)
            grouped[img_name][class_name].append(os.path.join(class_dir, fname))
    return grouped


def load_rgb(path: str) -> np.ndarray:
    """Load an icon as an RGB float array in [0,1]; alpha is composited on white."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("cv2 could not read the image")
    if img.ndim == 2:                                   # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:                             # BGRA -> composite
        bgr = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        white = np.full_like(bgr, 255.0)
        bgr = bgr * alpha + white * (1.0 - alpha)
        img = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
    else:                                               # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# PDF layout — a simple top-down cursor with automatic page breaks
# ---------------------------------------------------------------------------
class PdfBuilder:
    """Flows titles / headings / icon rows down a page, paging when full."""

    def __init__(self, pdf: PdfPages, page: str = "a4") -> None:
        self.pdf = pdf
        self.page_w, self.page_h = PAGE_SIZES.get(page, PAGE_SIZES["a4"])
        self.margin = 0.6           # inches
        # Layout metrics (inches).
        self.title_h = 0.45
        self.class_h = 0.30
        self.icon_h = 0.85          # displayed icon height
        self.icon_max_w = 1.7       # cap very wide icons
        self.gap = 0.16             # gap between icons
        self.row_gap = 0.18         # gap below an icon row
        self.fig = None
        self.y = 0.0                # inches from the TOP of the page

    # -- page management --------------------------------------------------
    def _new_page(self) -> None:
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.fig = plt.figure(figsize=(self.page_w, self.page_h))
        self.y = self.margin

    def _ensure(self, need_h: float) -> None:
        """Start a new page if ``need_h`` inches won't fit below the cursor."""
        if self.fig is None or self.y + need_h > self.page_h - self.margin:
            self._new_page()

    # -- coordinate helpers (inches-from-top -> figure fraction) ----------
    def _frac_y(self, top_in: float, h_in: float) -> float:
        return (self.page_h - top_in - h_in) / self.page_h

    def _add_text(self, text: str, top_in: float, h_in: float,
                  fontsize: float, weight: str, indent: float,
                  color: str = "black") -> None:
        x = (self.margin + indent) / self.page_w
        y = self._frac_y(top_in, h_in) + (h_in / self.page_h) * 0.15
        self.fig.text(x, y, text, fontsize=fontsize, fontweight=weight,
                      color=color, va="baseline", ha="left")

    # -- content ----------------------------------------------------------
    def add_image_title(self, name: str) -> None:
        """Every source image begins on its own fresh page."""
        self._new_page()
        self._add_text(name, self.y, self.title_h, 15, "bold", 0.0, "#1a3c6e")
        # Underline rule.
        y_rule = self._frac_y(self.y + self.title_h, 0.0)
        self.fig.add_artist(plt.Line2D(
            [self.margin / self.page_w, (self.page_w - self.margin) / self.page_w],
            [y_rule, y_rule], color="#1a3c6e", linewidth=1.0))
        self.y += self.title_h + 0.1

    def add_class_heading(self, name: str) -> None:
        self._ensure(self.class_h + self.icon_h)   # keep heading with >=1 row.
        self._add_text(name, self.y, self.class_h, 11.5, "bold", 0.15, "#333333")
        self.y += self.class_h

    def add_icons(self, paths: List[str]) -> None:
        usable_right = self.page_w - self.margin
        x = self.margin + 0.30                      # indent icons under heading
        row_top = self.y
        placed_in_row = False

        for path in paths:
            try:
                img = load_rgb(path)
            except Exception:
                continue
            h, w = img.shape[:2]
            disp_h = self.icon_h
            disp_w = min(self.icon_max_w, disp_h * (w / h if h else 1.0))

            # Wrap to a new row (or page) when this icon would overflow.
            if placed_in_row and x + disp_w > usable_right:
                self.y = row_top + self.icon_h + self.row_gap
                self._ensure(self.icon_h)
                row_top = self.y
                x = self.margin + 0.30
                placed_in_row = False

            if not placed_in_row:
                self._ensure(self.icon_h)
                row_top = self.y

            left = x / self.page_w
            bottom = self._frac_y(row_top, disp_h)
            ax = self.fig.add_axes([left, bottom,
                                    disp_w / self.page_w, disp_h / self.page_h])
            ax.imshow(img)
            ax.axis("off")

            x += disp_w + self.gap
            placed_in_row = True

        # Advance past the final row of this class.
        self.y = row_top + self.icon_h + self.row_gap

    def finish(self) -> None:
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
            self.fig = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a PDF of icon crops grouped by source image, then class.")
    parser.add_argument("input_folder",
                        help="Folder of class sub-folders holding icon crops.")
    parser.add_argument("--out", default="icons_report.pdf",
                        help="Output PDF path (default: icons_report.pdf).")
    parser.add_argument("--page", choices=list(PAGE_SIZES), default="a4",
                        help="Page size (default: a4).")
    args = parser.parse_args(argv)

    grouped = group_icons(args.input_folder)
    if not grouped:
        sys.exit(f"No icons found under: {args.input_folder}")

    n_imgs = len(grouped)
    n_icons = sum(len(p) for classes in grouped.values() for p in classes.values())
    print(f"Found {n_icons} icon(s) across {n_imgs} source image(s). "
          f"Writing {args.out} ...")

    with PdfPages(args.out) as pdf:
        builder = PdfBuilder(pdf, page=args.page)
        for img_name in sorted(grouped):
            builder.add_image_title(img_name)
            for class_name in sorted(grouped[img_name]):
                builder.add_class_heading(class_name)
                builder.add_icons(grouped[img_name][class_name])
        builder.finish()

    print(f"Done -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
