#!/usr/bin/env python3
"""Ensure every manuscript figure has both PNG and PDF representations.

This is principally useful for Figure 6 in the public release: its source-level
annotation reconstruction requires the withheld BIO corpus, but the final PNG
is distributed. If only PNG is present, this utility wraps that raster image in
a PDF without altering its pixels. Source-derived users with data/train.txt can
regenerate the native vector PDF through Script 10 instead.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

__author__ = "LIJUNHUA"

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
STEMS = (
    "fig01_schema",
    "fig02_extraction_funnel",
    "fig03_relation_composition",
    "fig04_threshold_sensitivity",
    "fig05_degree_structure",
    "fig06_annotation_sensitivity",
    "fig07_objective_ablation",
    "fig08_relation_lift_exact",
    "fig09_graph_map",
    "fig10_kgmmai_design",
)


def png_to_pdf(png: Path, pdf: Path):
    image = mpimg.imread(png)
    height, width = image.shape[:2]
    dpi = 150.0
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.axis("off")
    fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"created {pdf.relative_to(ROOT)} from distributed PNG")


def main():
    missing = []
    for stem in STEMS:
        png = FIG / f"{stem}.png"
        pdf = FIG / f"{stem}.pdf"
        if png.is_file() and not pdf.is_file():
            png_to_pdf(png, pdf)
        if not png.is_file():
            missing.append(str(png.relative_to(ROOT)))
        if not pdf.is_file():
            missing.append(str(pdf.relative_to(ROOT)))
    if missing:
        print("Missing figure assets after format completion:")
        for path in missing:
            print("  -", path)
        return 1
    print(f"All {len(STEMS)} manuscript figures have PNG + PDF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
