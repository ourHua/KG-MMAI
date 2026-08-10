#!/usr/bin/env python3
"""Generate Figure 10, the design specification of KG-MMAI.

The revised manuscript explicitly states that the multimodal classifier and
knowledge-constraint terms shown here are not implemented or evaluated in this
study. The figure mirrors the intended architecture without implying an
experimental result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

__author__ = "LIJUNHUA"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import figstyle as fs  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
INK = "#222222"; MUTED = "#666666"
BLUE = "#2B7A8B"; GREEN = "#4F805E"; PURPLE = "#66598A"
GOLD = "#C59A2D"; ORANGE = "#B86C3A"
LIGHT = {"b": "#EAF5F7", "g": "#EEF5EF", "p": "#F0EEF7",
         "y": "#FBF5E6", "o": "#FBF0EA"}


def box(ax, x, y, w, h, title, subtitle="", edge=BLUE, face="#fff", fs=9.3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.12",
        facecolor=face, edgecolor=edge, linewidth=1.5))
    ax.text(x+w/2, y+h*0.62, title, ha="center", va="center",
            fontsize=fs, fontweight="semibold", color=INK)
    if subtitle:
        ax.text(x+w/2, y+h*0.30, subtitle, ha="center", va="center",
                fontsize=7.2, color=MUTED)


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.3, rad=0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}"))


def main():
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.set_xlim(0, 15.5); ax.set_ylim(0, 8.3); ax.axis("off")
    for x, text in [(0.7, "Multimodal input"), (3.5, "Modality encoding"),
                    (6.8, "Knowledge-guided fusion"), (11.1, "Auditable output")]:
        ax.text(x, 7.9, text, fontsize=10.3, fontweight="bold", ha="center", color=INK)

    ys = [5.7, 3.7, 1.7]
    inputs = [("Tongue image", "visual"), ("Pulse series", "time series"),
              ("Inquiry text", "clinical text")]
    encoders = [("Residual encoder", "image"), ("1D-CNN encoder", "pulse"),
                ("Clinical LM encoder", "text")]
    for y, (title, subtitle) in zip(ys, inputs):
        box(ax, 0.05, y, 2.0, 1.25, title, subtitle, BLUE, LIGHT["b"])
    for y, (title, subtitle) in zip(ys, encoders):
        box(ax, 2.9, y, 2.2, 1.25, title, subtitle, GREEN, LIGHT["g"])
        arrow(ax, 2.05, y+0.625, 2.9, y+0.625)

    box(ax, 6.15, 2.85, 2.35, 2.0, "Cross-modal\nattention",
        "case-specific fusion", PURPLE, LIGHT["p"], 9.5)
    for y in ys:
        arrow(ax, 5.1, y+0.625, 6.15, 3.85, rad=(3.85-(y+0.625))*0.04)

    box(ax, 6.15, 5.65, 2.35, 1.25, "TCM knowledge graph",
        "embedded prior  q(s | h)", ORANGE, LIGHT["o"])
    box(ax, 9.05, 4.55, 1.85, 1.40, "Knowledge-guided\nfusion",
        "projection + KL term", ORANGE, LIGHT["o"], 8.9)
    arrow(ax, 8.5, 6.25, 9.05, 5.45, ORANGE)
    arrow(ax, 8.5, 3.85, 9.05, 5.05, PURPLE)

    box(ax, 9.05, 2.45, 1.85, 1.25, "Classifier",
        "syndrome distribution", PURPLE, LIGHT["p"])
    arrow(ax, 8.5, 3.65, 9.05, 3.05, PURPLE)
    box(ax, 11.55, 4.85, 1.85, 1.15, "Auditable path",
        "graph-informed trace", ORANGE, LIGHT["o"])
    arrow(ax, 10.9, 5.25, 11.55, 5.42, ORANGE)
    arrow(ax, 10.9, 3.05, 11.55, 5.08, PURPLE, rad=-0.12)

    outputs = [("Symptom", 6.4), ("Pathogenesis", 4.95),
               ("Syndrome", 3.5), ("Formula", 2.05)]
    for title, y in outputs:
        box(ax, 13.65, y, 1.55, 0.92, title, "", GOLD, LIGHT["y"], 8.8)
    for (_, y1), (_, y2) in zip(outputs[:-1], outputs[1:]):
        arrow(ax, 14.42, y1, 14.42, y2+0.92, GOLD, 1.0)
    arrow(ax, 13.4, 5.42, 13.65, 5.42, ORANGE)

    ax.text(0.1, 0.52,
            "DESIGN SPECIFICATION ONLY — the multimodal classifier and knowledge-constraint terms are not implemented or evaluated in this study.",
            fontsize=8.4, fontweight="bold", color="#9A3F3F", ha="left", va="center")
    ax.text(0.1, 0.13,
            "Figure 10 visualises the intended audit path; realisation requires syndrome entities and a syndrome-adjacency relation in the graph.",
            fontsize=7.4, color=MUTED, ha="left", va="center")
    fs.save_checked(fig, FIG / "fig10_kgmmai_design")


if __name__ == "__main__":
    main()
