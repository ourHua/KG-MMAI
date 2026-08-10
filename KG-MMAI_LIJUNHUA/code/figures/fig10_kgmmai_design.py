#!/usr/bin/env python3
"""Manuscript Figure 10 — intended KG-MMAI design specification."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _common import FIG, save_checked

__author__ = "LIJUNHUA"

INK = "#242424"
MUTED = "#777777"
BLUE = "#1B6C7F"
GREEN = "#4A7C59"
PURPLE = "#6D4C8C"
NAVY = "#345366"
GOLD = "#C9A227"
ORANGE = "#C4622D"

FILL_BLUE = "#E8F1F4"
FILL_GREEN = "#EEF4EF"
FILL_PURPLE = "#F1EEF7"
FILL_GRAY = "#EEF0F2"
FILL_GOLD = "#FFF9E8"
FILL_ORANGE = "#FFF3EC"


def rounded_box(ax, x, y, w, h, title, subtitle="", edge=BLUE, face="white",
                title_size=9.0, subtitle_size=7.2, radius=0.09):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=1.45,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h * (0.62 if subtitle else 0.52), title,
        ha="center", va="center", fontsize=title_size,
        fontweight="semibold", color=edge if edge != NAVY else NAVY,
    )
    if subtitle:
        ax.text(
            x + w / 2, y + h * 0.28, subtitle,
            ha="center", va="center", fontsize=subtitle_size, color=MUTED,
        )


def arrow(ax, start, end, color=MUTED, lw=1.2, rad=0.0, scale=11):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=scale,
        linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}",
        shrinkA=0, shrinkB=0,
    ))


fig, ax = plt.subplots(figsize=(10.45, 5.0))
ax.set_xlim(0, 15.3)
ax.set_ylim(0, 7.25)
ax.axis("off")

for x, label in [
    (1.55, "Multimodal input"),
    (5.03, "Modality encoding"),
    (8.70, "Knowledge-guided fusion"),
    (12.55, "Auditable output"),
]:
    ax.text(x, 6.96, label, ha="center", va="center",
            fontsize=10.0, fontweight="bold", color=INK)

ys = [5.30, 3.45, 1.60]
inputs = [
    ("Tongue image", "visual"),
    ("Pulse series", "time series"),
    ("Inquiry text", "clinical text"),
]
encoders = [
    ("ResNet encoder", ""),
    ("1D-CNN encoder", ""),
    ("Clinical LM encoder", ""),
]
for y, (title, subtitle) in zip(ys, inputs):
    rounded_box(ax, 0.18, y, 2.80, 1.36, title, subtitle,
                edge=BLUE, face=FILL_BLUE, title_size=8.8)
for y, (title, subtitle) in zip(ys, encoders):
    rounded_box(ax, 3.48, y, 2.80, 1.36, title, subtitle,
                edge=GREEN, face=FILL_GREEN, title_size=8.8)
    arrow(ax, (2.98, y + 0.68), (3.48, y + 0.68), color=MUTED)

rounded_box(
    ax, 6.80, 5.48, 3.00, 1.16,
    "TCM knowledge graph", "embedded prior  q(s | h)",
    edge=PURPLE, face=FILL_PURPLE, title_size=8.7,
)
rounded_box(
    ax, 6.80, 2.72, 3.00, 2.12,
    "Cross-modal\nattention", "",
    edge=NAVY, face=FILL_GRAY, title_size=9.0,
)

arrow(ax, (6.28, 5.98), (6.80, 3.98), color=MUTED, rad=0.03)
arrow(ax, (6.28, 4.13), (6.80, 3.80), color=MUTED)
arrow(ax, (6.28, 2.28), (6.80, 3.60), color=MUTED, rad=-0.03)

arrow(ax, (8.30, 5.48), (8.30, 4.84), color=PURPLE, lw=1.25)
ax.text(8.52, 5.10, r"KL divergence penalty  $\lambda^L L_{kg}$",
        fontsize=7.5, fontweight="semibold", color=PURPLE, va="center")

rounded_box(
    ax, 10.45, 3.28, 2.35, 1.23,
    "Classifier", "", edge=NAVY, face=FILL_GRAY, title_size=9.0,
)
arrow(ax, (9.80, 3.78), (10.45, 3.78), color=MUTED)

rounded_box(
    ax, 10.22, 5.40, 2.90, 1.05,
    "Auditable path", "", edge=ORANGE, face=FILL_ORANGE, title_size=8.8,
)
arrow(ax, (11.62, 4.51), (11.62, 5.40), color=ORANGE, lw=1.15)

out_x = 13.55
out_w = 1.45
out_h = 0.98
outputs = [
    ("Symptom", 5.90),
    ("Pathogenesis", 4.38),
    ("Syndrome", 2.86),
    ("Formula", 1.34),
]
for title, y in outputs:
    rounded_box(
        ax, out_x, y, out_w, out_h, title, "",
        edge=GOLD, face=FILL_GOLD, title_size=7.8, radius=0.08,
    )

arrow(ax, (13.12, 5.90), (13.55, 6.39), color=ORANGE, lw=1.1)
arrow(ax, (12.80, 3.90), (13.55, 3.35), color=MUTED, lw=1.1)
for (_, y1), (_, y2) in zip(outputs[:-1], outputs[1:]):
    arrow(ax, (out_x + out_w / 2, y1), (out_x + out_w / 2, y2 + out_h),
          color=GOLD, lw=1.05, scale=9)

ax.text(
    0.18, 0.31,
    "The graph constrains the predictive distribution during training; "
    "every prediction retains the graph-induced prior that informed it.",
    fontsize=7.3, color=MUTED, ha="left", va="center",
)

save_checked(fig, str(FIG / "fig10_kgmmai_design"))
