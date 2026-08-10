#!/usr/bin/env python3
"""Manuscript Figure 5 — degree structure of the core graph."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import FIG, INK, MUTED, RES, TYPE_COLORS, TYPE_LABELS, hgrid, save_checked, tag
from labels_en import label as en_label

__author__ = "LIJUNHUA"

degs = pd.read_csv(RES / "core_node_degrees.csv")
degs["label"] = [en_label(name, typ) for name, typ in zip(degs.name, degs.type)]

fig, axes = plt.subplots(
    1, 3, figsize=(7.6, 3.05),
    gridspec_kw={"width_ratios": [1, 1, 1.45]},
)
fig.subplots_adjust(left=0.078, right=0.988, top=0.795, bottom=0.225, wspace=0.50)

ax = axes[0]
degree = np.sort(degs.degree.values)
ccdf = 1.0 - np.arange(len(degree)) / len(degree)
ax.loglog(degree, ccdf, color="#1B6C7F", linewidth=1.9, zorder=3)
ax.set_xlabel("Degree $k$")
ax.set_ylabel("P(K ≥ k)")
ax.set_title("Degree distribution", loc="left", pad=7, fontsize=9.5)
ax.text(
    0.045, 0.09,
    f"skewness {degs.degree.skew():.2f}\nmax degree {degree.max()}",
    transform=ax.transAxes, ha="left", va="bottom", fontsize=7.4, color=MUTED,
)
ax.grid(which="both", linewidth=0.5, color="#EDEDED")
ax.set_axisbelow(True)

ax = axes[1]
order = ["SYM", "HER", "CAU", "PRE", "EFF"]
data = [degs.loc[degs.type == typ, "degree"].values for typ in order]
bp = ax.boxplot(
    data, widths=0.56, patch_artist=True, showfliers=False,
    medianprops=dict(color="white", linewidth=1.3),
    whiskerprops=dict(color=MUTED, linewidth=0.9),
    capprops=dict(color=MUTED, linewidth=0.9),
)
for patch, typ in zip(bp["boxes"], order):
    patch.set_facecolor(TYPE_COLORS[typ])
    patch.set_edgecolor("none")
for pos, typ in enumerate(order, 1):
    values = degs.loc[degs.type == typ, "degree"]
    ax.scatter(
        np.full(len(values), pos), values, s=2.6,
        color=TYPE_COLORS[typ], alpha=0.15, zorder=1, linewidths=0,
    )
ax.set_yscale("log")
ax.set_ylim(0.8, 400)
ax.set_xticks(range(1, 6))
ax.set_xticklabels(
    [f"{TYPE_LABELS[typ][:4]}.\n{len(degs[degs.type == typ])}" for typ in order],
    fontsize=7.4,
)
ax.set_ylabel("Core-graph degree (log)")
ax.set_title("Degree by entity type", loc="left", pad=7, fontsize=9.5)
ax.text(
    0.5, -0.235, "type (entities)", transform=ax.transAxes,
    ha="center", va="top", fontsize=8.2, color=INK,
)
hgrid(ax)

ax = axes[2]
hub = degs.nlargest(12, "degree").sort_values("degree")
ax.barh(
    np.arange(len(hub)), hub.degree, height=0.66,
    color=[TYPE_COLORS[typ] for typ in hub.type], zorder=3,
)
ax.set_yticks(np.arange(len(hub)))
ax.set_yticklabels(hub.label.tolist(), fontsize=7.4)
for idx, value in enumerate(hub.degree):
    ax.text(value + 4, idx, str(value), va="center", fontsize=7.0, color=INK)
ax.set_xlim(0, hub.degree.max() * 1.17)
ax.set_xticks([0, 100, 200])
ax.set_xlabel("Core-graph degree")
ax.set_title("Highest-degree entities", loc="left", pad=7, fontsize=9.5)
hgrid(ax, "x")

for idx, typ in enumerate(["CAU", "HER", "EFF", "PRE"]):
    fig.text(0.655 + idx * 0.087, 0.037, "■", color=TYPE_COLORS[typ],
             fontsize=8, ha="left", va="center")
    fig.text(0.671 + idx * 0.087, 0.037, TYPE_LABELS[typ][:4] + ".",
             fontsize=7.2, ha="left", va="center", color=INK)

tag(fig, "a", 0.004, 0.978)
tag(fig, "b", 0.348, 0.978)
tag(fig, "c", 0.652, 0.978)
save_checked(fig, str(FIG / "fig05_degree_structure"))
