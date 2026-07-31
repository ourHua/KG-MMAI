#!/usr/bin/env python3
"""05_figures_structure.py — Figures 2-5 (graph structure).

Layout policy: panel tags are drawn in figure coordinates via figstyle.tag(),
and inter-panel spacing is set explicitly with subplots_adjust, so no tag,
title, or axis label can collide with a neighbouring panel. All entity names
are rendered in Latin script by code/labels.py; no figure emits CJK.
"""

__author__ = "LIJUNHUA"
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from figstyle import (FAINT, INK, MUTED, REL_COLORS, TYPE_COLORS, TYPE_LABELS,
                      hgrid, save_checked, tag)

DATA, RES, FIG = ROOT / "data", ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

nodes = pd.read_csv(DATA / "nodes.csv", encoding="utf-8-sig")
edges = pd.read_csv(DATA / "edges.csv", encoding="utf-8-sig")
nodes_core = pd.read_csv(DATA / "nodes_core.csv", encoding="utf-8-sig")
core = edges[edges.weight >= 2]
prof = pd.read_csv(RES / "structural_profile.csv")
reld = pd.read_csv(RES / "relation_distribution.csv")

# =========================================================================== #
# FIGURE 2 — schema
# =========================================================================== #
fig, ax = plt.subplots(figsize=(7.2, 4.1))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")

pos = {"CAU": (1.5, 4.5), "SYM": (5.0, 4.5), "PRE": (1.5, 1.3),
       "HER": (5.0, 1.3), "EFF": (8.5, 1.3)}
c_all = nodes.type.value_counts()
c_core = nodes_core.type.value_counts()

for t, (x, y) in pos.items():
    ax.add_patch(FancyBboxPatch((x - 1.05, y - 0.52), 2.1, 1.04,
                                boxstyle="round,pad=0.02,rounding_size=0.14",
                                linewidth=1.6, edgecolor=TYPE_COLORS[t],
                                facecolor=TYPE_COLORS[t] + "1A", zorder=3))
    ax.text(x, y + 0.21, TYPE_LABELS[t], ha="center", va="center", fontsize=10,
            fontweight="bold", color=TYPE_COLORS[t], zorder=4)
    ax.text(x, y - 0.05, t, ha="center", va="center", fontsize=7.5,
            color=MUTED, zorder=4)
    ax.text(x, y - 0.30, f"{c_all[t]:,}  \u2192  {c_core[t]:,}", ha="center",
            va="center", fontsize=8, color=INK, zorder=4)

core_n = reld.set_index("relation").triples
for a, b, rel, frac, ox, oy in [
        ("CAU", "SYM", "CAUSES", 0.50, 0.0, 0.0),
        ("PRE", "HER", "CONTAINS", 0.50, 0.0, 0.0),
        ("HER", "EFF", "HAS_EFFECT", 0.50, 0.0, 0.0),
        ("HER", "SYM", "RELIEVES", 0.50, 0.0, 0.0),
        ("PRE", "SYM", "TREATS", 0.44, -0.66, 0.34)]:
    x1, y1 = pos[a]; x2, y2 = pos[b]
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=1.5,
                                 color=REL_COLORS[rel], shrinkA=58, shrinkB=58,
                                 zorder=2, alpha=0.9))
    mx = x1 + (x2 - x1) * frac + ox
    my = y1 + (y2 - y1) * frac + oy
    ax.text(mx, my + 0.17, rel, ha="center", va="center", fontsize=8,
            fontweight="bold", color=REL_COLORS[rel], zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.4))
    ax.text(mx, my - 0.10, f"{core_n[rel]:,} triples", ha="center", va="center",
            fontsize=7.2, color=MUTED, zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.1))

ax.text(0.10, 5.95, "Five entity types, five directional relation types",
        fontsize=10.5, fontweight="semibold", color=INK)
ax.text(0.10, 5.62,
        "Entity boxes: all extracted  \u2192  retained in core graph (weight \u2265 2)",
        fontsize=8.2, color=MUTED)
save_checked(fig, str(FIG / "fig01_schema"))

# =========================================================================== #
# FIGURE 3 — extraction funnel
# =========================================================================== #
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.15))
fig.subplots_adjust(left=0.135, right=0.985, top=0.79, bottom=0.165, wspace=0.42)

ax = axes[0]
stages = ["Extracted\nentities", "Entities in\n\u2265 1 relation", "Core-graph\nentities"]
vals = [len(nodes), len(set(edges.source_id) | set(edges.target_id)),
        int(prof.loc[prof.threshold == 2, "nodes"].iloc[0])]
ax.barh([2, 1, 0], vals, height=0.52,
        color=["#C9D6DA", "#6E9FAC", "#1B6C7F"], zorder=3)
for i, v in enumerate(vals):
    y = 2 - i
    ax.text(v + max(vals) * 0.025, y + 0.10, f"{v:,}", va="center",
            fontsize=8.8, fontweight="bold", color=INK)
    if i:
        ax.text(v + max(vals) * 0.025, y - 0.21,
                f"{100*v/vals[0]:.1f}% of extracted", va="center",
                fontsize=7.1, color=MUTED)
ax.set_yticks([2, 1, 0]); ax.set_yticklabels(stages, fontsize=8.4)
ax.set_xlim(0, max(vals) * 1.38)
ax.set_xlabel("Entities")
ax.set_title("Entity attrition", loc="left", pad=8)
hgrid(ax, "x")

ax = axes[1]
wd = pd.read_csv(RES / "weight_distribution.csv")
wd.columns = ["weight", "n", "cum"]
top = wd[wd.weight <= 10]
ax.bar(top.weight, top.n, width=0.7, color="#1B6C7F", zorder=3)
ax.bar([11], [wd[wd.weight > 10].n.sum()], width=0.7, color="#9EBCC4", zorder=3)
ax.axvline(1.5, color="#C4622D", linewidth=1.4, linestyle="--", zorder=4)
ax.set_yscale("log")
ax.set_ylim(30, wd.n.max() * 16)
ax.text(1.95, wd.n.max() * 5.5, "core threshold  w \u2265 2", fontsize=7.6,
        color="#C4622D", fontweight="semibold", va="center")
ax.annotate(f"{wd.iloc[0].n:,.0f} single-occurrence triples\n"
            f"({100*wd.iloc[0].n/wd.n.sum():.1f}% of all candidates)",
            xy=(1, wd.iloc[0].n), xytext=(4.3, wd.n.max() * 1.5),
            fontsize=7.4, color=INK, va="center",
            arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.9,
                            shrinkB=3))
ax.set_xticks(range(1, 12))
ax.set_xticklabels([str(i) for i in range(1, 11)] + [">10"], fontsize=8)
ax.set_xlabel("Co-occurrence weight")
ax.set_ylabel("Candidate triples (log)")
ax.set_title("Weight distribution", loc="left", pad=8)
hgrid(ax)

tag(fig, "a", 0.010, 0.965); tag(fig, "b", 0.545, 0.965)
save_checked(fig, str(FIG / "fig02_extraction_funnel"))

# =========================================================================== #
# FIGURE 4 — relation composition
# =========================================================================== #
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
fig.subplots_adjust(left=0.19, right=0.975, top=0.79, bottom=0.165, wspace=0.44)

ax = axes[0]
r = reld.sort_values("triples")
y = np.arange(len(r))
ax.barh(y, r.all_triples, height=0.60, color=FAINT, zorder=2,
        label="All candidates (w \u2265 1)")
ax.barh(y, r.triples, height=0.60, color=[REL_COLORS[x] for x in r.relation],
        zorder=3, label="Core graph (w \u2265 2)")
for i, row in enumerate(r.itertuples(index=False)):
    ax.text(row.all_triples + 400, i, f"{row.triples:,} / {row.all_triples:,}",
            va="center", fontsize=7.2, color=INK)
ax.set_yticks(y)
ax.set_yticklabels([f"{x}\n{s}" for x, s in zip(r.relation, r.schema)],
                   fontsize=7.7)
ax.set_xlim(0, r.all_triples.max() * 1.44)
ax.set_xlabel("Triples")
ax.set_title("Composition", loc="left", pad=8)
ax.legend(loc="lower right", fontsize=7.2, borderpad=0.2)
hgrid(ax, "x")

ax = axes[1]
r2 = reld.sort_values("survival_pct")
ax.barh(np.arange(len(r2)), r2.survival_pct, height=0.56,
        color=[REL_COLORS[x] for x in r2.relation], zorder=3)
for i, row in enumerate(r2.itertuples(index=False)):
    ax.text(row.survival_pct + 0.7, i + 0.11, f"{row.survival_pct:.1f}%",
            va="center", fontsize=8.0, fontweight="bold", color=INK)
    ax.text(row.survival_pct + 0.7, i - 0.21, f"mean w {row.mean_weight:.2f}",
            va="center", fontsize=6.8, color=MUTED)
overall = 100 * len(core) / len(edges)
ax.axvline(overall, color=INK, linewidth=1.0, linestyle=":", zorder=4)
ax.text(overall - 0.8, 4.30, f"overall {overall:.1f}%", fontsize=7.1,
        color=INK, ha="right", va="center")
ax.set_yticks(np.arange(len(r2)))
ax.set_yticklabels(r2.relation, fontsize=8)
ax.set_xlim(0, 35)
ax.set_xlabel("Surviving w \u2265 2 (%)")
ax.set_title("Survival rate", loc="left", pad=8)
hgrid(ax, "x")

tag(fig, "a", 0.010, 0.965); tag(fig, "b", 0.555, 0.965)
save_checked(fig, str(FIG / "fig03_relation_composition"))

# =========================================================================== #
# FIGURE 5 — threshold sensitivity (four single-axis panels; no twin axes,
# which removes the tick-label collisions twin axes create between neighbours)
# =========================================================================== #
fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.6))
fig.subplots_adjust(left=0.075, right=0.988, top=0.77, bottom=0.235, wspace=0.52)

ax = axes[0]
ax.plot(prof.threshold, prof.edges, "s-", color="#C4622D", linewidth=1.7,
        markersize=4.4, label="Triples", zorder=3)
ax.plot(prof.threshold, prof.nodes, "o-", color="#1B6C7F", linewidth=1.7,
        markersize=4.4, label="Entities", zorder=3)
ax.scatter([2], [9544], s=120, facecolor="none", edgecolor="#C4622D",
           linewidth=1.3, zorder=4)
ax.set_yscale("log"); ax.set_ylim(90, 400000)
ax.set_yticks([1e2, 1e3, 1e4, 1e5])
ax.set_xticks([1, 3, 5, 10])
ax.set_xlabel("Threshold \u03C4"); ax.set_ylabel("Count (log)")
ax.set_title("Graph size", loc="left", pad=7, fontsize=9.5)
ax.legend(fontsize=7.2, loc="upper right", borderpad=0.2, handlelength=1.4)
hgrid(ax)

ax = axes[1]
ax.plot(prof.threshold, prof.largest_component_pct, "o-", color="#4A7C59",
        linewidth=1.7, markersize=4.4, zorder=3)
for t, v, k in zip(prof.threshold, prof.largest_component_pct, prof.components):
    ax.annotate(str(int(k)), (t, v), xytext=(0, 7), textcoords="offset points",
                ha="center", fontsize=6.8, color=MUTED)
ax.set_ylim(97.9, 100.9)
ax.set_xticks([1, 3, 5, 10])
ax.set_yticks([98, 99, 100])
ax.set_xlabel("Threshold \u03C4"); ax.set_ylabel("Largest comp. (%)")
ax.set_title("Connectivity", loc="left", pad=7, fontsize=9.5)
hgrid(ax)

ax = axes[2]
ax.plot(prof.threshold, prof.mean_degree, "o-", color="#6D4C8C", linewidth=1.7,
        markersize=4.4, zorder=3)
ax.set_xticks([1, 3, 5, 10])
ax.set_ylim(2, 17)
ax.set_xlabel("Threshold \u03C4"); ax.set_ylabel("Mean degree")
ax.set_title("Mean degree", loc="left", pad=7, fontsize=9.5)
hgrid(ax)

ax = axes[3]
ax.plot(prof.threshold, prof.density * 1000, "^-", color="#C9A227",
        linewidth=1.7, markersize=4.6, zorder=3)
ax.set_xticks([1, 3, 5, 10])
ax.set_ylim(0, 12.5)
ax.set_xlabel("Threshold \u03C4")
ax.set_ylabel("Density \u00D7 10\u00B3")
ax.set_title("Density", loc="left", pad=7, fontsize=9.5)
hgrid(ax)

tag(fig, "a", 0.004, 0.975); tag(fig, "b", 0.256, 0.975)
tag(fig, "c", 0.506, 0.975); tag(fig, "d", 0.756, 0.975)
save_checked(fig, str(FIG / "fig04_threshold_sensitivity"))
print("structure figures done")
