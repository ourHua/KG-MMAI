#!/usr/bin/env python3
"""Manuscript Figure 6 — annotation audit and structural sensitivity."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import ACCENT, FIG, MUTED, REF, RES, TYPE_COLORS, choose, save_checked
from labels_en import label as en_label

__author__ = "LIJUNHUA"

structure_path = choose(
    RES / "sensitivity" / "sensitivity_structure.csv",
    REF / "annotation_sensitivity_structure.csv",
)
structure = pd.read_csv(structure_path).copy()
if "nodes" not in structure.columns:
    structure["nodes"] = structure["core_entities"]
if "edges" not in structure.columns:
    structure["edges"] = structure["core_triples"]

order = ["S0_as_annotated", "S1_adjudicated", "S2_majority_harmonised"]
structure = structure.set_index("condition").loc[order].reset_index()

hubs_path = RES / "sensitivity" / "sensitivity_hubs.csv"
if hubs_path.is_file():
    hubs = pd.read_csv(hubs_path)
    s0 = hubs[hubs.condition == "S0_as_annotated"].sort_values("rank").head(10)
else:
    s0 = pd.read_csv(RES / "core_node_degrees.csv").nlargest(10, "degree").copy()

collision_path = RES / "sensitivity" / "label_collisions.csv"
if collision_path.is_file():
    collision = pd.read_csv(collision_path)
    pair_counts = {}
    types = ["SYM", "CAU", "PRE", "HER", "EFF"]
    for row in collision.itertuples(index=False):
        present = tuple(sorted(typ for typ in types if getattr(row, typ) > 0))
        pair_counts[present] = pair_counts.get(present, 0) + 1
    type_set = pd.DataFrame(
        [("/".join(typeset), count) for typeset, count in
         sorted(pair_counts.items(), key=lambda item: -item[1])[:8]],
        columns=["type_set", "surface_forms"],
    )
else:
    type_set = pd.read_csv(REF / "annotation_collision_typesets_aggregate.csv")

fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1))
fig.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.18, wspace=0.52)

short = {
    "S0_as_annotated": "S0\nas annotated",
    "S1_adjudicated": "S1\nadjudicated",
    "S2_majority_harmonised": "S2\nmajority",
}
x = np.arange(len(structure))
width = 0.38

ax = axes[0]
ax.bar(x - width / 2, structure.nodes, width, color=TYPE_COLORS["SYM"], label="core entities")
ax.bar(x + width / 2, structure.edges, width, color=TYPE_COLORS["HER"], label="core triples")
for xi, (nodes, edges) in enumerate(zip(structure.nodes, structure.edges)):
    ax.text(xi - width / 2, nodes, f"{int(nodes):,}", ha="center", va="bottom", fontsize=7.5)
    ax.text(xi + width / 2, edges, f"{int(edges):,}", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels([short[c] for c in structure.condition])
ax.set_ylabel("count")
ax.set_ylim(0, structure.edges.max() * 1.22)
ax.set_title("(a) Core graph under the three conditions", loc="left")
ax.legend(loc="upper left", fontsize=7.5)

ax = axes[1]
colours = [
    ACCENT if typ == "PRE" else TYPE_COLORS.get(typ, MUTED)
    for typ in s0.type
]
y = np.arange(len(s0))[::-1]
ax.barh(y, s0.degree, color=colours, height=0.68)
ax.set_yticks(y)
ax.set_yticklabels(
    [f"{en_label(name, typ)} ({typ})" for name, typ in zip(s0.name, s0.type)],
    fontsize=7.0,
)
ax.set_xlabel("core degree")
ax.set_title(
    "(b) Top hubs as annotated (S0); mislabelled PRE hub in red",
    loc="left", fontsize=9.2,
)

ax = axes[2]
y = np.arange(len(type_set))[::-1]
ax.barh(y, type_set.surface_forms, color=TYPE_COLORS["CAU"], height=0.68)
ax.set_yticks(y)
ax.set_yticklabels(type_set.type_set, fontsize=7.5)
ax.set_xlabel("surface forms")
ax.set_title("(c) Multi-type surface forms by type set", loc="left")

save_checked(fig, str(FIG / "fig06_annotation_sensitivity"))
