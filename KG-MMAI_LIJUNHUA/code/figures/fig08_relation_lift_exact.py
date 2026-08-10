#!/usr/bin/env python3
"""Manuscript Figure 8 — relation MRR and exact random-ranking normalisation."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import FIG, INK, MUTED, REF, REL_COLORS, RES, choose, hgrid, save_checked

__author__ = "LIJUNHUA"

relation = pd.read_csv(choose(
    RES / "statistics" / "relation_lift_exact.csv",
    REF / "relation_lift_exact.csv",
))
if "mean_filtered_candidates" not in relation.columns:
    relation["mean_filtered_candidates"] = (
        relation["head_side_candidates"] + relation["tail_side_candidates"]
    ) / 2.0

fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))
fig.subplots_adjust(left=0.10, right=0.985, top=0.87, bottom=0.18, wspace=0.38)

ax = axes[0]
ax.scatter(
    relation.mean_filtered_candidates, relation.MRR, s=70,
    c=[REL_COLORS[name] for name in relation.relation], zorder=3,
)
for row in relation.itertuples(index=False):
    ax.annotate(
        row.relation, (row.mean_filtered_candidates, row.MRR),
        textcoords="offset points", xytext=(7, 4), fontsize=7.5,
    )
ax.set_xlabel("mean filtered candidate-set size")
ax.set_ylabel("raw MRR")
ax.set_title("(a) Raw MRR tracks candidate-set size", loc="left")
hgrid(ax, "both")

ax = axes[1]
ordered = relation.sort_values("lift")
y = np.arange(len(ordered))
ax.barh(
    y, ordered.lift,
    color=[REL_COLORS[name] for name in ordered.relation],
    height=0.62,
)
ax.errorbar(
    ordered.lift, y,
    xerr=[ordered.lift - ordered.lift_ci_low, ordered.lift_ci_high - ordered.lift],
    fmt="none", ecolor=INK, lw=1.0, capsize=3,
)
ax.set_yticks(y)
ax.set_yticklabels(ordered.relation, fontsize=8)
ax.set_xlabel("lift over an exact random-ranking baseline (×)")
ax.axvline(1.0, color=MUTED, lw=0.9, ls=":")
ax.set_title("(b) Normalisation reverses the ordering", loc="left")

save_checked(fig, str(FIG / "fig08_relation_lift_exact"))
