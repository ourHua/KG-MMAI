#!/usr/bin/env python3
"""Manuscript Figure 7 — controlled objective ablation and triple-level inference."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import ACCENT, FIG, MODEL_COLORS, MUTED, REF, RES, TYPE_COLORS, choose, save_checked

__author__ = "LIJUNHUA"

MODELS = ("TransE", "DistMult", "ComplEx", "RotatE")
OBJECTIVES = ("margin", "logistic", "selfadv")
OBJ_LABEL = {
    "margin": "O1 margin",
    "logistic": "O2 logistic",
    "selfadv": "O3 self-adversarial",
}
CFG_A = {"ComplEx": 0.225, "DistMult": 0.215, "TransE": 0.153, "RotatE": 0.114}

summary = pd.read_csv(choose(
    RES / "ablation" / "objective_ablation_summary.csv",
    REF / "objective_ablation_60ep.csv",
))
at60 = summary[summary.budget_epochs == 60] if "budget_epochs" in summary.columns else summary
pairwise = pd.read_csv(choose(
    RES / "statistics" / "model_pairwise_triplelevel.csv",
    REF / "model_pairwise_triplelevel.csv",
))
bootstrap = pd.read_csv(choose(
    RES / "statistics" / "model_bootstrap_clustered.csv",
    REF / "model_bootstrap_clustered.csv",
))

fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.2))
fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.09, hspace=0.42, wspace=0.28)

x = np.arange(len(OBJECTIVES))
width = 0.2

ax = axes[0, 0]
for index, model in enumerate(MODELS):
    values, sds = [], []
    for objective in OBJECTIVES:
        cell = at60[(at60.objective == objective) & (at60.model == model)]
        values.append(cell.MRR_mean.iloc[0])
        sds.append(cell.MRR_sd.iloc[0])
    ax.bar(
        x + (index - 1.5) * width, values, width, yerr=sds, capsize=2,
        color=MODEL_COLORS[model], label=model,
        error_kw=dict(lw=0.8, ecolor=MUTED),
    )
ax.set_xticks(x)
ax.set_xticklabels([OBJ_LABEL[obj] for obj in OBJECTIVES], fontsize=8)
ax.set_ylabel("test MRR")
ax.set_title("(a) One code base, three objectives (60 epochs)", loc="left")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7.5)
ax.set_ylim(0, max(at60.MRR_mean) * 1.35)

ax = axes[0, 1]
labels = ["Config. A\n(reported,\nseparate code base)"] + [
    OBJ_LABEL[obj].replace(" ", "\n", 1) for obj in OBJECTIVES
]
xs = np.arange(len(labels))
for model in MODELS:
    ranks = [sorted(CFG_A, key=CFG_A.get, reverse=True).index(model) + 1]
    for objective in OBJECTIVES:
        cell = at60[at60.objective == objective].sort_values("MRR_mean", ascending=False)
        ranks.append(list(cell.model).index(model) + 1)
    ax.plot(xs, ranks, "-o", color=MODEL_COLORS[model], lw=1.6, ms=5)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=8)
ax.set_yticks([1, 2, 3, 4])
ax.invert_yaxis()
ax.set_ylabel("rank by MRR")
ax.set_title("(b) The ordering follows the objective", loc="left")
ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")

ax = axes[1, 0]
bootstrap = bootstrap.sort_values("MRR_triple_level")
y = np.arange(len(bootstrap))
ax.errorbar(
    bootstrap.MRR_triple_level, y,
    xerr=[
        bootstrap.MRR_triple_level - bootstrap.ci_low_triple,
        bootstrap.ci_high_triple - bootstrap.MRR_triple_level,
    ],
    fmt="none", lw=1.4, capsize=3, ecolor=MUTED,
)
for yi, row in enumerate(bootstrap.itertuples(index=False)):
    ax.plot(row.MRR_triple_level, yi, "o", ms=6, color=MODEL_COLORS[row.model])
ax.set_yticks(y)
ax.set_yticklabels(bootstrap.model)
ax.set_xlabel(f"MRR (triple-level unit, n = {int(bootstrap.n_triples.iloc[0])})")
ax.set_title("(c) Cluster-bootstrap intervals overlap", loc="left")

ax = axes[1, 1]
objective_colours = {
    "margin": TYPE_COLORS["CAU"],
    "logistic": TYPE_COLORS["SYM"],
    "selfadv": TYPE_COLORS["HER"],
}
effect_data = pairwise.copy()
effect_data["abs_d"] = effect_data.cohens_d.abs()
effect_data = effect_data.sort_values(["objective", "abs_d"])
y = np.arange(len(effect_data))[::-1]
ax.barh(
    y, effect_data.abs_d,
    color=[objective_colours[obj] for obj in effect_data.objective],
    height=0.72,
)
ax.axvline(0.2, color=ACCENT, lw=1.1, ls="--")
ax.text(0.21, y.max(), "small-effect\nthreshold 0.2",
        color=ACCENT, fontsize=7.5, va="top")
ax.set_yticks(y)
ax.set_yticklabels([name.replace(" - ", "−") for name in effect_data.comparison], fontsize=6.2)
handles = [plt.Rectangle((0, 0), 1, 1, color=objective_colours[obj]) for obj in OBJECTIVES]
ax.legend(handles, [OBJ_LABEL[obj] for obj in OBJECTIVES], fontsize=7.5, loc="lower right")
ax.set_xlabel("|Cohen's d| on triple-level differences")
ax.set_xlim(0, max(0.62, effect_data.abs_d.max() * 1.18))
ax.set_title("(d) Pairwise effect sizes by objective", loc="left")

save_checked(fig, str(FIG / "fig07_objective_ablation"))
