#!/usr/bin/env python3
"""Generate revised-manuscript Figures 6--8.

Figure 6: annotation audit and S0/S1/S2 structural sensitivity.
Figure 7: controlled objective ablation inside one code base.
Figure 8: relation-level performance normalised by the exact per-query
          random-ranking baseline.

Historical fig10/fig11/fig12 aliases from an earlier revision are intentionally
not written because Figure 10 in the final manuscript is the KG-MMAI design.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__author__ = "LIJUNHUA"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import figstyle as fs  # noqa: E402
from labels_en import label as en_label  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
MODELS = ("TransE", "DistMult", "ComplEx", "RotatE")
OBJ_LABEL = {
    "margin": "O1 margin",
    "logistic": "O2 logistic",
    "selfadv": "O3 self-adversarial",
}
CFG_A = {"ComplEx": 0.225, "DistMult": 0.215, "TransE": 0.153, "RotatE": 0.114}


def require(*paths):
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing prerequisite result file(s):\n  "
            + "\n  ".join(str(p.relative_to(ROOT)) for p in missing)
        )


def figure6_annotation():
    structure_path = RES / "sensitivity" / "sensitivity_structure.csv"
    hubs_path = RES / "sensitivity" / "sensitivity_hubs.csv"
    collisions_path = RES / "sensitivity" / "label_collisions.csv"
    require(structure_path, hubs_path, collisions_path)

    structure = pd.read_csv(structure_path)
    hubs = pd.read_csv(hubs_path)
    collisions = pd.read_csv(collisions_path)
    order = ["S0_as_annotated", "S1_expert_corrected", "S2_majority_harmonised"]
    structure = structure.set_index("condition").loc[order].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.1))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.18, wspace=0.52)
    short = {
        "S0_as_annotated": "S0\nas annotated",
        "S1_expert_corrected": "S1\nexpert",
        "S2_majority_harmonised": "S2\nmajority",
    }
    x = np.arange(len(structure)); width = 0.38

    ax = axes[0]
    ax.bar(x-width/2, structure.nodes, width, color=fs.TYPE_COLORS["SYM"], label="core entities")
    ax.bar(x+width/2, structure.edges, width, color=fs.TYPE_COLORS["HER"], label="core triples")
    for xi, (nodes, edges) in enumerate(zip(structure.nodes, structure.edges)):
        ax.text(xi-width/2, nodes, f"{int(nodes):,}", ha="center", va="bottom", fontsize=7.5)
        ax.text(xi+width/2, edges, f"{int(edges):,}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([short[c] for c in structure.condition])
    ax.set_ylabel("count"); ax.set_ylim(0, structure.edges.max()*1.22)
    ax.set_title("(a) Core graph under the three conditions", loc="left")
    ax.legend(loc="upper left", fontsize=7.5)

    ax = axes[1]
    s0 = hubs[hubs.condition == "S0_as_annotated"].sort_values("rank").head(10)
    colours = [fs.ACCENT if typ == "PRE" else fs.TYPE_COLORS.get(typ, fs.MUTED)
               for typ in s0.type]
    y = np.arange(len(s0))[::-1]
    ax.barh(y, s0.degree, color=colours, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{en_label(name, typ)} ({typ})"
                        for name, typ in zip(s0.name, s0.type)], fontsize=7.0)
    ax.set_xlabel("core degree")
    ax.set_title("(b) Top hubs as annotated (S0); mislabelled PRE hub in red",
                 loc="left", fontsize=9.2)

    ax = axes[2]
    pair_counts = {}; types = ["SYM", "CAU", "PRE", "HER", "EFF"]
    for row in collisions.itertuples(index=False):
        present = tuple(sorted(typ for typ in types if getattr(row, typ) > 0))
        pair_counts[present] = pair_counts.get(present, 0) + 1
    top = sorted(pair_counts.items(), key=lambda item: -item[1])[:8]
    y = np.arange(len(top))[::-1]
    ax.barh(y, [count for _, count in top], color=fs.TYPE_COLORS["CAU"], height=0.68)
    ax.set_yticks(y); ax.set_yticklabels(["/".join(ts) for ts, _ in top], fontsize=7.5)
    ax.set_xlabel("surface forms")
    ax.set_title("(c) Multi-type surface forms by type set", loc="left")
    fs.save_checked(fig, FIG / "fig06_annotation_sensitivity")


def figure7_ablation():
    summary_path = RES / "ablation" / "objective_ablation_summary.csv"
    pairwise_path = RES / "statistics" / "model_pairwise_triplelevel.csv"
    bootstrap_path = RES / "statistics" / "model_bootstrap_clustered.csv"
    require(summary_path, pairwise_path, bootstrap_path)
    summary = pd.read_csv(summary_path)
    pairwise = pd.read_csv(pairwise_path)
    bootstrap = pd.read_csv(bootstrap_path)
    at60 = summary[summary.budget_epochs == 60]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.2))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.09,
                        hspace=0.42, wspace=0.28)
    objectives = ["margin", "logistic", "selfadv"]
    x = np.arange(len(objectives)); width = 0.2

    ax = axes[0, 0]
    for index, model in enumerate(MODELS):
        values, sds = [], []
        for obj in objectives:
            cell = at60[(at60.objective == obj) & (at60.model == model)]
            if len(cell) != 1:
                raise RuntimeError(f"Expected one 60-epoch cell for {obj}/{model}")
            values.append(cell.MRR_mean.iloc[0]); sds.append(cell.MRR_sd.iloc[0])
        ax.bar(x+(index-1.5)*width, values, width, yerr=sds, capsize=2,
               color=fs.MODEL_COLORS[model], label=model,
               error_kw=dict(lw=0.8, ecolor=fs.MUTED))
    ax.set_xticks(x); ax.set_xticklabels([OBJ_LABEL[obj] for obj in objectives], fontsize=8)
    ax.set_ylabel("test MRR")
    ax.set_title("(a) One code base, three objectives (60 epochs)", loc="left")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7.5)
    ax.set_ylim(0, max(at60.MRR_mean)*1.35)

    ax = axes[0, 1]
    labels = ["Config. A\n(reported,\nseparate code base)"] + [
        OBJ_LABEL[obj].replace(" ", "\n", 1) for obj in objectives
    ]
    xs = np.arange(len(labels))
    for model in MODELS:
        ranks = [sorted(CFG_A, key=CFG_A.get, reverse=True).index(model)+1]
        for obj in objectives:
            cell = at60[at60.objective == obj].sort_values("MRR_mean", ascending=False)
            ranks.append(list(cell.model).index(model)+1)
        ax.plot(xs, ranks, "-o", color=fs.MODEL_COLORS[model], lw=1.6, ms=5)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([1,2,3,4]); ax.invert_yaxis(); ax.set_ylabel("rank by MRR")
    ax.set_title("(b) The ordering follows the objective", loc="left")
    ax.axvline(0.5, color=fs.MUTED, lw=0.8, ls=":")

    ax = axes[1, 0]
    bootstrap = bootstrap.sort_values("MRR_triple_level")
    y = np.arange(len(bootstrap))
    ax.errorbar(bootstrap.MRR_triple_level, y,
                xerr=[bootstrap.MRR_triple_level-bootstrap.ci_low_triple,
                      bootstrap.ci_high_triple-bootstrap.MRR_triple_level],
                fmt="none", lw=1.4, capsize=3, ecolor=fs.MUTED)
    for yi, row in enumerate(bootstrap.itertuples(index=False)):
        ax.plot(row.MRR_triple_level, yi, "o", ms=6, color=fs.MODEL_COLORS[row.model])
    ax.set_yticks(y); ax.set_yticklabels(bootstrap.model)
    ax.set_xlabel(f"MRR (triple-level unit, n = {int(bootstrap.n_triples.iloc[0])})")
    ax.set_title("(c) Cluster-bootstrap intervals overlap", loc="left")

    ax = axes[1, 1]
    objective_colours = {
        "margin": fs.TYPE_COLORS["CAU"],
        "logistic": fs.TYPE_COLORS["SYM"],
        "selfadv": fs.TYPE_COLORS["HER"],
    }
    effect_data = pairwise.copy(); effect_data["abs_d"] = effect_data.cohens_d.abs()
    effect_data = effect_data.sort_values(["objective", "abs_d"])
    y = np.arange(len(effect_data))[::-1]
    ax.barh(y, effect_data.abs_d,
            color=[objective_colours[obj] for obj in effect_data.objective], height=0.72)
    ax.axvline(0.2, color=fs.ACCENT, lw=1.1, ls="--")
    ax.text(0.21, y.max(), "small-effect\nthreshold 0.2",
            color=fs.ACCENT, fontsize=7.5, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels([name.replace(" - ", "−") for name in effect_data.comparison], fontsize=6.2)
    handles = [plt.Rectangle((0,0),1,1,color=objective_colours[obj]) for obj in objectives]
    ax.legend(handles, [OBJ_LABEL[obj] for obj in objectives], fontsize=7.5, loc="lower right")
    ax.set_xlabel("|Cohen's d| on triple-level differences")
    ax.set_xlim(0, max(0.62, effect_data.abs_d.max()*1.18))
    ax.set_title("(d) Pairwise effect sizes by objective", loc="left")
    fs.save_checked(fig, FIG / "fig07_objective_ablation")


def figure8_relation_lift():
    relation_path = RES / "statistics" / "relation_lift_exact.csv"
    require(relation_path)
    relation = pd.read_csv(relation_path)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.87, bottom=0.18, wspace=0.38)

    ax = axes[0]
    ax.scatter(relation.mean_filtered_candidates, relation.MRR, s=70,
               c=[fs.REL_COLORS[name] for name in relation.relation], zorder=3)
    for row in relation.itertuples(index=False):
        ax.annotate(row.relation, (row.mean_filtered_candidates, row.MRR),
                    textcoords="offset points", xytext=(7,4), fontsize=7.5)
    ax.set_xlabel("mean filtered candidate-set size"); ax.set_ylabel("raw MRR")
    ax.set_title("(a) Raw MRR tracks candidate-set size", loc="left")
    fs.hgrid(ax, "both")

    ax = axes[1]
    ordered = relation.sort_values("lift"); y = np.arange(len(ordered))
    ax.barh(y, ordered.lift,
            color=[fs.REL_COLORS[name] for name in ordered.relation], height=0.62)
    ax.errorbar(ordered.lift, y,
                xerr=[ordered.lift-ordered.lift_ci_low,
                      ordered.lift_ci_high-ordered.lift],
                fmt="none", ecolor=fs.INK, lw=1.0, capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(ordered.relation, fontsize=8)
    ax.set_xlabel("lift over an exact random-ranking baseline (×)")
    ax.axvline(1.0, color=fs.MUTED, lw=0.9, ls=":")
    ax.set_title("(b) Normalisation reverses the ordering", loc="left")
    if "TREATS" in set(ordered.relation):
        idx = list(ordered.relation).index("TREATS")
        row = ordered.iloc[idx]
        ax.text(row.lift_ci_high + 0.25, idx, "n=20 queries",
                fontsize=6.8, va="center", color=fs.MUTED)
    fs.save_checked(fig, FIG / "fig08_relation_lift_exact")


def main():
    try:
        figure6_annotation()
    except FileNotFoundError as exc:
        print("[figure6] skipped:", exc)
        print("[figure6] use the committed asset, or rebuild locally with an authorised data/train.txt")
    figure7_ablation()
    figure8_relation_lift()


if __name__ == "__main__":
    main()
