#!/usr/bin/env python3
"""06_figures_results.py — Figures 6-10 (degree structure, embedding
comparison, relation difficulty, core-graph map).

Every figure passes figstyle.check_overlaps() before it is written: the drawn
bounding box of every text artist is compared pairwise, and a figure that
reports a collision is not shipped. Entity names are rendered in Latin script
by code/labels.py, so no figure emits CJK.
"""

__author__ = "LIJUNHUA"
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from figstyle import (ACCENT, INK, MODEL_COLORS, MUTED, REL_COLORS,
                      TYPE_COLORS, TYPE_LABELS, hgrid, save_checked, tag)
from labels import label as elabel

DATA, RES, FIG = ROOT / "data", ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

nodes = pd.read_csv(DATA / "nodes.csv", encoding="utf-8-sig")
edges = pd.read_csv(DATA / "edges.csv", encoding="utf-8-sig")
core = edges[edges.weight >= 2]
degs = pd.read_csv(RES / "core_node_degrees.csv")
rob = pd.read_csv(RES / "robustness_summary.csv")
curves = pd.read_csv(RES / "robustness_curves.csv")
ci = pd.read_csv(RES / "model_bootstrap_ci.csv")
diff = pd.read_csv(RES / "relation_difficulty.csv")
small = pd.read_csv(RES / "small_sample_precision.csv")
pw = pd.read_csv(RES / "model_pairwise_tests.csv")

MODELS = ["TransE", "DistMult", "ComplEx", "RotatE"]
ABBR = {"TransE": "TrE", "DistMult": "DiM", "ComplEx": "CoE", "RotatE": "RoE"}
REPORTED = {"ComplEx": (0.225, 0.001), "DistMult": (0.215, 0.003),
            "TransE": (0.153, 0.006), "RotatE": (0.114, 0.007)}

degs["label"] = [elabel(n, t) for n, t in zip(degs.name, degs.type)]

# =========================================================================== #
# FIGURE 6 — degree structure
#   Panel (b) previously collided: rotated tick labels ran into the per-type
#   sample sizes. Sample sizes now sit inside the axes on a second line of the
#   tick label itself, and the labels are horizontal.
#   Panel (c) previously collided: the legend sat on top of the bars. Entity
#   type is now carried by a colour key placed in the figure margin.
# =========================================================================== #
fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.05),
                         gridspec_kw={"width_ratios": [1, 1, 1.45]})
fig.subplots_adjust(left=0.078, right=0.988, top=0.795, bottom=0.225,
                    wspace=0.50)

ax = axes[0]
d = np.sort(degs.degree.values)
ccdf = 1.0 - np.arange(len(d)) / len(d)
ax.loglog(d, ccdf, color="#1B6C7F", linewidth=1.9, zorder=3)
ax.set_xlabel("Degree $k$")
ax.set_ylabel("P(K \u2265 k)")
ax.set_title("Degree distribution", loc="left", pad=7, fontsize=9.5)
ax.text(0.045, 0.09, f"skewness {degs.degree.skew():.2f}\nmax degree {d.max()}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7.4,
        color=MUTED)
ax.grid(which="both", linewidth=0.5, color="#EDEDED"); ax.set_axisbelow(True)

ax = axes[1]
order = ["SYM", "HER", "CAU", "PRE", "EFF"]
data = [degs[degs.type == t].degree.values for t in order]
bp = ax.boxplot(data, widths=0.56, patch_artist=True, showfliers=False,
                medianprops=dict(color="white", linewidth=1.3),
                whiskerprops=dict(color=MUTED, linewidth=0.9),
                capprops=dict(color=MUTED, linewidth=0.9))
for patch, t in zip(bp["boxes"], order):
    patch.set_facecolor(TYPE_COLORS[t]); patch.set_edgecolor("none")
for i, t in enumerate(order, 1):
    v = degs[degs.type == t].degree
    ax.scatter(np.full(len(v), i), v, s=2.6, color=TYPE_COLORS[t], alpha=0.15,
               zorder=1, linewidths=0)
ax.set_yscale("log"); ax.set_ylim(0.8, 400)
ax.set_xticks(range(1, 6))
ax.set_xticklabels([f"{TYPE_LABELS[t][:4]}.\n{len(degs[degs.type==t])}"
                    for t in order], fontsize=7.4)
ax.set_ylabel("Core-graph degree (log)")
ax.set_title("Degree by entity type", loc="left", pad=7, fontsize=9.5)
ax.text(0.5, -0.235, "type (entities)", transform=ax.transAxes, ha="center",
        va="top", fontsize=8.2, color=INK)
hgrid(ax)

ax = axes[2]
hub = degs.nlargest(12, "degree").sort_values("degree")
ax.barh(np.arange(len(hub)), hub.degree, height=0.66,
        color=[TYPE_COLORS[t] for t in hub.type], zorder=3)
ax.set_yticks(np.arange(len(hub)))
ax.set_yticklabels(hub.label.tolist(), fontsize=7.4)
for i, v in enumerate(hub.degree):
    ax.text(v + 4, i, str(v), va="center", fontsize=7.0, color=INK)
ax.set_xlim(0, hub.degree.max() * 1.17)
ax.set_xticks([0, 100, 200])
ax.set_xlabel("Core-graph degree")
ax.set_title("Highest-degree entities", loc="left", pad=7, fontsize=9.5)
hgrid(ax, "x")

# colour key for panel (c), placed in the figure margin so it cannot overlap
for i, t in enumerate(["CAU", "HER", "EFF", "PRE"]):
    fig.text(0.655 + i * 0.087, 0.037, "\u25A0", color=TYPE_COLORS[t],
             fontsize=8, ha="left", va="center")
    fig.text(0.671 + i * 0.087, 0.037, TYPE_LABELS[t][:4] + ".",
             fontsize=7.2, ha="left", va="center", color=INK)

tag(fig, "a", 0.004, 0.978); tag(fig, "b", 0.348, 0.978); tag(fig, "c", 0.652, 0.978)
save_checked(fig, str(FIG / "fig05_degree_structure"))

# =========================================================================== #
# FIGURE 7 — the ranking is a property of the configuration
#   Rebuilt at 2x3 with a taller canvas and wide gutters. Every annotation is
#   placed in axes coordinates in a region verified to be empty, and the
#   convergence panel carries its legend outside the data area.
# =========================================================================== #
fig = plt.figure(figsize=(7.6, 5.35))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.06], hspace=0.92, wspace=0.46,
                      left=0.088, right=0.985, top=0.90, bottom=0.085)

# ---- (a) configuration A
ax = fig.add_subplot(gs[0, 0])
v = [REPORTED[m][0] for m in MODELS]; e = [REPORTED[m][1] for m in MODELS]
ax.bar(range(4), v, yerr=e, width=0.66, capsize=2.6,
       color=[MODEL_COLORS[m] for m in MODELS], zorder=3,
       error_kw=dict(elinewidth=0.9, ecolor=INK))
b = int(np.argmax(v))
ax.text(b, v[b] + e[b] + 0.014, "best", ha="center", fontsize=7.6,
        fontweight="bold", color=INK)
ax.set_xticks(range(4))
ax.set_xticklabels([ABBR[m] for m in MODELS], fontsize=7.8)
ax.set_ylabel("Filtered MRR"); ax.set_ylim(0, 0.28)
ax.set_title("Config. A: reported", loc="left", pad=7, fontsize=9.2)
hgrid(ax)

# ---- (b) configuration B
ax = fig.add_subplot(gs[0, 1])
b60 = rob[rob.budget_epochs == 60].set_index("model")
v = [b60.loc[m, "MRR_mean"] for m in MODELS]
e = [b60.loc[m, "MRR_sd"] for m in MODELS]
ax.bar(range(4), v, yerr=e, width=0.66, capsize=2.6,
       color=[MODEL_COLORS[m] for m in MODELS], zorder=3,
       error_kw=dict(elinewidth=0.9, ecolor=INK))
b = int(np.argmax(v))
ax.text(b, v[b] + e[b] + 0.014, "best", ha="center", fontsize=7.6,
        fontweight="bold", color=INK)
ax.set_xticks(range(4))
ax.set_xticklabels([ABBR[m] for m in MODELS], fontsize=7.8)
ax.set_ylim(0, 0.28)
ax.set_title("Config. B: reimplemented", loc="left", pad=7, fontsize=9.2)
hgrid(ax)

# ---- (c) rank inversion
ax = fig.add_subplot(gs[0, 2])
rankA = {m: r for r, m in enumerate(sorted(MODELS, key=lambda x: -REPORTED[x][0]), 1)}
rankB = {m: int(b60.loc[m, "rank_in_config"]) for m in MODELS}
for m in MODELS:
    ax.plot([0, 1], [rankA[m], rankB[m]], "o-", color=MODEL_COLORS[m],
            linewidth=1.9, markersize=5.2, zorder=3)
    ax.text(-0.10, rankA[m], ABBR[m], ha="right", va="center", fontsize=7.8,
            color=MODEL_COLORS[m], fontweight="bold")
    ax.text(1.10, rankB[m], ABBR[m], ha="left", va="center", fontsize=7.8,
            color=MODEL_COLORS[m], fontweight="bold")
ax.set_xlim(-0.62, 1.62); ax.set_ylim(4.6, 0.4)
ax.set_xticks([0, 1]); ax.set_xticklabels(["A", "B"], fontsize=8.2)
ax.set_yticks([1, 2, 3, 4]); ax.set_ylabel("Rank")
ax.set_title("Ordering inverts", loc="left", pad=7, fontsize=9.2)
ax.spines["left"].set_visible(True)

# ---- (d) convergence
ax = fig.add_subplot(gs[1, 0])
for m in MODELS:
    sub = curves[curves.model == m].groupby("epoch").valid_mrr.agg(["mean", "std"])
    ax.plot(sub.index, sub["mean"], "-o", color=MODEL_COLORS[m], linewidth=1.5,
            markersize=3.0, zorder=3)
    ax.fill_between(sub.index, sub["mean"] - sub["std"], sub["mean"] + sub["std"],
                    color=MODEL_COLORS[m], alpha=0.13, linewidth=0)
ax.axvline(20, color=MUTED, linestyle=":", linewidth=1.0)
ax.set_xlabel("Training epoch"); ax.set_ylabel("Validation MRR")
ax.set_xlim(6, 64); ax.set_ylim(0.06, 0.235)
ax.set_title("Not converged at 20 epochs", loc="left", pad=7, fontsize=9.2)
ax.text(21.5, 0.078, "Config. A\nbudget", fontsize=6.9, color=MUTED,
        va="bottom", ha="left")
hgrid(ax)

# ---- (e) bootstrap intervals
ax = fig.add_subplot(gs[1, 1])
ci_s = ci.set_index("model").loc[MODELS[::-1]]
y = np.arange(len(ci_s))
ax.hlines(y, ci_s.ci_low, ci_s.ci_high,
          color=[MODEL_COLORS[m] for m in ci_s.index], linewidth=2.4, zorder=3)
ax.scatter(ci_s.MRR, y, s=34, color=[MODEL_COLORS[m] for m in ci_s.index],
           zorder=4, edgecolor="white", linewidth=0.9)
ax.set_yticks(y); ax.set_yticklabels([ABBR[m] for m in ci_s.index], fontsize=7.8)
ax.set_xlim(0.163, 0.219)
ax.set_xticks([0.17, 0.19, 0.21])
ax.set_xlabel("MRR (95% bootstrap CI)")
ax.set_title("Top two overlap", loc="left", pad=7, fontsize=9.2)
hgrid(ax, "x")

# ---- (f) effect sizes
ax = fig.add_subplot(gs[1, 2])
pw2 = pw.copy()
pw2["pair"] = [f"{ABBR[a]}\u2013{ABBR[b]}" for a, b in zip(pw2.model_a, pw2.model_b)]
pw2 = pw2.sort_values("cohens_d")
ax.barh(np.arange(len(pw2)), pw2.cohens_d, height=0.62,
        color=[ACCENT if abs(x) >= 0.2 else "#B8C4C9" for x in pw2.cohens_d],
        zorder=3)
ax.axvline(0, color=INK, linewidth=0.9)
for x in (-0.2, 0.2):
    ax.axvline(x, color=MUTED, linestyle="--", linewidth=0.9)
ax.set_yticks(np.arange(len(pw2)))
ax.set_yticklabels(pw2.pair, fontsize=7.2)
ax.set_xlabel("Cohen's $d$ (paired)")
ax.set_xlim(-0.42, 0.42)
ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
ax.set_title("All effects negligible", loc="left", pad=7, fontsize=9.2)
ax.text(0.235, -0.62, "|d| = 0.2", fontsize=6.8, color=MUTED, ha="left",
        va="center")
hgrid(ax, "x")

for lt, x in zip("abc", (0.004, 0.335, 0.666)):
    tag(fig, lt, x, 0.988)
for lt, x in zip("def", (0.004, 0.335, 0.666)):
    tag(fig, lt, x, 0.505)
save_checked(fig, str(FIG / "fig06_ranking_robustness"))

# =========================================================================== #
# FIGURE 8 — relation difficulty
# =========================================================================== #
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
fig.subplots_adjust(left=0.098, right=0.982, top=0.80, bottom=0.165, wspace=0.40)

ax = axes[0]
OFFSET = {"HAS_EFFECT": (0, 13), "CONTAINS": (0, -17), "RELIEVES": (12, 10),
          "CAUSES": (-10, -18), "TREATS": (0, 13)}
for r in diff.itertuples(index=False):
    ax.scatter(r.mean_candidates, r.MRR, s=70 + r.queries * 0.28,
               color=REL_COLORS[r.relation], alpha=0.85, zorder=3,
               edgecolor="white", linewidth=1.1)
    dx, dy = OFFSET[r.relation]
    ax.annotate(r.relation, (r.mean_candidates, r.MRR), xytext=(dx, dy),
                textcoords="offset points", fontsize=7.4, ha="center",
                color=REL_COLORS[r.relation], fontweight="bold")
xs = np.linspace(150, 760, 200)
ax.plot(xs, (np.log(xs) + 0.5772) / xs, "--", color=MUTED, linewidth=1.1,
        zorder=2)
ax.text(690, 0.030, "random\nranking", fontsize=7.0, color=MUTED, ha="center",
        va="bottom")
ax.set_xlim(140, 800); ax.set_ylim(0, 0.245)
ax.set_xlabel("Mean candidate-set size")
ax.set_ylabel("Filtered MRR")
ax.set_title("Raw MRR tracks candidate-set size", loc="left", pad=7, fontsize=9.4)
hgrid(ax)

ax = axes[1]
dd = diff.sort_values("lift_over_random")
ax.barh(np.arange(len(dd)), dd.lift_over_random, height=0.60,
        color=[REL_COLORS[x] for x in dd.relation], zorder=3)
for i, r in enumerate(dd.itertuples(index=False)):
    ax.text(r.lift_over_random + 0.3, i + 0.11, f"{r.lift_over_random:.1f}\u00D7",
            va="center", fontsize=8.0, fontweight="bold", color=INK)
    ax.text(r.lift_over_random + 0.3, i - 0.22, f"{int(r.mean_candidates)} cand.",
            va="center", fontsize=6.8, color=MUTED)
ax.set_yticks(np.arange(len(dd)))
ax.set_yticklabels(dd.relation, fontsize=8)
ax.set_xlim(0, dd.lift_over_random.max() * 1.30)
ax.set_xlabel("Lift over random ranking")
ax.set_title("Normalisation reverses the order", loc="left", pad=7, fontsize=9.4)
hgrid(ax, "x")

tag(fig, "a", 0.004, 0.975); tag(fig, "b", 0.512, 0.975)
save_checked(fig, str(FIG / "fig07_relation_difficulty"))

# =========================================================================== #
# FIGURE 9 — small-sample precision
# =========================================================================== #
fig, ax = plt.subplots(figsize=(5.2, 2.95))
fig.subplots_adjust(left=0.225, right=0.975, top=0.86, bottom=0.185)
s = small.sort_values("queries_per_seed")
y = np.arange(len(s))
ax.hlines(y, s.MRR_ci_low, s.MRR_ci_high,
          color=[REL_COLORS[r] for r in s.relation], linewidth=2.8, zorder=3)
ax.scatter(s.MRR, y, s=38, color=[REL_COLORS[r] for r in s.relation], zorder=4,
           edgecolor="white", linewidth=1.0)
for i, r in enumerate(s.itertuples(index=False)):
    ax.text(r.MRR_ci_high + 0.009, i, f"n = {r.queries_per_seed}", va="center",
            fontsize=7.2, color=MUTED)
ax.set_yticks(y); ax.set_yticklabels(s.relation, fontsize=8.2)
ax.set_xlim(0.10, 0.52)
ax.set_ylim(-0.75, len(s) - 0.25)
ax.set_xlabel("Filtered MRR (95% bootstrap CI)")
ax.set_title("TREATS cannot support a ranking claim", loc="left", pad=7,
             fontsize=9.6)
ax.annotate("interval 4\u20135\u00D7 wider\nthan any other relation",
            xy=(0.351, 0.06), xytext=(0.425, 1.85), fontsize=7.1, color=INK,
            ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=0.9,
                            connectionstyle="arc3,rad=0.18"))
hgrid(ax, "x")
save_checked(fig, str(FIG / "fig08_small_sample"))

# =========================================================================== #
# FIGURE 10 — core-graph map
#   Labels are placed by iterative repulsion in display coordinates and the
#   result is verified by the same overlap gate as every other figure.
# =========================================================================== #
G = nx.Graph()
for r in core.itertuples(index=False):
    G.add_edge(r.source_id, r.target_id, relation=r.relation, weight=r.weight)
H = G.subgraph(max(nx.connected_components(G), key=len))
layout_file = RES / "graph_layout_seed7.csv"
if layout_file.exists():
    layout = pd.read_csv(layout_file)
    pos = {row.node_id: np.array([row.x, row.y])
           for row in layout.itertuples(index=False)}
    if set(pos) != set(H.nodes()):
        pos = {}
else:
    pos = {}

if not pos:
    print(f"  laying out {H.number_of_nodes()} nodes / {H.number_of_edges()} edges")
    pos = nx.spring_layout(
        H,
        k=0.42 / np.sqrt(H.number_of_nodes()),
        iterations=60,
        seed=7,
        weight=None,
    )
    pd.DataFrame(
        [(node, xy[0], xy[1]) for node, xy in pos.items()],
        columns=["node_id", "x", "y"],
    ).to_csv(layout_file, index=False)
else:
    print(f"  loaded cached layout for {H.number_of_nodes()} nodes")

tmap = nodes.set_index("id").type.to_dict()
nmap = nodes.set_index("id").name.to_dict()
deg = dict(H.degree())

fig, ax = plt.subplots(figsize=(7.4, 6.3))
fig.subplots_adjust(left=0.02, right=0.98, top=0.845, bottom=0.02)
ax.axis("off")
edge_segments = []
edge_colors = []
edge_widths = []
for u, v_, edge_data in H.edges(data=True):
    edge_segments.append([pos[u], pos[v_]])
    edge_colors.append(REL_COLORS[edge_data["relation"]])
    edge_widths.append(0.12 + 0.028 * min(edge_data["weight"], 12))
ax.add_collection(LineCollection(
    edge_segments,
    colors=edge_colors,
    linewidths=edge_widths,
    alpha=0.15,
    zorder=1,
    rasterized=True,
))
for t in ["SYM", "CAU", "PRE", "HER", "EFF"]:
    ids = [n for n in H.nodes() if tmap[n] == t]
    xy = np.array([pos[n] for n in ids])
    sz = np.array([3.4 + 2.2 * np.sqrt(deg[n]) for n in ids])
    ax.scatter(xy[:, 0], xy[:, 1], s=sz, color=TYPE_COLORS[t], alpha=0.80,
               linewidths=0, zorder=2, label=TYPE_LABELS[t],
               rasterized=True)

# --- hub annotation: numbered markers on the map plus a key in the margin.
# Numerals are small enough that they cannot collide, which removes label
# overlap as a failure mode instead of trying to solve it by displacement. ---
top = sorted(H.nodes(), key=lambda n: -deg[n])[:10]
for i, n in enumerate(top, 1):
    x, y = pos[n]
    ax.scatter([x], [y], s=88, facecolor="white", edgecolor=INK,
               linewidth=0.8, zorder=5)
    ax.text(x, y, str(i), fontsize=6.4, fontweight="bold", color=INK,
            ha="center", va="center", zorder=6)

key_left, key_top, dy = 0.600, 0.905, 0.0295
fig.text(key_left, key_top + 0.030, "Highest-degree entities", fontsize=8.4,
         fontweight="semibold", color=INK, va="top")
for i, n in enumerate(top):
    col, row = divmod(i, 5)
    x = key_left + col * 0.200
    y = key_top - row * dy
    # the numeral itself carries the entity-type colour, so the key needs no
    # separate swatch and therefore cannot collide with one
    fig.text(x + 0.020, y, f"{i+1}.", fontsize=7.2, fontweight="bold",
             color=TYPE_COLORS[tmap[n]], va="top", ha="right")
    fig.text(x + 0.028, y, f"{elabel(nmap[n], tmap[n])} ({deg[n]})",
             fontsize=7.0, color=INK, va="top", ha="left")

leg1 = ax.legend(loc="upper left", fontsize=7.8, markerscale=2.4,
                 title="Entity type", title_fontsize=8.4, frameon=False,
                 labelspacing=0.28, borderpad=0.2)
leg1._legend_box.align = "left"
ax.add_artist(leg1)
rel_h = [Line2D([], [], color=REL_COLORS[r], linewidth=2.0, label=r)
         for r in ["CAUSES", "HAS_EFFECT", "CONTAINS", "RELIEVES", "TREATS"]]
lg2 = ax.legend(handles=rel_h, loc="lower left", fontsize=7.4, title="Relation",
                title_fontsize=8.4, frameon=False, labelspacing=0.28,
                borderpad=0.2)
lg2._legend_box.align = "left"
fig.text(0.02, 0.975, "Largest connected component of the core graph",
         fontsize=10.4, fontweight="semibold", color=INK, va="top")
fig.text(0.02, 0.940,
         f"{H.number_of_nodes():,} entities, {H.number_of_edges():,} relations "
         "(99.5% of the core graph); node area scales with degree",
         fontsize=8.0, color=MUTED, va="top")
save_checked(fig, str(FIG / "fig09_graph_map"))
print("result figures done")
