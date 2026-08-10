#!/usr/bin/env python3
"""Manuscript Figure 9 — largest connected component of the S0 core graph."""
from __future__ import annotations
__author__ = "LIJUNHUA"
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from _common import DATA, RES, FIG, INK, MUTED, TYPE_COLORS, TYPE_LABELS, REL_COLORS, save_checked
from labels_en import label as en_label
nodes = pd.read_csv(DATA / 'nodes.csv', encoding='utf-8-sig')
edges = pd.read_csv(DATA / 'edges.csv', encoding='utf-8-sig')
core = edges[edges.weight >= 2]
G = nx.Graph()
for r in core.itertuples(index=False):
    G.add_edge(r.source_id, r.target_id, relation=r.relation, weight=r.weight)
H = G.subgraph(max(nx.connected_components(G), key=len)).copy()
layout_file = RES / 'graph_layout_seed7.csv'
pos = {}
if layout_file.exists():
    layout = pd.read_csv(layout_file)
    pos = {row.node_id: np.array([row.x, row.y]) for row in layout.itertuples(index=False)}
    if set(pos) != set(H.nodes()):
        pos = {}
if not pos:
    pos = nx.spring_layout(H, k=0.42 / np.sqrt(H.number_of_nodes()), iterations=60, seed=7, weight=None)
    pd.DataFrame([(n, xy[0], xy[1]) for n, xy in pos.items()], columns=['node_id', 'x', 'y']).to_csv(layout_file, index=False)
tmap = nodes.set_index('id').type.to_dict()
nmap = nodes.set_index('id').name.to_dict()
deg = dict(H.degree())
fig, ax = plt.subplots(figsize=(7.4, 6.3))
fig.subplots_adjust(left=0.02, right=0.98, top=0.845, bottom=0.02)
ax.axis('off')
seg = []
cols = []
widths = []
for u, v, d in H.edges(data=True):
    seg.append([pos[u], pos[v]])
    cols.append(REL_COLORS[d['relation']])
    widths.append(0.12 + 0.028 * min(d['weight'], 12))
ax.add_collection(LineCollection(seg, colors=cols, linewidths=widths, alpha=0.15, zorder=1, rasterized=True))
for t in ['SYM', 'CAU', 'PRE', 'HER', 'EFF']:
    ids = [n for n in H.nodes if tmap[n] == t]
    xy = np.array([pos[n] for n in ids])
    sz = np.array([3.4 + 2.2 * np.sqrt(deg[n]) for n in ids])
    ax.scatter(xy[:, 0], xy[:, 1], s=sz, color=TYPE_COLORS[t], alpha=0.8, linewidths=0, zorder=2, label=TYPE_LABELS[t], rasterized=True)
top = sorted(H.nodes, key=lambda n: -deg[n])[:10]
for i, n in enumerate(top, 1):
    x, y = pos[n]
    ax.scatter([x], [y], s=88, facecolor='white', edgecolor=INK, linewidth=0.8, zorder=5)
    ax.text(x, y, str(i), fontsize=6.4, fontweight='bold', color=INK, ha='center', va='center', zorder=6)
key_left, key_top, dy = (0.6, 0.905, 0.0295)
fig.text(key_left, key_top + 0.03, 'Highest-degree entities', fontsize=8.4, fontweight='semibold', color=INK, va='top')
for i, n in enumerate(top):
    col, row = divmod(i, 5)
    x = key_left + col * 0.2
    y = key_top - row * dy
    fig.text(x + 0.02, y, f'{i + 1}.', fontsize=7.2, fontweight='bold', color=TYPE_COLORS[tmap[n]], va='top', ha='right')
    fig.text(x + 0.028, y, f'{en_label(nmap[n], tmap[n])} ({deg[n]})', fontsize=7, color=INK, va='top', ha='left')
leg = ax.legend(loc='upper left', fontsize=7.8, markerscale=2.4, title='Entity type', title_fontsize=8.4, frameon=False, labelspacing=0.28, borderpad=0.2)
leg._legend_box.align = 'left'
ax.add_artist(leg)
rel_h = [Line2D([], [], color=REL_COLORS[r], linewidth=2, label=r) for r in ['CAUSES', 'HAS_EFFECT', 'CONTAINS', 'RELIEVES', 'TREATS']]
lg = ax.legend(handles=rel_h, loc='lower left', fontsize=7.4, title='Relation', title_fontsize=8.4, frameon=False, labelspacing=0.28, borderpad=0.2)
lg._legend_box.align = 'left'
fig.text(0.02, 0.975, 'Largest connected component of the core graph (S0)', fontsize=10.4, fontweight='semibold', color=INK, va='top')
fig.text(0.02, 0.94, f'{H.number_of_nodes():,} entities, {H.number_of_edges():,} relations; node area scales with degree', fontsize=8, color=MUTED, va='top')
save_checked(fig, str(FIG / 'fig09_graph_map'))
