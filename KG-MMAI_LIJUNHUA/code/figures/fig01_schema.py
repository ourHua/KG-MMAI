#!/usr/bin/env python3
"""Manuscript Figure 1 — schema of the induced knowledge graph."""
from __future__ import annotations
__author__ = "LIJUNHUA"
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from _common import DATA, RES, FIG, INK, MUTED, TYPE_COLORS, TYPE_LABELS, REL_COLORS, save_checked
nodes = pd.read_csv(DATA / 'nodes.csv', encoding='utf-8-sig')
core_nodes = pd.read_csv(DATA / 'nodes_core.csv', encoding='utf-8-sig')
reld = pd.read_csv(RES / 'relation_distribution.csv')
fig, ax = plt.subplots(figsize=(7.2, 4.1))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.2)
ax.axis('off')
pos = {'CAU': (1.5, 4.5), 'SYM': (5.0, 4.5), 'PRE': (1.5, 1.3), 'HER': (5.0, 1.3), 'EFF': (8.5, 1.3)}
c_all = nodes.type.value_counts()
c_core = core_nodes.type.value_counts()
for t, (x, y) in pos.items():
    ax.add_patch(FancyBboxPatch((x - 1.05, y - 0.52), 2.1, 1.04, boxstyle='round,pad=0.02,rounding_size=0.14', linewidth=1.6, edgecolor=TYPE_COLORS[t], facecolor=TYPE_COLORS[t] + '1A', zorder=3))
    ax.text(x, y + 0.21, TYPE_LABELS[t], ha='center', va='center', fontsize=10, fontweight='bold', color=TYPE_COLORS[t])
    ax.text(x, y - 0.05, t, ha='center', va='center', fontsize=7.5, color=MUTED)
    ax.text(x, y - 0.3, f'{c_all[t]:,}  →  {c_core[t]:,}', ha='center', va='center', fontsize=8, color=INK)
core_n = reld.set_index('relation').triples
for a, b, rel, frac, ox, oy in [('CAU', 'SYM', 'CAUSES', 0.5, 0, 0), ('PRE', 'HER', 'CONTAINS', 0.5, 0, 0), ('HER', 'EFF', 'HAS_EFFECT', 0.5, 0, 0), ('HER', 'SYM', 'RELIEVES', 0.5, 0, 0), ('PRE', 'SYM', 'TREATS', 0.44, -0.66, 0.34)]:
    x1, y1 = pos[a]
    x2, y2 = pos[b]
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13, linewidth=1.5, color=REL_COLORS[rel], shrinkA=58, shrinkB=58, zorder=2, alpha=0.9))
    mx = x1 + (x2 - x1) * frac + ox
    my = y1 + (y2 - y1) * frac + oy
    ax.text(mx, my + 0.17, rel, ha='center', va='center', fontsize=8, fontweight='bold', color=REL_COLORS[rel], bbox=dict(facecolor='white', edgecolor='none', pad=1.4))
    ax.text(mx, my - 0.1, f'{core_n[rel]:,} triples', ha='center', va='center', fontsize=7.2, color=MUTED, bbox=dict(facecolor='white', edgecolor='none', pad=1.1))
ax.text(0.1, 5.95, 'Five entity types, five directional relation types', fontsize=10.5, fontweight='semibold', color=INK)
ax.text(0.1, 5.62, 'Entity boxes: all extracted  →  retained in core graph (weight ≥ 2)', fontsize=8.2, color=MUTED)
save_checked(fig, str(FIG / 'fig01_schema'))
