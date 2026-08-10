#!/usr/bin/env python3
"""Manuscript Figure 5 — core-graph degree structure."""
from __future__ import annotations
__author__ = "LIJUNHUA"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import RES, FIG, INK, MUTED, TYPE_COLORS, TYPE_LABELS, hgrid, tag, save_checked
from labels_en import label as en_label
degs = pd.read_csv(RES / 'core_node_degrees.csv')
degs['label'] = [en_label(n, t) for n, t in zip(degs.name, degs.type)]
fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.05), gridspec_kw={'width_ratios': [1, 1, 1.45]})
fig.subplots_adjust(left=0.078, right=0.988, top=0.795, bottom=0.225, wspace=0.5)
ax = axes[0]
d = np.sort(degs.degree.values)
ccdf = 1.0 - np.arange(len(d)) / len(d)
ax.loglog(d, ccdf, color='#1B6C7F', linewidth=1.9, zorder=3)
ax.set_xlabel('Degree $k$')
ax.set_ylabel('P(K ≥ k)')
ax.set_title('Degree distribution', loc='left', pad=7, fontsize=9.5)
ax.text(0.045, 0.09, f'skewness {degs.degree.skew():.2f}\nmax degree {d.max()}', transform=ax.transAxes, ha='left', va='bottom', fontsize=7.4, color=MUTED)
ax.grid(which='both', linewidth=0.5, color='#EDEDED')
ax.set_axisbelow(True)
ax = axes[1]
order = ['SYM', 'HER', 'CAU', 'PRE', 'EFF']
data = [degs[degs.type == t].degree.values for t in order]
bp = ax.boxplot(data, widths=0.56, patch_artist=True, showfliers=False, medianprops=dict(color='white', linewidth=1.3), whiskerprops=dict(color=MUTED, linewidth=0.9), capprops=dict(color=MUTED, linewidth=0.9))
for patch, t in zip(bp['boxes'], order):
    patch.set_facecolor(TYPE_COLORS[t])
    patch.set_edgecolor('none')
for i, t in enumerate(order, 1):
    v = degs[degs.type == t].degree
    ax.scatter(np.full(len(v), i), v, s=2.6, color=TYPE_COLORS[t], alpha=0.15, zorder=1, linewidths=0)
ax.set_yscale('log')
ax.set_ylim(0.8, 400)
ax.set_xticks(range(1, 6))
ax.set_xticklabels([f'{t}\n{len(degs[degs.type == t])}' for t in order], fontsize=7.4)
ax.set_ylabel('Core-graph degree (log)')
ax.set_title('Degree by entity type', loc='left', pad=7, fontsize=9.5)
hgrid(ax)
ax = axes[2]
hub = degs.nlargest(12, 'degree').sort_values('degree')
ax.barh(np.arange(len(hub)), hub.degree, height=0.66, color=[TYPE_COLORS[t] for t in hub.type], zorder=3)
ax.set_yticks(np.arange(len(hub)))
ax.set_yticklabels(hub.label.tolist(), fontsize=7.2)
for i, v in enumerate(hub.degree):
    ax.text(v + 4, i, str(v), va='center', fontsize=7, color=INK)
ax.set_xlim(0, hub.degree.max() * 1.17)
ax.set_xticks([0, 100, 200])
ax.set_xlabel('Core-graph degree')
ax.set_title('Highest-degree entities', loc='left', pad=7, fontsize=9.5)
hgrid(ax, 'x')
tag(fig, 'a', 0.004, 0.978)
tag(fig, 'b', 0.348, 0.978)
tag(fig, 'c', 0.652, 0.978)
save_checked(fig, str(FIG / 'fig05_degree_structure'))
