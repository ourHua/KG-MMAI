#!/usr/bin/env python3
"""Manuscript Figure 3 — relation composition and survival after thresholding."""
from __future__ import annotations
__author__ = "LIJUNHUA"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import DATA, RES, FIG, FAINT, INK, MUTED, REL_COLORS, hgrid, tag, save_checked
edges = pd.read_csv(DATA / 'edges.csv', encoding='utf-8-sig')
reld = pd.read_csv(RES / 'relation_distribution.csv')
core = edges[edges.weight >= 2]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
fig.subplots_adjust(left=0.19, right=0.975, top=0.79, bottom=0.165, wspace=0.44)
ax = axes[0]
r = reld.sort_values('triples')
y = np.arange(len(r))
ax.barh(y, r.all_triples, height=0.6, color=FAINT, zorder=2, label='All candidates (w ≥ 1)')
ax.barh(y, r.triples, height=0.6, color=[REL_COLORS[x] for x in r.relation], zorder=3, label='Core graph (w ≥ 2)')
for i, row in enumerate(r.itertuples(index=False)):
    ax.text(row.all_triples + 400, i, f'{row.triples:,} / {row.all_triples:,}', va='center', fontsize=7.2, color=INK)
ax.set_yticks(y)
ax.set_yticklabels([f'{x}\n{s}' for x, s in zip(r.relation, r.schema)], fontsize=7.7)
ax.set_xlim(0, r.all_triples.max() * 1.44)
ax.set_xlabel('Triples')
ax.set_title('Composition', loc='left', pad=8)
ax.legend(loc='lower right', fontsize=7.2, borderpad=0.2)
hgrid(ax, 'x')
ax = axes[1]
r2 = reld.sort_values('survival_pct')
ax.barh(np.arange(len(r2)), r2.survival_pct, height=0.56, color=[REL_COLORS[x] for x in r2.relation], zorder=3)
for i, row in enumerate(r2.itertuples(index=False)):
    ax.text(row.survival_pct + 0.7, i + 0.11, f'{row.survival_pct:.1f}%', va='center', fontsize=8, fontweight='bold', color=INK)
    ax.text(row.survival_pct + 0.7, i - 0.21, f'mean w {row.mean_weight:.2f}', va='center', fontsize=6.8, color=MUTED)
overall = 100 * len(core) / len(edges)
ax.axvline(overall, color=INK, linewidth=1, linestyle=':', zorder=4)
ax.text(overall - 0.8, 4.3, f'overall {overall:.1f}%', fontsize=7.1, color=INK, ha='right', va='center')
ax.set_yticks(np.arange(len(r2)))
ax.set_yticklabels(r2.relation, fontsize=8)
ax.set_xlim(0, 35)
ax.set_xlabel('Surviving w ≥ 2 (%)')
ax.set_title('Survival rate', loc='left', pad=8)
hgrid(ax, 'x')
tag(fig, 'a', 0.01, 0.965)
tag(fig, 'b', 0.555, 0.965)
save_checked(fig, str(FIG / 'fig03_relation_composition'))
