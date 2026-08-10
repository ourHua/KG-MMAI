#!/usr/bin/env python3
"""Manuscript Figure 2 — entity attrition and candidate-edge weight distribution."""
from __future__ import annotations
__author__ = "LIJUNHUA"
import matplotlib.pyplot as plt
import pandas as pd
from _common import DATA, RES, FIG, INK, MUTED, hgrid, tag, save_checked
nodes = pd.read_csv(DATA / 'nodes.csv', encoding='utf-8-sig')
edges = pd.read_csv(DATA / 'edges.csv', encoding='utf-8-sig')
prof = pd.read_csv(RES / 'structural_profile.csv')
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.15))
fig.subplots_adjust(left=0.135, right=0.985, top=0.79, bottom=0.165, wspace=0.42)
ax = axes[0]
stages = ['Extracted\nentities', 'Entities in\n≥ 1 relation', 'Core-graph\nentities']
vals = [len(nodes), len(set(edges.source_id) | set(edges.target_id)), int(prof.loc[prof.threshold == 2, 'nodes'].iloc[0])]
ax.barh([2, 1, 0], vals, height=0.52, color=['#C9D6DA', '#6E9FAC', '#1B6C7F'], zorder=3)
for i, v in enumerate(vals):
    y = 2 - i
    ax.text(v + max(vals) * 0.025, y + 0.1, f'{v:,}', va='center', fontsize=8.8, fontweight='bold', color=INK)
    if i:
        ax.text(v + max(vals) * 0.025, y - 0.21, f'{100 * v / vals[0]:.1f}% of extracted', va='center', fontsize=7.1, color=MUTED)
ax.set_yticks([2, 1, 0])
ax.set_yticklabels(stages, fontsize=8.4)
ax.set_xlim(0, max(vals) * 1.38)
ax.set_xlabel('Entities')
ax.set_title('Entity attrition', loc='left', pad=8)
hgrid(ax, 'x')
ax = axes[1]
wd = pd.read_csv(RES / 'weight_distribution.csv')
wd.columns = ['weight', 'n', 'cum']
top = wd[wd.weight <= 10]
ax.bar(top.weight, top.n, width=0.7, color='#1B6C7F', zorder=3)
ax.bar([11], [wd[wd.weight > 10].n.sum()], width=0.7, color='#9EBCC4', zorder=3)
ax.axvline(1.5, color='#C4622D', linewidth=1.4, linestyle='--', zorder=4)
ax.set_yscale('log')
ax.set_ylim(30, wd.n.max() * 16)
ax.text(1.95, wd.n.max() * 5.5, 'core threshold  w ≥ 2', fontsize=7.6, color='#C4622D', fontweight='semibold', va='center')
ax.annotate(f'{wd.iloc[0].n:,.0f} single-occurrence triples\n({100 * wd.iloc[0].n / wd.n.sum():.1f}% of all candidates)', xy=(1, wd.iloc[0].n), xytext=(4.3, wd.n.max() * 1.5), fontsize=7.4, color=INK, va='center', arrowprops=dict(arrowstyle='->', color=MUTED, linewidth=0.9, shrinkB=3))
ax.set_xticks(range(1, 12))
ax.set_xticklabels([str(i) for i in range(1, 11)] + ['>10'], fontsize=8)
ax.set_xlabel('Co-occurrence weight')
ax.set_ylabel('Candidate triples (log)')
ax.set_title('Weight distribution', loc='left', pad=8)
hgrid(ax)
tag(fig, 'a', 0.01, 0.965)
tag(fig, 'b', 0.545, 0.965)
save_checked(fig, str(FIG / 'fig02_extraction_funnel'))
