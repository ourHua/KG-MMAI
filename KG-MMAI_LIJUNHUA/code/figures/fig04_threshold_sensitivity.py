#!/usr/bin/env python3
"""Manuscript Figure 4 — graph structure across co-occurrence thresholds."""
from __future__ import annotations
__author__ = "LIJUNHUA"
import matplotlib.pyplot as plt
import pandas as pd
from _common import RES, FIG, MUTED, hgrid, tag, save_checked
prof = pd.read_csv(RES / 'structural_profile.csv')
fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.6))
fig.subplots_adjust(left=0.075, right=0.988, top=0.77, bottom=0.235, wspace=0.52)
ax = axes[0]
ax.plot(prof.threshold, prof.edges, 's-', color='#C4622D', linewidth=1.7, markersize=4.4, label='Triples', zorder=3)
ax.plot(prof.threshold, prof.nodes, 'o-', color='#1B6C7F', linewidth=1.7, markersize=4.4, label='Entities', zorder=3)
ax.scatter([2], [9544], s=120, facecolor='none', edgecolor='#C4622D', linewidth=1.3, zorder=4)
ax.set_yscale('log')
ax.set_ylim(90, 400000)
ax.set_yticks([100.0, 1000.0, 10000.0, 100000.0])
ax.set_xticks([1, 3, 5, 10])
ax.set_xlabel('Threshold τ')
ax.set_ylabel('Count (log)')
ax.set_title('Graph size', loc='left', pad=7, fontsize=9.5)
ax.legend(fontsize=7.2, loc='upper right', borderpad=0.2, handlelength=1.4)
hgrid(ax)
ax = axes[1]
ax.plot(prof.threshold, prof.largest_component_pct, 'o-', color='#4A7C59', linewidth=1.7, markersize=4.4, zorder=3)
for t, v, k in zip(prof.threshold, prof.largest_component_pct, prof.components):
    ax.annotate(str(int(k)), (t, v), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=6.8, color=MUTED)
ax.set_ylim(97.9, 100.9)
ax.set_xticks([1, 3, 5, 10])
ax.set_yticks([98, 99, 100])
ax.set_xlabel('Threshold τ')
ax.set_ylabel('Largest comp. (%)')
ax.set_title('Connectivity', loc='left', pad=7, fontsize=9.5)
hgrid(ax)
ax = axes[2]
ax.plot(prof.threshold, prof.mean_degree, 'o-', color='#6D4C8C', linewidth=1.7, markersize=4.4, zorder=3)
ax.set_xticks([1, 3, 5, 10])
ax.set_ylim(2, 17)
ax.set_xlabel('Threshold τ')
ax.set_ylabel('Mean degree')
ax.set_title('Mean degree', loc='left', pad=7, fontsize=9.5)
hgrid(ax)
ax = axes[3]
ax.plot(prof.threshold, prof.density * 1000, '^-', color='#C9A227', linewidth=1.7, markersize=4.6, zorder=3)
ax.set_xticks([1, 3, 5, 10])
ax.set_ylim(0, 12.5)
ax.set_xlabel('Threshold τ')
ax.set_ylabel('Density × 10³')
ax.set_title('Density', loc='left', pad=7, fontsize=9.5)
hgrid(ax)
tag(fig, 'a', 0.004, 0.975)
tag(fig, 'b', 0.256, 0.975)
tag(fig, 'c', 0.506, 0.975)
tag(fig, 'd', 0.756, 0.975)
save_checked(fig, str(FIG / 'fig04_threshold_sensitivity'))
