#!/usr/bin/env python3
"""Manuscript Figure 8 — relation MRR and lift over the exact random-ranking baseline."""
from __future__ import annotations
__author__ = "LIJUNHUA"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import RES, REF, FIG, INK, MUTED, REL_COLORS, choose, save_checked
R = pd.read_csv(choose(RES / 'statistics/relation_lift_exact.csv', REF / 'relation_lift_exact.csv'))
if 'mean_filtered_candidates' not in R.columns:
    R['mean_filtered_candidates'] = (R.head_side_candidates + R.tail_side_candidates) / 2.0
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.3))
a = ax[0]
a.scatter(R.mean_filtered_candidates, R.MRR, s=70, c=[REL_COLORS[r] for r in R.relation], zorder=3)
for r in R.itertuples(index=False):
    a.annotate(r.relation, (r.mean_filtered_candidates, r.MRR), textcoords='offset points', xytext=(6, 4), fontsize=7.5)
a.set_xlabel('mean filtered candidate-set size')
a.set_ylabel('raw MRR')
a.set_title('(a) Raw MRR against candidate-set size', loc='left')
a.grid(True, axis='both', lw=0.5, alpha=0.5)
a = ax[1]
R2 = R.sort_values('lift')
y = np.arange(len(R2))
a.barh(y, R2.lift, color=[REL_COLORS[r] for r in R2.relation], height=0.62)
a.errorbar(R2.lift, y, xerr=[R2.lift - R2.lift_ci_low, R2.lift_ci_high - R2.lift], fmt='none', ecolor=INK, lw=1, capsize=3)
a.set_yticks(y)
a.set_yticklabels(R2.relation, fontsize=8)
a.set_xlabel('lift over exact random-ranking baseline (×)')
a.axvline(1, color=MUTED, lw=0.9, ls=':')
a.set_title('(b) Exact-baseline lift with clustered CI', loc='left')
fig.tight_layout()
save_checked(fig, str(FIG / 'fig08_relation_lift_exact'))
