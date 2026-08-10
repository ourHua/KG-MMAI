#!/usr/bin/env python3
"""Manuscript Figure 7 — controlled objective ablation and triple-level inference."""
from __future__ import annotations
__author__ = "LIJUNHUA"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import RES, REF, FIG, ACCENT, MUTED, TYPE_COLORS, MODEL_COLORS, choose, save_checked
MODELS = ('TransE', 'DistMult', 'ComplEx', 'RotatE')
OBJ_LABEL = {'margin': 'O1 margin', 'logistic': 'O2 logistic', 'selfadv': 'O3 self-adversarial'}
CFG_A = {'ComplEx': 0.225, 'DistMult': 0.215, 'TransE': 0.153, 'RotatE': 0.114}
A = pd.read_csv(choose(RES / 'ablation/objective_ablation_summary.csv', REF / 'objective_ablation_60ep.csv'))
a60 = A[A.budget_epochs == 60] if 'budget_epochs' in A else A
P = pd.read_csv(choose(RES / 'statistics/model_pairwise_triplelevel.csv', REF / 'model_pairwise_triplelevel.csv'))
B = pd.read_csv(choose(RES / 'statistics/model_bootstrap_clustered.csv', REF / 'model_bootstrap_clustered.csv'))
fig, ax = plt.subplots(2, 2, figsize=(11, 6.2))
objs = ['margin', 'logistic', 'selfadv']
x = np.arange(3)
w = 0.2
a = ax[0, 0]
for k, m in enumerate(MODELS):
    vals = [a60[(a60.objective == o) & (a60.model == m)].MRR_mean.iloc[0] for o in objs]
    sds = [a60[(a60.objective == o) & (a60.model == m)].MRR_sd.iloc[0] for o in objs]
    a.bar(x + (k - 1.5) * w, vals, w, yerr=sds, capsize=2, color=MODEL_COLORS[m], label=m, error_kw=dict(lw=0.8, ecolor=MUTED))
a.set_xticks(x)
a.set_xticklabels([OBJ_LABEL[o] for o in objs])
a.set_ylabel('test MRR')
a.set_title('(a) One code base, three objectives (60 epochs)', loc='left')
a.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.02), fontsize=8)
a.set_ylim(0, max(a60.MRR_mean) * 1.35)
a = ax[0, 1]
cols = ['Config. A\n(recorded)'] + [OBJ_LABEL[o].replace(' ', '\n', 1) for o in objs]
xs = np.arange(len(cols))
for m in MODELS:
    ranks = [sorted(CFG_A, key=CFG_A.get, reverse=True).index(m) + 1]
    for o in objs:
        ranks.append(list(a60[a60.objective == o].sort_values('MRR_mean', ascending=False).model).index(m) + 1)
    a.plot(xs, ranks, '-o', color=MODEL_COLORS[m], lw=1.6, ms=5, label=m)
a.set_xticks(xs)
a.set_xticklabels(cols, fontsize=8)
a.set_yticks([1, 2, 3, 4])
a.invert_yaxis()
a.set_ylabel('rank by MRR')
a.set_title('(b) Model ordering follows the objective', loc='left')
a.axvline(0.5, color=MUTED, lw=0.8, ls=':')
a = ax[1, 0]
B = B.sort_values('MRR_triple_level')
y = np.arange(len(B))
a.errorbar(B.MRR_triple_level, y, xerr=[B.MRR_triple_level - B.ci_low_triple, B.ci_high_triple - B.MRR_triple_level], fmt='none', lw=1.4, capsize=3, ecolor=MUTED)
for yi, row in enumerate(B.itertuples(index=False)):
    a.plot(row.MRR_triple_level, yi, 'o', ms=6, color=MODEL_COLORS[row.model])
a.set_yticks(y)
a.set_yticklabels(B.model)
a.set_xlabel(f'MRR (triple-level unit, n = {int(B.n_triples.iloc[0])})')
a.set_title('(c) Cluster-bootstrap 95% intervals', loc='left')
a = ax[1, 1]
OBJC = {'margin': TYPE_COLORS['CAU'], 'logistic': TYPE_COLORS['SYM'], 'selfadv': TYPE_COLORS['HER']}
d = P.copy()
d['abs_d'] = d.cohens_d.abs()
d = d.sort_values(['objective', 'abs_d'])
y = np.arange(len(d))[::-1]
a.barh(y, d.abs_d, color=[OBJC[o] for o in d.objective], height=0.72)
a.axvline(0.2, color=ACCENT, lw=1.1, ls='--')
a.text(0.21, y.max(), 'small-effect\nthreshold 0.2', color=ACCENT, fontsize=7.5, va='top')
a.set_yticks(y)
a.set_yticklabels([c.replace(' - ', '−') for c in d.comparison], fontsize=6.2)
a.set_xlabel("|Cohen's d| on triple-level differences")
a.set_xlim(0, max(0.62, d.abs_d.max() * 1.18))
a.set_title('(d) Pairwise effect sizes by objective', loc='left')
fig.tight_layout()
save_checked(fig, str(FIG / 'fig07_objective_ablation'))
