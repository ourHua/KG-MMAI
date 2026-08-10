#!/usr/bin/env python3
"""Manuscript Figure 6 — annotation audit and structural sensitivity.

When authorised raw-corpus outputs exist, all three panels use those outputs.
The public release can still render a documented reference version: Table 6
for panel (a), the released S0 graph for panel (b), and only manuscript-reported
collision aggregates for panel (c). No unreported collision breakdown is invented.
"""
from __future__ import annotations
__author__ = "LIJUNHUA"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import RES, REF, FIG, ACCENT, MUTED, TYPE_COLORS, choose, save_checked
from labels_en import label as en_label
S = pd.read_csv(choose(RES / 'sensitivity/sensitivity_structure.csv', REF / 'annotation_sensitivity_structure.csv'))
if (RES / 'sensitivity/sensitivity_hubs.csv').is_file():
    H = pd.read_csv(RES / 'sensitivity/sensitivity_hubs.csv')
    s0 = H[H.condition == 'S0_as_annotated'].sort_values('rank').head(10)
else:
    D = pd.read_csv(RES / 'core_node_degrees.csv').nlargest(10, 'degree').copy()
    D['condition'] = 'S0_as_annotated'
    s0 = D[['condition', 'name', 'type', 'degree']]
exact_coll = (RES / 'sensitivity/label_collisions.csv').is_file()
if exact_coll:
    C = pd.read_csv(RES / 'sensitivity/label_collisions.csv')
else:
    C = pd.read_csv(REF / 'annotation_collision_typesets_aggregate.csv')
fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
short = {'S0_as_annotated': 'S0\nas annotated', 'S1_expert_corrected': 'S1\nexpert', 'S2_majority_harmonised': 'S2\nmajority'}
x = np.arange(len(S))
w = 0.38
a = ax[0]
a.bar(x - w / 2, S.nodes, w, color=TYPE_COLORS['SYM'], label='core entities')
a.bar(x + w / 2, S.edges, w, color=TYPE_COLORS['HER'], label='core triples')
for xi, (n, e) in enumerate(zip(S.nodes, S.edges)):
    a.text(xi - w / 2, n, f'{n:,}', ha='center', va='bottom', fontsize=7.5)
    a.text(xi + w / 2, e, f'{e:,}', ha='center', va='bottom', fontsize=7.5)
a.set_xticks(x)
a.set_xticklabels([short[c] for c in S.condition])
a.set_ylabel('count')
a.set_ylim(0, S.edges.max() * 1.22)
a.set_title('(a) Core graph under the three conditions', loc='left')
a.legend(loc='upper left', fontsize=8)
a = ax[1]
cols = [ACCENT if t == 'PRE' else TYPE_COLORS.get(t, MUTED) for t in s0.type]
y = np.arange(len(s0))[::-1]
a.barh(y, s0.degree, color=cols, height=0.68)
a.set_yticks(y)
a.set_yticklabels([f'{en_label(n, t)} ({t})' for n, t in zip(s0.name, s0.type)], fontsize=7)
a.set_xlabel('core degree')
a.set_title('(b) Top hubs as annotated (S0)', loc='left', fontsize=9.5)
a = ax[2]
if exact_coll:
    pairs = {}
    types = ['SYM', 'CAU', 'PRE', 'HER', 'EFF']
    for r in C.itertuples(index=False):
        present = tuple(sorted((t for t in types if getattr(r, t) > 0)))
        pairs[present] = pairs.get(present, 0) + 1
    top = sorted(pairs.items(), key=lambda kv: -kv[1])[:8]
    labels = ['/'.join(k) for k, _ in top]
    vals = [v for _, v in top]
    subtitle = 'exact raw-corpus audit'
else:
    labels = C.type_set.tolist()
    vals = C.surface_forms.tolist()
    subtitle = 'public reference aggregate'
y = np.arange(len(vals))[::-1]
a.barh(y, vals, color=TYPE_COLORS['CAU'], height=0.68)
a.set_yticks(y)
a.set_yticklabels(labels, fontsize=7.5)
a.set_xlabel('surface forms')
a.set_title(f'(c) Multi-type surface forms\n{subtitle}', loc='left', fontsize=9.2)
fig.tight_layout()
save_checked(fig, str(FIG / 'fig06_annotation_sensitivity'))
