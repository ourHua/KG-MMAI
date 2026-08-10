#!/usr/bin/env python3
"""Manuscript Figure 10 — intended KG-MMAI design specification.

This is a design diagram, not an implemented diagnostic experiment. The script
therefore draws the architecture described in Section 5 without generating any
performance values.
"""
from __future__ import annotations
__author__ = "LIJUNHUA"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from _common import FIG, INK, MUTED, save_checked
fig, ax = plt.subplots(figsize=(10.2, 5.8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

def box(x, y, w, h, title, sub='', fc='#F7F9FA', ec='#4F6D7A', lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03,rounding_size=0.12', facecolor=fc, edgecolor=ec, linewidth=lw))
    ax.text(x + w / 2, y + h * 0.62, title, ha='center', va='center', fontsize=9, fontweight='semibold', color=INK)
    if sub:
        ax.text(x + w / 2, y + h * 0.3, sub, ha='center', va='center', fontsize=7.3, color=MUTED)

def arrow(x1, y1, x2, y2, label=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=12, linewidth=1.2, color='#667780'))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label, ha='center', fontsize=7, color=MUTED)
box(0.4, 6.7, 1.7, 1.0, 'Tongue image', 'visual', fc='#EEF7F8', ec='#4F8C96')
box(0.4, 4.5, 1.7, 1.0, 'Pulse series', 'time series', fc='#EEF7F8', ec='#4F8C96')
box(0.4, 2.3, 1.7, 1.0, 'Inquiry text', 'clinical text', fc='#EEF7F8', ec='#4F8C96')
box(2.8, 6.7, 2.0, 1.0, 'Residual encoder', fc='#F1F7F1', ec='#5E8669')
box(2.8, 4.5, 2.0, 1.0, '1D-CNN encoder', fc='#F1F7F1', ec='#5E8669')
box(2.8, 2.3, 2.0, 1.0, 'Clinical LM encoder', fc='#F1F7F1', ec='#5E8669')
for y in (7.2, 5.0, 2.8):
    arrow(2.1, y, 2.8, y)
box(5.7, 3.8, 2.1, 2.4, 'Cross-modal\nattention', 'fused representation h', fc='#F2F4F8', ec='#687A9E')
for y in (7.2, 5.0, 2.8):
    arrow(4.8, y, 5.7, 5.0)
box(8.5, 6.2, 2.2, 1.2, 'TCM knowledge graph', 'embedded prototypes', fc='#FFF7EC', ec='#B98243')
box(8.5, 3.9, 2.2, 1.5, 'Knowledge-guided\nfusion', 'graph-induced prior q(s|h)', fc='#FFF7EC', ec='#B98243')
arrow(7.8, 5.0, 8.5, 4.65)
arrow(9.6, 6.2, 9.6, 5.4, 'embedding')
box(11.2, 4.1, 1.8, 1.1, 'Classifier', 'prediction p', fc='#F6F2F7', ec='#876090')
arrow(10.7, 4.65, 11.2, 4.65)
box(10.8, 7.1, 2.4, 0.8, 'Auditable path', fc='#FFF4EE', ec='#B06C45')
ys = [6.0, 4.9, 3.8, 2.7]
labels = ['Symptom', 'Pathogenesis', 'Syndrome', 'Formula']
for y, l in zip(ys, labels):
    box(13.15, y, 0.7, 0.72, l, fc='#FFFBEA', ec='#B9A33B', lw=1.1)
arrow(12.1, 5.2, 12.1, 7.1)
arrow(13.0, 7.5, 13.5, 6.72)
for i in range(len(ys) - 1):
    arrow(13.5, ys[i], 13.5, ys[i + 1] + 0.72)
ax.text(0.45, 8.55, 'KG-MMAI: intended knowledge-guided multimodal design', fontsize=13, fontweight='bold', color=INK)
ax.text(0.45, 0.65, 'DESIGN SPECIFICATION ONLY — no multimodal component is implemented or evaluated in this study.', fontsize=9, fontweight='semibold', color='#A34B36')
ax.text(0.45, 0.25, 'The present experiments build and audit the knowledge substrate on which these terms could later be defined.', fontsize=8, color=MUTED)
save_checked(fig, str(FIG / 'fig10_kgmmai_design'))
