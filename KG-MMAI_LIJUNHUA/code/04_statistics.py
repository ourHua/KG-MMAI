#!/usr/bin/env python3
"""Compute statistical comparisons for the embedding experiments.

The analysis uses per-query reciprocal ranks for paired tests, bootstrap
confidence intervals for model MRR, and relation-specific precision estimates
for small test sets.
"""

__author__ = "LIJUNHUA"
import numpy as np
import pandas as pd
from scipy import stats as sps

from kge_core import (
    ALL_TRUE, MODELS, RES, SCHEMA, SEEDS, rels, te, train_epochs, type_members
)


def train_budget(name, seed, total):
    """Use the same training routine as the robustness experiment."""
    return train_epochs(name, seed, total)
BUDGET = 60
BOOT = 5000
rng_global = np.random.default_rng(20260101)


def reciprocal_ranks(M, split):
    rr, rel_of = [], []
    for h, r, t in split:
        htype, ttype = SCHEMA[rels[r]]
        true = M.score(M.E[h], M.R[r], M.E[t])
        cand = type_members[ttype]
        sc = M.score(M.E[h][None, :], M.R[r][None, :], M.E[cand])
        bad = np.array([c != t and (h, r, int(c)) in ALL_TRUE for c in cand])
        rr.append(1.0 / (1 + int(((sc > true) & ~bad).sum())))
        rel_of.append(r)
        cand = type_members[htype]
        sc = M.score(M.E[cand], M.R[r][None, :], M.E[t][None, :])
        bad = np.array([c != h and (int(c), r, t) in ALL_TRUE for c in cand])
        rr.append(1.0 / (1 + int(((sc > true) & ~bad).sum())))
        rel_of.append(r)
    return np.asarray(rr), np.asarray(rel_of)


print("training models for the statistical comparison ...")
RR, REL = {}, None
for name in MODELS:
    per_seed = []
    for seed in SEEDS:
        M = train_budget(name, seed, BUDGET)
        rr, rel_of = reciprocal_ranks(M, te)
        per_seed.append(rr)
        REL = rel_of
    RR[name] = np.mean(per_seed, axis=0)          # average over seeds, per query
    print(f"  {name:9s} MRR {RR[name].mean():.4f}")

# ---------------------------------------------------------------- (a) pairs
rows = []
for i, a in enumerate(MODELS):
    for b in MODELS[i + 1:]:
        d = RR[a] - RR[b]
        t_stat, p_t = sps.ttest_rel(RR[a], RR[b])
        try:
            w_stat, p_w = sps.wilcoxon(RR[a], RR[b])
        except ValueError:
            w_stat, p_w = np.nan, np.nan
        boot = np.array([d[rng_global.integers(0, len(d), len(d))].mean()
                         for _ in range(BOOT)])
        rows.append({
            "model_a": a, "model_b": b,
            "mrr_a": round(RR[a].mean(), 4), "mrr_b": round(RR[b].mean(), 4),
            "diff": round(d.mean(), 4),
            "ci_low": round(float(np.percentile(boot, 2.5)), 4),
            "ci_high": round(float(np.percentile(boot, 97.5)), 4),
            "cohens_d": round(float(d.mean() / d.std(ddof=1)), 3),
            "p_paired_t": f"{p_t:.2e}",
            "p_wilcoxon": f"{p_w:.2e}",
            "significant_0.05": bool(p_t < 0.05),
        })
pd.DataFrame(rows).to_csv(RES / "model_pairwise_tests.csv", index=False)

# ------------------------------------------------------------ (b) bootstrap
rows = []
for name in MODELS:
    rr = RR[name]
    boot = np.array([rr[rng_global.integers(0, len(rr), len(rr))].mean()
                     for _ in range(BOOT)])
    rows.append({
        "model": name, "MRR": round(rr.mean(), 4),
        "ci_low": round(float(np.percentile(boot, 2.5)), 4),
        "ci_high": round(float(np.percentile(boot, 97.5)), 4),
        "queries": len(rr),
    })
pd.DataFrame(rows).sort_values("MRR", ascending=False).to_csv(
    RES / "model_bootstrap_ci.csv", index=False)

# --------------------------------------------------- (c) small-sample check
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


best = max(MODELS, key=lambda m: RR[m].mean())
rr = RR[best]
rows = []
for r_i, rname in enumerate(rels):
    mask = REL == r_i
    n = int(mask.sum())
    mrr = float(rr[mask].mean())
    h10 = float((rr[mask] >= 1 / 10).mean())
    lo, hi = wilson(int(round(h10 * n)), n)
    boot = np.array([rr[mask][rng_global.integers(0, n, n)].mean()
                     for _ in range(BOOT)]) if n else np.zeros(1)
    rows.append({
        "relation": rname, "queries_per_seed": n,
        "MRR": round(mrr, 4),
        "MRR_ci_low": round(float(np.percentile(boot, 2.5)), 4),
        "MRR_ci_high": round(float(np.percentile(boot, 97.5)), 4),
        "ci_width": round(float(np.percentile(boot, 97.5)
                                - np.percentile(boot, 2.5)), 4),
        "Hits@10": round(h10, 4),
        "H10_ci_low": round(lo, 4), "H10_ci_high": round(hi, 4),
    })
pd.DataFrame(rows).sort_values("queries_per_seed").to_csv(
    RES / "small_sample_precision.csv", index=False)

print("\nPairwise tests:")
print(pd.read_csv(RES / "model_pairwise_tests.csv").to_string(index=False))
print("\nBootstrap CIs:")
print(pd.read_csv(RES / "model_bootstrap_ci.csv").to_string(index=False))
print(f"\nPer-relation precision ({best}):")
print(pd.read_csv(RES / "small_sample_precision.csv").to_string(index=False))
