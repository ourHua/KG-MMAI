#!/usr/bin/env python3
"""Revised statistical analysis for the controlled embedding comparison.

The analysis follows the revised manuscript: the held-out triple is the primary
unit of inference (n=886 in the released S0 split), intervals use 5,000 cluster
bootstrap resamples, pairwise p-values are Holm-Bonferroni adjusted, and the
relation-level random-ranking baseline is computed exactly for each realised
filtered candidate set.

A second bootstrap clusters triples by their shared head entity as a dependence
sensitivity check. Script 07 stores the required head metadata in the NPZ file.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

__author__ = "LIJUNHUA"

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.special import digamma

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "results" / "ablation" / "ablation_per_query_ranks.npz"
OUT = ROOT / "results" / "statistics"
OUT.mkdir(parents=True, exist_ok=True)

BOOT = 5000
BOOT_SEED = 20260101
MODELS = ("TransE", "DistMult", "ComplEx", "RotatE")
SEEDS = (42, 1337, 2024)
OBJECTIVES = ("margin", "logistic", "selfadv")
PRIMARY_OBJECTIVE = "selfadv"
EULER_GAMMA = 0.5772156649015329


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for i, index in enumerate(order):
        value = (len(p) - i) * p[index]
        running = max(running, value)
        adjusted[index] = min(1.0, running)
    return adjusted


def cluster_bootstrap(values, clusters, n=BOOT, seed=BOOT_SEED):
    """Resample complete clusters and return bootstrap means."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    index = {cluster: np.where(clusters == cluster)[0] for cluster in unique}
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for b in range(n):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([index[c] for c in sampled])
        out[b] = values[positions].mean()
    return out


def ci95(samples):
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def paired_d(diff):
    sd = np.asarray(diff, dtype=float).std(ddof=1)
    return float(np.mean(diff) / sd) if sd else 0.0


def effect_label(value):
    value = abs(value)
    if value < 0.05:
        return "none"
    if value < 0.2:
        return "negligible"
    if value < 0.5:
        return "small"
    if value < 0.8:
        return "medium"
    return "large"


def aggregate_query_to_triple(query_values, triple_id, n_triples):
    """Average the two ranking-query values belonging to each test triple."""
    acc = np.zeros(n_triples)
    count = np.zeros(n_triples)
    np.add.at(acc, triple_id, query_values)
    np.add.at(count, triple_id, 1.0)
    return acc / count


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Run code/07_objective_ablation.py first: {INPUT}")

    data = np.load(INPUT)
    relation = data["_relation"].astype(int)
    triple = data["_triple"].astype(int)
    side = data["_side"].astype(int)
    n_candidates = data["_ncand"].astype(float)
    n_triples = int(triple.max()) + 1

    if "_head" in data.files:
        head_query = data["_head"].astype(int)
        head_of_triple = np.zeros(n_triples, dtype=int)
        for q, tri in enumerate(triple):
            head_of_triple[tri] = head_query[q]
    else:
        # Compatibility with an NPZ produced by the pre-revision Script 07.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from kge_core import te  # noqa: E402
        if len(te) != n_triples:
            raise RuntimeError("Cannot infer head clusters from the legacy NPZ.")
        head_of_triple = te[:, 0].astype(int)

    relation_of_triple = np.zeros(n_triples, dtype=int)
    for q, tri in enumerate(triple):
        relation_of_triple[tri] = relation[q]

    def ranks_for(objective, model):
        return [data[f"{objective}|{model}|{seed}"].astype(float) for seed in SEEDS]

    def rr_query(objective):
        return {
            model: np.mean([1.0 / rank for rank in ranks_for(objective, model)], axis=0)
            for model in MODELS
        }

    def rr_triple(objective):
        query = rr_query(objective)
        return {
            model: aggregate_query_to_triple(query[model], triple, n_triples)
            for model in MODELS
        }

    # Pairwise comparisons for all three objectives.
    pairwise_frames = []
    for objective in OBJECTIVES:
        triple_rr = rr_triple(objective)
        rows, raw_p = [], []
        for a, b in itertools.combinations(MODELS, 2):
            diff = triple_rr[a] - triple_rr[b]
            _, p_t = sps.ttest_rel(triple_rr[a], triple_rr[b])
            try:
                _, p_w = sps.wilcoxon(triple_rr[a], triple_rr[b])
            except ValueError:
                p_w = np.nan

            lo, hi = ci95(cluster_bootstrap(diff, np.arange(n_triples)))
            hlo, hhi = ci95(cluster_bootstrap(diff, head_of_triple, seed=BOOT_SEED + 1))
            d = paired_d(diff)
            rows.append({
                "objective": objective,
                "comparison": f"{a} - {b}",
                "n_units": n_triples,
                "mrr_a": round(float(triple_rr[a].mean()), 4),
                "mrr_b": round(float(triple_rr[b].mean()), 4),
                "diff": round(float(diff.mean()), 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "head_cluster_ci_low": round(hlo, 4),
                "head_cluster_ci_high": round(hhi, 4),
                "cohens_d": round(d, 3),
                "effect": effect_label(d),
                "p_raw": float(p_t),
                "p_wilcoxon": float(p_w) if np.isfinite(p_w) else np.nan,
            })
            raw_p.append(p_t)

        adjusted = holm(raw_p)
        for row, p_adj in zip(rows, adjusted):
            row["p_holm"] = float(p_adj)
            row["sig_holm_0.05"] = bool(p_adj < 0.05)
        pairwise_frames.append(pd.DataFrame(rows))

    pairwise = pd.concat(pairwise_frames, ignore_index=True)
    pairwise.to_csv(OUT / "model_pairwise_triplelevel.csv", index=False)
    pairwise[pairwise.objective == PRIMARY_OBJECTIVE].to_csv(
        OUT / "model_pairwise_primary.csv", index=False
    )

    # Primary-objective model intervals, with both dependence assumptions.
    primary_q = rr_query(PRIMARY_OBJECTIVE)
    primary_t = {
        model: aggregate_query_to_triple(primary_q[model], triple, n_triples)
        for model in MODELS
    }
    interval_rows = []
    for model in MODELS:
        tri_lo, tri_hi = ci95(cluster_bootstrap(primary_t[model], np.arange(n_triples)))
        qry_lo, qry_hi = ci95(cluster_bootstrap(primary_q[model], triple, seed=BOOT_SEED + 2))
        head_lo, head_hi = ci95(cluster_bootstrap(primary_t[model], head_of_triple, seed=BOOT_SEED + 3))
        interval_rows.append({
            "model": model,
            "MRR_query_level": round(float(primary_q[model].mean()), 4),
            "MRR_triple_level": round(float(primary_t[model].mean()), 4),
            "ci_low_triple": round(tri_lo, 4),
            "ci_high_triple": round(tri_hi, 4),
            "ci_low_clustered_query": round(qry_lo, 4),
            "ci_high_clustered_query": round(qry_hi, 4),
            "ci_low_head_cluster": round(head_lo, 4),
            "ci_high_head_cluster": round(head_hi, 4),
            "n_triples": n_triples,
            "n_queries": len(triple),
            "n_head_clusters": int(np.unique(head_of_triple).size),
        })
    intervals = pd.DataFrame(interval_rows).sort_values("MRR_triple_level", ascending=False)
    intervals.to_csv(OUT / "model_bootstrap_clustered.csv", index=False)

    # Exact random-ranking baseline and lift by relation.
    harmonic = digamma(n_candidates + 1) + EULER_GAMMA
    expected_rr = harmonic / n_candidates
    expected_h10 = np.minimum(10.0, n_candidates) / n_candidates

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from kge_core import rels  # noqa: E402
    relation_names = dict(enumerate(rels))

    best = max(MODELS, key=lambda model: primary_t[model].mean())
    best_rr = primary_q[best]
    best_ranks = ranks_for(PRIMARY_OBJECTIVE, best)
    best_h10 = np.mean([(rank <= 10).astype(float) for rank in best_ranks], axis=0)

    relation_rows = []
    for rel_id, rel_name in relation_names.items():
        qmask = relation == rel_id
        tmask = relation_of_triple == rel_id
        observed = best_rr[qmask]
        baseline = float(expected_rr[qmask].mean())
        boot = cluster_bootstrap(observed, triple[qmask], seed=BOOT_SEED + 10 + rel_id)
        lo, hi = ci95(boot)
        relation_rows.append({
            "relation": rel_name,
            "test_triples": int(tmask.sum()),
            "queries": int(qmask.sum()),
            "mean_filtered_candidates": round(float(n_candidates[qmask].mean()), 1),
            "head_side_candidates": round(float(n_candidates[qmask & (side == 0)].mean()), 1),
            "tail_side_candidates": round(float(n_candidates[qmask & (side == 1)].mean()), 1),
            "MRR": round(float(observed.mean()), 4),
            "expected_random_MRR_exact": round(baseline, 5),
            "lift": round(float(observed.mean() / baseline), 1),
            "lift_ci_low": round(float(lo / baseline), 1),
            "lift_ci_high": round(float(hi / baseline), 1),
            "Hits@10": round(float(best_h10[qmask].mean()), 4),
            "expected_random_Hits@10": round(float(expected_h10[qmask].mean()), 4),
        })

    relation_table = pd.DataFrame(relation_rows).sort_values("MRR", ascending=False)
    relation_table.to_csv(OUT / "relation_lift_exact.csv", index=False)

    print(f"primary objective: {PRIMARY_OBJECTIVE}; best model: {best}")
    print("\nPairwise comparisons (triple-level, Holm-adjusted):")
    print(pairwise.to_string(index=False))
    print("\nCluster-bootstrap intervals:")
    print(intervals.to_string(index=False))
    print("\nExact random-ranking baseline and lift:")
    print(relation_table.to_string(index=False))


if __name__ == "__main__":
    main()
