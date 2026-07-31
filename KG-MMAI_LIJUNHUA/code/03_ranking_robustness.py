#!/usr/bin/env python3
"""Evaluate whether model rankings are stable across training budgets.

The script trains TransE, DistMult, ComplEx, and RotatE for 20 and 60 epochs
using the shared NumPy implementation, then exports per-seed results, aggregate
rankings, validation curves, and relation-level difficulty estimates.
"""

__author__ = "LIJUNHUA"
import numpy as np
import pandas as pd

from kge_core import (
    MODELS, RES, SCHEMA, SEEDS, evaluate, rels, te, train_epochs, type_members
)
BUDGETS = (20, 60)
EVAL_EVERY = 10


def train_budget(name, seed, total, curve=None):
    """Train one model for a specified number of epochs."""
    return train_epochs(
        name, seed, total, curve=curve, eval_every=EVAL_EVERY
    )


if __name__ == "__main__":
    rows, curves, per_rel = [], [], []
    for budget in BUDGETS:
        for name in MODELS:
            for seed in SEEDS:
                track = curves if budget == max(BUDGETS) else None
                M = train_budget(name, seed, budget, curve=track)
                m = evaluate(M, te, per_relation=True)
                rows.append({"budget_epochs": budget, "model": name, "seed": seed,
                             **{k: v for k, v in m.items() if k != "per_relation"}})
                print(f"  {budget:2d}ep {name:9s} seed {seed:5d}  "
                      f"MRR {m['MRR']:.4f}  H@10 {m['Hits@10']:.4f}")
                if budget == max(BUDGETS):
                    for rel, v in m["per_relation"].items():
                        per_rel.append({"model": name, "seed": seed,
                                        "relation": rel, **v})

    ps = pd.DataFrame(rows)
    ps.to_csv(RES / "robustness_per_seed.csv", index=False)
    pd.DataFrame(curves).to_csv(RES / "robustness_curves.csv", index=False)

    agg = (ps.groupby(["budget_epochs", "model"])
           .agg(MRR_mean=("MRR", "mean"), MRR_sd=("MRR", "std"),
                H1_mean=("Hits@1", "mean"), H3_mean=("Hits@3", "mean"),
                H10_mean=("Hits@10", "mean"), H10_sd=("Hits@10", "std"),
                Med=("MedianRank", "mean"))
           .round(4).reset_index())
    agg["rank_in_config"] = (agg.groupby("budget_epochs")
                             .MRR_mean.rank(ascending=False).astype(int))
    agg = agg.sort_values(["budget_epochs", "rank_in_config"])
    agg.to_csv(RES / "robustness_summary.csv", index=False)

    # --------- per-relation difficulty vs candidate-space size ------------- #
    cand_size = {r: len(type_members[SCHEMA[r][1]]) for r in rels}
    head_size = {r: len(type_members[SCHEMA[r][0]]) for r in rels}
    pr = pd.DataFrame(per_rel)
    pr = (pr[pr.model == "ComplEx"].groupby("relation")
          .agg(queries=("queries", "mean"), MRR=("MRR", "mean"),
               H10=("Hits@10", "mean")).reset_index())
    pr["tail_candidates"] = pr.relation.map(cand_size)
    pr["head_candidates"] = pr.relation.map(head_size)
    pr["mean_candidates"] = (pr.tail_candidates + pr.head_candidates) / 2
    pr["expected_random_MRR"] = (
        np.log(pr.mean_candidates) + 0.5772) / pr.mean_candidates
    pr["lift_over_random"] = (pr.MRR / pr.expected_random_MRR).round(1)
    pr = pr.round(4).sort_values("MRR", ascending=False)
    pr.to_csv(RES / "relation_difficulty.csv", index=False)

    print("\nRanking by configuration:")
    print(agg.to_string(index=False))
    print("\nPer-relation difficulty vs candidate-space size:")
    print(pr.to_string(index=False))
