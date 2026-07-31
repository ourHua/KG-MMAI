#!/usr/bin/env python3
"""Run the default typed, filtered link-prediction experiment.

The reusable model and evaluation code is stored in ``kge_core.py``. This
entry point writes the deterministic split table, trains four embedding models
with three random seeds, and exports aggregate and relation-level metrics.
"""

from __future__ import annotations

__author__ = "LIJUNHUA"

import pandas as pd

from kge_core import (
    MODELS,
    N_E,
    N_R,
    RES,
    SEEDS,
    evaluate,
    te,
    tr,
    train,
    va,
    write_split_table,
)

if __name__ == "__main__":
    write_split_table()
    print(f"entities {N_E} | relations {N_R} | "
          f"train {len(tr)} valid {len(va)} test {len(te)} | "
          f"queries/seed {2*len(te)}")

    rows, curves, per_rel_best = [], [], []
    for name in MODELS:
        for seed in SEEDS:
            M = train(name, seed, curve=curves)
            m = evaluate(M, te, per_relation=(name == "ComplEx"))
            rows.append({"model": name, "seed": seed,
                         **{k: v for k, v in m.items() if k != "per_relation"}})
            print(f"  {name:9s} seed {seed:5d}  "
                  f"MRR {m['MRR']:.3f}  H@1 {m['Hits@1']:.3f}  "
                  f"H@10 {m['Hits@10']:.3f}  med {m['MedianRank']:.1f}")
            if name == "ComplEx":
                for rel, v in m["per_relation"].items():
                    per_rel_best.append({"seed": seed, "relation": rel, **v})

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(RES / "lp_per_seed.csv", index=False)
    pd.DataFrame(curves).to_csv(RES / "lp_training_curves.csv", index=False)

    agg = (per_seed.groupby("model")
           .agg(MRR_mean=("MRR", "mean"), MRR_sd=("MRR", "std"),
                H1_mean=("Hits@1", "mean"), H1_sd=("Hits@1", "std"),
                H3_mean=("Hits@3", "mean"), H3_sd=("Hits@3", "std"),
                H10_mean=("Hits@10", "mean"), H10_sd=("Hits@10", "std"),
                Med_mean=("MedianRank", "mean"))
           .round(4).reset_index()
           .sort_values("MRR_mean", ascending=False))
    agg.to_csv(RES / "lp_summary.csv", index=False)

    prb = pd.DataFrame(per_rel_best)
    prb_agg = (prb.groupby("relation")
               .agg(queries=("queries", "mean"),
                    MRR_mean=("MRR", "mean"), MRR_sd=("MRR", "std"),
                    H10_mean=("Hits@10", "mean"), H10_sd=("Hits@10", "std"))
               .round(4).reset_index().sort_values("MRR_mean", ascending=False))
    prb_agg.to_csv(RES / "lp_relation_complex.csv", index=False)

    print("\nSummary across seeds:")
    print(agg.to_string(index=False))
    print("\nPer-relation (ComplEx):")
    print(prb_agg.to_string(index=False))

