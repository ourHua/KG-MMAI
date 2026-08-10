#!/usr/bin/env python3
"""Re-run the primary link-prediction protocol on a corrected graph.

The edge table is selected before Script 07 imports ``kge_core``. This is
important: the sensitivity experiment must actually train and evaluate on the
S0/S1/S2 rebuilt graph rather than silently reusing ``data/edges.csv``.

Examples
--------
python code/11_sensitivity_linkpred.py --condition S1 \
  --edges results/sensitivity/edges_S1_expert_corrected.csv
python code/11_sensitivity_linkpred.py --condition S2 \
  --edges results/sensitivity/edges_S2_majority_harmonised.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

__author__ = "LIJUNHUA"

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results" / "sensitivity"
OUT.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=("S0", "S1", "S2"))
    parser.add_argument("--edges", required=True, help="Rebuilt edge CSV for this condition")
    return parser.parse_args()


def load_ablation(edges_path):
    path = Path(edges_path).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    # Script 07 reads KG_EDGES during module import and rebuilds the shared KGE
    # context before training begins.
    os.environ["KG_EDGES"] = str(path)
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("objective_ablation_sensitivity", HERE / "07_objective_ablation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def main():
    args = parse_args()
    ablation, edge_path = load_ablation(args.edges)
    print(f"condition {args.condition}; edges {edge_path}")

    rows = []
    for model_name in ablation.MODELS:
        for seed in ablation.SEEDS:
            model = ablation.train_objective(model_name, seed, 60, "selfadv")
            metrics, _, _ = ablation.evaluate_fast(model)
            rows.append({
                "condition": args.condition,
                "model": model_name,
                "seed": seed,
                **metrics,
            })
            print(
                f"  {args.condition} {model_name:9s} seed {seed:5d} "
                f"MRR {metrics['MRR']:.4f}",
                flush=True,
            )

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT / f"linkpred_{args.condition}.csv", index=False)
    summary = (
        per_seed.groupby("model")
        .agg(
            MRR_mean=("MRR", "mean"),
            MRR_sd=("MRR", "std"),
            H10_mean=("Hits@10", "mean"),
        )
        .round(4)
        .sort_values("MRR_mean", ascending=False)
    )
    summary.to_csv(OUT / f"linkpred_{args.condition}_summary.csv")
    print("\n" + summary.to_string())


if __name__ == "__main__":
    main()
