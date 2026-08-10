#!/usr/bin/env python3
"""Audit regenerated results against headline claims in the revised manuscript.

This is a release gate, not a source of scientific results. A failed hard check
means the code, regenerated results, and manuscript must be reconciled before a
release is tagged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

__author__ = "LIJUNHUA"

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "revision_claims.csv"

CFG_A = {"ComplEx": 0.225, "DistMult": 0.215, "TransE": 0.153, "RotatE": 0.114}
EXPECTED_60 = {
    "margin": {"DistMult": 0.0991, "TransE": 0.0942, "ComplEx": 0.0932, "RotatE": 0.0845},
    "logistic": {"RotatE": 0.2711, "TransE": 0.2577, "DistMult": 0.1627, "ComplEx": 0.1546},
    "selfadv": {"RotatE": 0.2119, "TransE": 0.1930, "DistMult": 0.1859, "ComplEx": 0.1831},
}
EXPECTED_STRUCTURE = {
    "S0_as_annotated": (8024, 48566, 1905, 9544, 99.48, 225),
    "S1_expert_corrected": (8019, 48401, 1903, 9440, 99.47, 225),
    "S2_majority_harmonised": (7922, 48978, 1946, 9908, 99.23, 250),
}
EXPECTED_LINKPRED = {
    "S0": {"RotatE": (0.212, 0.014), "TransE": (0.193, 0.011),
           "DistMult": (0.186, 0.016), "ComplEx": (0.183, 0.014)},
    "S1": {"RotatE": (0.216, 0.004), "TransE": (0.195, 0.011),
           "DistMult": (0.169, 0.007), "ComplEx": (0.172, 0.004)},
    "S2": {"RotatE": (0.213, 0.004), "TransE": (0.200, 0.008),
           "DistMult": (0.191, 0.005), "ComplEx": (0.185, 0.009)},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-sensitivity", action="store_true",
                        help="Require S0/S1/S2 source-derived outputs to be present.")
    return parser.parse_args()


def main():
    args = parse_args(); rows = []

    def add(section, claim, value, expected=None, ok=None, note=""):
        if ok is None and expected is not None:
            ok = value == expected
        rows.append({"section": section, "claim": claim, "value": value,
                     "expected": expected,
                     "status": "PASS" if ok is True else "FAIL" if ok is False else "INFO",
                     "note": note})

    ablation_path = RES / "ablation" / "objective_ablation_summary.csv"
    if not ablation_path.is_file():
        add("objective_ablation", "summary file", "missing", "present", ok=False)
    else:
        A = pd.read_csv(ablation_path); A60 = A[A.budget_epochs == 60].copy()
        for obj, expected in EXPECTED_60.items():
            sub = A60[A60.objective == obj].set_index("model")
            for model, target in expected.items():
                got = float(sub.loc[model, "MRR_mean"])
                add("objective_ablation", f"{obj}:{model}:MRR60", round(got, 4), target,
                    ok=abs(got-target) <= 5e-4)
        rank_a = {m: r+1 for r, m in enumerate(sorted(CFG_A, key=CFG_A.get, reverse=True))}
        for obj, target in (("margin", 0.4), ("selfadv", -1.0)):
            sub = A60[A60.objective == obj].sort_values("MRR_mean", ascending=False)
            rank_o = {m: r+1 for r, m in enumerate(sub.model)}; order = list(CFG_A)
            rho = float(spearmanr([rank_a[m] for m in order],
                                  [rank_o[m] for m in order]).statistic)
            add("objective_ablation", f"Spearman(ConfigA,{obj})", round(rho, 3), target,
                ok=abs(rho-target) < 1e-12)

    p_path = RES / "statistics" / "model_pairwise_triplelevel.csv"
    b_path = RES / "statistics" / "model_bootstrap_clustered.csv"
    if p_path.is_file() and b_path.is_file():
        P = pd.read_csv(p_path); o3 = P[P.objective == "selfadv"]
        sig = int((pd.to_numeric(o3.p_holm_t, errors="coerce") < 0.05).sum())
        add("statistics", "O3 Holm-significant t-test comparisons", sig, 3, ok=sig == 3)
        max_o3 = float(o3.cohens_d.abs().max())
        add("statistics", "O3 maximum |paired Cohen d|", round(max_o3, 3), "<=0.171",
            ok=max_o3 <= 0.171 + 1e-12)
        add("statistics", "Holm-adjusted Wilcoxon column",
            "present" if "p_holm_wilcoxon" in P.columns else "missing", "present",
            ok="p_holm_wilcoxon" in P.columns)
        B = pd.read_csv(b_path)
        add("statistics", "test triples", int(B.n_triples.iloc[0]), 886,
            ok=int(B.n_triples.iloc[0]) == 886)
        add("statistics", "ranking queries", int(B.n_queries.iloc[0]), 1772,
            ok=int(B.n_queries.iloc[0]) == 1772)
        add("statistics", "head clusters", int(B.n_head_clusters.iloc[0]), 343,
            ok=int(B.n_head_clusters.iloc[0]) == 343)
    else:
        add("statistics", "revised statistical tables", "missing", "present", ok=False)

    r_path = RES / "statistics" / "relation_lift_exact.csv"
    if r_path.is_file():
        R = pd.read_csv(r_path).set_index("relation")
        expected_lift = {"CAUSES": 8.9, "HAS_EFFECT": 8.1, "CONTAINS": 8.8,
                         "RELIEVES": 15.2, "TREATS": 14.6}
        for rel, target in expected_lift.items():
            got = float(R.loc[rel, "lift"])
            add("random_baseline", f"{rel} lift", round(got, 1), target,
                ok=abs(got-target) <= 0.2)
        well_sampled = R.drop(index="TREATS", errors="ignore")
        add("random_baseline", "largest well-sampled relation lift",
            well_sampled.lift.idxmax(), "RELIEVES",
            ok=well_sampled.lift.idxmax() == "RELIEVES")
    else:
        add("random_baseline", "relation_lift_exact.csv", "missing", "present", ok=False)

    s_path = RES / "sensitivity" / "sensitivity_structure.csv"
    c_path = RES / "sensitivity" / "label_collisions.csv"
    if s_path.is_file() and c_path.is_file():
        S = pd.read_csv(s_path).set_index("condition")
        for condition, targets in EXPECTED_STRUCTURE.items():
            u, all_t, nodes, edges, lcc, maxdeg = targets
            checks = {"unique_entities": u, "all_triples": all_t, "nodes": nodes,
                      "edges": edges, "largest_component_pct": lcc, "max_degree": maxdeg}
            for col, target in checks.items():
                got = float(S.loc[condition, col])
                add("annotation_sensitivity", f"{condition}:{col}", got, target,
                    ok=abs(got-target) < 1e-9)
        C = pd.read_csv(c_path)
        add("annotation_sensitivity", "multi-type surface forms", len(C), 102, ok=len(C) == 102)
        add("annotation_sensitivity", "mentions on multi-type forms",
            int(C.total_mentions.sum()), 2412, ok=int(C.total_mentions.sum()) == 2412)
    else:
        add("annotation_sensitivity", "source-derived sensitivity tables",
            "not present in public checkout", "required for full local rerun",
            ok=False if args.strict_sensitivity else None,
            note="The raw BIO corpus is not redistributed.")

    for short in ("S0", "S1", "S2"):
        candidates = [RES / "sensitivity" / f"linkpred_{short}_summary.csv",
                      RES / "sensitivity" / f"linkpred_{short}.csv"]
        existing = next((p for p in candidates if p.is_file()), None)
        if existing is None:
            if args.strict_sensitivity:
                add("linkpred_sensitivity", f"{short} results", "missing", "present", ok=False)
            continue
        T = pd.read_csv(existing)
        if "seed" in T.columns:
            T = T.groupby("model").MRR.agg(["mean", "std"]).reset_index()
            T.columns = ["model", "MRR_mean", "MRR_sd"]
        elif "model" not in T.columns:
            T = T.rename(columns={T.columns[0]: "model"})
        T = T.set_index("model")
        for model, (target_mean, target_sd) in EXPECTED_LINKPRED[short].items():
            got_m = float(T.loc[model, "MRR_mean"]); got_s = float(T.loc[model, "MRR_sd"])
            add("linkpred_sensitivity", f"{short}:{model}:MRR", round(got_m, 3), target_mean,
                ok=abs(got_m-target_mean) <= 0.0015)
            add("linkpred_sensitivity", f"{short}:{model}:SD", round(got_s, 3), target_sd,
                ok=abs(got_s-target_sd) <= 0.0015)

    out = pd.DataFrame(rows); OUT.parent.mkdir(exist_ok=True); out.to_csv(OUT, index=False)
    print(out.to_string(index=False)); failed = out[out.status == "FAIL"]
    print(f"\nWrote {OUT}")
    if len(failed):
        print(f"ERROR: {len(failed)} manuscript-alignment check(s) failed.")
        return 2
    print("All available hard checks passed."); return 0


if __name__ == "__main__":
    raise SystemExit(main())
