#!/usr/bin/env python3
"""Audit regenerated or archived reference results against the revised manuscript.

The public repository intentionally excludes the raw BIO corpus, so a normal
release check may use the small manuscript-reference tables committed with the
package. Pass ``--strict-sensitivity`` for a local source-level audit that
requires the S0/S1/S2 outputs rebuilt from an authorised corpus copy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

__author__ = "LIJUNHUA"

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
REF = RES / "manuscript_reference"
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
COND_NAMES = {
    "S0": "S0_as_annotated",
    "S1": "S1_expert_corrected",
    "S2": "S2_majority_harmonised",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-sensitivity",
        action="store_true",
        help="Require source-derived S0/S1/S2 sensitivity outputs rather than archived reference tables.",
    )
    return parser.parse_args()


def first_existing(*paths: Path) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def canonical_type_set(value: str) -> str:
    """Treat CAU/SYM and SYM/CAU as the same unordered annotation type set."""
    parts = [part.strip() for part in str(value).split("/") if part.strip()]
    return "/".join(sorted(parts))


def main():
    args = parse_args()
    rows = []

    def add(section, claim, value, expected=None, ok=None, note=""):
        if ok is None and expected is not None:
            ok = value == expected
        rows.append({
            "section": section,
            "claim": claim,
            "value": value,
            "expected": expected,
            "status": "PASS" if ok is True else "FAIL" if ok is False else "INFO",
            "note": note,
        })

    ablation_path = first_existing(
        RES / "ablation" / "objective_ablation_summary.csv",
        REF / "objective_ablation_60ep.csv",
    )
    if ablation_path is None:
        add("objective_ablation", "summary file", "missing", "present", ok=False)
    else:
        A = pd.read_csv(ablation_path)
        A60 = A[A.budget_epochs == 60].copy() if "budget_epochs" in A.columns else A.copy()
        note = "regenerated output" if "ablation" in ablation_path.parts else "archived manuscript reference"
        for obj, expected in EXPECTED_60.items():
            sub = A60[A60.objective == obj].set_index("model")
            for model, target in expected.items():
                got = float(sub.loc[model, "MRR_mean"])
                add("objective_ablation", f"{obj}:{model}:MRR60", round(got, 4), target,
                    ok=abs(got - target) <= 5e-4, note=note)
        rank_a = {m: r + 1 for r, m in enumerate(sorted(CFG_A, key=CFG_A.get, reverse=True))}
        for obj, target in (("margin", 0.4), ("selfadv", -1.0)):
            sub = A60[A60.objective == obj].sort_values("MRR_mean", ascending=False)
            rank_o = {m: r + 1 for r, m in enumerate(sub.model)}
            order = list(CFG_A)
            rho = float(spearmanr([rank_a[m] for m in order], [rank_o[m] for m in order]).statistic)
            add("objective_ablation", f"Spearman(ConfigA,{obj})", round(rho, 3), target,
                ok=abs(rho - target) < 1e-12, note=note)

    p_path = first_existing(
        RES / "statistics" / "model_pairwise_triplelevel.csv",
        REF / "model_pairwise_triplelevel.csv",
    )
    b_path = first_existing(
        RES / "statistics" / "model_bootstrap_clustered.csv",
        REF / "model_bootstrap_clustered.csv",
    )
    if p_path is not None and b_path is not None:
        P = pd.read_csv(p_path)
        o3 = P[P.objective == "selfadv"]
        sig = int((pd.to_numeric(o3.p_holm_t, errors="coerce") < 0.05).sum())
        add("statistics", "O3 Holm-significant t-test comparisons", sig, 3, ok=sig == 3)
        max_o3 = float(o3.cohens_d.abs().max())
        add("statistics", "O3 maximum |paired Cohen d|", round(max_o3, 3), "<=0.171",
            ok=max_o3 <= 0.171 + 1e-12)
        add("statistics", "Holm-adjusted Wilcoxon column",
            "present" if "p_holm_wilcoxon" in P.columns else "missing", "present",
            ok="p_holm_wilcoxon" in P.columns)
        B = pd.read_csv(b_path)
        add("statistics", "test triples", int(B.n_triples.iloc[0]), 886, ok=int(B.n_triples.iloc[0]) == 886)
        add("statistics", "ranking queries", int(B.n_queries.iloc[0]), 1772, ok=int(B.n_queries.iloc[0]) == 1772)
        add("statistics", "head clusters", int(B.n_head_clusters.iloc[0]), 343, ok=int(B.n_head_clusters.iloc[0]) == 343)
    else:
        add("statistics", "revised statistical tables", "missing", "present", ok=False)

    r_path = first_existing(
        RES / "statistics" / "relation_lift_exact.csv",
        REF / "relation_lift_exact.csv",
    )
    if r_path is not None:
        R = pd.read_csv(r_path).set_index("relation")
        expected_lift = {"CAUSES": 8.9, "HAS_EFFECT": 8.1, "CONTAINS": 8.8, "RELIEVES": 15.2, "TREATS": 14.6}
        for rel, target in expected_lift.items():
            got = float(R.loc[rel, "lift"])
            add("random_baseline", f"{rel} lift", round(got, 1), target, ok=abs(got - target) <= 0.2)
        well_sampled = R.drop(index="TREATS", errors="ignore")
        add("random_baseline", "largest well-sampled relation lift", well_sampled.lift.idxmax(), "RELIEVES",
            ok=well_sampled.lift.idxmax() == "RELIEVES")
    else:
        add("random_baseline", "relation_lift_exact.csv", "missing", "present", ok=False)

    s_source = RES / "sensitivity" / "sensitivity_structure.csv"
    c_source = RES / "sensitivity" / "label_collisions.csv"
    if s_source.is_file() and c_source.is_file():
        S = pd.read_csv(s_source)
        C = pd.read_csv(c_source)
        source_note = "source-derived"
    elif args.strict_sensitivity:
        S = C = None
        add("annotation_sensitivity", "source-derived sensitivity tables", "missing", "required", ok=False,
            note="Place an authorised data/train.txt locally and rebuild S0/S1/S2 before strict audit.")
    else:
        s_ref = REF / "annotation_sensitivity_structure.csv"
        c_ref = REF / "annotation_collision_typesets_aggregate.csv"
        if s_ref.is_file() and c_ref.is_file():
            S = pd.read_csv(s_ref)
            C = pd.read_csv(c_ref)
            source_note = "archived manuscript reference; raw BIO corpus is intentionally not redistributed"
        else:
            S = C = None
            add("annotation_sensitivity", "public reference tables", "missing", "present", ok=False)

    if S is not None:
        S = S.rename(columns={
            "candidate_triples": "all_triples",
            "core_entities": "nodes",
            "core_triples": "edges",
        }).set_index("condition")
        for condition, targets in EXPECTED_STRUCTURE.items():
            u, all_t, nodes, edges, lcc, maxdeg = targets
            checks = {
                "unique_entities": u,
                "all_triples": all_t,
                "nodes": nodes,
                "edges": edges,
                "largest_component_pct": lcc,
                "max_degree": maxdeg,
            }
            for col, target in checks.items():
                got = float(S.loc[condition, col])
                add("annotation_sensitivity", f"{condition}:{col}", got, target,
                    ok=abs(got - target) < 1e-9, note=source_note)
        if "total_mentions" in C.columns:
            add("annotation_sensitivity", "multi-type surface forms", len(C), 102,
                ok=len(C) == 102, note=source_note)
            add("annotation_sensitivity", "mentions on multi-type forms", int(C.total_mentions.sum()), 2412,
                ok=int(C.total_mentions.sum()) == 2412, note=source_note)
        else:
            total_forms = int(C.surface_forms.sum())
            add("annotation_sensitivity", "multi-type surface forms", total_forms, 102,
                ok=total_forms == 102, note=source_note)
            pair_counts = {
                canonical_type_set(type_set): int(surface_forms)
                for type_set, surface_forms in zip(C.type_set, C.surface_forms)
            }
            sym_cau = canonical_type_set("SYM/CAU")
            pre_her = canonical_type_set("PRE/HER")
            add("annotation_sensitivity", "SYM/CAU surface forms", int(pair_counts.get(sym_cau, -1)), 88,
                ok=pair_counts.get(sym_cau) == 88, note=source_note)
            add("annotation_sensitivity", "PRE/HER surface forms", int(pair_counts.get(pre_her, -1)), 5,
                ok=pair_counts.get(pre_her) == 5, note=source_note)

    ref_link = REF / "annotation_sensitivity_linkpred.csv"
    ref_link_table = pd.read_csv(ref_link) if ref_link.is_file() and not args.strict_sensitivity else None
    for short in ("S0", "S1", "S2"):
        candidates = [
            RES / "sensitivity" / f"linkpred_{short}_summary.csv",
            RES / "sensitivity" / f"linkpred_{short}.csv",
        ]
        existing = first_existing(*candidates)
        if existing is not None:
            T = pd.read_csv(existing)
            if "seed" in T.columns:
                T = T.groupby("model").MRR.agg(["mean", "std"]).reset_index()
                T.columns = ["model", "MRR_mean", "MRR_sd"]
            elif "model" not in T.columns:
                T = T.rename(columns={T.columns[0]: "model"})
            note = "regenerated condition-specific output"
        elif ref_link_table is not None:
            T = ref_link_table[ref_link_table.condition == COND_NAMES[short]][["model", "MRR_mean", "MRR_sd"]].copy()
            note = "archived manuscript reference"
        else:
            if args.strict_sensitivity:
                add("linkpred_sensitivity", f"{short} results", "missing", "present", ok=False)
            continue
        T = T.set_index("model")
        for model, (target_mean, target_sd) in EXPECTED_LINKPRED[short].items():
            got_m = float(T.loc[model, "MRR_mean"])
            got_s = float(T.loc[model, "MRR_sd"])
            add("linkpred_sensitivity", f"{short}:{model}:MRR", round(got_m, 3), target_mean,
                ok=abs(got_m - target_mean) <= 0.0015, note=note)
            add("linkpred_sensitivity", f"{short}:{model}:SD", round(got_s, 3), target_sd,
                ok=abs(got_s - target_sd) <= 0.0015, note=note)

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    failed = out[out.status == "FAIL"]
    print(f"\nWrote {OUT}")
    if len(failed):
        print(f"ERROR: {len(failed)} manuscript-alignment check(s) failed.")
        return 2
    print("All available hard checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
