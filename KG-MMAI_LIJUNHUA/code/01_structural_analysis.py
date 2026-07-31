#!/usr/bin/env python3
"""
01_structural_analysis.py
Recomputes every structural claim in the manuscript directly from nodes.csv /
edges.csv, plus additional analyses added in revision.

Outputs (results/):
  structural_profile.csv        threshold sensitivity  -> Table 4
  entity_distribution.csv       entity counts by type  -> Table 2
  relation_distribution.csv     relation stats         -> Table 3
  degree_stats.csv              degree distribution per entity type
  component_detail.csv          weakly connected components of the core graph
  hub_entities.csv              top entities by core degree
  weight_distribution.csv       edge-weight histogram
  structural_summary.json       machine-readable digest
"""

__author__ = "LIJUNHUA"
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
RES.mkdir(exist_ok=True)

SCHEMA = {
    "CAUSES": ("CAU", "SYM"),
    "CONTAINS": ("PRE", "HER"),
    "TREATS": ("PRE", "SYM"),
    "HAS_EFFECT": ("HER", "EFF"),
    "RELIEVES": ("HER", "SYM"),
}
TYPE_LABEL = {
    "SYM": "Symptom / sign",
    "CAU": "Cause / pathogenesis / condition",
    "PRE": "Prescription / formula",
    "HER": "Herb",
    "EFF": "Effect / treatment action",
}

nodes = pd.read_csv(DATA / "nodes.csv", encoding="utf-8-sig")
edges = pd.read_csv(DATA / "edges.csv", encoding="utf-8-sig")
nodes_core = pd.read_csv(DATA / "nodes_core.csv", encoding="utf-8-sig")

# --------------------------------------------------------------------------- #
# 1. entity distribution (Table 2)
# --------------------------------------------------------------------------- #
ent = (
    pd.DataFrame({"code": list(TYPE_LABEL)})
    .assign(interpretation=lambda d: d.code.map(TYPE_LABEL))
    .merge(nodes.type.value_counts().rename("all_entities"),
           left_on="code", right_index=True)
    .merge(nodes_core.type.value_counts().rename("core_entities"),
           left_on="code", right_index=True)
)
ent["retention_pct"] = (100 * ent.core_entities / ent.all_entities).round(2)
ent = ent.sort_values("all_entities", ascending=False)
ent.to_csv(RES / "entity_distribution.csv", index=False)

# --------------------------------------------------------------------------- #
# 2. relation distribution (Table 3)
# --------------------------------------------------------------------------- #
core = edges[edges.weight >= 2]
rel = (
    core.groupby("relation")
    .weight.agg(triples="count", mean_weight="mean", max_weight="max",
                total_weight="sum")
    .reset_index()
)
rel["schema"] = rel.relation.map(lambda r: f"{SCHEMA[r][0]} \u2192 {SCHEMA[r][1]}")
allrel = edges.relation.value_counts().rename("all_triples")
rel = rel.merge(allrel, left_on="relation", right_index=True)
rel["survival_pct"] = (100 * rel.triples / rel.all_triples).round(2)
rel["mean_weight"] = rel.mean_weight.round(2)
rel = rel[["relation", "schema", "all_triples", "triples", "survival_pct",
           "mean_weight", "max_weight", "total_weight"]]
rel = rel.sort_values("triples", ascending=False)
rel.to_csv(RES / "relation_distribution.csv", index=False)

# --------------------------------------------------------------------------- #
# 3. threshold sensitivity (Table 4)
# --------------------------------------------------------------------------- #
def profile(sub, tau):
    G = nx.DiGraph()
    for r in sub.itertuples(index=False):
        G.add_edge(r.source_id, r.target_id, relation=r.relation, weight=r.weight)
    n, m = G.number_of_nodes(), G.number_of_edges()
    comps = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    deg = np.array([d for _, d in G.degree()], dtype=float)
    und = G.to_undirected()

    # schema audit
    tviol = sum(
        1 for r in sub.itertuples(index=False)
        if SCHEMA.get(r.relation, (None, None)) != (r.source_type, r.target_type)
    )
    dup = int(sub.duplicated(subset=["source_id", "relation", "target_id"]).sum())

    return {
        "threshold": tau,
        "nodes": n,
        "edges": m,
        "components": len(comps),
        "largest_component_pct": round(100 * len(comps[0]) / n, 2) if n else 0.0,
        "mean_degree": round(deg.mean(), 2) if n else 0.0,
        "median_degree": float(np.median(deg)) if n else 0.0,
        "degree_skewness": round(float(sps.skew(deg)), 2) if n else 0.0,
        "max_degree": int(deg.max()) if n else 0,
        "density": round(nx.density(G), 6),
        "isolated_nodes": int(nx.number_of_isolates(G)),
        "reciprocity": round(nx.reciprocity(G) or 0.0, 4),
        "clustering_undirected": round(nx.average_clustering(und), 4),
        "type_violations": tviol,
        "duplicate_triples": dup,
    }


rows = [profile(edges[edges.weight >= t], t) for t in (1, 2, 3, 4, 5, 10)]
prof = pd.DataFrame(rows)
prof.to_csv(RES / "structural_profile.csv", index=False)

# --------------------------------------------------------------------------- #
# 4. core-graph detail: degrees, components, hubs
# --------------------------------------------------------------------------- #
Gc = nx.DiGraph()
for r in core.itertuples(index=False):
    Gc.add_edge(r.source_id, r.target_id, relation=r.relation, weight=r.weight)

nmap = nodes.set_index("id")[["name", "type"]]
degdf = pd.DataFrame({
    "id": list(Gc.nodes()),
    "in_degree": [Gc.in_degree(v) for v in Gc.nodes()],
    "out_degree": [Gc.out_degree(v) for v in Gc.nodes()],
}).join(nmap, on="id")
degdf["degree"] = degdf.in_degree + degdf.out_degree
degdf["strength"] = [
    sum(d["weight"] for _, _, d in Gc.in_edges(v, data=True))
    + sum(d["weight"] for _, _, d in Gc.out_edges(v, data=True))
    for v in degdf.id
]

deg_stats = (
    degdf.groupby("type")
    .degree.agg(entities="count", mean="mean", median="median",
                p90=lambda s: s.quantile(0.90), max="max")
    .round(2).reset_index()
)
deg_stats.to_csv(RES / "degree_stats.csv", index=False)

degdf.sort_values("degree", ascending=False).head(40).to_csv(
    RES / "hub_entities.csv", index=False)
degdf.to_csv(RES / "core_node_degrees.csv", index=False)

comps = sorted(nx.weakly_connected_components(Gc), key=len, reverse=True)
comp_rows = []
for i, cset in enumerate(comps, 1):
    sub = degdf[degdf.id.isin(cset)]
    comp_rows.append({
        "component": i,
        "nodes": len(cset),
        "share_pct": round(100 * len(cset) / Gc.number_of_nodes(), 3),
        "edges": Gc.subgraph(cset).number_of_edges(),
        "types_present": "/".join(sorted(sub.type.unique())),
        "example_entities": ", ".join(sub.nlargest(3, "degree").name.tolist()),
    })
pd.DataFrame(comp_rows).to_csv(RES / "component_detail.csv", index=False)

# --------------------------------------------------------------------------- #
# 5. weight distribution
# --------------------------------------------------------------------------- #
wd = (edges.weight.value_counts().sort_index().rename("edges")
      .reset_index().rename(columns={"index": "weight"}))
wd["cumulative_pct"] = (100 * wd.edges.cumsum() / wd.edges.sum()).round(3)
wd.to_csv(RES / "weight_distribution.csv", index=False)

# --------------------------------------------------------------------------- #
summary = {
    "corpus_entities_total": int(len(nodes)),
    "candidate_triples_total": int(len(edges)),
    "entities_with_at_least_one_edge": int(
        len(set(edges.source_id) | set(edges.target_id))),
    "single_occurrence_share_pct": round(100 * float((edges.weight == 1).mean()), 2),
    "core_entities": int(len(nodes_core)),
    "core_triples": int(len(core)),
    "core_largest_component_pct": rows[1]["largest_component_pct"],
    "core_components": rows[1]["components"],
    "core_mean_degree": rows[1]["mean_degree"],
    "core_density": rows[1]["density"],
    "core_degree_skewness": rows[1]["degree_skewness"],
    "core_clustering": rows[1]["clustering_undirected"],
    "type_violations_all_thresholds": int(prof.type_violations.sum()),
    "duplicate_triples_all_thresholds": int(prof.duplicate_triples.sum()),
}
(RES / "structural_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

print(json.dumps(summary, indent=2, ensure_ascii=False))
print("\nThreshold profile:")
print(prof.to_string(index=False))
print("\nComponents of the core graph:")
print(pd.DataFrame(comp_rows).to_string(index=False))
