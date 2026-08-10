#!/usr/bin/env python3
"""Audit annotation inconsistency and rebuild the graph under three conditions.

This reviewer-requested analysis starts from the local BIO-tagged corpus,
identifies surface forms assigned to more than one entity type, and rebuilds
the graph under S0 (as annotated), S1 (expert correction of the five PRE/HER
collisions), and S2 (majority harmonisation of all multi-type forms).

The raw corpus is intentionally not redistributed. Place the source file at
``data/train.txt`` before running this script.

Outputs are written to ``results/sensitivity/``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

__author__ = "LIJUNHUA"

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
OUT = RES / "sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

SCHEMA = {
    "CAUSES": ("CAU", "SYM"),
    "CONTAINS": ("PRE", "HER"),
    "TREATS": ("PRE", "SYM"),
    "HAS_EFFECT": ("HER", "EFF"),
    "RELIEVES": ("HER", "SYM"),
}
RULES = {(h, t): r for r, (h, t) in SCHEMA.items()}
TYPES = ("SYM", "CAU", "PRE", "HER", "EFF")

EXPERT_MAP = {
    "苍术": "HER",       # Cangzhu / Atractylodis Rhizoma
    "麦门冬": "HER",     # Maimendong / Ophiopogonis Radix
    "橘皮": "HER",       # Jupi / Citri Reticulatae Pericarpium
    "紫雪丹": "PRE",     # Zixue Dan
    "金水六君煎": "PRE", # Jinshui Liujun Jian
}


def read_samples(path=DATA / "train.txt"):
    """Read comma-separated character/BIO rows into samples."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Raw BIO corpus not found: {path}. The source corpus is not "
            "redistributed; place the authorised local copy at data/train.txt."
        )
    samples, cur = [], []
    n_tokens = 0
    repairs = Counter()
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                if cur:
                    samples.append(cur)
                    cur = []
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            char, tag = parts[0], parts[-1].strip()
            n_tokens += 1
            if tag != "O":
                prefix, _, typ = tag.partition("-")
                clean = "".join(ch for ch in typ if ch.isalpha()).upper()[:3]
                if clean != typ:
                    repairs[typ] += 1
                tag = f"{prefix}-{clean}"
            cur.append((char, tag))
    if cur:
        samples.append(cur)
    return samples, n_tokens, repairs


def spans(sample):
    """Reconstruct B-X followed by I-X spans of the same entity type."""
    out, buf, cur_t = [], [], None
    for char, tag in sample:
        if tag.startswith("B-"):
            if buf:
                out.append((cur_t, "".join(buf)))
            buf, cur_t = [char], tag[2:]
        elif tag.startswith("I-") and cur_t == tag[2:]:
            buf.append(char)
        else:
            if buf:
                out.append((cur_t, "".join(buf)))
            buf, cur_t = [], None
    if buf:
        out.append((cur_t, "".join(buf)))
    return [(typ, name) for typ, name in out if typ in TYPES and name]


def build(sample_mentions, remap=None):
    """Build node and weighted candidate-edge tables under a type remapping."""
    remap = remap or {}
    freq, edge_w = Counter(), Counter()
    for mentions in sample_mentions:
        fixed = [(remap.get(name, typ), name) for typ, name in mentions]
        for typ, name in fixed:
            freq[(typ, name)] += 1
        by_type = defaultdict(set)
        for typ, name in fixed:
            by_type[typ].add(name)
        for (head_t, tail_t), rel in RULES.items():
            for head, tail in product(
                sorted(by_type.get(head_t, ())),
                sorted(by_type.get(tail_t, ())),
            ):
                if head != tail:
                    edge_w[(head_t, head, rel, tail_t, tail)] += 1

    ids, counter = {}, Counter()
    ordered = sorted(freq.items(), key=lambda kv: (kv[0][0], -kv[1], kv[0][1]))
    for (typ, name), _ in ordered:
        ids[(typ, name)] = f"{typ}_{counter[typ]:05d}"
        counter[typ] += 1

    nodes = pd.DataFrame([
        {"id": ids[(typ, name)], "name": name, "type": typ, "frequency": count}
        for (typ, name), count in freq.items()
    ]).sort_values("id").reset_index(drop=True)

    edges = pd.DataFrame([
        {
            "source_id": ids[(ht, head)], "source_name": head, "source_type": ht,
            "relation": rel,
            "target_id": ids[(tt, tail)], "target_name": tail, "target_type": tt,
            "weight": weight,
        }
        for (ht, head, rel, tt, tail), weight in edge_w.items()
    ]).sort_values(["relation", "source_id", "target_id"]).reset_index(drop=True)
    return nodes, edges


def profile(edges, tau=2, label=""):
    """Return the structural profile of the thresholded core graph."""
    sub = edges[edges.weight >= tau]
    graph = nx.DiGraph()
    for row in sub.itertuples(index=False):
        graph.add_edge(row.source_id, row.target_id)
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    components = sorted(nx.weakly_connected_components(graph), key=len, reverse=True)
    degrees = np.array([d for _, d in graph.degree()], dtype=float)
    violations = sum(
        1 for row in sub.itertuples(index=False)
        if SCHEMA.get(row.relation, (None, None)) != (row.source_type, row.target_type)
    )
    return {
        "condition": label,
        "threshold": tau,
        "all_entities": int(edges[["source_id", "target_id"]].stack().nunique()),
        "all_triples": len(edges),
        "nodes": n,
        "edges": m,
        "components": len(components),
        "largest_component_pct": round(100 * len(components[0]) / n, 2) if n else 0.0,
        "mean_degree": round(float(degrees.mean()), 2) if n else 0.0,
        "median_degree": float(np.median(degrees)) if n else 0.0,
        "degree_skewness": round(float(sps.skew(degrees)), 2) if n else 0.0,
        "max_degree": int(degrees.max()) if n else 0,
        "density": round(nx.density(graph), 6),
        "type_violations": violations,
        "duplicate_triples": int(sub.duplicated(
            subset=["source_id", "relation", "target_id"]
        ).sum()),
    }, graph


def hubs(graph, nodes, label, k=10):
    degree = pd.Series(dict(graph.degree())).sort_values(ascending=False).head(k)
    node_map = nodes.set_index("id")
    return pd.DataFrame([
        {
            "condition": label,
            "rank": i + 1,
            "id": entity,
            "name": node_map.loc[entity, "name"],
            "type": node_map.loc[entity, "type"],
            "degree": int(value),
        }
        for i, (entity, value) in enumerate(degree.items())
    ])


def main():
    samples, n_tokens, repairs = read_samples()
    mentions = [spans(sample) for sample in samples]
    n_mentions = sum(len(m) for m in mentions)
    print(
        f"samples {len(samples)}  character tokens {n_tokens}  "
        f"mentions {n_mentions}  label repairs {dict(repairs)}"
    )

    surface = defaultdict(Counter)
    for sample_mentions in mentions:
        for typ, name in sample_mentions:
            surface[name][typ] += 1
    collisions = {name: counts for name, counts in surface.items() if len(counts) > 1}

    rows = []
    for name, counts in sorted(collisions.items(), key=lambda kv: -sum(kv[1].values())):
        row = {
            "name": name,
            "total_mentions": sum(counts.values()),
            "n_types": len(counts),
            "majority_type": counts.most_common(1)[0][0],
        }
        row.update({typ: counts.get(typ, 0) for typ in TYPES})
        rows.append(row)
    collision_df = pd.DataFrame(rows)
    collision_df.to_csv(OUT / "label_collisions.csv", index=False)
    collision_mentions = int(collision_df.total_mentions.sum())
    print(
        f"multi-type surface forms: {len(collisions)} "
        f"({collision_mentions} mentions, {100 * collision_mentions / n_mentions:.2f}% of all mentions)"
    )

    majority_map = {name: counts.most_common(1)[0][0] for name, counts in collisions.items()}
    conditions = {
        "S0_as_annotated": {},
        "S1_expert_corrected": EXPERT_MAP,
        "S2_majority_harmonised": majority_map,
    }

    profile_rows, hub_rows = [], []
    for label, remap in conditions.items():
        nodes, edges = build(mentions, remap)
        nodes.to_csv(OUT / f"nodes_{label}.csv", index=False)
        edges.to_csv(OUT / f"edges_{label}.csv", index=False)
        stats, graph = profile(edges, tau=2, label=label)
        stats["unique_entities"] = len(nodes)
        profile_rows.append(stats)
        hub_rows.append(hubs(graph, nodes, label))
        print(
            f"{label:24s} entities {len(nodes):6d} triples {len(edges):7d} "
            f"core {stats['nodes']:5d}/{stats['edges']:6d} "
            f"LCC {stats['largest_component_pct']}% maxdeg {stats['max_degree']}"
        )

    profiles = pd.DataFrame(profile_rows)
    profiles.to_csv(OUT / "sensitivity_structure.csv", index=False)
    pd.concat(hub_rows, ignore_index=True).to_csv(OUT / "sensitivity_hubs.csv", index=False)
    print("\n" + profiles.to_string(index=False))


if __name__ == "__main__":
    main()
