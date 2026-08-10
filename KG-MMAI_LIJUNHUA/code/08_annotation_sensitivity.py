#!/usr/bin/env python3
"""Audit annotation inconsistency and rebuild the graph under three conditions.

This reviewer-requested analysis starts from the local BIO-tagged corpus,
identifies surface forms assigned to more than one entity type, and rebuilds
the graph under S0 (as annotated), S1 (expert correction of the five PRE/HER
collisions), and S2 (majority harmonisation of all multi-type forms).

The raw corpus is intentionally not redistributed. Place an authorised local
copy at ``data/train.txt`` or pass ``--corpus PATH``.

Outputs are written to ``results/sensitivity/``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
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

# Expert adjudication of the five surface forms carrying both PRE and HER.
EXPERT_MAP = {
    "苍术": "HER",        # Atractylodis Rhizoma
    "麦门冬": "HER",      # Ophiopogonis Radix
    "橘皮": "HER",        # Citri Reticulatae Pericarpium
    "紫雪丹": "PRE",      # Zixue Dan
    "金水六君煎": "PRE",  # Jinshui Liujun Jian
}


def read_samples(path=DATA / "train.txt"):
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
                pre, _, typ = tag.partition("-")
                clean = "".join(ch for ch in typ if ch.isalpha()).upper()[:3]
                if clean != typ:
                    repairs[typ] += 1
                tag = f"{pre}-{clean}"
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
    return [(t, n) for t, n in out if t in TYPES and n]


def build(sample_mentions, remap=None):
    """Build typed nodes and weighted within-sample co-occurrence triples."""
    remap = remap or {}
    freq = Counter()
    edge_w = Counter()
    for ments in sample_mentions:
        fixed = [(remap.get(n, t), n) for t, n in ments]
        for t, n in fixed:
            freq[(t, n)] += 1
        by_type = defaultdict(set)
        for t, n in fixed:
            by_type[t].add(n)
        for (ht, tt), rel in RULES.items():
            for h, t_ in product(sorted(by_type.get(ht, ())),
                                 sorted(by_type.get(tt, ()))):
                if h == t_:
                    continue
                edge_w[(ht, h, rel, tt, t_)] += 1

    ids, counter = {}, Counter()
    for (t, n), _ in sorted(freq.items(), key=lambda kv: (kv[0][0], -kv[1], kv[0][1])):
        ids[(t, n)] = f"{t}_{counter[t]:05d}"
        counter[t] += 1

    nodes = pd.DataFrame(
        [{"id": ids[(t, n)], "name": n, "type": t, "frequency": c}
         for (t, n), c in freq.items()]
    ).sort_values("id").reset_index(drop=True)

    edges = pd.DataFrame(
        [{"source_id": ids[(ht, h)], "source_name": h, "source_type": ht,
          "relation": rel, "target_id": ids[(tt, t_)], "target_name": t_,
          "target_type": tt, "weight": w}
         for (ht, h, rel, tt, t_), w in edge_w.items()]
    ).sort_values(["relation", "source_id", "target_id"]).reset_index(drop=True)
    return nodes, edges


def profile(edges, tau=2, label=""):
    sub = edges[edges.weight >= tau]
    G = nx.DiGraph()
    for r in sub.itertuples(index=False):
        G.add_edge(r.source_id, r.target_id)
    n, m = G.number_of_nodes(), G.number_of_edges()
    comps = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    deg = np.array([d for _, d in G.degree()], dtype=float)
    tviol = sum(
        1 for r in sub.itertuples(index=False)
        if SCHEMA.get(r.relation, (None, None)) != (r.source_type, r.target_type)
    )
    return {
        "condition": label,
        "threshold": tau,
        "all_entities": int(edges[["source_id", "target_id"]].stack().nunique()),
        "all_triples": len(edges),
        "nodes": n,
        "edges": m,
        "components": len(comps),
        "largest_component_pct": round(100 * len(comps[0]) / n, 2) if n else 0.0,
        "mean_degree": round(float(deg.mean()), 2) if n else 0.0,
        "median_degree": float(np.median(deg)) if n else 0.0,
        "degree_skewness": round(float(sps.skew(deg)), 2) if n else 0.0,
        "max_degree": int(deg.max()) if n else 0,
        "density": round(nx.density(G), 6),
        "type_violations": tviol,
        "duplicate_triples": int(sub.duplicated(
            subset=["source_id", "relation", "target_id"]
        ).sum()),
    }, G


def hubs(G, nodes, label, k=10):
    deg = pd.Series(dict(G.degree())).sort_values(ascending=False).head(k)
    nm = nodes.set_index("id")
    return pd.DataFrame([
        {"condition": label, "rank": i + 1, "id": e,
         "name": nm.loc[e, "name"], "type": nm.loc[e, "type"], "degree": int(d)}
        for i, (e, d) in enumerate(deg.items())
    ])


def semantic_core_edges(edges, tau=2):
    """Condition-comparable semantic keys independent of generated node IDs."""
    sub = edges[edges.weight >= tau]
    return set(zip(
        sub.source_type, sub.source_name, sub.relation,
        sub.target_type, sub.target_name,
    ))


MANUSCRIPT_EXPECTED = {
    "S0_as_annotated": {
        "unique_entities": 8024, "all_triples": 48566, "nodes": 1905,
        "edges": 9544, "largest_component_pct": 99.48, "max_degree": 225,
    },
    "S1_expert_corrected": {
        "unique_entities": 8019, "all_triples": 48401, "nodes": 1903,
        "edges": 9440, "largest_component_pct": 99.47, "max_degree": 225,
    },
    "S2_majority_harmonised": {
        "unique_entities": 7922, "all_triples": 48978, "nodes": 1946,
        "edges": 9908, "largest_component_pct": 99.23, "max_degree": 250,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", default=str(DATA / "train.txt"),
        help="Authorised local BIO corpus. The public release does not redistribute it.",
    )
    parser.add_argument(
        "--strict-manuscript", action="store_true",
        help="Fail if regenerated headline structural values differ from the revised manuscript.",
    )
    return parser.parse_args()


def compare_to_manuscript(profile_table):
    failures = []
    table = profile_table.set_index("condition")
    for condition, expected in MANUSCRIPT_EXPECTED.items():
        if condition not in table.index:
            failures.append(f"missing condition: {condition}")
            continue
        row = table.loc[condition]
        for key, target in expected.items():
            got = float(row[key]) if isinstance(target, float) else int(row[key])
            ok = abs(got - target) < 1e-9 if isinstance(target, float) else got == target
            if not ok:
                failures.append(f"{condition}:{key}={got} (manuscript {target})")
    return failures


if __name__ == "__main__":
    args = parse_args()
    corpus = Path(args.corpus).expanduser().resolve()
    if not corpus.is_file():
        raise SystemExit(
            "The raw BIO corpus is intentionally not redistributed. "
            "Place an authorised copy at data/train.txt (or pass --corpus PATH) "
            "to rebuild S0/S1/S2 from source."
        )

    samples, n_tokens, repairs = read_samples(corpus)
    mentions = [spans(s) for s in samples]
    print(f"samples {len(samples)}  character tokens {n_tokens}  "
          f"mentions {sum(len(m) for m in mentions)}  label repairs {dict(repairs)}")

    surface = defaultdict(Counter)
    for ments in mentions:
        for t, n in ments:
            surface[n][t] += 1
    coll = {n: c for n, c in surface.items() if len(c) > 1}
    rows = []
    for n, c in sorted(coll.items(), key=lambda kv: -sum(kv[1].values())):
        row = {"name": n, "total_mentions": sum(c.values()),
               "n_types": len(c), "majority_type": c.most_common(1)[0][0]}
        row.update({t: c.get(t, 0) for t in TYPES})
        rows.append(row)
    collisions = pd.DataFrame(rows)
    collisions.to_csv(OUT / "label_collisions.csv", index=False)
    tot_ment = sum(len(m) for m in mentions)
    print(f"multi-type surface forms: {len(coll)} "
          f"({collisions.total_mentions.sum()} mentions, "
          f"{100 * collisions.total_mentions.sum() / tot_ment:.2f}% of all mentions)")

    majority_map = {n: c.most_common(1)[0][0] for n, c in coll.items()}
    conditions = {
        "S0_as_annotated": {},
        "S1_expert_corrected": EXPERT_MAP,
        "S2_majority_harmonised": majority_map,
    }

    prof_rows, hub_rows = [], []
    edge_sets = {}
    for label, remap in conditions.items():
        nodes, edges = build(mentions, remap)
        nodes.to_csv(OUT / f"nodes_{label}.csv", index=False)
        edges.to_csv(OUT / f"edges_{label}.csv", index=False)
        p, G = profile(edges, 2, label)
        p["unique_entities"] = len(nodes)
        prof_rows.append(p)
        hub_rows.append(hubs(G, nodes, label))
        edge_sets[label] = semantic_core_edges(edges)
        print(f"{label:24s} entities {len(nodes):6d}  triples {len(edges):7d}  "
              f"core {p['nodes']:5d}/{p['edges']:6d}  "
              f"LCC {p['largest_component_pct']}%  maxdeg {p['max_degree']}")

    profile_table = pd.DataFrame(prof_rows)
    profile_table.to_csv(OUT / "sensitivity_structure.csv", index=False)
    pd.concat(hub_rows).to_csv(OUT / "sensitivity_hubs.csv", index=False)

    # Keep the manuscript's net-count change distinct from edge membership
    # turnover measured by symmetric semantic set difference.
    s0 = edge_sets["S0_as_annotated"]
    s0_edges = int(profile_table.loc[
        profile_table.condition == "S0_as_annotated", "edges"
    ].iloc[0])
    changes = []
    for condition in ("S1_expert_corrected", "S2_majority_harmonised"):
        current = edge_sets[condition]
        current_edges = int(profile_table.loc[
            profile_table.condition == condition, "edges"
        ].iloc[0])
        added = len(current - s0)
        removed = len(s0 - current)
        changes.append({
            "condition": condition,
            "net_core_edge_count_change": current_edges - s0_edges,
            "abs_net_change": abs(current_edges - s0_edges),
            "abs_net_change_pct_of_S0": round(
                100 * abs(current_edges - s0_edges) / s0_edges, 3
            ),
            "added_semantic_edges": added,
            "removed_semantic_edges": removed,
            "symmetric_difference": added + removed,
            "symmetric_difference_pct_of_S0": round(
                100 * (added + removed) / len(s0), 3
            ),
        })
    pd.DataFrame(changes).to_csv(OUT / "sensitivity_changes_vs_s0.csv", index=False)

    print("\n", profile_table.to_string(index=False))
    print("\nChanges versus S0:")
    print(pd.DataFrame(changes).to_string(index=False))

    failures = compare_to_manuscript(profile_table)
    if failures:
        print("\nMANUSCRIPT ALIGNMENT WARNING:")
        for item in failures:
            print("  -", item)
        if args.strict_manuscript:
            raise SystemExit(2)
    else:
        print("\nHeadline S0/S1/S2 structural values match the revised manuscript.")
