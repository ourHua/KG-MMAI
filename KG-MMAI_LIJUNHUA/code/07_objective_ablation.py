#!/usr/bin/env python3
"""Controlled ablation of the training objective within a single code base.

Reviewer-requested experiment. Configurations A (PyTorch, margin-based) and B
(NumPy, self-adversarial logistic) reported in the original manuscript differed
in two respects: the training objective and the implementation environment.
The ordering inversion observed between them is therefore confounded.

This script removes the confound. It holds the code base, data, deterministic
split, typed negative sampler, random-number stream, embedding dimension,
optimiser, learning rate, batch size, seeds, and filtered typed evaluation
protocol fixed, and varies only the loss function.

Outputs (results/ablation/):
    objective_ablation_per_seed.csv
    objective_ablation_summary.csv
    ablation_per_query_ranks.npz
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__author__ = "LIJUNHUA"

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kge_core as _kc  # noqa: E402


def _configure_edge_source_from_env():
    """Optionally rebuild the shared KGE context from ``KG_EDGES``.

    The default remains ``data/edges.csv``. The override is used for the S1/S2
    annotation-sensitivity re-runs so that only the graph changes while the
    split, model implementation, sampler, optimiser, and evaluation stay fixed.
    """
    raw = os.environ.get("KG_EDGES")
    if not raw:
        return
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (_kc.ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"KG_EDGES does not exist: {path}")

    edges = pd.read_csv(path, encoding="utf-8-sig")
    core = edges[edges.weight >= 2].reset_index(drop=True)
    ents = sorted(set(core.source_id) | set(core.target_id))
    e2i = {e: i for i, e in enumerate(ents)}
    rels = sorted(core.relation.unique())
    r2i = {r: i for i, r in enumerate(rels)}

    etype = {}
    for row in core.itertuples(index=False):
        etype[e2i[row.source_id]] = row.source_type
        etype[e2i[row.target_id]] = row.target_type
    type_members = {}
    for i, typ in etype.items():
        type_members.setdefault(typ, []).append(i)
    type_members = {typ: np.array(sorted(v)) for typ, v in type_members.items()}

    triples = np.array(
        [[e2i[r.source_id], r2i[r.relation], e2i[r.target_id]]
         for r in core.itertuples(index=False)],
        dtype=np.int64,
    )

    _kc.edges, _kc.core = edges, core
    _kc.ents, _kc.e2i, _kc.rels, _kc.r2i = ents, e2i, rels, r2i
    _kc.etype, _kc.type_members = etype, type_members
    _kc.triples, _kc.N_E, _kc.N_R = triples, len(ents), len(rels)
    _kc.TAG = _kc.make_split()
    _kc.tr = triples[_kc.TAG == "train"]
    _kc.va = triples[_kc.TAG == "valid"]
    _kc.te = triples[_kc.TAG == "test"]
    _kc.ALL_TRUE = set(map(tuple, triples.tolist()))
    print(f"using edge table: {path}", flush=True)


_configure_edge_source_from_env()

from kge_core import (  # noqa: E402
    ALL_TRUE, BATCH, KGE, MODELS, N_NEG, RES, SCHEMA, SEEDS,
    rels, te, tr, type_members,
)


def sample_negatives_fast(batch, rng):
    """Vectorised typed sampler used identically by every objective."""
    n_batch = len(batch)
    negs = np.repeat(batch[:, None, :], N_NEG, axis=1).copy()
    corrupt_head = rng.random((n_batch, N_NEG)) < 0.5
    r_ids = batch[:, 1]
    for r in np.unique(r_ids):
        idx = np.where(r_ids == r)[0]
        htype, ttype = SCHEMA[rels[r]]
        hc, tc = type_members[htype], type_members[ttype]
        ch = corrupt_head[idx]
        hsel = hc[rng.integers(len(hc), size=(len(idx), N_NEG))]
        tsel = tc[rng.integers(len(tc), size=(len(idx), N_NEG))]
        blk = negs[idx]
        blk[..., 0] = np.where(ch, hsel, blk[..., 0])
        blk[..., 2] = np.where(ch, blk[..., 2], tsel)
        negs[idx] = blk
    return negs


def build_eval_index(split):
    """Precompute typed, filtered candidate sets shared by all evaluations."""
    idx = []
    for j, (h, r, t) in enumerate(split):
        htype, ttype = SCHEMA[rels[r]]
        tc = type_members[ttype]
        badt = np.array([c != t and (h, r, int(c)) in ALL_TRUE for c in tc])
        hc = type_members[htype]
        badh = np.array([c != h and (int(c), r, t) in ALL_TRUE for c in hc])
        idx.append((int(h), int(r), int(t), tc[~badt], hc[~badh], j))
    return idx


EVAL_INDEX = build_eval_index(te)


def evaluate_fast(model):
    """Return metrics, per-query ranks, and clustering metadata."""
    ranks, rel_of, trp_of, side_of, nc_of, head_of = [], [], [], [], [], []
    for h, r, t, tc, hc, j in EVAL_INDEX:
        true = model.score(model.E[h], model.R[r], model.E[t])
        sc = model.score(model.E[h][None, :], model.R[r][None, :], model.E[tc])
        ranks.append(1 + int((sc > true).sum()))
        rel_of.append(r); trp_of.append(j); side_of.append(1)
        nc_of.append(len(tc)); head_of.append(h)

        sc = model.score(model.E[hc], model.R[r][None, :], model.E[t][None, :])
        ranks.append(1 + int((sc > true).sum()))
        rel_of.append(r); trp_of.append(j); side_of.append(0)
        nc_of.append(len(hc)); head_of.append(h)

    ranks = np.asarray(ranks, dtype=float)
    metrics = {
        "MRR": float((1 / ranks).mean()),
        "Hits@1": float((ranks <= 1).mean()),
        "Hits@3": float((ranks <= 3).mean()),
        "Hits@10": float((ranks <= 10).mean()),
        "MedianRank": float(np.median(ranks)),
        "queries": int(len(ranks)),
    }
    meta = (
        np.asarray(rel_of), np.asarray(trp_of), np.asarray(side_of),
        np.asarray(nc_of, dtype=float), np.asarray(head_of, dtype=int),
    )
    return metrics, ranks, meta


OBJECTIVES = ("margin", "logistic", "selfadv")
BUDGETS = (20, 60)
GAMMA_MARGIN = 1.0
ADV_TEMP = 1.0
OUT = RES / "ablation"
OUT.mkdir(parents=True, exist_ok=True)


def grad_margin(sp, sn):
    viol = (GAMMA_MARGIN - (sp[:, None] - sn)) > 0
    gp = -viol.sum(axis=1) / N_NEG
    gn = viol.astype(float) / N_NEG
    return gp, gn


def grad_logistic(sp, sn):
    gp = -(1.0 / (1.0 + np.exp(sp)))
    gn = (1.0 / N_NEG) * (1.0 / (1.0 + np.exp(-sn)))
    return gp, gn


def grad_selfadv(sp, sn):
    w = np.exp(ADV_TEMP * (sn - sn.max(axis=1, keepdims=True)))
    w /= w.sum(axis=1, keepdims=True)
    gp = -(1.0 / (1.0 + np.exp(sp)))
    gn = w * (1.0 / (1.0 + np.exp(-sn)))
    return gp, gn


GRADS = {"margin": grad_margin, "logistic": grad_logistic, "selfadv": grad_selfadv}


def train_objective(model_name, seed, epochs, objective):
    """Train one model while changing only the requested loss gradient."""
    grad_fn = GRADS[objective]
    rng = np.random.default_rng(seed)
    model = KGE(model_name, rng)
    order = np.arange(len(tr))
    for _ in range(epochs):
        rng.shuffle(order)
        for s in range(0, len(order), BATCH):
            batch = tr[order[s:s + BATCH]]
            if len(batch) < 2:
                continue
            negs = sample_negatives_fast(batch, rng)

            hp, rp, tp = model.E[batch[:, 0]], model.R[batch[:, 1]], model.E[batch[:, 2]]
            hn, rn, tn = model.E[negs[..., 0]], model.R[negs[..., 1]], model.E[negs[..., 2]]
            sp = model.score(hp, rp, tp)
            sn = model.score(hn, rn, tn)
            gp, gn = grad_fn(sp, sn)

            dhp, drp, dtp = model.grads(hp, rp, tp, gp)
            dhn, drn, dtn = model.grads(hn, rn, tn, gn)
            idx_e = np.concatenate([
                batch[:, 0], batch[:, 2], negs[..., 0].ravel(), negs[..., 2].ravel()
            ])
            grad_e = np.concatenate([
                dhp, dtp, dhn.reshape(-1, dhn.shape[-1]), dtn.reshape(-1, dtn.shape[-1])
            ])
            idx_r = np.concatenate([batch[:, 1], negs[..., 1].ravel()])
            grad_r = np.concatenate([drp, drn.reshape(-1, drn.shape[-1])])
            model.step(idx_e, grad_e, idx_r, grad_r)
    return model


def main():
    rows, store = [], {}
    meta_saved = False
    for objective in OBJECTIVES:
        for budget in BUDGETS:
            for name in MODELS:
                for seed in SEEDS:
                    model = train_objective(name, seed, budget, objective)
                    metrics, ranks, meta = evaluate_fast(model)
                    rows.append({
                        "objective": objective,
                        "budget_epochs": budget,
                        "model": name,
                        "seed": seed,
                        **metrics,
                    })
                    print(
                        f"  {objective:9s} {budget:2d}ep {name:9s} "
                        f"seed {seed:5d} MRR {metrics['MRR']:.4f} "
                        f"H@10 {metrics['Hits@10']:.4f}",
                        flush=True,
                    )
                    if budget == max(BUDGETS):
                        store[f"{objective}|{name}|{seed}"] = ranks
                        if not meta_saved:
                            rel, trp, side, nc, head = meta
                            store.update({
                                "_relation": rel,
                                "_triple": trp,
                                "_side": side,
                                "_ncand": nc,
                                "_head": head,
                            })
                            meta_saved = True

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT / "objective_ablation_per_seed.csv", index=False)
    summary = (
        per_seed.groupby(["objective", "budget_epochs", "model"])
        .agg(
            MRR_mean=("MRR", "mean"), MRR_sd=("MRR", "std"),
            H1_mean=("Hits@1", "mean"), H3_mean=("Hits@3", "mean"),
            H10_mean=("Hits@10", "mean"), H10_sd=("Hits@10", "std"),
            Med=("MedianRank", "mean"),
        )
        .round(4)
        .reset_index()
    )
    summary["rank_in_cell"] = (
        summary.groupby(["objective", "budget_epochs"])
        .MRR_mean.rank(ascending=False).astype(int)
    )
    summary = summary.sort_values(["objective", "budget_epochs", "rank_in_cell"])
    summary.to_csv(OUT / "objective_ablation_summary.csv", index=False)
    np.savez_compressed(OUT / "ablation_per_query_ranks.npz", **store)
    print("\nControlled objective ablation (single code base):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
