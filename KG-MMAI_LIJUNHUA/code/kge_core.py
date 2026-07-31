"""Shared data preparation, KGE models, training, and evaluation utilities.

The functions in this module are used by the link-prediction, robustness, and
statistical-analysis scripts. Keeping them here avoids dynamic code execution
and ensures that all stages use the same split and scoring implementation.
"""

from __future__ import annotations

__author__ = "LIJUNHUA"

from pathlib import Path

import numpy as np
import pandas as pd

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
DIM, EPOCHS, N_NEG, BATCH = 64, 20, 8, 256
LR, MARGIN, ADV_TEMP = 0.01, 9.0, 1.0
SEEDS = (42, 1337, 2024)
MODELS = ("TransE", "DistMult", "ComplEx", "RotatE")

# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
edges = pd.read_csv(DATA / "edges.csv", encoding="utf-8-sig")
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
for i, t in etype.items():
    type_members.setdefault(t, []).append(i)
type_members = {t: np.array(sorted(v)) for t, v in type_members.items()}

triples = np.array([[e2i[r.source_id], r2i[r.relation], e2i[r.target_id]]
                    for r in core.itertuples(index=False)], dtype=np.int64)
N_E, N_R = len(ents), len(rels)


def make_split(seed=20240101):
    """80/10/10 stratified by relation, repaired for entity coverage."""
    rng = np.random.default_rng(seed)
    tag = np.empty(len(triples), dtype=object)
    for r in range(N_R):
        idx = np.where(triples[:, 1] == r)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr, n_va = int(round(0.8 * n)), int(round(0.1 * n))
        tag[idx[:n_tr]] = "train"
        tag[idx[n_tr:n_tr + n_va]] = "valid"
        tag[idx[n_tr + n_va:]] = "test"

    # repair: every entity appearing in valid/test must appear in train
    seen = set(triples[tag == "train"][:, [0, 2]].ravel().tolist())
    for i in np.where(tag != "train")[0]:
        h, _, t = triples[i]
        if h not in seen or t not in seen:
            tag[i] = "train"
            seen.update((int(h), int(t)))
    return tag


TAG = make_split()
tr = triples[TAG == "train"]
va = triples[TAG == "valid"]
te = triples[TAG == "test"]

def write_split_table(path=RES / "splits.csv"):
    """Write the deterministic split assignment used by the experiments."""
    pd.DataFrame({
        "source_id": core.source_id,
        "relation": core.relation,
        "target_id": core.target_id,
        "weight": core.weight,
        "split": TAG,
    }).to_csv(path, index=False)
    return path

ALL_TRUE = set(map(tuple, triples.tolist()))

# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
class KGE:
    """Score functions and gradients, all vectorised over (batch, n_cand)."""

    def __init__(self, name, rng):
        self.name = name
        d = DIM
        scale = 6.0 / np.sqrt(d)
        self.E = rng.uniform(-scale, scale, (N_E, d))
        self.R = rng.uniform(-scale, scale, (N_R, d))
        if name == "RotatE":                       # phases in [-pi, pi]
            self.R = rng.uniform(-np.pi, np.pi, (N_R, d // 2))
        self.mE = np.zeros_like(self.E); self.vE = np.zeros_like(self.E)
        self.mR = np.zeros_like(self.R); self.vR = np.zeros_like(self.R)
        self.t = 0

    # ---- scoring -------------------------------------------------------- #
    def score(self, h, r, t):
        """h,t: (...,d)  r: (...,d) or (...,d/2) -> (...)"""
        if self.name == "TransE":
            return MARGIN - np.linalg.norm(h + r - t, ord=1, axis=-1)
        if self.name == "DistMult":
            return np.sum(h * r * t, axis=-1)
        if self.name == "ComplEx":
            d = DIM // 2
            hr, hi = h[..., :d], h[..., d:]
            rr, ri = r[..., :d], r[..., d:]
            tr_, ti = t[..., :d], t[..., d:]
            return np.sum(hr * rr * tr_ + hr * ri * ti
                          + hi * rr * ti - hi * ri * tr_, axis=-1)
        if self.name == "RotatE":
            d = DIM // 2
            hr, hi = h[..., :d], h[..., d:]
            tr_, ti = t[..., :d], t[..., d:]
            cr, ci = np.cos(r), np.sin(r)
            re = hr * cr - hi * ci - tr_
            im = hr * ci + hi * cr - ti
            return MARGIN - np.sum(np.sqrt(re ** 2 + im ** 2 + 1e-12), axis=-1)
        raise ValueError(self.name)

    # ---- numeric gradient-free update via analytic grads ----------------- #
    def grads(self, h, r, t, g):
        """g: (...) upstream gradient of the loss wrt the score.
        Returns dh, dr, dt with the same shapes as h, r, t."""
        g = g[..., None]
        if self.name == "TransE":
            s = np.sign(h + r - t)
            return -g * s, -g * s, g * s
        if self.name == "DistMult":
            return g * r * t, g * h * t, g * h * r
        if self.name == "ComplEx":
            d = DIM // 2
            hr, hi = h[..., :d], h[..., d:]
            rr, ri = r[..., :d], r[..., d:]
            tr_, ti = t[..., :d], t[..., d:]
            dhr = rr * tr_ + ri * ti
            dhi = rr * ti - ri * tr_
            drr = hr * tr_ + hi * ti
            dri = hr * ti - hi * tr_
            dtr = hr * rr - hi * ri
            dti = hr * ri + hi * rr
            cat = lambda a, b: np.concatenate([a, b], axis=-1)
            return g * cat(dhr, dhi), g * cat(drr, dri), g * cat(dtr, dti)
        if self.name == "RotatE":
            d = DIM // 2
            hr, hi = h[..., :d], h[..., d:]
            tr_, ti = t[..., :d], t[..., d:]
            cr, ci = np.cos(r), np.sin(r)
            re = hr * cr - hi * ci - tr_
            im = hr * ci + hi * cr - ti
            nrm = np.sqrt(re ** 2 + im ** 2 + 1e-12)
            ur, ui = re / nrm, im / nrm            # d(norm)/d(re), d(im)
            dhr = -(ur * cr + ui * ci)
            dhi = -(-ur * ci + ui * cr)
            dtr, dti = ur, ui
            drot = -(ur * (-hr * ci - hi * cr) + ui * (hr * cr - hi * ci))
            cat = lambda a, b: np.concatenate([a, b], axis=-1)
            return g * cat(dhr, dhi), g * drot, g * cat(dtr, dti)
        raise ValueError(self.name)

    # ---- Adam ------------------------------------------------------------ #
    def step(self, idxE, gE, idxR, gR, lr=LR, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for idx, g, P, m, v in ((idxE, gE, self.E, self.mE, self.vE),
                                (idxR, gR, self.R, self.mR, self.vR)):
            acc = np.zeros_like(P)
            np.add.at(acc, idx, g)
            m *= b1; m += (1 - b1) * acc
            v *= b2; v += (1 - b2) * acc ** 2
            mh = m / (1 - b1 ** self.t)
            vh = v / (1 - b2 ** self.t)
            P -= lr * mh / (np.sqrt(vh) + eps)


def sample_negatives(batch, rng):
    """8 typed corruptions per positive: half head-side, half tail-side."""
    B = len(batch)
    negs = np.repeat(batch[:, None, :], N_NEG, axis=1).copy()
    corrupt_head = rng.random((B, N_NEG)) < 0.5
    for pos in range(B):
        h, r, t = batch[pos]
        htype, ttype = SCHEMA[rels[r]]
        hcand, tcand = type_members[htype], type_members[ttype]
        for k in range(N_NEG):
            if corrupt_head[pos, k]:
                negs[pos, k, 0] = hcand[rng.integers(len(hcand))]
            else:
                negs[pos, k, 2] = tcand[rng.integers(len(tcand))]
    return negs


def train_epochs(model_name, seed, epochs, curve=None, eval_every=1):
    rng = np.random.default_rng(seed)
    M = KGE(model_name, rng)
    order = np.arange(len(tr))
    for ep in range(epochs):
        rng.shuffle(order)
        for s in range(0, len(order), BATCH):
            b = tr[order[s:s + BATCH]]
            if len(b) < 2:
                continue
            negs = sample_negatives(b, rng)

            hp, rp, tp = M.E[b[:, 0]], M.R[b[:, 1]], M.E[b[:, 2]]
            hn, rn, tn = M.E[negs[..., 0]], M.R[negs[..., 1]], M.E[negs[..., 2]]
            sp = M.score(hp, rp, tp)                     # (B,)
            sn = M.score(hn, rn, tn)                     # (B, N_NEG)

            # self-adversarial weighting of negatives (RotatE, Sun et al.)
            w = np.exp(ADV_TEMP * (sn - sn.max(axis=1, keepdims=True)))
            w /= w.sum(axis=1, keepdims=True)

            gp = -(1.0 / (1.0 + np.exp(sp)))             # d/ds -logsigmoid(s)
            gn = w * (1.0 / (1.0 + np.exp(-sn)))         # d/ds -logsigmoid(-s)

            dhp, drp, dtp = M.grads(hp, rp, tp, gp)
            dhn, drn, dtn = M.grads(hn, rn, tn, gn)

            idxE = np.concatenate([b[:, 0], b[:, 2],
                                   negs[..., 0].ravel(), negs[..., 2].ravel()])
            gEc = np.concatenate([dhp, dtp,
                                  dhn.reshape(-1, dhn.shape[-1]),
                                  dtn.reshape(-1, dtn.shape[-1])])
            idxR = np.concatenate([b[:, 1], negs[..., 1].ravel()])
            gRc = np.concatenate([drp, drn.reshape(-1, drn.shape[-1])])
            M.step(idxE, gEc, idxR, gRc)

        if curve is not None and (ep + 1) % eval_every == 0:
            curve.append({
                "model": model_name,
                "seed": seed,
                "epoch": ep + 1,
                "valid_mrr": round(evaluate(M, va)["MRR"], 5),
            })
    return M


def train(model_name, seed, curve=None):
    """Train one model with the default 20-epoch configuration."""
    return train_epochs(model_name, seed, EPOCHS, curve=curve, eval_every=1)


def evaluate(M, split, per_relation=False):
    ranks, rel_of = [], []
    for h, r, t in split:
        htype, ttype = SCHEMA[rels[r]]
        # ---- tail ranking ----
        cand = type_members[ttype]
        sc = M.score(M.E[h][None, :], M.R[r][None, :], M.E[cand])
        true = M.score(M.E[h], M.R[r], M.E[t])
        bad = np.array([c != t and (h, r, int(c)) in ALL_TRUE for c in cand])
        ranks.append(1 + int(((sc > true) & ~bad).sum()))
        rel_of.append(r)
        # ---- head ranking ----
        cand = type_members[htype]
        sc = M.score(M.E[cand], M.R[r][None, :], M.E[t][None, :])
        bad = np.array([c != h and (int(c), r, t) in ALL_TRUE for c in cand])
        ranks.append(1 + int(((sc > true) & ~bad).sum()))
        rel_of.append(r)

    ranks = np.asarray(ranks, dtype=float)
    out = {
        "MRR": float((1 / ranks).mean()),
        "Hits@1": float((ranks <= 1).mean()),
        "Hits@3": float((ranks <= 3).mean()),
        "Hits@10": float((ranks <= 10).mean()),
        "MedianRank": float(np.median(ranks)),
        "queries": int(len(ranks)),
    }
    if per_relation:
        rel_of = np.asarray(rel_of)
        out["per_relation"] = {
            rels[r]: {
                "queries": int((rel_of == r).sum()),
                "MRR": float((1 / ranks[rel_of == r]).mean()),
                "Hits@10": float((ranks[rel_of == r] <= 10).mean()),
            } for r in range(N_R)
        }
    return out


