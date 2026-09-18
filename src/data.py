"""Data loading, leakage-safe scaling, stratified splits and STRICT INDUCTIVE
graph construction.

The inductive contract, which the rest of the code depends on:

  * The training graph contains training DMUs only. No held-out node, and no
    statistic derived from one, ever touches training.
  * A held-out DMU is scored by attaching it to its k nearest neighbours
    *among training DMUs only*, with edges pointing INTO it (train -> new).
    Held-out DMUs are never connected to each other, so each one is scored
    exactly as a single unseen bank would be, and val/test stay independent.
  * Because messages only ever flow train -> held-out, training-node
    embeddings are unchanged by the presence of held-out nodes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors

from .config import FEATURES, FRONTIER_EPS, TARGET, Config


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_dataset(cfg: Config) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X_raw [n,4], y [n], feature_names)."""
    df = pd.read_excel(cfg.data_path, sheet_name=cfg.sheet, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(
            f"columns {missing} not found in sheet '{cfg.sheet}'. "
            f"Found: {list(df.columns)}"
        )

    df = df[FEATURES + [TARGET]].dropna()
    X = df[FEATURES].to_numpy(dtype=np.float64)
    y = df[TARGET].to_numpy(dtype=np.float64)

    # Sanity checks tied to the DEA semantics of the file.
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("non-finite values in the data")
    if y.min() <= 0 or y.max() > 1 + FRONTIER_EPS:
        raise ValueError(
            f"BCC scores must lie in (0, 1]; got [{y.min()}, {y.max()}]"
        )
    if X.min() < 0:
        raise ValueError("DEA inputs/outputs must be non-negative")
    return X, y, list(FEATURES)


def target_bins(y: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bins over the target, used to stratify every split so each
    fold carries the full efficiency range including the thin low tail and
    the DMUs on the frontier."""
    edges = np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(y, edges)


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
def make_splits(y: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test indices for the showcase run."""
    strata = target_bins(y, cfg.n_bins)
    idx = np.arange(len(y))

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=cfg.test_size, random_state=cfg.seed
    )
    rest_i, test_i = next(sss.split(idx, strata))

    # val_size is expressed as a fraction of the whole dataset
    val_frac = cfg.val_size / (1.0 - cfg.test_size)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=cfg.seed
    )
    tr_j, val_j = next(sss2.split(rest_i, strata[rest_i]))

    train_i, val_i = rest_i[tr_j], rest_i[val_j]
    assert not (set(train_i) & set(val_i)), "train/val overlap"
    assert not (set(train_i) & set(test_i)), "train/test overlap"
    assert not (set(val_i) & set(test_i)), "val/test overlap"
    return np.sort(train_i), np.sort(val_i), np.sort(test_i)


def make_folds(y: np.ndarray, cfg: Config):
    """Stratified k-fold indices for cross-validation.

    Each fold yields (train, val, test): the held-out fold is the test set and
    a slice of the remaining data becomes the val set used for early stopping,
    so the test fold never influences when training stops.
    """
    strata = target_bins(y, cfg.n_bins)
    idx = np.arange(len(y))
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    for fold, (rest_i, test_i) in enumerate(skf.split(idx, strata)):
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=cfg.val_size, random_state=cfg.seed + fold
        )
        tr_j, val_j = next(sss.split(rest_i, strata[rest_i]))
        yield fold, np.sort(rest_i[tr_j]), np.sort(rest_i[val_j]), np.sort(test_i)


# --------------------------------------------------------------------------
# Scaling  (fit on TRAIN ONLY - the single most important leakage guard)
# --------------------------------------------------------------------------
@dataclass
class LogStandardScaler:
    """log1p, then z-score with statistics estimated on the training fold.

    log1p is used because the four columns span five orders of magnitude and
    are heavily right-skewed. It is monotone, so it preserves the size
    ordering that a VRS (BCC) score genuinely depends on - it compresses
    scale without discarding it.
    """
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> "LogStandardScaler":
        L = np.log1p(X_train)
        self.mean_ = L.mean(axis=0)
        self.std_ = L.std(axis=0)
        self.std_[self.std_ < 1e-12] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("scaler used before fit()")
        return (np.log1p(X) - self.mean_) / self.std_

    def to_dict(self) -> dict:
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist()}


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------
def _dedup(edge_index: np.ndarray) -> np.ndarray:
    if edge_index.size == 0:
        return edge_index.reshape(2, 0)
    key = edge_index[0].astype(np.int64) * (edge_index.max() + 1) + edge_index[1]
    _, keep = np.unique(key, return_index=True)
    return edge_index[:, np.sort(keep)]


def _dominators(Z_ref: np.ndarray, Z_q: np.ndarray, in_cols, out_cols,
                cap: int) -> list[np.ndarray]:
    """For each query DMU, reference DMUs that weakly dominate it: fewer or
    equal inputs AND greater or equal outputs.

    Dominance is exactly the partial order whose Pareto-maximal elements form
    the DEA frontier, so these edges encode the mechanism that produced the
    label rather than a generic similarity heuristic. Monotone per-dimension
    transforms (log1p, z-score) preserve the order, so standardised values
    can be compared directly.
    """
    res = []
    for q in Z_q:
        le_in = np.all(Z_ref[:, in_cols] <= q[in_cols] + 1e-12, axis=1)
        ge_out = np.all(Z_ref[:, out_cols] >= q[out_cols] - 1e-12, axis=1)
        cand = np.flatnonzero(le_in & ge_out)
        if cand.size > cap:  # keep the closest ones
            d = np.linalg.norm(Z_ref[cand] - q, axis=1)
            cand = cand[np.argsort(d)[:cap]]
        res.append(cand)
    return res


def build_train_edges(Z_tr: np.ndarray, cfg: Config,
                      in_cols, out_cols) -> np.ndarray:
    """Symmetric k-NN graph over training DMUs only.

    No self-loops: PyG's SAGEConv already carries a root weight (out =
    W_l * mean(x_neighbours) + W_r * x_self), so adding a self-loop would
    also fold the node's own features into the neighbour aggregate and
    double-count it.
    """
    n = len(Z_tr)
    k = min(cfg.k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z_tr)
    _, idx = nn.kneighbors(Z_tr)
    idx = idx[:, 1:]                                   # drop self

    dst = np.repeat(np.arange(n), k)
    src = idx.reshape(-1)
    ei = np.stack([np.concatenate([src, dst]),         # symmetrise
                   np.concatenate([dst, src])])

    if cfg.edges == "dominance":
        # dominance edges are added ON TOP of k-NN so every node stays
        # connected even when nothing dominates it (i.e. it is on the frontier)
        doms = _dominators(Z_tr, Z_tr, in_cols, out_cols, cap=cfg.k)
        pairs = [(u, i) for i, d in enumerate(doms) for u in d if u != i]
        if pairs:
            arr = np.asarray(pairs, dtype=np.int64).T
            ei = np.concatenate([ei, arr], axis=1)

    return _dedup(ei)


def build_eval_graph(Z_tr: np.ndarray, Z_eval: np.ndarray, cfg: Config,
                     in_cols, out_cols,
                     train_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Graph for scoring held-out DMUs inductively.

    Nodes are [train nodes | held-out nodes]. Held-out node j lives at index
    n_train + j. Edges are the training edges plus, for each held-out node,
    incoming edges from its k nearest TRAINING neighbours. Nothing points out
    of a held-out node, so training representations are untouched and held-out
    nodes cannot see one another.

    Returns (X_combined, edge_index, eval_positions).
    """
    n_tr, n_ev = len(Z_tr), len(Z_eval)
    k = min(cfg.k, n_tr)

    nn = NearestNeighbors(n_neighbors=k).fit(Z_tr)
    _, idx = nn.kneighbors(Z_eval)                     # train neighbours only

    src = idx.reshape(-1)
    dst = n_tr + np.repeat(np.arange(n_ev), k)
    add = np.stack([src, dst])

    if cfg.edges == "dominance":
        doms = _dominators(Z_tr, Z_eval, in_cols, out_cols, cap=cfg.k)
        pairs = [(u, n_tr + i) for i, d in enumerate(doms) for u in d]
        if pairs:
            arr = np.asarray(pairs, dtype=np.int64).T
            add = np.concatenate([add, arr], axis=1)

    ei = _dedup(np.concatenate([train_edges, add], axis=1))
    X = np.concatenate([Z_tr, Z_eval], axis=0)
    eval_pos = np.arange(n_tr, n_tr + n_ev)

    # Contract check: nothing may originate from a held-out node, so a
    # held-out DMU can never influence a training representation.
    assert bool((ei[0] < n_tr).all()), \
        "inductive violation: a held-out node is a message source"
    return X, ei, eval_pos


def to_tensor(x: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype)


def edges_to_tensor(ei: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(ei), dtype=torch.long)
