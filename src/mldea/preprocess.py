"""Preprocessing: zero correction, Box-Cox skewness correction, stratified
splits, and the three normalisations under comparison.

Order of operations, and why:

  1. load the four DEA columns and the BCC score
  2. correct the single zero in `net profit`      (Box-Cox needs x > 0)
  3. Box-Cox per feature                          (skew 6.7-12.3 -> ~0)
  4. stratified train / val / test split          (quantile bins of the target)
  5. fit the normaliser ON THE TRAINING SPLIT ONLY, then apply it everywhere

Step 5 is the leakage guard that matters: the scaler's centre and spread are
estimated from training DMUs alone, so no test DMU influences the numbers any
model is fitted on. Step 3 is run on the full feature matrix by default
because that is the specified pipeline; it is unsupervised (the BCC score is
never involved) and `--boxcox-fit train` re-runs it train-only as a check.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import special, stats
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from ..config import FEATURES, TARGET
from ..data import load_dataset, target_bins
from .config import NORMALISERS, MLConfig

NET_PROFIT = "net profit"
NET_PROFIT_COL = FEATURES.index(NET_PROFIT)


# --------------------------------------------------------------------------
# 2. the zero in `net profit`
# --------------------------------------------------------------------------
def fix_zeros(X: np.ndarray, col: int, mode: str, shift_frac: float) -> tuple[np.ndarray, dict]:
    """Make column `col` strictly positive. Returns (X_fixed, provenance)."""
    x = X[:, col].astype(np.float64)
    bad = x <= 0
    pos = x[x > 0]
    if pos.size == 0:
        raise ValueError(f"column {col} has no positive values")
    info = {
        "column": FEATURES[col],
        "mode": mode,
        "n_nonpositive": int(bad.sum()),
        "min_positive": float(pos.min()),
        "rows": np.flatnonzero(bad).tolist(),
    }
    if not bad.any():
        info["applied"] = False
        return X.copy(), info

    Xf = X.copy()
    if mode == "halfmin":
        fill = float(pos.min() / 2.0)
        Xf[bad, col] = fill
        info["replacement_value"] = fill
    elif mode == "shift":
        eps = float(pos.min() * shift_frac)
        Xf[:, col] = x + eps
        info["shift_added_to_whole_column"] = eps
        info["replacement_value"] = eps
    else:
        raise ValueError(f"unknown zero_fix mode {mode!r}")
    info["applied"] = True
    return Xf, info


# --------------------------------------------------------------------------
# 3. Box-Cox
# --------------------------------------------------------------------------
@dataclass
class BoxCox:
    """Per-feature Box-Cox with lambdas by maximum likelihood.

    Monotone on x > 0, so it compresses the five-order-of-magnitude spread of
    these columns without reordering any DMU - the size ordering a VRS score
    genuinely depends on survives the transform.
    """
    lambdas_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "BoxCox":
        if X.min() <= 0:
            raise ValueError("Box-Cox requires strictly positive values; "
                             "run fix_zeros() first")
        self.lambdas_ = np.array(
            [stats.boxcox_normmax(X[:, j], method="mle") for j in range(X.shape[1])],
            dtype=np.float64)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.lambdas_ is None:
            raise RuntimeError("BoxCox used before fit()")
        out = np.column_stack(
            [special.boxcox(X[:, j], self.lambdas_[j]) for j in range(X.shape[1])])
        if not np.all(np.isfinite(out)):
            raise ValueError("Box-Cox produced non-finite values")
        return out

    def to_dict(self) -> dict:
        return {"lambdas": self.lambdas_.tolist()}


def skewness(X: np.ndarray) -> list[float]:
    return [float(stats.skew(X[:, j])) for j in range(X.shape[1])]


def kurtosis(X: np.ndarray) -> list[float]:
    return [float(stats.kurtosis(X[:, j])) for j in range(X.shape[1])]


# --------------------------------------------------------------------------
# 4. splits
# --------------------------------------------------------------------------
def make_splits(y: np.ndarray, cfg: MLConfig):
    """Stratified train / val / test over quantile bins of the BCC score, so
    every split carries the full efficiency range including the thin low tail
    and the DMUs sitting on the frontier."""
    strata = target_bins(y, cfg.n_bins)
    idx = np.arange(len(y))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg.test_size,
                                 random_state=cfg.seed)
    rest_i, test_i = next(sss.split(idx, strata))
    val_frac = cfg.val_size / (1.0 - cfg.test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac,
                                  random_state=cfg.seed)
    tr_j, val_j = next(sss2.split(rest_i, strata[rest_i]))
    train_i, val_i = rest_i[tr_j], rest_i[val_j]

    assert not (set(train_i) & set(val_i)), "train/val overlap"
    assert not (set(train_i) & set(test_i)), "train/test overlap"
    assert not (set(val_i) & set(test_i)), "val/test overlap"
    return np.sort(train_i), np.sort(val_i), np.sort(test_i)


# --------------------------------------------------------------------------
# 5. the three normalisations
# --------------------------------------------------------------------------
def make_normaliser(kind: str):
    if kind == "zscore":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()
    if kind == "robust":
        return RobustScaler()          # median / IQR
    raise ValueError(f"unknown normaliser {kind!r}")


@dataclass
class Prepared:
    X_raw: np.ndarray
    X_fixed: np.ndarray
    X_bc: np.ndarray
    y: np.ndarray
    names: list[str]
    train_i: np.ndarray
    val_i: np.ndarray
    test_i: np.ndarray
    boxcox: BoxCox | None
    views: dict = field(default_factory=dict)      # norm -> dict of matrices
    provenance: dict = field(default_factory=dict)

    def split_sizes(self) -> dict:
        return {"train": int(len(self.train_i)), "val": int(len(self.val_i)),
                "test": int(len(self.test_i))}


def prepare(cfg: MLConfig, norms=NORMALISERS) -> Prepared:
    X_raw, y, names = load_dataset(cfg)

    X_fixed, zinfo = fix_zeros(X_raw, NET_PROFIT_COL, cfg.zero_fix, cfg.shift_frac)
    train_i, val_i, test_i = make_splits(y, cfg)

    if cfg.boxcox:
        bc = BoxCox().fit(X_fixed if cfg.boxcox_fit == "all" else X_fixed[train_i])
        X_bc = bc.transform(X_fixed)
    else:
        bc, X_bc = None, X_fixed.copy()

    prov = {
        "zero_fix": zinfo,
        "boxcox": {"applied": cfg.boxcox, "fit_on": cfg.boxcox_fit,
                   **(bc.to_dict() if bc else {})},
        "skew_before": skewness(X_fixed),
        "skew_after": skewness(X_bc),
        "kurtosis_before": kurtosis(X_fixed),
        "kurtosis_after": kurtosis(X_bc),
        "target": {"name": TARGET, "min": float(y.min()), "max": float(y.max()),
                   "mean": float(y.mean()), "std": float(y.std()),
                   "skew": float(stats.skew(y))},
    }

    p = Prepared(X_raw=X_raw, X_fixed=X_fixed, X_bc=X_bc, y=y, names=names,
                 train_i=train_i, val_i=val_i, test_i=test_i, boxcox=bc,
                 provenance=prov)

    for kind in norms:
        sc = make_normaliser(kind).fit(X_bc[train_i])          # TRAIN ONLY
        Z = sc.transform(X_bc)
        p.views[kind] = {
            "scaler": sc,
            "Z": Z,
            "Ztr": Z[train_i], "Zva": Z[val_i], "Zte": Z[test_i],
            "ytr": y[train_i], "yva": y[val_i], "yte": y[test_i],
        }
    return p
