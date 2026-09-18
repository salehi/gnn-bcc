"""Evaluation metrics - regression quality AND ranking quality.

In DEA, DMUs are usually acted on by rank order, so a model can post a
respectable R^2 while scrambling the ordering. Spearman rho and Kendall tau
are reported for exactly that reason.

A note on frontier detection. The DEA-native definition of "efficient" is a
score of exactly 1.0 (29 DMUs here). Applying that threshold to a continuous
regressor is degenerate - a linear-output network essentially never emits
exactly 1.0, so precision/recall would both read 0 and tell you nothing. What
is informative is the *calibration gap*: how far below 1.0 the model places
the DMUs that really are on the frontier. That is reported as
`frontier_pred_mean`, alongside honest near-frontier classification at the
softer 0.95 and 0.90 cuts.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .config import FRONTIER_EPS


def clamp_predictions(y_pred: np.ndarray) -> np.ndarray:
    """A BCC score lives in (0, 1]; predictions outside that are not
    meaningful DEA scores, so they are clipped rather than reported raw."""
    return np.clip(y_pred, 1e-6, 1.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = clamp_predictions(np.asarray(y_pred, dtype=np.float64).ravel())
    err = y_pred - y_true

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))

    out = {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mape": float(np.mean(np.abs(err / y_true)) * 100.0),
        "max_error": float(np.max(np.abs(err))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "bias": float(np.mean(err)),
    }

    # Ranking quality - what DEA practitioners actually use the scores for.
    if len(y_true) > 2:
        out["spearman"] = float(stats.spearmanr(y_true, y_pred).statistic)
        out["kendall"] = float(stats.kendalltau(y_true, y_pred).statistic)
    else:
        out["spearman"] = out["kendall"] = float("nan")

    # Frontier calibration: DMUs that truly sit at 1.0, and where the model
    # puts them. Under-prediction here is expected and worth reporting.
    on_frontier = y_true >= 1.0 - FRONTIER_EPS
    out["frontier_n"] = int(on_frontier.sum())
    if on_frontier.any():
        out["frontier_pred_mean"] = float(y_pred[on_frontier].mean())
        out["frontier_gap"] = float(1.0 - y_pred[on_frontier].mean())
        out["frontier_mae"] = float(np.mean(np.abs(err[on_frontier])))
    else:
        out["frontier_pred_mean"] = out["frontier_gap"] = out["frontier_mae"] = float("nan")

    # Near-frontier detection at operational thresholds.
    for thr in (0.95, 0.90):
        key = f"p{int(thr * 100)}"
        t, p = y_true >= thr, y_pred >= thr
        tp = int((t & p).sum()); fp = int((~t & p).sum()); fn = int((t & ~p).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = (2 * prec * rec / (prec + rec)
              if prec == prec and rec == rec and (prec + rec) > 0 else float("nan"))
        out[f"{key}_precision"], out[f"{key}_recall"], out[f"{key}_f1"] = prec, rec, f1
    return out


def error_by_decile(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10):
    """MAE and signed bias per decile of the true BCC score. Exposes whether
    error concentrates near the efficient frontier."""
    y_true = np.asarray(y_true).ravel()
    y_pred = clamp_predictions(np.asarray(y_pred).ravel())
    edges = np.quantile(y_true, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    labels, maes, biases, counts = [], [], [], []
    for i in range(n_bins):
        m = (y_true >= edges[i]) & (y_true < edges[i + 1])
        if not m.any():
            continue
        labels.append(f"{edges[i]:.2f}-{edges[i+1]:.2f}")
        maes.append(float(np.mean(np.abs(y_pred[m] - y_true[m]))))
        biases.append(float(np.mean(y_pred[m] - y_true[m])))
        counts.append(int(m.sum()))
    return labels, maes, biases, counts


def format_row(name: str, m: dict) -> str:
    return (f"{name:<8s} n={m['n']:<5d} RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  "
            f"R2={m['r2']:+.4f}  rho={m['spearman']:+.4f}  tau={m['kendall']:+.4f}")
