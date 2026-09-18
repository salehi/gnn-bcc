"""5-fold strict-inductive cross-validation.

Each fold rebuilds everything from scratch - scaler, k-NN graph, model - from
that fold's training DMUs only, so the held-out fold is never involved in
fitting, scaling or graph construction. The showcase split in train.py gives
the pretty training curves; these are the numbers to actually quote.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import plots
from .config import Config, build_parser, config_from_args
from .data import load_dataset, make_folds
from .metrics import compute_metrics, format_row
from .train import fit, figdir, predict_heldout, suffix

AGG_KEYS = ("rmse", "mae", "mape", "r2", "spearman", "kendall", "max_error",
            "bias", "frontier_gap", "p95_precision", "p95_recall", "p95_f1")


def aggregate(fold_metrics: list[dict]) -> dict:
    mean, std = {}, {}
    for k in AGG_KEYS:
        v = np.array([m.get(k, np.nan) for m in fold_metrics], dtype=float)
        mean[k] = float(np.nanmean(v))
        std[k] = float(np.nanstd(v))
    return {"mean": mean, "std": std, "folds": fold_metrics}


def run_cv(cfg: Config, X, y, verbose: bool = True) -> dict:
    fold_metrics = []
    for fold, tr_i, va_i, te_i in make_folds(y, cfg):
        fcfg = Config(**{**cfg.to_dict(), "hidden_dims": tuple(cfg.hidden_dims),
                         "seed": cfg.seed + fold})
        model, b, hist, info = fit(fcfg, X, y, tr_i, va_i, te_i,
                                   live=False, verbose=False)
        m = compute_metrics(y[te_i], predict_heldout(model, b, "test"))
        m["fold"] = fold + 1
        m["best_epoch"] = info["best_epoch"]
        m["seconds"] = info["seconds"]
        fold_metrics.append(m)
        if verbose:
            print("  " + format_row(f"fold {fold+1}", m) +
                  f"  [{info['epochs_run']} ep, {info['seconds']}s]")
    return aggregate(fold_metrics)


def main():
    p = build_parser("5-fold strict-inductive cross-validation.")
    cfg = config_from_args(p.parse_args())

    print("=" * 78)
    print(f"  {cfg.n_folds}-fold strict-inductive cross-validation  |  "
          f"{cfg.model} {list(cfg.hidden_dims)}  k={cfg.k}")
    print("=" * 78)

    X, y, names = load_dataset(cfg)
    if cfg.shuffle_labels:
        y = np.random.default_rng(cfg.seed).permutation(y)
        print("  !! --shuffle-labels active: R2 must come out ~ 0\n")

    res = run_cv(cfg, X, y)

    print("-" * 78)
    m, s = res["mean"], res["std"]
    print(f"  RMSE      {m['rmse']:.4f} +/- {s['rmse']:.4f}")
    print(f"  MAE       {m['mae']:.4f} +/- {s['mae']:.4f}")
    print(f"  MAPE      {m['mape']:.2f}% +/- {s['mape']:.2f}%")
    print(f"  R2        {m['r2']:+.4f} +/- {s['r2']:.4f}")
    print(f"  Spearman  {m['spearman']:+.4f} +/- {s['spearman']:.4f}")
    print(f"  Kendall   {m['kendall']:+.4f} +/- {s['kendall']:.4f}")
    print(f"  frontier under-prediction gap  {m['frontier_gap']:+.4f} "
          f"+/- {s['frontier_gap']:.4f}")
    print("-" * 78)

    sfx = suffix(cfg)
    with open(os.path.join(cfg.out_dir, f"cv_results{sfx}.json"), "w") as f:
        json.dump({"config": cfg.to_dict(), "cv": res}, f, indent=2)
    path = plots.plot_cv_results(res["folds"],
                                 f"{figdir(cfg)}/08_cv_results{sfx}.png")
    print(f"\n  {os.path.relpath(path, cfg.out_dir)}")
    print(f"  cv_results{sfx}.json")


if __name__ == "__main__":
    main()
