"""Run the twelve ML-DEA combinations and write every artefact.

  python -m src.mldea.run              # all 4 methods x 3 normalisations
  python -m src.mldea.run --methods BPNN-DEA --norms zscore

Writes into outputs/ml_dea/:
  metrics.json                  every number, plus provenance and config
  results.csv                   one row per (method, normalisation)
  history_<method>_<norm>.csv   per-iteration loss / learning curves
  figures/*.png                 preprocessing, curves, predictions, comparison
  report.md                     the write-up, assembled from metrics.json
"""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np
from scipy import stats
from joblib import Parallel, delayed
from sklearn.base import clone

from ..metrics import clamp_predictions, compute_metrics
from . import nn, plots, svm
from .config import (ITERATIVE, MLConfig, build_parser, config_from_args,
                     selected)
from .preprocess import prepare


# --------------------------------------------------------------------------
def evaluate(y_true: np.ndarray, y_pred_raw: np.ndarray) -> dict:
    """Regression + ranking metrics on DEA-valid (clipped) predictions.

    Predictions are clipped into (0, 1] first, because a BCC score outside
    that interval is not a score. `mse_raw` keeps the unclipped figure so the
    clipping can never be mistaken for a way of hiding a bad model.
    """
    y_true = np.asarray(y_true, float).ravel()
    y_raw = np.asarray(y_pred_raw, float).ravel()
    y_pred = clamp_predictions(y_raw)

    m = compute_metrics(y_true, y_raw)
    m["mse"] = float(np.mean((y_pred - y_true) ** 2))
    m["mse_raw"] = float(np.mean((y_raw - y_true) ** 2))
    m["n_clipped"] = int(np.sum(y_raw != y_pred))
    r = stats.pearsonr(y_true, y_pred)
    m["pearson_r"] = float(r.statistic)
    m["pearson_p"] = float(r.pvalue)
    m["pearson_r2"] = float(r.statistic ** 2)
    return m


def with_restarts(trainer, cfg: MLConfig, v: dict) -> dict:
    """Run `trainer` from n_restarts independent seeds and keep the champion.

    Selection is on the VALIDATION split - the same rule that decides when
    training stops - so the test split still sees a model chosen without it.
    The losing runs are kept as a spread, because a single run of either
    network was measurably unstable and a lucky seed must not be allowed to
    pass for a better method. Reported cost is the whole batch, not just the
    winner's.

    The restarts are independent, so they run in parallel across cores. That
    is the only reason the restart protocol is affordable at all.
    """
    seeds = [cfg.seed + i for i in range(max(1, cfg.n_restarts))]
    args = (v["Ztr"], v["ytr"], v["Zva"], v["yva"])
    t0 = time.time()
    if len(seeds) == 1:
        fits = [trainer(cfg, *args, seeds[0])]
    else:
        n_jobs = (min(len(seeds), os.cpu_count() or 1)
                  if cfg.restart_jobs <= 0 else cfg.restart_jobs)
        fits = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(trainer)(cfg, *args, s) for s in seeds)
    wall = time.time() - t0

    runs = [{"seed": s, "val_mse": f["best_val_mse"], "n_iter": f["n_iter"],
             "best_iter": f["best_iter"], "seconds": f["seconds"]}
            for s, f in zip(seeds, fits)]
    best = min(fits, key=lambda f: f["best_val_mse"])
    vals = np.array([r["val_mse"] for r in runs])

    best["restarts"] = runs
    best["restart_spread"] = {
        "n": len(runs), "best": float(vals.min()),
        "median": float(np.median(vals)), "worst": float(vals.max()),
        "ratio_worst_to_best": (float(vals.max() / vals.min())
                                if vals.min() > 0 else float("nan")),
    }
    best["seconds"] = wall                                   # wall clock
    best["cpu_seconds"] = float(sum(r["seconds"] for r in runs))
    best["evaluations"] = int(best["evaluations"]) * len(runs)
    return best


def fit_one(method: str, cfg: MLConfig, v: dict) -> dict:
    """Fit one learner on one normalisation. `v` is a preprocessed view."""
    if method == "BPNN-DEA":
        f = with_restarts(nn.train_bpnn, cfg, v)
        f["predict"] = lambda X, f=f: nn.predict(f, X)
    elif method == "GANN-DEA":
        f = with_restarts(nn.train_gann, cfg, v)
        f["predict"] = lambda X, f=f: nn.predict(f, X)
    elif method == "SVM-DEA":
        f = svm.fit_svm(cfg, v["Ztr"], v["ytr"], v["Zva"], v["yva"])
        f["learning_curve"] = svm.learning_curve_for(
            clone(f["model"]), v["Ztr"], v["ytr"], cfg)
        f["predict"] = lambda X, f=f: svm.predict(f, X)
    elif method == "ISVM-DEA":
        f = svm.fit_isvm(cfg, v["Ztr"], v["ytr"], v["Zva"], v["yva"])
        f["learning_curve"] = svm.learning_curve_for(
            clone(f["model"]), v["Ztr"], v["ytr"], cfg)
        f["predict"] = lambda X, f=f: svm.predict(f, X)
    else:
        raise ValueError(f"unknown method {method!r}")
    return f


def fit_summary(f: dict) -> dict:
    """The JSON-safe part of a fitted model."""
    keep = ("seconds", "evaluations", "n_iter", "best_iter", "best_val_mse",
            "n_params", "n_support", "support_frac", "best_params", "cv_mse",
            "n_candidates", "n_fits", "curve_kind", "epsilon_vs_target_std",
            "cv_top5", "grid", "cpu_seconds", "restarts", "restart_spread")
    out = {k: f[k] for k in keep if k in f}
    if "best_params" in out:
        out["best_params"] = {k: str(v) for k, v in out["best_params"].items()}
    return out


def write_history(path: str, hist: dict) -> None:
    cols = list(hist)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for row in zip(*[hist[c] for c in cols]):
            w.writerow([f"{x:.10g}" if isinstance(x, float) else x for x in row])


# --------------------------------------------------------------------------
def main() -> None:
    p = build_parser("Hybrid ML-DEA: 4 learners x 3 normalisations.")
    args = p.parse_args()
    cfg = config_from_args(args)
    methods, norms = selected(args)
    os.makedirs(os.path.join(cfg.out_dir, "figures"), exist_ok=True)

    t_all = time.time()
    print("=" * 84)
    print("  Hybrid ML-DEA - predicting BCC efficiency scores")
    print("=" * 84)

    prep = prepare(cfg, norms)
    ss = prep.split_sizes()
    prov = prep.provenance
    z = prov["zero_fix"]
    print(f"  DMUs               {len(prep.y)}   features {prep.names}")
    print(f"  zero correction    {z['n_nonpositive']} non-positive cell(s) in "
          f"'{z['column']}' -> mode={z['mode']}"
          + (f", value={z['replacement_value']:.6g}" if z.get("applied") else ""))
    if cfg.boxcox:
        print(f"  Box-Cox (fit on {prov['boxcox']['fit_on']})   lambda = "
              + ", ".join(f"{l:+.4f}" for l in prov["boxcox"]["lambdas"]))
        print("  skewness           "
              + ", ".join(f"{a:+.2f}->{b:+.2f}"
                          for a, b in zip(prov["skew_before"], prov["skew_after"])))
    print(f"  split              train {ss['train']} / val {ss['val']} / "
          f"test {ss['test']}  (stratified on {cfg.n_bins} score bins)")
    print(f"  target             BCC in [{prov['target']['min']:.4f}, "
          f"{prov['target']['max']:.4f}], sd = {prov['target']['std']:.4f}")
    print("-" * 84)

    results, fitted = {}, {}
    for kind in norms:
        v = prep.views[kind]
        for m in methods:
            t0 = time.time()
            f = fit_one(m, cfg, v)
            r = {
                "method": m, "norm": kind,
                "train": evaluate(v["ytr"], f["predict"](v["Ztr"])),
                "val": evaluate(v["yva"], f["predict"](v["Zva"])),
                "test": evaluate(v["yte"], f["predict"](v["Zte"])),
                "fit": fit_summary(f),
                "y_test": v["yte"].tolist(),
                "pred_test": clamp_predictions(f["predict"](v["Zte"])).tolist(),
            }
            if "learning_curve" in f:
                r["learning_curve"] = f["learning_curve"]
            results[(m, kind)] = r
            fitted[(m, kind)] = f

            t = r["test"]
            sp = f.get("restart_spread", {})
            extra = (f"iters={f.get('n_iter','-')}@best {f.get('best_iter','-')}"
                     f"  val {sp.get('best',0):.5f}-{sp.get('worst',0):.5f} "
                     f"over {sp.get('n',0)} restarts"
                     if m in ITERATIVE else
                     f"nSV={f.get('n_support','-')} "
                     f"({f.get('support_frac', 0)*100:.0f}%)")
            print(f"  {m:<9s} {kind:<7s} test MSE={t['mse']:.5f}  "
                  f"R2={t['r2']:+.4f}  r={t['pearson_r']:+.4f}  "
                  f"rho={t['spearman']:+.4f}  {extra}  {time.time()-t0:5.1f}s")

    # ---- artefacts ----------------------------------------------------
    for (m, kind), f in fitted.items():
        if "history" in f:
            write_history(os.path.join(cfg.out_dir,
                                       f"history_{m}_{kind}.csv"), f["history"])
        if "learning_curve" in f:
            write_history(os.path.join(cfg.out_dir,
                                       f"history_{m}_{kind}.csv"), f["learning_curve"])

    rows = []
    for m in methods:
        for kind in norms:
            r = results[(m, kind)]
            rows.append({
                "method": m, "normalisation": kind,
                "test_mse": r["test"]["mse"], "test_rmse": r["test"]["rmse"],
                "test_mae": r["test"]["mae"], "test_r2": r["test"]["r2"],
                "pearson_r": r["test"]["pearson_r"],
                "pearson_p": r["test"]["pearson_p"],
                "spearman_rho": r["test"]["spearman"],
                "kendall_tau": r["test"]["kendall"],
                "train_mse": r["train"]["mse"], "val_mse": r["val"]["mse"],
                "overfit_ratio": (r["test"]["mse"] / r["train"]["mse"]
                                  if r["train"]["mse"] > 0 else float("nan")),
                "seconds": r["fit"].get("seconds", float("nan")),
            })
    csv_path = os.path.join(cfg.out_dir, "results.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.8g}" if isinstance(v, float) else v)
                        for k, v in row.items()})

    payload = {
        "config": cfg.to_dict(),
        "methods": methods, "norms": norms,
        "provenance": prov,
        "splits": ss,
        "features": prep.names,
        "results": [{**{k: v for k, v in results[(m, kind)].items()}}
                    for m in methods for kind in norms],
        "ranking": sorted(
            [{"method": m, "norm": k, "test_mse": results[(m, k)]["test"]["mse"],
              "test_r2": results[(m, k)]["test"]["r2"],
              "pearson_r": results[(m, k)]["test"]["pearson_r"]}
             for m in methods for k in norms],
            key=lambda d: d["test_mse"]),
        "total_seconds": time.time() - t_all,
    }
    with open(os.path.join(cfg.out_dir, "metrics.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    # ---- figures ------------------------------------------------------
    if cfg.figures:
        fd = os.path.join(cfg.out_dir, "figures")
        print("-" * 84)
        made = [plots.plot_preprocessing(prep, f"{fd}/01_preprocessing.png"),
                plots.plot_normalisations(prep, norms, f"{fd}/02_normalisations.png")]
        n = 3
        for m in methods:
            per_norm = {k: fitted[(m, k)] for k in norms}
            if m in ITERATIVE:
                made.append(plots.plot_loss_curves(m, per_norm, f"{fd}/{n:02d}_loss_{m}.png"))
            else:
                made.append(plots.plot_learning_curves(m, per_norm,
                                                       f"{fd}/{n:02d}_learning_{m}.png"))
            n += 1
        made += [
            plots.plot_pred_vs_actual(results, methods, norms, f"{fd}/{n:02d}_pred_vs_actual.png"),
            plots.plot_comparison(results, methods, norms, f"{fd}/{n+1:02d}_comparison.png"),
            plots.plot_heatmap(results, methods, norms, f"{fd}/{n+2:02d}_metric_heatmap.png"),
            plots.plot_residuals(results, methods, norms, f"{fd}/{n+3:02d}_residuals.png"),
            plots.plot_optimiser_duel(results, methods, norms, f"{fd}/{n+4:02d}_optimiser_duel.png"),
        ]
        print(f"  {len([x for x in made if x])} figures -> {fd}")

    best = payload["ranking"][0]
    print("-" * 84)
    print(f"  best combination   {best['method']} + {best['norm']}   "
          f"test MSE={best['test_mse']:.5f}  R2={best['test_r2']:+.4f}  "
          f"r={best['pearson_r']:+.4f}")
    print(f"  wall clock         {payload['total_seconds']:.1f}s")
    print(f"  artefacts          {cfg.out_dir}")
    print("=" * 84)

    from .report import write_report
    write_report(cfg.out_dir)
    print(f"  report             {os.path.join(cfg.out_dir, 'report.md')}")


if __name__ == "__main__":
    main()
