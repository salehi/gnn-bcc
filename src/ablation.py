"""Two questions the report has to answer honestly.

1. Does the graph actually help? GraphSAGE is compared against the identical
   trunk with message passing removed (4 -> 24 -> 12 -> 1 MLP). Same depth,
   same widths, same normalisation, same optimiser, same folds. If the MLP
   wins, that is the finding and it goes in the report.

2. How many DEA peers should a DMU see? k is swept over a range and scored
   the same way.

Everything is measured with 5-fold strict-inductive CV, so these are
generalisation numbers, not training-set numbers.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

from . import plots
from .config import Config, build_parser, config_from_args
from .cv import run_cv
from .data import load_dataset
from .train import figdir, suffix

K_GRID = (3, 5, 8, 10, 15, 20)


def _cfg(cfg: Config, **over) -> Config:
    d = {**cfg.to_dict(), **over}
    d["hidden_dims"] = tuple(d["hidden_dims"])
    return Config(**d)


def main():
    p = build_parser("Ablations: GNN vs edges-removed MLP, and k sensitivity.")
    p.add_argument("--k-grid", default=",".join(map(str, K_GRID)))
    p.add_argument("--skip-k-sweep", action="store_true")
    args = p.parse_args()
    cfg = config_from_args(args)
    sfx = suffix(cfg)
    fd = figdir(cfg)

    X, y, names = load_dataset(cfg)

    # ---- 1. does the graph earn its place? --------------------------------
    print("=" * 78)
    print("  ABLATION 1  -  does message passing help?")
    print("=" * 78)
    results = {}
    for label, over in (
        (f"GraphSAGE (k={cfg.k})", {"model": "sage"}),
        ("MLP (no edges)", {"model": "mlp"}),
    ):
        print(f"\n{label}")
        results[label] = run_cv(_cfg(cfg, **over), X, y)

    print("\n" + "-" * 78)
    hdr = f"  {'model':<22s} {'R2':>16s} {'RMSE':>16s} {'Spearman':>16s}"
    print(hdr); print("-" * 78)
    for label, r in results.items():
        m, s = r["mean"], r["std"]
        print(f"  {label:<22s} {m['r2']:>8.4f}+/-{s['r2']:<6.4f} "
              f"{m['rmse']:>8.4f}+/-{s['rmse']:<6.4f} "
              f"{m['spearman']:>8.4f}+/-{s['spearman']:<6.4f}")
    print("-" * 78)

    # Both models are scored on the SAME folds, so the comparison is paired.
    # A paired test is far more powerful than eyeballing mean +/- pooled sd,
    # and it is the honest way to decide whether the graph earned its place.
    names_l = list(results)
    r2_sage = np.array([f["r2"] for f in results[names_l[0]]["folds"]])
    r2_mlp = np.array([f["r2"] for f in results[names_l[1]]["folds"]])
    diff = r2_sage - r2_mlp
    d_r2 = float(diff.mean())

    t_stat, p_val = stats.ttest_rel(r2_sage, r2_mlp)
    try:
        w_stat, w_p = stats.wilcoxon(r2_sage, r2_mlp)
    except ValueError:
        w_p = float("nan")

    sd_sage = float(r2_sage.std(ddof=1))
    sd_mlp = float(r2_mlp.std(ddof=1))

    if p_val < 0.05:
        verdict = ("the graph HELPS (significant)" if d_r2 > 0
                   else "the MLP is BETTER (significant)")
    elif sd_sage < sd_mlp * 0.6:
        verdict = ("no significant mean difference, but the graph is markedly "
                   "MORE STABLE across folds")
    else:
        verdict = "no significant difference - the graph does not earn its place here"

    print(f"\n  per-fold R2 difference (SAGE - MLP): "
          f"{', '.join(f'{d:+.4f}' for d in diff)}")
    print(f"  mean delta R2 = {d_r2:+.4f}   "
          f"paired t={float(t_stat):+.3f}, p={float(p_val):.3f}   "
          f"Wilcoxon p={float(w_p):.3f}")
    print(f"  fold-to-fold sd of R2:  SAGE {sd_sage:.4f}   MLP {sd_mlp:.4f}")
    print(f"  -> {verdict}")

    plots.plot_model_comparison(results, f"{fd}/09_gnn_vs_mlp{sfx}.png")

    payload = {"config": cfg.to_dict(),
               "graph_vs_no_graph": {k: {"mean": v["mean"], "std": v["std"]}
                                     for k, v in results.items()},
               "paired_test": {
                   "per_fold_delta_r2": diff.tolist(),
                   "mean_delta_r2": d_r2,
                   "paired_t": float(t_stat),
                   "p_value": float(p_val),
                   "wilcoxon_p": float(w_p),
                   "sd_r2_sage": sd_sage,
                   "sd_r2_mlp": sd_mlp,
               },
               "delta_r2_sage_minus_mlp": d_r2,
               "verdict": verdict}

    # ---- 2. how many peers? ----------------------------------------------
    if not args.skip_k_sweep:
        ks = [int(x) for x in args.k_grid.split(",") if x.strip()]
        print("\n" + "=" * 78)
        print(f"  ABLATION 2  -  k sensitivity over {ks}")
        print("=" * 78)
        per_k = {}
        for k in ks:
            r = run_cv(_cfg(cfg, model="sage", k=k), X, y, verbose=False)
            per_k[k] = r
            print(f"  k={k:<3d} R2 {r['mean']['r2']:+.4f}+/-{r['std']['r2']:.4f}   "
                  f"RMSE {r['mean']['rmse']:.4f}   rho {r['mean']['spearman']:+.4f}")
        best_k = max(per_k, key=lambda k: per_k[k]["mean"]["r2"])
        print(f"\n  best k by mean R2: {best_k}")
        plots.plot_k_sensitivity(ks, per_k, f"{fd}/10_k_sensitivity{sfx}.png")
        payload["k_sweep"] = {str(k): {"mean": v["mean"], "std": v["std"]}
                              for k, v in per_k.items()}
        payload["best_k"] = best_k

    with open(os.path.join(cfg.out_dir, f"ablation{sfx}.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  figures/09_gnn_vs_mlp{sfx}.png")
    if not args.skip_k_sweep:
        print(f"  figures/10_k_sensitivity{sfx}.png")
    print(f"  ablation{sfx}.json")


if __name__ == "__main__":
    main()
