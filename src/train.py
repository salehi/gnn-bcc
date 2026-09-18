"""Train the GraphSAGE BCC-score regressor on the showcase split.

Writes, into outputs/:
  figures/live_training.png   rewritten every few epochs WHILE training runs
  figures/01..07, 11, 12      static charts
  history.csv                 one row per epoch
  metrics.json                all metrics for this run
  best_model.pt               weights + scaler + config
"""
from __future__ import annotations

import copy
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from . import plots
from .config import (DEA_INPUTS, DEA_OUTPUTS, FEATURES, Config, build_parser,
                     config_from_args)
from .data import (LogStandardScaler, build_eval_graph, build_train_edges,
                   edges_to_tensor, load_dataset, make_splits, to_tensor)
from .metrics import (clamp_predictions, compute_metrics, error_by_decile,
                      format_row)
from .models import build_model, count_params, layer_summary

IN_COLS = [FEATURES.index(c) for c in DEA_INPUTS]
OUT_COLS = [FEATURES.index(c) for c in DEA_OUTPUTS]


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def figdir(cfg: Config) -> str:
    d = os.path.join(cfg.out_dir, "figures")
    os.makedirs(d, exist_ok=True)
    return d


def suffix(cfg: Config) -> str:
    return f"_{cfg.tag}" if cfg.tag else ""


# --------------------------------------------------------------------------
def prepare_graphs(cfg, X, y, train_i, val_i, test_i):
    """Scale (train-fit only) and build the strict-inductive graphs."""
    scaler = LogStandardScaler().fit(X[train_i])
    Z = scaler.transform(X)
    Z_tr = Z[train_i]

    train_edges = build_train_edges(Z_tr, cfg, IN_COLS, OUT_COLS)

    bundle = {
        "scaler": scaler,
        "Z": Z,
        "Z_tr": Z_tr,
        "train_edges": train_edges,
        "x_tr": to_tensor(Z_tr),
        "ei_tr": edges_to_tensor(train_edges),
        "y_tr": to_tensor(y[train_i]),
    }
    for name, idx in (("val", val_i), ("test", test_i)):
        if idx is None or len(idx) == 0:
            continue
        Xc, ei, pos = build_eval_graph(Z_tr, Z[idx], cfg, IN_COLS, OUT_COLS, train_edges)
        bundle[name] = {
            "x": to_tensor(Xc),
            "ei": edges_to_tensor(ei),
            "pos": torch.as_tensor(pos, dtype=torch.long),
            "y": y[idx],
        }
    return bundle


@torch.no_grad()
def predict_heldout(model, b, name) -> np.ndarray:
    model.eval()
    g = b[name]
    out = model(g["x"], g["ei"])
    return out[g["pos"]].cpu().numpy()


@torch.no_grad()
def predict_train(model, b) -> np.ndarray:
    model.eval()
    return model(b["x_tr"], b["ei_tr"]).cpu().numpy()


def fit(cfg: Config, X, y, train_i, val_i, test_i, live: bool = False,
        verbose: bool = True):
    """Train one model. Returns (model, bundle, history, info)."""
    set_seed(cfg.seed)
    b = prepare_graphs(cfg, X, y, train_i, val_i, test_i)

    model = build_model(cfg)
    crit = (nn.MSELoss() if cfg.loss == "mse"
            else nn.SmoothL1Loss(beta=cfg.huber_beta))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg.sched_factor,
        patience=cfg.sched_patience, min_lr=cfg.min_lr)

    hist = {k: [] for k in ("epoch", "train_loss", "val_loss", "train_rmse",
                            "val_rmse", "val_mae", "train_r2", "val_r2", "lr")}
    best = {"rmse": float("inf"), "epoch": 0, "state": None}
    y_tr_np, y_va_np = y[train_i], y[val_i]
    live_path = os.path.join(figdir(cfg), f"live_training{suffix(cfg)}.png")
    t0 = time.time()

    if verbose:
        print(f"  {'epoch':>6s} {'train_loss':>11s} {'val_loss':>10s} "
              f"{'val_RMSE':>9s} {'val_MAE':>8s} {'val_R2':>8s} {'lr':>9s}")

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(b["x_tr"], b["ei_tr"])
        loss = crit(pred, b["y_tr"])
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        tr_pred = predict_train(model, b)
        va_pred = predict_heldout(model, b, "val")
        with torch.no_grad():
            val_loss = float(crit(torch.as_tensor(va_pred),
                                  torch.as_tensor(y_va_np, dtype=torch.float32)))
        m_tr = compute_metrics(y_tr_np, tr_pred)
        m_va = compute_metrics(y_va_np, va_pred)
        lr_now = opt.param_groups[0]["lr"]

        for k, v in (("epoch", epoch), ("train_loss", float(loss)),
                     ("val_loss", val_loss), ("train_rmse", m_tr["rmse"]),
                     ("val_rmse", m_va["rmse"]), ("val_mae", m_va["mae"]),
                     ("train_r2", m_tr["r2"]), ("val_r2", m_va["r2"]),
                     ("lr", lr_now)):
            hist[k].append(v)

        sched.step(m_va["rmse"])

        if m_va["rmse"] < best["rmse"] - 1e-6:
            best = {"rmse": m_va["rmse"], "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}

        if verbose and (epoch % cfg.log_every == 0 or epoch == 1):
            print(f"  {epoch:6d} {float(loss):11.6f} {val_loss:10.6f} "
                  f"{m_va['rmse']:9.4f} {m_va['mae']:8.4f} {m_va['r2']:8.4f} "
                  f"{lr_now:9.2e}")
        if live and (epoch % cfg.live_every == 0 or epoch == 1):
            plots.plot_live(hist, live_path, epoch, best["epoch"], best["rmse"])

        if epoch - best["epoch"] >= cfg.patience:
            if verbose:
                print(f"  early stop at epoch {epoch} "
                      f"(no val improvement for {cfg.patience} epochs)")
            break

    model.load_state_dict(best["state"])
    info = {
        "best_epoch": best["epoch"],
        "epochs_run": hist["epoch"][-1],
        "best_val_rmse": best["rmse"],
        "seconds": round(time.time() - t0, 1),
        "n_params": count_params(model),
        "train_edges": int(b["train_edges"].shape[1]),
        "mean_in_degree": float(np.bincount(
            b["train_edges"][1], minlength=len(train_i)).mean()),
    }
    if live:
        plots.plot_live(hist, live_path, info["epochs_run"],
                        best["epoch"], best["rmse"])
    return model, b, hist, info


# --------------------------------------------------------------------------
def main():
    p = build_parser("Train the GraphSAGE BCC-score regressor (showcase split).")
    cfg = config_from_args(p.parse_args())
    fd = figdir(cfg)
    sfx = suffix(cfg)

    print("=" * 78)
    print("  BCC efficiency score  |  GraphSAGE node regression  |  strict inductive")
    print("=" * 78)

    X, y, names = load_dataset(cfg)
    print(f"\nLoaded {len(y)} DMUs x {X.shape[1]} features from "
          f"{os.path.basename(cfg.data_path)} [sheet '{cfg.sheet}']")
    print(f"  BCC score: min {y.min():.4f}  mean {y.mean():.4f}  max {y.max():.4f}  "
          f"| {int((y >= 1 - 1e-6).sum())} DMUs on the frontier")

    if cfg.shuffle_labels:
        rng = np.random.default_rng(cfg.seed)
        y = rng.permutation(y)
        print("  !! --shuffle-labels active: targets permuted. "
              "A correct pipeline must now score R2 ~ 0.")

    train_i, val_i, test_i = make_splits(y, cfg)
    print(f"  split: train {len(train_i)} | val {len(val_i)} | test {len(test_i)}"
          f"  (stratified over {cfg.n_bins} target bins)")

    print(f"\nArchitecture: {cfg.model} {list(cfg.hidden_dims)}  "
          f"dropout={cfg.dropout}  loss={cfg.loss}  k={cfg.k}  edges={cfg.edges}")
    print(f"Training (max {cfg.max_epochs} epochs, early stop patience {cfg.patience})\n")

    model, b, hist, info = fit(cfg, X, y, train_i, val_i, test_i, live=True)

    # ---- final evaluation -------------------------------------------------
    preds = {
        "train": (y[train_i], clamp_predictions(predict_train(model, b))),
        "val": (y[val_i], clamp_predictions(predict_heldout(model, b, "val"))),
        "test": (y[test_i], clamp_predictions(predict_heldout(model, b, "test"))),
    }
    mets = {k: compute_metrics(t, pdd) for k, (t, pdd) in preds.items()}

    print(f"\nDone in {info['seconds']}s  |  best epoch {info['best_epoch']} / "
          f"{info['epochs_run']}  |  {info['n_params']:,} parameters")
    print("-" * 78)
    for k in ("train", "val", "test"):
        print("  " + format_row(k, mets[k]))
    print("-" * 78)
    t = mets["test"]
    print(f"  frontier DMUs in test: {t['frontier_n']}  |  mean prediction on them "
          f"{t['frontier_pred_mean']:.4f}  (gap {t['frontier_gap']:+.4f})")
    print(f"  near-frontier (>=0.95) detection: precision {t['p95_precision']:.3f}  "
          f"recall {t['p95_recall']:.3f}  F1 {t['p95_f1']:.3f}")

    # ---- persist ----------------------------------------------------------
    with open(os.path.join(cfg.out_dir, f"history{sfx}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hist.keys())
        w.writerows(zip(*hist.values()))

    payload = {
        "run": "showcase_split",
        "config": cfg.to_dict(),
        "dea": {"inputs": DEA_INPUTS, "outputs": DEA_OUTPUTS,
                "note": "assumed orientation; confirm against the thesis"},
        "splits": {"train": len(train_i), "val": len(val_i), "test": len(test_i)},
        "training": info,
        "metrics": mets,
    }
    with open(os.path.join(cfg.out_dir, f"metrics{sfx}.json"), "w") as f:
        json.dump(payload, f, indent=2)

    torch.save({"state_dict": model.state_dict(),
                "config": cfg.to_dict(),
                "scaler": b["scaler"].to_dict(),
                "features": names,
                "metrics": mets},
               os.path.join(cfg.out_dir, f"best_model{sfx}.pt"))

    # ---- charts -----------------------------------------------------------
    print("\nRendering charts...")
    made = []
    made.append(plots.plot_data_overview(X, y, names, f"{fd}/01_data_overview{sfx}.png"))
    made.append(plots.plot_correlations(np.log1p(X), y, names, f"{fd}/02_correlations{sfx}.png"))
    made.append(plots.plot_graph_structure(b["Z_tr"], b["train_edges"], y[train_i],
                                           f"{fd}/03_graph_structure{sfx}.png", cfg.k))
    made.append(plots.plot_training_curves(hist, f"{fd}/04_training_curves{sfx}.png",
                                           info["best_epoch"],
                                           f" - {cfg.model} {list(cfg.hidden_dims)}, k={cfg.k}"))
    made.append(plots.plot_pred_vs_actual(
        {k: (preds[k][0], preds[k][1], mets[k]) for k in ("train", "val", "test")},
        f"{fd}/05_pred_vs_actual{sfx}.png"))
    made.append(plots.plot_residuals(*preds["test"], f"{fd}/06_residuals{sfx}.png"))
    made.append(plots.plot_error_by_decile(*error_by_decile(*preds["test"]),
                                           f"{fd}/07_error_by_decile{sfx}.png"))

    with torch.no_grad():
        model.eval()
        g = b["test"]
        emb_test = model.embed(g["x"], g["ei"])[g["pos"]].cpu().numpy()
    made.append(plots.plot_embeddings(emb_test, y[test_i],
                                      f"{fd}/11_embeddings{sfx}.png", "held-out test"))
    made.append(plots.plot_architecture(layer_summary(model), info["n_params"],
                                        f"{fd}/12_architecture{sfx}.png",
                                        "GraphSAGE" if cfg.model == "sage" else "MLP",
                                        k=cfg.k))
    for m in made:
        print(f"  {os.path.relpath(m, cfg.out_dir)}")
    print(f"\nOutputs in {cfg.out_dir}")


if __name__ == "__main__":
    main()
