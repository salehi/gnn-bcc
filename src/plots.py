"""Every PNG chart. Headless Agg backend, 150 dpi.

The live chart (`live_training.png`) is rewritten every few epochs while
training runs, so it can be kept open in an image viewer and watched.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy import stats

DPI = 150
C_TRAIN = "#3F6FB0"
C_VAL = "#E08A3C"
C_TEST = "#4C9A6B"
C_ACC = "#C0504D"
C_GREY = "#8A8F98"
CMAP = "viridis"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "figure.autolayout": False,
})


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 1. Data overview
# --------------------------------------------------------------------------
def plot_data_overview(X_raw, y, names, path):
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    for j, nm in enumerate(names):
        ax = axes[0, j]
        ax.hist(X_raw[:, j], bins=50, color=C_GREY, edgecolor="white", linewidth=0.3)
        ax.set_yscale("log")
        ax.set_title(f"{nm}\n(raw)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

        ax = axes[1, j]
        ax.hist(np.log1p(X_raw[:, j]), bins=50, color=C_TRAIN,
                edgecolor="white", linewidth=0.3)
        ax.set_title(f"{nm}\n(log1p)", fontsize=8)
        ax.tick_params(labelsize=7)

    ax = axes[2, 0]
    ax.hist(y, bins=40, color=C_TEST, edgecolor="white", linewidth=0.3)
    ax.axvline(1.0, color=C_ACC, ls="--", lw=1.2, label="frontier (1.0)")
    ax.set_title("BCC score distribution"); ax.set_xlabel("BCC score"); ax.legend(fontsize=7)

    ax = axes[2, 1]
    ax.plot(np.sort(y), np.linspace(0, 1, len(y)), color=C_TEST, lw=1.5)
    ax.set_title("BCC score ECDF"); ax.set_xlabel("BCC score")

    ax = axes[2, 2]
    ax.boxplot([y], vert=True, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor=C_TEST, alpha=0.5))
    ax.set_xticks([1]); ax.set_xticklabels(["BCC"]); ax.set_title("BCC score spread")

    ax = axes[2, 3]
    eff = int((y >= 1 - 1e-6).sum())
    ax.bar(["efficient\n(=1.0)", "inefficient"], [eff, len(y) - eff],
           color=[C_ACC, C_GREY])
    for i, v in enumerate([eff, len(y) - eff]):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_title("Frontier membership")

    fig.suptitle(f"Dataset overview - {len(y)} DMUs, 4 features, BCC target",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, path)


# --------------------------------------------------------------------------
# 2. Correlations
# --------------------------------------------------------------------------
def plot_correlations(X_log, y, names, path):
    labels = [n.replace(" ", "\n") for n in names] + ["BCC\nscore"]
    M = np.column_stack([X_log, y])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (title, fn) in zip(axes, [
        ("Pearson (on log1p features)", lambda A: np.corrcoef(A, rowvar=False)),
        ("Spearman (rank)", lambda A: stats.spearmanr(A).statistic),
    ]):
        C = fn(M)
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
        ax.grid(False)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(C[i, j]) > 0.55 else "black")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Feature / target correlation - weak linear signal motivates a nonlinear model",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path)


# --------------------------------------------------------------------------
# 3. Graph structure
# --------------------------------------------------------------------------
def plot_graph_structure(Z, edge_index, y, path, k, max_edges=4000):
    from sklearn.decomposition import PCA
    P = PCA(n_components=2, random_state=0).fit_transform(Z)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ei = edge_index
    if ei.shape[1] > max_edges:
        sel = np.random.default_rng(0).choice(ei.shape[1], max_edges, replace=False)
        ei = ei[:, sel]
    segs = np.stack([P[ei[0]], P[ei[1]]], axis=1)
    ax.add_collection(LineCollection(segs, colors=C_GREY, linewidths=0.2, alpha=0.35))
    sc = ax.scatter(P[:, 0], P[:, 1], c=y, cmap=CMAP, s=14,
                    edgecolors="white", linewidths=0.3)
    ax.autoscale_view()
    fig.colorbar(sc, ax=ax, label="BCC score")
    # A handful of DMUs are enormous (the data spans 5 orders of magnitude even
    # after log1p), and left alone they compress everything else into a line.
    # The view is clipped to the bulk; the count of hidden points is stated.
    (x0, x1), (y0, y1) = (np.percentile(P[:, i], [0.5, 99.5]) for i in (0, 1))
    padx, pady = 0.08 * (x1 - x0) + 1e-9, 0.08 * (y1 - y0) + 1e-9
    x0, x1, y0, y1 = x0 - padx, x1 + padx, y0 - pady, y1 + pady
    hidden = int((~((P[:, 0] >= x0) & (P[:, 0] <= x1) &
                    (P[:, 1] >= y0) & (P[:, 1] <= y1))).sum())
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_title(f"k-NN peer graph (k={k}), PCA projection\n"
                 f"{edge_index.shape[1]} edges"
                 + (f" ({max_edges} drawn)" if edge_index.shape[1] > max_edges else "")
                 + (f", {hidden} outliers outside view" if hidden else ""))
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    ax = axes[1]
    deg = np.bincount(edge_index[1], minlength=len(Z))
    ax.hist(deg, bins=range(deg.min(), deg.max() + 2), color=C_TRAIN,
            edgecolor="white", linewidth=0.4)
    ax.set_title(f"In-degree distribution\nmean {deg.mean():.1f}, min {deg.min()}, max {deg.max()}")
    ax.set_xlabel("in-degree"); ax.set_ylabel("nodes")

    # Do neighbours actually share efficiency? If they do, message passing has
    # something real to pass. This is the empirical case for using a GNN.
    ax = axes[2]
    nb_mean = np.zeros(len(Z))
    cnt = np.bincount(edge_index[1], minlength=len(Z)).astype(float)
    np.add.at(nb_mean, edge_index[1], y[edge_index[0]])
    ok = cnt > 0
    nb_mean[ok] /= cnt[ok]
    r = np.corrcoef(y[ok], nb_mean[ok])[0, 1]
    ax.scatter(y[ok], nb_mean[ok], s=10, alpha=0.5, color=C_TEST)
    lim = [min(y.min(), nb_mean[ok].min()), 1.02]
    ax.plot(lim, lim, ls="--", color=C_ACC, lw=1)
    ax.set_xlabel("own BCC score"); ax.set_ylabel("mean BCC of graph neighbours")
    ax.set_title(f"Neighbourhood homophily\nr = {r:.3f}")

    fig.suptitle("Graph structure - built from training DMUs only",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, path)


# --------------------------------------------------------------------------
# 4 / live. Training curves
# --------------------------------------------------------------------------
def _curves(h, fig, axes, best_epoch=None, title=None):
    ep = h["epoch"]
    ax = axes[0, 0]
    ax.plot(ep, h["train_loss"], color=C_TRAIN, lw=1.4, label="train")
    ax.plot(ep, h["val_loss"], color=C_VAL, lw=1.4, label="val")
    ax.set_yscale("log"); ax.set_title("Loss"); ax.set_xlabel("epoch"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(ep, h["train_rmse"], color=C_TRAIN, lw=1.4, label="train RMSE")
    ax.plot(ep, h["val_rmse"], color=C_VAL, lw=1.4, label="val RMSE")
    ax.plot(ep, h["val_mae"], color=C_VAL, lw=1.0, ls="--", label="val MAE")
    ax.set_title("Error"); ax.set_xlabel("epoch"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(ep, h["train_r2"], color=C_TRAIN, lw=1.4, label="train")
    ax.plot(ep, h["val_r2"], color=C_VAL, lw=1.4, label="val")
    ax.set_title("$R^2$"); ax.set_xlabel("epoch"); ax.legend(fontsize=8)
    # The first few epochs sit at R^2 ~ -20 and would flatten everything that
    # matters, so the axis is set from the settled part of the curve.
    tail = np.asarray(h["val_r2"][max(1, len(ep) // 10):], dtype=float)
    lo = float(np.nanmin(tail)) if tail.size else -0.05
    ax.set_ylim(max(-1.0, min(-0.05, lo)), 1.02)

    ax = axes[1, 1]
    ax.plot(ep, h["lr"], color=C_ACC, lw=1.4)
    ax.set_yscale("log"); ax.set_title("Learning rate"); ax.set_xlabel("epoch")

    if best_epoch is not None:
        for ax in axes.ravel():
            ax.axvline(best_epoch, color=C_GREY, ls=":", lw=1.2)
        axes[0, 0].text(best_epoch, axes[0, 0].get_ylim()[1], " best",
                        va="top", fontsize=7, color=C_GREY)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")


def plot_live(history: dict, path: str, epoch: int, best_epoch: int, best_rmse: float):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    _curves(history, fig, axes, best_epoch,
            f"Training in progress - epoch {epoch}  |  best val RMSE {best_rmse:.4f} @ {best_epoch}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=110, bbox_inches="tight")
    plt.close(fig)
    os.replace(tmp, path)   # atomic, so a viewer never reads a half-written file
    return path


def plot_training_curves(history, path, best_epoch, cfg_title=""):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _curves(history, fig, axes, best_epoch, f"Training history{cfg_title}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path)


# --------------------------------------------------------------------------
# 5. Predicted vs actual
# --------------------------------------------------------------------------
def plot_pred_vs_actual(splits: dict, path):
    n = len(splits)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.8), squeeze=False)
    colors = {"train": C_TRAIN, "val": C_VAL, "test": C_TEST}
    for ax, (name, (yt, yp, m)) in zip(axes[0], splits.items()):
        ax.scatter(yt, yp, s=14, alpha=0.55, color=colors.get(name, C_GREY),
                   edgecolors="none")
        lo = min(yt.min(), yp.min()) - 0.02
        ax.plot([lo, 1.02], [lo, 1.02], ls="--", color=C_ACC, lw=1.2)
        ax.set_xlim(lo, 1.03); ax.set_ylim(lo, 1.03)
        ax.set_xlabel("true BCC score"); ax.set_ylabel("predicted BCC score")
        ax.set_title(f"{name}  (n={m['n']})\n"
                     f"$R^2$={m['r2']:.3f}  RMSE={m['rmse']:.4f}  "
                     f"$\\rho$={m['spearman']:.3f}")
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Predicted vs actual BCC score", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, path)


# --------------------------------------------------------------------------
# 6. Residuals
# --------------------------------------------------------------------------
def plot_residuals(y_true, y_pred, path, label="test"):
    res = y_pred - y_true
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.scatter(y_pred, res, s=14, alpha=0.55, color=C_TEST, edgecolors="none")
    ax.axhline(0, color=C_ACC, ls="--", lw=1.2)
    ax.set_xlabel("predicted"); ax.set_ylabel("residual (pred - true)")
    ax.set_title("Residual vs predicted")

    ax = axes[1]
    ax.hist(res, bins=35, color=C_TEST, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color=C_ACC, ls="--", lw=1.2)
    ax.set_title(f"Residual distribution\nmean {res.mean():+.4f}, sd {res.std():.4f}")
    ax.set_xlabel("residual")

    ax = axes[2]
    stats.probplot(res, dist="norm", plot=ax)
    ax.get_lines()[0].set(marker="o", markersize=3, alpha=0.6, color=C_TEST)
    ax.get_lines()[1].set(color=C_ACC, lw=1.2)
    ax.set_title("Normal Q-Q plot")

    fig.suptitle(f"Residual diagnostics ({label} set)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, path)


# --------------------------------------------------------------------------
# 7. Error by decile
# --------------------------------------------------------------------------
def plot_error_by_decile(labels, maes, biases, counts, path, label="test"):
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    ax = axes[0]
    bars = ax.bar(x, maes, color=C_TRAIN)
    bars[-1].set_color(C_ACC)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("MAE"); ax.set_title("MAE per true-BCC decile")
    for i, (v, c) in enumerate(zip(maes, counts)):
        ax.text(i, v, f"n={c}", ha="center", va="bottom", fontsize=6)

    ax = axes[1]
    cols = [C_ACC if b < 0 else C_TEST for b in biases]
    ax.bar(x, biases, color=cols)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("mean (pred - true)")
    ax.set_title("Signed bias per decile\n(negative = under-predicted)")

    fig.suptitle(f"Where the error lives ({label} set) - the top decile holds the frontier DMUs",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, path)


# --------------------------------------------------------------------------
# 8. Cross-validation
# --------------------------------------------------------------------------
def plot_cv_results(fold_metrics: list[dict], path):
    keys = [("rmse", "RMSE", False), ("mae", "MAE", False),
            ("r2", "$R^2$", True), ("spearman", "Spearman $\\rho$", True)]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    x = np.arange(len(fold_metrics))
    for ax, (k, title, higher) in zip(axes, keys):
        v = np.array([m[k] for m in fold_metrics])
        ax.bar(x, v, color=C_TRAIN, alpha=0.85)
        ax.axhline(v.mean(), color=C_ACC, ls="--", lw=1.4,
                   label=f"mean {v.mean():.4f}")
        ax.fill_between([-0.6, len(v) - 0.4], v.mean() - v.std(), v.mean() + v.std(),
                        color=C_ACC, alpha=0.12, label=f"$\\pm$sd {v.std():.4f}")
        ax.set_xlim(-0.6, len(v) - 0.4)
        ax.set_xticks(x); ax.set_xticklabels([f"fold {i+1}" for i in x], fontsize=8)
        ax.set_title(f"{title}\n{v.mean():.4f} $\\pm$ {v.std():.4f}")
        ax.legend(fontsize=7)
        for i, val in enumerate(v):
            ax.text(i, val, f"{val:.3f}", ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=7)
    fig.suptitle(f"{len(fold_metrics)}-fold strict-inductive cross-validation (held-out folds)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, path)


# --------------------------------------------------------------------------
# 9. GNN vs MLP
# --------------------------------------------------------------------------
def plot_model_comparison(results: dict, path, title="GraphSAGE vs edges-removed MLP"):
    keys = [("r2", "$R^2$", True), ("rmse", "RMSE", False), ("mae", "MAE", False),
            ("spearman", "Spearman $\\rho$", True), ("kendall", "Kendall $\\tau$", True)]
    names = list(results.keys())
    colors = [C_TRAIN, C_VAL, C_TEST, C_ACC, C_GREY]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 4.6))
    for ax, (k, lbl, higher) in zip(axes, keys):
        vals = [results[n]["mean"][k] for n in names]
        errs = [results[n]["std"][k] for n in names]
        ax.bar(range(len(names)), vals, yerr=errs, capsize=4,
               color=colors[:len(names)], alpha=0.9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8, rotation=15, ha="right")
        ax.set_title(f"{lbl}\n({'higher' if higher else 'lower'} is better)")
        for i, (v, e) in enumerate(zip(vals, errs)):
            ax.text(i, v + e, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle(title + "  -  mean $\\pm$ sd over folds", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, path)


# --------------------------------------------------------------------------
# 10. k sensitivity
# --------------------------------------------------------------------------
def plot_k_sensitivity(ks, per_k: dict, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (k, lbl) in zip(axes, [("r2", "$R^2$"), ("rmse", "RMSE"),
                                   ("spearman", "Spearman $\\rho$")]):
        m = np.array([per_k[kk]["mean"][k] for kk in ks])
        s = np.array([per_k[kk]["std"][k] for kk in ks])
        ax.plot(ks, m, "o-", color=C_TRAIN, lw=1.6, markersize=5)
        ax.fill_between(ks, m - s, m + s, color=C_TRAIN, alpha=0.15)
        ax.set_xlabel("k (neighbours per DMU)"); ax.set_title(lbl)
        ax.set_xticks(ks)
    fig.suptitle("Sensitivity to graph density - how many DEA peers should a DMU see?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, path)


# --------------------------------------------------------------------------
# 11. Learned embeddings
# --------------------------------------------------------------------------
def plot_embeddings(emb, y, path, split_name="all"):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    P = PCA(n_components=2, random_state=0).fit_transform(emb)
    sc = axes[0].scatter(P[:, 0], P[:, 1], c=y, cmap=CMAP, s=16,
                         edgecolors="white", linewidths=0.3)
    axes[0].set_title("PCA of the learned 12-d embedding")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    fig.colorbar(sc, ax=axes[0], label="true BCC score")

    perp = max(5, min(30, (len(emb) - 1) // 3))
    T = TSNE(n_components=2, perplexity=perp, random_state=0, init="pca").fit_transform(emb)
    sc = axes[1].scatter(T[:, 0], T[:, 1], c=y, cmap=CMAP, s=16,
                         edgecolors="white", linewidths=0.3)
    axes[1].set_title(f"t-SNE of the same embedding (perplexity {perp})")
    fig.colorbar(sc, ax=axes[1], label="true BCC score")

    fig.suptitle(f"Hidden representation after block 2 ({split_name} nodes)\n"
                 "a smooth colour gradient means the network has ordered DMUs by efficiency",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, path)


# --------------------------------------------------------------------------
# 12. Architecture diagram
# --------------------------------------------------------------------------
def plot_architecture(rows, total_params, path, model_name="GraphSAGE",
                      in_dim=4, k=8):
    """Two panels: the block diagram on top, the parameter table below, so
    neither can collide with the other."""
    is_mlp = model_name.lower().startswith("mlp")
    fig, (ax, axt) = plt.subplots(
        2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [2.1, 1]})

    for a in (ax, axt):
        a.axis("off"); a.grid(False)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)

    agg = "Linear" if is_mlp else "SAGEConv"
    sub1 = "no message passing" if is_mlp else f"mean over {k} peers"
    sub2 = "no message passing" if is_mlp else "mean over peers"
    body = C_VAL if is_mlp else C_TRAIN

    blocks = [
        ("input", "4 DMU features", 4, C_GREY),
        (f"{agg} 4\u219224", sub1, 24, body),
        ("LayerNorm", "+ ReLU", 24, "#A9C2DD"),
        (f"{agg} 24\u219212", sub2, 12, body),
        ("LayerNorm", "+ ReLU", 12, "#A9C2DD"),
        ("Linear 12\u21921", "BCC score", 1, C_ACC),
    ]

    n = len(blocks)
    gap = 2.6
    w = (100 - 4 - gap * (n - 1)) / n
    x = 2.0
    for title, sub, width, color in blocks:
        h = 26 + 34 * (width / 24) ** 0.5
        yb = 52 - h / 2
        ax.add_patch(plt.Rectangle((x, yb), w, h, facecolor=color, alpha=0.85,
                                   edgecolor="#2b2b2b", linewidth=0.9,
                                   zorder=2))
        light = color in (C_TRAIN, C_ACC, C_GREY, C_VAL)
        ax.text(x + w / 2, 52 + 3.5, title, ha="center", va="center",
                fontsize=8.5, fontweight="bold", zorder=3,
                color="white" if light else "#1a1a1a")
        ax.text(x + w / 2, 52 - 4.5, sub, ha="center", va="center",
                fontsize=7, zorder=3,
                color="#f0f0f0" if light else "#333333")
        ax.text(x + w / 2, yb + h + 3, f"{width} unit{'s' if width > 1 else ''}",
                ha="center", va="bottom", fontsize=8, color="#444",
                fontweight="bold")
        if x > 2.0:
            ax.annotate("", xy=(x - 0.5, 52), xytext=(x - gap + 0.5, 52),
                        arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#555"),
                        zorder=1)
        x += w + gap

    header = f"{'layer':<26s}{'shape':<16s}{'params':>8s}"
    lines = [f"{nm:<26s}{sh:<16s}{pc:>8,d}" for nm, sh, pc in rows]
    rule = "-" * len(header)
    axt.text(0.5, 0.95, "\n".join([header, rule] + lines + [rule,
             f"{'TOTAL':<26s}{'':<16s}{total_params:>8,d}"]),
             transform=axt.transAxes, ha="center", va="top",
             fontsize=8.5, family="monospace", linespacing=1.5)

    fig.suptitle(f"{model_name} regressor   4 \u2192 24 \u2192 12 \u2192 1"
                 f"   ({total_params:,} trainable parameters)",
                 fontsize=13, fontweight="bold", y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, path)
