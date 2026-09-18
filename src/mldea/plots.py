"""Figures for the ML-DEA study. Headless Agg, 150 dpi, styled to match the
rest of the repository.

Curve semantics differ by method and the charts say so explicitly:
BPNN and GANN are iterative and get a genuine loss-per-iteration curve; SVR
and ISVR solve a convex problem in one shot and get a learning curve - error
against training-set size - which is the standard stand-in.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..plots import (C_ACC, C_GREY, C_TEST, C_TRAIN, C_VAL, DPI, _save, plt)
from .config import ITERATIVE, METHOD_LABEL, NORM_LABEL

M_COLOR = {"BPNN-DEA": C_TRAIN, "GANN-DEA": C_ACC,
           "SVM-DEA": C_GREY, "ISVM-DEA": C_TEST}
N_COLOR = {"zscore": "#3F6FB0", "minmax": "#E08A3C", "robust": "#4C9A6B"}


def _short(name: str) -> str:
    return name.replace("-DEA", "")


# --------------------------------------------------------------------------
# 1. preprocessing
# --------------------------------------------------------------------------
def plot_preprocessing(prep, path):
    names, X0, X1, y = prep.names, prep.X_fixed, prep.X_bc, prep.y
    sb = prep.provenance["skew_before"]; sa = prep.provenance["skew_after"]
    lam = prep.provenance["boxcox"].get("lambdas", [np.nan] * len(names))
    zinfo = prep.provenance["zero_fix"]

    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    for j, nm in enumerate(names):
        ax = axes[0, j]
        ax.hist(X0[:, j], bins=50, color=C_GREY, edgecolor="white", linewidth=0.3)
        ax.set_yscale("log")
        ax.set_title(f"{nm}\nraw (zero-corrected)  skew={sb[j]:+.2f}", fontsize=8)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.tick_params(labelsize=7)
        if zinfo.get("applied") and nm == zinfo["column"]:
            ax.axvline(zinfo["replacement_value"], color=C_ACC, ls="--", lw=1.2,
                       label="corrected zero")
            ax.legend(fontsize=6)

        ax = axes[1, j]
        ax.hist(X1[:, j], bins=50, color=C_TRAIN, edgecolor="white", linewidth=0.3)
        ax.set_title(f"Box-Cox  $\\lambda$={lam[j]:+.4f}  skew={sa[j]:+.2f}", fontsize=8)
        ax.tick_params(labelsize=7)

        ax = axes[2, j]
        stats.probplot(X1[:, j], dist="norm", plot=ax)
        ax.get_lines()[0].set(marker="o", ms=2, color=C_TRAIN, alpha=0.5)
        ax.get_lines()[1].set(color=C_ACC, lw=1.2)
        ax.set_title("normal Q-Q after Box-Cox", fontsize=8)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(labelsize=7)

    fig.suptitle("Preprocessing: one zero corrected in `net profit`, then "
                 "per-feature Box-Cox", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, path)


def plot_normalisations(prep, norms, path):
    fig, axes = plt.subplots(1, len(norms), figsize=(5 * len(norms), 4.2),
                             squeeze=False)
    short = [n.replace("total ", "").replace("operational ", "op. ")
             for n in prep.names]
    for a, kind in enumerate(norms):
        ax = axes[0, a]
        Ztr = prep.views[kind]["Ztr"]
        bp = ax.boxplot([Ztr[:, j] for j in range(Ztr.shape[1])],
                        patch_artist=True, widths=0.55,
                        flierprops=dict(marker=".", ms=2, alpha=0.35))
        for b in bp["boxes"]:
            b.set(facecolor=N_COLOR[kind], alpha=0.55)
        for med in bp["medians"]:
            med.set(color="black", lw=1.2)
        ax.set_xticklabels(short, rotation=20, ha="right", fontsize=7)
        ax.axhline(0, color=C_GREY, lw=0.8, ls=":")
        ax.set_title(NORM_LABEL[kind])
        if a == 0:
            ax.set_ylabel("normalised value (training rows)")
    fig.suptitle("The three normalisations, each fitted on the training split only",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, path)


# --------------------------------------------------------------------------
# 2. curves
# --------------------------------------------------------------------------
def plot_loss_curves(method, per_norm, path):
    """Train/validation MSE per iteration for the two iterative methods."""
    norms = list(per_norm)
    fig, axes = plt.subplots(1, len(norms), figsize=(5 * len(norms), 4.0),
                             squeeze=False)
    for a, kind in enumerate(norms):
        ax = axes[0, a]
        fit = per_norm[kind]
        h = fit["history"]
        if method == "GANN-DEA":
            x = h["gen"]
            ax.plot(x, h["best_train_mse"], color=C_TRAIN, lw=1.4,
                    label="best-of-generation (train)")
            ax.plot(x, h["mean_train_mse"], color=C_GREY, lw=1.0, alpha=0.8,
                    label="population mean (train)")
            ax.plot(x, h["val_mse"], color=C_VAL, lw=1.2, label="champion (val)")
            ax.set_xlabel("generation")
        else:
            x = h["epoch"]
            ax.plot(x, h["train_mse"], color=C_TRAIN, lw=1.3, label="train")
            ax.plot(x, h["val_mse"], color=C_VAL, lw=1.3, label="validation")
            ax.set_xlabel("epoch")
        ax.axvline(fit["best_iter"], color=C_ACC, ls="--", lw=1.1,
                   label=f"checkpoint @ {fit['best_iter']}")
        ax.set_yscale("log")
        ax.set_ylabel("MSE")
        ax.set_title(f"{NORM_LABEL[kind]}\nbest val MSE = {fit['best_val_mse']:.5f}",
                     fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"{METHOD_LABEL[method]} - loss per iteration",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, path)


def plot_learning_curves(method, per_norm, path):
    """Error against training-set size, for the convex (one-shot) solvers."""
    norms = list(per_norm)
    fig, axes = plt.subplots(1, len(norms), figsize=(5 * len(norms), 4.0),
                             squeeze=False)
    for a, kind in enumerate(norms):
        ax = axes[0, a]
        lc = per_norm[kind]["learning_curve"]
        s = np.asarray(lc["train_sizes"])
        tr = np.asarray(lc["train_mse"]); trs = np.asarray(lc["train_mse_std"])
        cv = np.asarray(lc["cv_mse"]); cvs = np.asarray(lc["cv_mse_std"])
        ax.plot(s, tr, "o-", color=C_TRAIN, ms=3.5, lw=1.3, label="train MSE")
        ax.fill_between(s, tr - trs, tr + trs, color=C_TRAIN, alpha=0.15)
        ax.plot(s, cv, "s-", color=C_VAL, ms=3.5, lw=1.3,
                label="cross-validated MSE")
        ax.fill_between(s, cv - cvs, cv + cvs, color=C_VAL, alpha=0.15)
        ax.set_xlabel("training DMUs used")
        ax.set_ylabel("MSE")
        p = per_norm[kind].get("best_params", {})
        ptxt = ", ".join(f"{k}={v}" for k, v in p.items() if k != "kernel")
        ax.set_title(f"{NORM_LABEL[kind]}\n{ptxt}", fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle(f"{METHOD_LABEL[method]} - learning curve "
                 "(convex solver: no loss-per-epoch exists)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, path)


# --------------------------------------------------------------------------
# 3. prediction vs actual
# --------------------------------------------------------------------------
def plot_pred_vs_actual(results, methods, norms, path):
    fig, axes = plt.subplots(len(methods), len(norms),
                             figsize=(4.1 * len(norms), 3.9 * len(methods)),
                             squeeze=False)
    for i, m in enumerate(methods):
        for j, k in enumerate(norms):
            ax = axes[i, j]
            r = results[(m, k)]
            yt = np.asarray(r["y_test"]); yp = np.asarray(r["pred_test"])
            ax.scatter(yt, yp, s=9, alpha=0.55, color=M_COLOR[m],
                       edgecolors="none")
            lo = min(yt.min(), yp.min()) - 0.02
            hi = max(yt.max(), yp.max()) + 0.02
            ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls="--",
                    label="y = x")
            b, a = np.polyfit(yt, yp, 1)
            xs = np.linspace(lo, hi, 20)
            ax.plot(xs, b * xs + a, color=C_ACC, lw=1.2, label="fit")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            mt = r["test"]
            ax.text(0.03, 0.97,
                    f"MSE={mt['mse']:.5f}\n$R^2$={mt['r2']:+.4f}\n"
                    f"r={mt['pearson_r']:+.4f}\n$\\rho$={mt['spearman']:+.4f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8,
                              ec=C_GREY, lw=0.5))
            if i == 0:
                ax.set_title(NORM_LABEL[k], fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{_short(m)}\npredicted BCC", fontsize=9)
            if i == len(methods) - 1:
                ax.set_xlabel("actual BCC score")
            if i == 0 and j == 0:
                ax.legend(fontsize=6.5, loc="lower right")
    fig.suptitle("Predicted vs actual BCC score on the held-out test DMUs "
                 f"(n={len(results[(methods[0], norms[0])]['y_test'])})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return _save(fig, path)


# --------------------------------------------------------------------------
# 4. comparison
# --------------------------------------------------------------------------
def plot_comparison(results, methods, norms, path):
    keys = [("mse", "test MSE (lower is better)", False),
            ("r2", "test $R^2$ (higher is better)", True),
            ("pearson_r", "Pearson r, predicted vs actual", True)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    width = 0.8 / len(norms)
    xs = np.arange(len(methods))
    for ax, (key, title, higher) in zip(axes, keys):
        for j, k in enumerate(norms):
            vals = [results[(m, k)]["test"][key] for m in methods]
            pos = xs + (j - (len(norms) - 1) / 2) * width
            ax.bar(pos, vals, width * 0.92, label=NORM_LABEL[k],
                   color=N_COLOR[k], edgecolor="white", linewidth=0.6)
            for p, v in zip(pos, vals):
                ax.annotate(f"{v:.4f}" if key != "mse" else f"{v:.4f}",
                            (p, v), ha="center", fontsize=6.2, rotation=90,
                            xytext=(0, 3 if v >= 0 else -12),
                            textcoords="offset points")
        ax.set_xticks(xs); ax.set_xticklabels([_short(m) for m in methods], fontsize=9)
        ax.set_title(title)
        ax.axhline(0, color="black", lw=0.8)
        if key == "mse":
            ax.set_yscale("log")
        ax.legend(fontsize=7)
    fig.suptitle("Twelve combinations on the held-out test set",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, path)


def plot_heatmap(results, methods, norms, path):
    keys = [("mse", "test MSE", "viridis_r"), ("r2", "test $R^2$", "viridis"),
            ("pearson_r", "Pearson r", "viridis"),
            ("spearman", "Spearman $\\rho$ (rank quality)", "viridis")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.9))
    for ax, (key, title, cmap) in zip(axes, keys):
        M = np.array([[results[(m, k)]["test"][key] for k in norms] for m in methods])
        im = ax.imshow(M, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(norms))); ax.set_xticklabels(norms, fontsize=8)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([_short(m) for m in methods], fontsize=8)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                rel = (v - M.min()) / (np.ptp(M) + 1e-12)
                ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=8,
                        color="white" if rel < 0.5 else "black")
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Metric by method and normalisation", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, path)


def plot_residuals(results, methods, norms, path):
    fig, axes = plt.subplots(2, len(methods), figsize=(4.2 * len(methods), 7.2),
                            squeeze=False)
    for i, m in enumerate(methods):
        best_k = min(norms, key=lambda k: results[(m, k)]["test"]["mse"])
        r = results[(m, best_k)]
        yt = np.asarray(r["y_test"]); yp = np.asarray(r["pred_test"])
        res = yp - yt

        ax = axes[0, i]
        ax.hist(res, bins=30, color=M_COLOR[m], edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="black", lw=1.0, ls="--")
        ax.axvline(res.mean(), color=C_ACC, lw=1.2,
                   label=f"bias {res.mean():+.4f}")
        ax.set_title(f"{_short(m)}  ({best_k})", fontsize=9)
        ax.set_xlabel("residual (pred - actual)")
        ax.legend(fontsize=7)

        ax = axes[1, i]
        ax.scatter(yp, res, s=9, alpha=0.55, color=M_COLOR[m], edgecolors="none")
        ax.axhline(0, color="black", lw=1.0, ls="--")
        ax.set_xlabel("predicted BCC"); ax.set_ylabel("residual" if i == 0 else "")
        ax.set_title("heteroscedasticity check", fontsize=8)
    fig.suptitle("Residuals on the test set, each method at its best normalisation",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, path)


def plot_optimiser_duel(results, methods, norms, path):
    """BPNN vs GANN: the same network, two searches. Iterations are not
    comparable units, so budget is reported in model evaluations and seconds."""
    pair = [m for m in ("BPNN-DEA", "GANN-DEA") if m in methods]
    if len(pair) < 2:
        return ""
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    xs = np.arange(len(norms)); w = 0.36

    for i, m in enumerate(pair):
        ax = axes[0]
        ax.bar(xs + (i - 0.5) * w, [results[(m, k)]["test"]["mse"] for k in norms],
               w * 0.9, label=_short(m), color=M_COLOR[m])
        ax = axes[1]
        ax.bar(xs + (i - 0.5) * w, [results[(m, k)]["fit"]["seconds"] for k in norms],
               w * 0.9, label=_short(m), color=M_COLOR[m])
        ax = axes[2]
        ax.bar(xs + (i - 0.5) * w,
               [results[(m, k)]["fit"]["evaluations"] for k in norms],
               w * 0.9, label=_short(m), color=M_COLOR[m])

    # Seed-to-seed spread. This panel is the reason the other three can be
    # read at all: if the restarts of one method straddle the bar of another,
    # the difference between those bars is seed luck, not method quality.
    ax = axes[3]
    for i, m in enumerate(pair):
        for j, k in enumerate(norms):
            vals = [r["val_mse"] for r in
                    results[(m, k)]["fit"].get("restarts", [])]
            if not vals:
                continue
            x = j + (i - 0.5) * w
            ax.scatter([x] * len(vals), vals, s=22, color=M_COLOR[m],
                       alpha=0.75, edgecolors="none",
                       label=_short(m) if j == 0 else None)
            ax.plot([x - 0.1, x + 0.1], [min(vals)] * 2, color=M_COLOR[m], lw=1.6)
    ax.set_xticks(xs); ax.set_xticklabels(norms, fontsize=8)
    ax.set_yscale("log")
    ax.set_title("validation MSE per restart\n(bar = the seed that was kept)")
    ax.legend(fontsize=8)

    for ax, t, logy in ((axes[0], "test MSE", True),
                        (axes[1], "wall clock (s, all restarts)", False),
                        (axes[2], "model evaluations\n(grad steps vs fitness evals)", True)):
        ax.set_xticks(xs); ax.set_xticklabels(norms, fontsize=8)
        ax.set_title(t)
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=8)
    fig.suptitle("Same 4-24-12-1 network, two optimisers: backpropagation vs "
                 "genetic search", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, path)
