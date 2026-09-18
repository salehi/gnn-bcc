"""SVM-DEA and ISVM-DEA.

SVM-DEA is an RBF support vector regressor left at fixed library defaults
(C=1, epsilon=0.1, gamma='scale'). ISVM-DEA is the same estimator with
C / gamma / epsilon chosen by exhaustive GridSearchCV inside the training
split. The grid search never sees validation or test DMUs: it cross-validates
within the training rows only, so "improved" here means better-tuned, not
better-informed.

Why the untuned baseline is expected to look bad, stated up front rather than
discovered in the results: the BCC score has a standard deviation of about
0.155, and epsilon=0.1 makes the insensitive tube two thirds as wide as the
entire signal. Everything inside the tube costs nothing, so the fixed SVR is
structurally encouraged to return something close to a constant. That gap is
the whole point of the ISVM comparison.

Neither estimator is iterative in any sense that yields a loss-per-epoch
curve; libsvm solves a convex QP to optimality. The honest analogue is the
learning curve - error as a function of how much training data the solver is
given - which is what `learning_curve_for` computes.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, learning_curve
from sklearn.svm import SVR

from .config import ISVM_GRID, MLConfig


def fit_svm(cfg: MLConfig, Ztr, ytr, Zva, yva) -> dict:
    """Fixed-parameter SVR - the untuned baseline."""
    t0 = time.time()
    params = dict(kernel=cfg.svm_kernel, C=cfg.svm_C,
                  epsilon=cfg.svm_epsilon, gamma=cfg.svm_gamma)
    model = SVR(**params).fit(Ztr, ytr)
    va = float(np.mean((model.predict(Zva) - yva) ** 2))
    return {
        "model": model, "params": params, "best_params": params,
        "n_support": int(model.support_.size),
        "support_frac": float(model.support_.size / len(ytr)),
        "best_val_mse": va, "seconds": time.time() - t0,
        "curve_kind": "learning", "n_fits": 1,
        "epsilon_vs_target_std": float(cfg.svm_epsilon / np.std(ytr)),
    }


def fit_isvm(cfg: MLConfig, Ztr, ytr, Zva, yva) -> dict:
    """SVR with an exhaustive grid search over C, gamma and epsilon,
    cross-validated inside the training split."""
    t0 = time.time()
    cv = KFold(n_splits=cfg.isvm_cv, shuffle=True, random_state=cfg.seed)
    gs = GridSearchCV(
        SVR(kernel=cfg.svm_kernel), ISVM_GRID, cv=cv,
        scoring="neg_mean_squared_error", n_jobs=cfg.isvm_jobs, refit=True,
    ).fit(Ztr, ytr)

    model = gs.best_estimator_
    va = float(np.mean((model.predict(Zva) - yva) ** 2))
    cvres = gs.cv_results_
    order = np.argsort(-cvres["mean_test_score"])
    top = [{"params": cvres["params"][i],
            "cv_mse": float(-cvres["mean_test_score"][i]),
            "cv_std": float(cvres["std_test_score"][i])} for i in order[:5]]

    return {
        "model": model, "params": dict(gs.best_params_),
        "best_params": dict(gs.best_params_),
        "grid": {k: [str(v) for v in vs] for k, vs in ISVM_GRID.items()},
        "n_candidates": int(len(cvres["params"])),
        "n_fits": int(len(cvres["params"]) * cfg.isvm_cv),
        "cv_mse": float(-gs.best_score_),
        "cv_top5": top,
        "n_support": int(model.support_.size),
        "support_frac": float(model.support_.size / len(ytr)),
        "best_val_mse": va, "seconds": time.time() - t0,
        "curve_kind": "learning",
        "epsilon_vs_target_std": float(gs.best_params_["epsilon"] / np.std(ytr)),
    }


def learning_curve_for(estimator, Ztr, ytr, cfg: MLConfig, n_points: int = 8) -> dict:
    """Error against training-set size, cross-validated.

    Stands in for the loss curve that a convex solver does not have. A gap
    that stays wide as data is added means variance; two curves that meet at a
    high error mean bias.
    """
    cv = KFold(n_splits=cfg.isvm_cv, shuffle=True, random_state=cfg.seed)
    sizes, train_scores, val_scores = learning_curve(
        estimator, Ztr, ytr,
        train_sizes=np.linspace(0.15, 1.0, n_points),
        cv=cv, scoring="neg_mean_squared_error", n_jobs=cfg.isvm_jobs,
        shuffle=True, random_state=cfg.seed,
    )
    return {
        "train_sizes": sizes.tolist(),
        "train_mse": (-train_scores.mean(axis=1)).tolist(),
        "train_mse_std": train_scores.std(axis=1).tolist(),
        "cv_mse": (-val_scores.mean(axis=1)).tolist(),
        "cv_mse_std": val_scores.std(axis=1).tolist(),
    }


def predict(fitted: dict, X: np.ndarray) -> np.ndarray:
    return fitted["model"].predict(X)
