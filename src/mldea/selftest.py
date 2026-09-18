"""Self-tests for the ML-DEA study.

Run with:  ./run.sh ml-dea-selftest

Each check guards a property that, if broken, would silently change the
conclusions rather than raise an error:

  * the hand-written backpropagation gradient is verified against central
    finite differences - a wrong gradient would make BPNN look bad and GANN
    look good for no real reason
  * the vectorised population forward pass must agree exactly with the
    single-network forward pass, or the GA is ranking networks by a different
    function than the one being reported
  * Box-Cox must reduce skew and preserve every DMU's rank
  * normalisers must be fitted on training rows only
"""
from __future__ import annotations

import sys

import numpy as np
from scipy import stats

from ..config import FRONTIER_EPS
from . import nn, svm
from .config import MLConfig, build_parser, config_from_args
from .preprocess import NET_PROFIT_COL, make_normaliser, prepare
from .run import evaluate, fit_one

PASS, FAIL = "  PASS  ", "  FAIL  "
_results = []


def check(name, cond, detail=""):
    _results.append(bool(cond))
    print(f"{PASS if cond else FAIL} {name}" + (f"   {detail}" if detail else ""))


def main():
    args = build_parser("ML-DEA self-tests.").parse_args()
    cfg = config_from_args(args)
    # small budgets: these tests check correctness, not convergence
    fast = MLConfig(**{**cfg.to_dict(), "hidden_dims": tuple(cfg.hidden_dims),
                       "bp_epochs": 80, "bp_patience": 80,
                       "ga_pop": 40, "ga_gens": 40, "ga_patience_frac": 1.0,
                       "isvm_cv": 3})

    print("=" * 84)
    print("  ML-DEA self-tests")
    print("=" * 84)

    # --- gradient check -------------------------------------------------
    rng = np.random.default_rng(0)
    spec = nn.MLPSpec((4, 5, 3, 1), "tanh")
    theta = rng.normal(0, 0.7, spec.n_params)
    Xg = rng.normal(0, 1, (20, 4))
    yg = rng.uniform(0.2, 1.0, 20)
    _, g = nn.loss_grad(theta, Xg, yg, spec)
    eps = 1e-6
    idx = rng.choice(spec.n_params, size=40, replace=False)
    num = np.empty(len(idx))
    for a, i in enumerate(idx):
        tp = theta.copy(); tp[i] += eps
        tm = theta.copy(); tm[i] -= eps
        num[a] = (nn.mse(tp, Xg, yg, spec) - nn.mse(tm, Xg, yg, spec)) / (2 * eps)
    rel = np.max(np.abs(num - g[idx]) / (np.abs(num) + np.abs(g[idx]) + 1e-12))
    check("backprop gradient matches central finite differences",
          rel < 1e-6, f"max relative error = {rel:.2e} over 40 coordinates")

    _, g2 = nn.loss_grad(theta, Xg, yg, spec, l2=0.1)
    wm = nn.weight_mask(spec)
    check("L2 penalises weights and never biases",
          np.allclose(g2 - g, 0.1 * theta * wm) and wm.sum() == 5 * 4 + 5 * 3 + 3,
          f"{int(wm.sum())} weights, {int((1 - wm).sum())} biases")

    # --- population forward ---------------------------------------------
    pop = nn.init_theta(spec, rng, size=7)
    P = nn.forward(pop, Xg, spec)
    S = np.stack([nn.forward(pop[i], Xg, spec) for i in range(7)])
    check("vectorised population forward == single-network forward",
          np.array_equal(P, S), f"max |diff| = {np.abs(P - S).max():.2e}")
    check("genome length matches the architecture",
          spec.n_params == 4 * 5 + 5 + 5 * 3 + 3 + 3 * 1 + 1, f"{spec.n_params}")

    # --- data + preprocessing -------------------------------------------
    prep = prepare(fast)
    z = prep.provenance["zero_fix"]
    check("dataset is 927 x 4", prep.X_raw.shape == (927, 4), str(prep.X_raw.shape))
    check("the zero in `net profit` is found and corrected",
          z["n_nonpositive"] == 1 and prep.X_fixed[:, NET_PROFIT_COL].min() > 0,
          f"{z['n_nonpositive']} cell, replaced with {z.get('replacement_value', 0):,.0f}")
    check("corrected DMU is still the column minimum",
          int(np.argmin(prep.X_fixed[:, NET_PROFIT_COL])) == z["rows"][0])
    check("all features strictly positive before Box-Cox", prep.X_fixed.min() > 0)

    sb, sa = prep.provenance["skew_before"], prep.provenance["skew_after"]
    check("Box-Cox reduces |skew| on every feature",
          all(abs(b) < abs(a) for a, b in zip(sb, sa)),
          ", ".join(f"{a:+.1f}->{b:+.2f}" for a, b in zip(sb, sa)))
    ranks_ok = all(
        stats.spearmanr(prep.X_fixed[:, j], prep.X_bc[:, j]).statistic > 1 - 1e-12
        for j in range(prep.X_bc.shape[1]))
    check("Box-Cox is monotone: no DMU changes rank on any feature", ranks_ok)

    # --- splits -----------------------------------------------------------
    tr, va, te = prep.train_i, prep.val_i, prep.test_i
    check("splits are disjoint and cover the dataset",
          not (set(tr) & set(va)) and not (set(tr) & set(te))
          and not (set(va) & set(te)) and len(tr) + len(va) + len(te) == len(prep.y),
          f"{len(tr)}/{len(va)}/{len(te)}")
    check("every split carries frontier DMUs (score = 1)",
          all(int((prep.y[s] >= 1 - FRONTIER_EPS).sum()) > 0 for s in (tr, va, te)),
          f"{[int((prep.y[s] >= 1 - FRONTIER_EPS).sum()) for s in (tr, va, te)]}")

    # --- normalisers fitted on train only ---------------------------------
    for kind in prep.views:
        sc_tr = prep.views[kind]["scaler"]
        sc_all = make_normaliser(kind).fit(prep.X_bc)
        a = np.asarray(sc_tr.transform(prep.X_bc[:3]))
        b = np.asarray(sc_all.transform(prep.X_bc[:3]))
        check(f"[{kind}] scaler statistics come from training rows only",
              not np.allclose(a, b), f"max |diff| on 3 rows = {np.abs(a - b).max():.4f}")
    Ztr = prep.views["zscore"]["Ztr"]
    check("[zscore] training rows are centred, test rows are not forced to be",
          abs(Ztr.mean()) < 1e-10 and abs(prep.views["zscore"]["Zte"].mean()) > 1e-10,
          f"train mean {Ztr.mean():+.2e}, test mean "
          f"{prep.views['zscore']['Zte'].mean():+.4f}")

    # --- does a test DMU influence the training matrix? -------------------
    # With --boxcox-fit train nothing about a test row can reach training.
    # With the default (fit on all rows) the lambdas are estimated with test
    # rows present; that is the documented caveat, and this quantifies it.
    rng2 = np.random.default_rng(1)
    for mode in ("train", "all"):
        c = MLConfig(**{**fast.to_dict(), "hidden_dims": tuple(fast.hidden_dims),
                        "boxcox_fit": mode})
        base = prepare(c, ("zscore",))
        import src.mldea.preprocess as pp
        real_load = pp.load_dataset

        def corrupt(cfg_, _r=real_load, _rng=rng2, _te=base.test_i):
            X, y, n = _r(cfg_)
            X = X.copy()
            X[_te] = X[_te] * _rng.uniform(0.5, 2.0, X[_te].shape)
            return X, y, n

        pp.load_dataset = corrupt
        try:
            pert = prepare(c, ("zscore",))
        finally:
            pp.load_dataset = real_load
        d = float(np.abs(base.views["zscore"]["Ztr"]
                         - pert.views["zscore"]["Ztr"]).max())
        if mode == "train":
            check("[boxcox-fit=train] corrupting test rows leaves the training "
                  "matrix bit-identical", d == 0.0, f"max |diff| = {d:.2e}")
        else:
            print(f"  (info)  [boxcox-fit=all] the same corruption moves the "
                  f"training matrix by {d:.2e} - unsupervised, but non-zero; "
                  f"see the caveats section of the report")

    # --- the learners -----------------------------------------------------
    v = prep.views["zscore"]
    spec_full = nn.MLPSpec((4, *fast.hidden_dims, 1), fast.activation)
    theta0 = nn.init_theta(spec_full, np.random.default_rng(fast.seed))
    mse0 = float(nn.mse(theta0, v["Ztr"], v["ytr"], spec_full))

    bp = nn.train_bpnn(fast, v["Ztr"], v["ytr"], v["Zva"], v["yva"], fast.seed)
    bp_mse = float(nn.mse(bp["theta"], v["Ztr"], v["ytr"], spec_full))
    check("BPNN improves on its own initialisation", bp_mse < mse0,
          f"train MSE {mse0:.5f} -> {bp_mse:.5f} in {bp['n_iter']} epochs")
    check("BPNN's training loss is finite and non-increasing overall",
          np.isfinite(bp["history"]["train_mse"]).all()
          and bp["history"]["train_mse"][-1] < bp["history"]["train_mse"][0])

    ga = nn.train_gann(fast, v["Ztr"], v["ytr"], v["Zva"], v["yva"], fast.seed)
    check("GANN improves on generation zero",
          ga["history"]["best_train_mse"][-1] < ga["history"]["best_train_mse"][0],
          f"best-of-generation {ga['history']['best_train_mse'][0]:.5f} -> "
          f"{ga['history']['best_train_mse'][-1]:.5f}")
    check("GANN's best-of-generation is monotone (elitism preserves the best)",
          all(b <= a + 1e-15 for a, b in zip(ga["history"]["best_train_mse"],
                                             ga["history"]["best_train_mse"][1:])))
    check("both networks have the same parameter count",
          bp["n_params"] == ga["n_params"] == spec_full.n_params,
          f"{spec_full.n_params} weights, identical architecture")

    sv = svm.fit_svm(fast, v["Ztr"], v["ytr"], v["Zva"], v["yva"])
    isv = svm.fit_isvm(fast, v["Ztr"], v["ytr"], v["Zva"], v["yva"])
    check("ISVM's grid search improves on the fixed SVR's validation MSE",
          isv["best_val_mse"] <= sv["best_val_mse"],
          f"{sv['best_val_mse']:.5f} -> {isv['best_val_mse']:.5f} "
          f"(best params {isv['best_params']})")
    check("fixed epsilon really is wide relative to the target",
          sv["epsilon_vs_target_std"] > 0.5,
          f"epsilon = {sv['epsilon_vs_target_std']:.0%} of the target's sd")

    # --- determinism ------------------------------------------------------
    p1 = fit_one("BPNN-DEA", fast, v)["predict"](v["Zte"])
    p2 = fit_one("BPNN-DEA", fast, v)["predict"](v["Zte"])
    check("BPNN is deterministic at a fixed seed", np.array_equal(p1, p2))
    q1 = fit_one("GANN-DEA", fast, v)["predict"](v["Zte"])
    q2 = fit_one("GANN-DEA", fast, v)["predict"](v["Zte"])
    check("GANN is deterministic at a fixed seed", np.array_equal(q1, q2))

    # --- metrics ----------------------------------------------------------
    m = evaluate(v["yte"], v["yte"].copy())
    check("a perfect prediction scores R2 = 1, MSE = 0, r = 1",
          abs(m["r2"] - 1) < 1e-12 and m["mse"] < 1e-24
          and abs(m["pearson_r"] - 1) < 1e-12)
    import warnings
    with warnings.catch_warnings():        # a constant prediction has no correlation
        warnings.simplefilter("ignore")
        m2 = evaluate(v["yte"], np.full_like(v["yte"], v["ytr"].mean()))
        m3c = evaluate(v["yte"], np.full_like(v["yte"], 5.0))
    check("predicting the training mean scores R2 <= 0",
          m2["r2"] <= 0, f"R2 = {m2['r2']:+.4f} (the baseline every model must beat)")
    check("predictions outside (0, 1] are clipped and counted",
          m3c["n_clipped"] == len(v["yte"]) and m3c["mse"] < m3c["mse_raw"])

    print("=" * 84)
    n_ok = sum(_results)
    print(f"  {n_ok}/{len(_results)} checks passed")
    print("=" * 84)
    sys.exit(0 if n_ok == len(_results) else 1)


if __name__ == "__main__":
    main()
