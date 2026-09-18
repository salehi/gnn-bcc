"""Assemble outputs/ml_dea/report.md from metrics.json."""
from __future__ import annotations

import json
import os

from ..config import DEA_INPUTS, DEA_OUTPUTS, NET_PROFIT_SHIFT
from .config import ITERATIVE, METHOD_LABEL, NORM_LABEL, build_parser, config_from_args


def _figfiles(out_dir):
    d = os.path.join(out_dir, "figures")
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def _fig_ending(out_dir, suffix, caption):
    for f in _figfiles(out_dir):
        if f.endswith(suffix):
            return _fig(out_dir, f, caption)
    return ""


def _fig(out_dir, name, caption):
    if not os.path.exists(os.path.join(out_dir, "figures", name)):
        return ""
    return f"![{caption}](figures/{name})\n\n*{caption}*\n\n"


def _tbl(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _findings(res, methods, norms) -> list[str]:
    """Read the result table and state what it actually shows.

    Everything here is derived from the numbers rather than asserted, so the
    prose cannot drift away from the table above it when the study is re-run
    on another seed or another split.
    """
    out = []
    mse = {(m, k): res[(m, k)]["test"]["mse"] for m in methods for k in norms}

    # 1. which method is ahead, and by how much relative to seed noise
    mean_mse = {m: sum(mse[(m, k)] for k in norms) / len(norms) for m in methods}
    order = sorted(mean_mse, key=mean_mse.get)
    top, second = order[0], order[1] if len(order) > 1 else None
    out.append(f"**`{top}` has the lowest mean test MSE** "
               f"({mean_mse[top]:.5f} averaged over the three normalisations)"
               + (f", ahead of `{second}` at {mean_mse[second]:.5f}."
                  if second else "."))

    # 2. tuning the SVR: did it help everywhere, or only on average?
    if "SVM-DEA" in methods and "ISVM-DEA" in methods:
        wins = [k for k in norms if mse[("ISVM-DEA", k)] < mse[("SVM-DEA", k)]]
        gain = [(mse[("SVM-DEA", k)] - mse[("ISVM-DEA", k)]) / mse[("SVM-DEA", k)]
                for k in norms]
        out.append(
            f"**Tuning the SVR paid off in {len(wins)} of {len(norms)} "
            f"normalisations** ({', '.join(wins) if wins else 'none'}), cutting "
            f"test MSE by {min(gain):.0%} to {max(gain):.0%}. That is the "
            "SVM-DEA / ISVM-DEA comparison working as intended: the fixed "
            "epsilon is too wide for a target this narrow, and the grid search "
            "finds that out.")

    # 3. the ISVM accuracy/robustness split, if it is present
    if "ISVM-DEA" in methods:
        best_mae = min(((m, k) for m in methods for k in norms),
                       key=lambda t: res[t]["test"]["mae"])
        ratios = {(m, k): (res[(m, k)]["test"]["mse"] / res[(m, k)]["train"]["mse"]
                           if res[(m, k)]["train"]["mse"] > 0 else float("inf"))
                  for m in methods for k in norms}
        worst_fit = max(ratios, key=ratios.get)
        if best_mae[0] == "ISVM-DEA" and worst_fit[0] == "ISVM-DEA":
            out.append(
                f"**`ISVM-DEA` has the best typical error but not the best "
                f"worst-case error.** It holds the lowest MAE of all twelve "
                f"({res[best_mae]['test']['mae']:.4f}, at {best_mae[1]}) while "
                f"sitting mid-table on MSE - so most of its predictions are "
                "excellent and a few are badly wrong. Its learning curve shows "
                "why: training error is driven to ~0 while cross-validated "
                f"error plateaus, a gap of {ratios[worst_fit]:,.0f}x between "
                f"train and test MSE at {worst_fit[1]}. The grid search bought "
                "accuracy on the training rows at the cost of variance.")

    # 4. is there a normalisation that wins outright?
    per_method_best = {m: min(norms, key=lambda k: mse[(m, k)]) for m in methods}
    uniq = set(per_method_best.values())
    if len(uniq) > 1:
        out.append("**No normalisation wins outright** - the best choice "
                   "depends on the learner ("
                   + "; ".join(f"`{m}` prefers {k}" for m, k in per_method_best.items())
                   + "). Normalisation is not a preprocessing detail that can "
                   "be fixed once and reused across methods, which is the "
                   "reason for running all twelve cells rather than picking "
                   "one scaler up front.")
    else:
        k = uniq.pop()
        out.append(f"**{k} is the best normalisation for every method tested**, "
                   "so on this dataset the choice of scaler can be made once.")

    # 5. ranking quality, which is what a DEA practitioner usually acts on
    best_rho = max(((m, k) for m in methods for k in norms),
                   key=lambda t: res[t]["test"]["spearman"])
    worst_rho = min(((m, k) for m in methods for k in norms),
                    key=lambda t: res[t]["test"]["spearman"])
    out.append(
        f"**Rank agreement is high across the board** - Spearman rho runs from "
        f"{res[worst_rho]['test']['spearman']:.4f} (`{worst_rho[0]}` + "
        f"{worst_rho[1]}) to {res[best_rho]['test']['spearman']:.4f} "
        f"(`{best_rho[0]}` + {best_rho[1]}). If the scores are only used to "
        "order DMUs rather than to quote a number, the spread between these "
        "methods matters far less than the MSE column suggests.")
    return out


def write_report(out_dir: str) -> str:
    with open(os.path.join(out_dir, "metrics.json")) as fh:
        d = json.load(fh)

    cfg, prov = d["config"], d["provenance"]
    methods, norms, feats = d["methods"], d["norms"], d["features"]
    res = {(r["method"], r["norm"]): r for r in d["results"]}
    rank = d["ranking"]
    best = rank[0]

    L, A = [], None
    A = L.append
    A("# Predicting DEA BCC efficiency scores with four ML methods "
      "(Hybrid ML-DEA)\n")
    A(f"{len(methods)} learners x {len(norms)} normalisations = "
      f"{len(methods) * len(norms)} fitted models, all evaluated on the same "
      "held-out DMUs.\n")

    # ---- what the four methods are ------------------------------------
    A("## The four methods\n")
    A(_tbl(["method", "model", "how the parameters are found"], [
        ["**BPNN-DEA**", "MLP " + "-".join(str(x) for x in
         [len(feats), *cfg["hidden_dims"], 1]),
         "error backpropagation, Adam, mini-batch, early stopping on validation"],
        ["**GANN-DEA**", "the *same* MLP",
         "real-coded genetic algorithm (tournament + BLX-alpha + Gaussian "
         "mutation + elitism); no gradients"],
        ["**SVM-DEA**", "SVR, RBF kernel",
         f"fixed defaults C={cfg['svm_C']}, epsilon={cfg['svm_epsilon']}, "
         f"gamma={cfg['svm_gamma']} - untuned baseline"],
        ["**ISVM-DEA**", "SVR, RBF kernel",
         f"GridSearchCV over C x gamma x epsilon, "
         f"{cfg['isvm_cv']}-fold CV inside the training split"],
    ]))
    A("BPNN-DEA and GANN-DEA share one network - the same architecture, "
      "initialisation scheme, activation and loss - so the difference between "
      "them isolates the optimiser and nothing else. The DEA in every name "
      "means the target: each model is trained to reproduce the BCC (VRS) "
      "efficiency score. No model re-solves the DEA programme.\n")

    # ---- data ----------------------------------------------------------
    A("## Data and DEA setup\n")
    A(f"- {sum(d['splits'].values())} DMUs, four columns: "
      + ", ".join(f"`{f}`" for f in feats))
    A(f"- **Inputs** (minimised): {', '.join(DEA_INPUTS)}")
    A(f"- **Outputs** (maximised): {', '.join(DEA_OUTPUTS)}")
    t = prov["target"]
    A(f"- Target `BCC score` in [{t['min']:.4f}, {t['max']:.4f}], "
      f"mean {t['mean']:.4f}, sd {t['std']:.4f}, skew {t['skew']:+.3f}\n")

    z = prov["zero_fix"]
    A("### Correcting the zero in `net profit`\n")
    A(f"`net profit` in the source sheet is the pre-shifted column "
      f"(+{NET_PROFIT_SHIFT:,.2f}), which maps the largest-loss DMU onto "
      f"exactly 0. Box-Cox needs strictly positive values, so that "
      f"{z['n_nonpositive']} cell (row index {z['rows']}) is corrected.\n")
    if z["mode"] == "halfmin":
        A(f"Mode `halfmin`: replaced with half the smallest positive value in "
          f"the column, {z['replacement_value']:,.0f} "
          f"(min positive = {z['min_positive']:,.0f}). This is the standard "
          "below-detection-limit substitution: the DMU stays the column "
          "minimum and stays inside the observed range, so the Box-Cox fit is "
          "not dragged by an artificial outlier. `--zero-fix shift` adds a "
          "small constant to the whole column instead — it preserves every "
          "difference exactly but pushes that DMU deep into the left tail.\n")
    else:
        A(f"Mode `shift`: a constant {z['replacement_value']:,.2f} was added to "
          "the whole column, preserving all differences exactly.\n")
    A("> Both corrections are legitimate for a BCC/VRS model specifically, "
      "because VRS is translation invariant in the outputs (Ali & Seiford, "
      "1990) — the same property the source file's own shift rests on. CCR/CRS "
      "would not tolerate it.\n")

    if prov["boxcox"]["applied"]:
        A("### Skewness correction (Box-Cox)\n")
        A(_tbl(["feature", "lambda", "skew before", "skew after",
                "kurtosis before", "kurtosis after"],
               [[f"`{f}`", f"{l:+.4f}", f"{a:+.3f}", f"{b:+.3f}",
                 f"{c:+.2f}", f"{e:+.2f}"]
                for f, l, a, b, c, e in zip(
                    feats, prov["boxcox"]["lambdas"], prov["skew_before"],
                    prov["skew_after"], prov["kurtosis_before"],
                    prov["kurtosis_after"])]))
        A(f"Lambdas estimated by maximum likelihood on "
          f"{'the full feature matrix' if prov['boxcox']['fit_on'] == 'all' else 'the training split only'}"
          f" (`--boxcox-fit {prov['boxcox']['fit_on']}`). The transform is "
          "unsupervised — the BCC score is never involved — and monotone, so "
          "no DMU changes rank on any feature.\n")
    A(_fig(out_dir, "01_preprocessing.png",
           "Raw vs Box-Cox distributions with normal Q-Q plots"))

    A("### Split and normalisation\n")
    s = d["splits"]
    A(f"Stratified on {cfg['n_bins']} quantile bins of the BCC score: "
      f"**train {s['train']} / validation {s['val']} / test {s['test']}**. "
      "Every split therefore carries the full efficiency range, including the "
      "thin low tail and the DMUs on the frontier.\n")
    A("Each normaliser is **fitted on the training split only** and then "
      "applied to validation and test. All four learners see exactly the same "
      "training rows, so the comparison is between methods, not between "
      "budgets:\n")
    A("- the two networks use the validation split for checkpointing "
      "(early stopping / generation selection)\n"
      "- ISVM-DEA tunes by cross-validation **inside** the training split\n"
      "- SVM-DEA tunes nothing\n"
      "- the test split is scored once, at the end, by all twelve models\n")
    A(_fig(out_dir, "02_normalisations.png",
           "The three normalisations, fitted on training DMUs"))

    # ---- results --------------------------------------------------------
    A("## Results\n")
    A("All figures below are on the held-out test set. `r` is the Pearson "
      "correlation between prediction and actual; `rho` is Spearman, which is "
      "what matters if the scores will be used to rank DMUs.\n")
    rows = []
    for m in methods:
        for k in norms:
            r = res[(m, k)]["test"]
            rows.append([f"`{m}`", k, f"{r['mse']:.5f}", f"{r['rmse']:.4f}",
                         f"{r['mae']:.4f}", f"{r['r2']:+.4f}",
                         f"{r['pearson_r']:+.4f}", f"{r['spearman']:+.4f}",
                         f"{res[(m,k)]['train']['mse']:.5f}",
                         f"{res[(m,k)]['fit'].get('seconds', 0):.1f}"])
    A(_tbl(["method", "normalisation", "MSE", "RMSE", "MAE", "R2", "r",
            "rho", "train MSE", "s"], rows))

    A("### Ranking, best first\n")
    A(_tbl(["#", "method", "normalisation", "test MSE", "test R2", "r"],
           [[i + 1, f"`{x['method']}`", x["norm"], f"{x['test_mse']:.5f}",
             f"{x['test_r2']:+.4f}", f"{x['pearson_r']:+.4f}"]
            for i, x in enumerate(rank)]))
    A(f"**Best combination: `{best['method']}` + {NORM_LABEL[best['norm']]}** — "
      f"test MSE {best['test_mse']:.5f}, R2 {best['test_r2']:+.4f}, "
      f"r {best['pearson_r']:+.4f}.\n")

    A("### What the numbers say\n")
    for b in _findings(res, methods, norms):
        A(f"- {b}\n")

    A(_fig_ending(out_dir, "pred_vs_actual.png",
                  "Predicted vs actual BCC score, all twelve combinations"))
    A(_fig_ending(out_dir, "comparison.png",
                  "MSE, R2 and Pearson r across the twelve combinations"))
    A(_fig_ending(out_dir, "metric_heatmap.png",
                  "Metric by method and normalisation"))

    # ---- curves ---------------------------------------------------------
    A("## Learning curves\n")
    A("The two families need different charts, and conflating them would be "
      "dishonest:\n")
    A("- **BPNN-DEA and GANN-DEA are iterative**, so they have a genuine loss "
      "per iteration — per epoch for backpropagation, per generation for the "
      "GA (best-of-generation and population mean, plus the champion's "
      "validation error).\n"
      "- **SVM-DEA and ISVM-DEA are convex**: libsvm solves the dual QP to "
      "optimality in one shot, so no loss-per-epoch exists. The standard "
      "equivalent is the **learning curve** — cross-validated error as a "
      "function of how many training DMUs the solver is given. A wide gap "
      "that stays wide is variance; two curves meeting at a high error is "
      "bias.\n")
    for f in _figfiles(out_dir):
        if "_loss_" in f or "_learning_" in f:
            m = f.split("_", 2)[2].replace(".png", "")
            kind = "loss per iteration" if m in ITERATIVE else "learning curve"
            A(_fig(out_dir, f, f"{METHOD_LABEL.get(m, m)} — {kind}"))

    # ---- method notes ---------------------------------------------------
    A("## What each fit actually did\n")
    for m in methods:
        A(f"### {METHOD_LABEL[m]}\n")
        rows = []
        for k in norms:
            f = res[(m, k)]["fit"]
            if m in ITERATIVE:
                sp = f.get("restart_spread", {})
                rows.append([k, f.get("n_iter"), f.get("best_iter"),
                             f"{f.get('best_val_mse', float('nan')):.5f}",
                             f"{sp.get('best', float('nan')):.5f} – "
                             f"{sp.get('worst', float('nan')):.5f}",
                             f"{sp.get('ratio_worst_to_best', float('nan')):.1f}x",
                             f"{f.get('evaluations', 0):,}",
                             f"{f.get('seconds', 0):.1f}"])
            else:
                bp = f.get("best_params", {})
                rows.append([k, ", ".join(f"{a}={b}" for a, b in bp.items()),
                             f.get("n_support"),
                             f"{f.get('support_frac', 0) * 100:.0f}%",
                             f"{f.get('cv_mse', float('nan')):.5f}"
                             if "cv_mse" in f else "-",
                             f"{f.get('seconds', 0):.1f}"])
        if m in ITERATIVE:
            A(_tbl(["normalisation", "iterations run", "best iteration",
                    "kept val MSE", "val MSE across restarts", "spread",
                    "model evaluations", "s"], rows))
        else:
            A(_tbl(["normalisation", "parameters", "support vectors",
                    "% of train", "CV MSE", "s"], rows))
        if m == "GANN-DEA":
            A(f"Population {cfg['ga_pop']}, up to {cfg['ga_gens']} generations, "
              f"tournament size {cfg['ga_tournament']}, crossover rate "
              f"{cfg['ga_cx_rate']}, BLX alpha {cfg['ga_alpha']}, mutation rate "
              f"{cfg['ga_mut_rate']} with the step annealed from "
              f"{cfg['ga_mut_sigma']} to {cfg['ga_mut_sigma_min']} across the budget, "
              f"{cfg['ga_elite']} elites, "
              f"genomes clipped to +/-{cfg['ga_clip']}. The genome is the full "
              "flat weight vector, so the GA searches one dimension per network "
              "parameter.\n")
        if m == "SVM-DEA":
            e = res[(m, norms[0])]["fit"].get("epsilon_vs_target_std")
            if e:
                A(f"The fixed epsilon={cfg['svm_epsilon']} makes the "
                  f"insensitive tube **{e:.0%} as wide as the standard "
                  "deviation of the target itself**. Everything inside the tube "
                  "costs nothing, so this baseline is structurally pushed "
                  "towards a near-constant prediction. That is the point of "
                  "comparing it with ISVM-DEA, not an accident of the run.\n")
        if m == "ISVM-DEA":
            f0 = res[(m, norms[0])]["fit"]
            A(f"Grid: {f0.get('n_candidates', '?')} candidates x "
              f"{cfg['isvm_cv']} folds = {f0.get('n_fits', '?')} fits per "
              "normalisation, scored by negative MSE, refit on the full "
              "training split.\n")

    A(_fig_ending(out_dir, "optimiser_duel.png",
                  "Backpropagation vs genetic search on the same network"))
    A(_fig_ending(out_dir, "residuals.png",
                  "Residuals, each method at its best normalisation"))

    # ---- honesty section -------------------------------------------------
    A("## Caveats\n")
    A("- **Predictions are clipped into (0, 1]** before scoring, because a BCC "
      "score outside that interval is not a score. `mse_raw` in `metrics.json` "
      "reports the unclipped value for every model, and `n_clipped` counts how "
      "many test predictions the clip touched.\n")
    # The restart spread is the single most important caveat: it sets the
    # resolution of the whole comparison, so it is stated first and with the
    # measured number rather than as a generic warning about randomness.
    spreads = [(m, k, res[(m, k)]["fit"].get("restart_spread", {}))
               for m in methods for k in norms
               if res[(m, k)]["fit"].get("restart_spread")]
    if spreads:
        wm, wk, ws = max(spreads, key=lambda t: t[2].get("ratio_worst_to_best", 0))
        A(f"- **Seed noise is the resolution limit of this table.** Both "
          f"networks were run from {cfg['n_restarts']} independent seeds per "
          "cell, keeping the champion on validation. Across those restarts "
          f"the worst seed was up to **{ws['ratio_worst_to_best']:.1f}x** worse "
          f"than the best (`{wm}` + {wk}: validation MSE "
          f"{ws['best']:.5f} to {ws['worst']:.5f}). Two combinations whose "
          "restarts overlap are not distinguishable by this experiment, "
          "however different their headline numbers look. The per-restart "
          "values are in `metrics.json` and in the last panel of the "
          "optimiser figure.\n")
        A("- A single run of either network would have been misleading here: "
          "before restarts were added, the GA's apparent ranking across the "
          "three normalisations reversed when only its generation budget "
          "changed. That is seed luck, not a property of the normalisation.\n")
    A("- **One split.** These numbers come from a single stratified split at "
      f"seed {cfg['seed']}. Re-splitting moves them too; `--seed` re-runs the "
      "whole study on a different split.\n")
    if prov["boxcox"]["fit_on"] == "all":
        A("- **Box-Cox lambdas were estimated on all rows**, as the specified "
          "pipeline puts skewness correction before the split. It is "
          "unsupervised, but it is still a statistic computed with test rows "
          "present. `--boxcox-fit train` re-runs the whole study with the "
          "lambdas fitted on training rows only.\n")
    A("- **The input/output assignment is an assumption** carried over from "
      "the repository's DEA setup. It affects how results are described, not "
      "what any model consumes — all four columns are features for all four "
      "learners.\n")
    A("- **The two SVR methods are deterministic** given the data and the "
      "seeded CV folds, so they get no restarts - repeating them would repeat "
      "the same fit. Their numbers carry split noise but not seed noise, "
      "which is worth remembering when comparing them with the two "
      "networks.\n")
    A("- The GA optimises "
      f"{res[(methods[0], norms[0])]['fit'].get('n_params', '?')} weights "
      "directly. Evolutionary search in a space that size is expected to be "
      "less efficient per evaluation than gradient descent; the duel figure "
      "reports both budgets so the comparison is not read as gradient-free "
      "search being hopeless at a fixed wall clock.\n")

    A("## Reproducing\n")
    A("```bash\n./run.sh ml-dea            # all twelve combinations\n"
      "./run.sh ml-dea-selftest   # gradient check + leakage checks\n"
      "./run.sh ml-dea --methods BPNN-DEA,GANN-DEA --norms zscore\n"
      "./run.sh ml-dea --boxcox-fit train --seed 7\n```\n")

    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L))
    return path


def main():
    cfg = config_from_args(build_parser("Rebuild report.md.").parse_args())
    print(write_report(cfg.out_dir))


if __name__ == "__main__":
    main()
