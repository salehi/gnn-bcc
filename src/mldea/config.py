"""Configuration for the Hybrid ML-DEA study.

Four learners x three normalisations = twelve fitted models. The DEA side of
every name means the same thing: the model is trained to reproduce the BCC
(VRS) efficiency score that DEA assigns to a DMU from its four
input/output columns. No learner re-solves the DEA programme.

  BPNN-DEA   MLP, weights found by error backpropagation with Adam
  GANN-DEA   the SAME MLP, weights found by a real-coded genetic algorithm
  SVM-DEA    support vector regression, parameters left at their fixed
             library defaults - the untuned baseline
  ISVM-DEA   the same SVR with C / gamma / epsilon selected by GridSearchCV

BPNN and GANN share one network so the comparison isolates the optimiser,
which is the only thing that differs between them.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

METHODS = ("BPNN-DEA", "GANN-DEA", "SVM-DEA", "ISVM-DEA")
NORMALISERS = ("zscore", "minmax", "robust")

# Long-form labels used in the report and the figures.
METHOD_LABEL = {
    "BPNN-DEA": "BPNN-DEA (MLP + backprop/Adam)",
    "GANN-DEA": "GANN-DEA (same MLP + genetic algorithm)",
    "SVM-DEA": "SVM-DEA (SVR, fixed parameters)",
    "ISVM-DEA": "ISVM-DEA (SVR + GridSearchCV)",
}
NORM_LABEL = {
    "zscore": "z-score (StandardScaler)",
    "minmax": "min-max (MinMaxScaler)",
    "robust": "robust (median / IQR)",
}
ITERATIVE = ("BPNN-DEA", "GANN-DEA")   # have a real per-iteration loss curve


@dataclass
class MLConfig:
    # --- data ---------------------------------------------------------
    data_path: str = "/app/data/data_with_bcc_score.xlsx"
    sheet: str = "Data"
    out_dir: str = "/app/outputs/ml_dea"

    # --- preprocessing ------------------------------------------------
    # `net profit` carries one exact zero (the largest-loss DMU, which the
    # source file's +9,101,232,179.17 shift mapped onto 0). Box-Cox needs
    # x > 0, so that single cell has to be corrected:
    #   halfmin  replace it with half the smallest positive value in the
    #            column - the standard below-detection-limit substitution.
    #            It keeps the DMU as the column minimum and inside the
    #            observed range.
    #   shift    add a small constant to the whole column instead, which
    #            preserves every difference exactly but pushes that one DMU
    #            far into the left tail of the Box-Cox transform.
    zero_fix: str = "halfmin"
    shift_frac: float = 1e-3     # only used by zero_fix="shift"
    boxcox: bool = True
    # Where the Box-Cox lambdas are estimated. "all" follows the specified
    # pipeline (skewness correction before the split); "train" is the
    # leakage-free variant reported as a sensitivity check. Both are
    # unsupervised - the target is never involved.
    boxcox_fit: str = "all"

    # --- splits -------------------------------------------------------
    val_size: float = 0.15       # early stopping / generation selection
    test_size: float = 0.15      # touched once, at the very end
    n_bins: int = 10             # quantile strata over the BCC score
    seed: int = 42

    # --- shared network (BPNN and GANN) -------------------------------
    hidden_dims: tuple[int, ...] = (24, 12)
    activation: str = "tanh"

    # --- BPNN ---------------------------------------------------------
    bp_lr: float = 5e-3
    bp_epochs: int = 1000
    bp_batch: int = 32
    bp_patience: int = 100
    bp_l2: float = 0.0           # weights only; early stopping is the regulariser

    # --- GANN ---------------------------------------------------------
    ga_pop: int = 150
    ga_gens: int = 3000
    ga_tournament: int = 3
    ga_cx_rate: float = 0.9
    ga_alpha: float = 0.5        # BLX-alpha
    ga_mut_rate: float = 0.1     # per gene
    # Mutation step anneals geometrically from ga_mut_sigma to ga_mut_sigma_min
    # ACROSS the whole budget, so raising ga_gens lengthens the search instead
    # of silently freezing it - a per-generation decay constant would make the
    # schedule depend on a number that is supposed to be free.
    ga_mut_sigma: float = 0.15
    ga_mut_sigma_min: float = 0.01
    ga_elite: int = 2
    # Early stopping is a safety net, not the schedule: the mutation anneal is
    # what converges the search, so patience is a FRACTION of the budget. A
    # fixed generation count fires mid-exploration on a long run, and a
    # half-annealed population would then be reported as a converged one.
    ga_patience_frac: float = 0.2
    # BLX-alpha widens the parents' interval every generation, so genomes are
    # kept inside a box. tanh saturates well before this, and it stops a
    # runaway individual from turning the fitness curve into noise.
    ga_clip: float = 10.0

    # --- SVM-DEA (fixed, untuned) -------------------------------------
    svm_C: float = 1.0
    svm_epsilon: float = 0.1
    svm_gamma: str = "scale"
    svm_kernel: str = "rbf"

    # --- ISVM-DEA (grid search) ---------------------------------------
    isvm_cv: int = 5
    isvm_jobs: int = -1

    # --- restarts -----------------------------------------------------
    # Both networks are stochastic: backpropagation depends on its
    # initialisation and mini-batch order, the GA on everything. A single run
    # per cell was measurably unstable for the GA - different schedules landed
    # in different basins - so BOTH networks get n_restarts independent seeds
    # and the champion is chosen on the VALIDATION split, never the test
    # split. The spread across restarts is reported alongside the winner, so
    # a lucky seed cannot pass for a better method. SVR and ISVR are
    # deterministic given the data and the seeded CV folds, so restarting
    # them would only repeat the same fit.
    n_restarts: int = 5
    # Restarts are independent, so they run across cores. <=0 means "as many
    # workers as restarts, capped by the cores the container can see".
    restart_jobs: int = -1

    # --- misc ---------------------------------------------------------
    log_every: int = 50
    figures: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden_dims"] = list(self.hidden_dims)
        return d


# The ISVM grid. Epsilon is swept low because the target lives in (0, 1]
# with a standard deviation of ~0.155: the fixed default epsilon=0.1 is an
# insensitive tube two thirds as wide as the signal itself.
ISVM_GRID = {
    "C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
    "gamma": ["scale", 0.001, 0.01, 0.1, 1.0, 10.0],
    "epsilon": [0.001, 0.005, 0.01, 0.05, 0.1],
}


def build_parser(description: str) -> argparse.ArgumentParser:
    d = MLConfig()
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-path", default=d.data_path)
    p.add_argument("--sheet", default=d.sheet)
    p.add_argument("--out-dir", default=d.out_dir)
    p.add_argument("--zero-fix", default=d.zero_fix, choices=["halfmin", "shift"])
    p.add_argument("--no-boxcox", dest="boxcox", action="store_false", default=True)
    p.add_argument("--boxcox-fit", default=d.boxcox_fit, choices=["all", "train"])
    p.add_argument("--methods", default=",".join(METHODS),
                   help="comma-separated subset of " + ",".join(METHODS))
    p.add_argument("--norms", default=",".join(NORMALISERS),
                   help="comma-separated subset of " + ",".join(NORMALISERS))
    p.add_argument("--hidden-dims", default="24,12")
    p.add_argument("--bp-epochs", type=int, default=d.bp_epochs)
    p.add_argument("--bp-lr", type=float, default=d.bp_lr)
    p.add_argument("--bp-batch", type=int, default=d.bp_batch)
    p.add_argument("--ga-pop", type=int, default=d.ga_pop)
    p.add_argument("--ga-gens", type=int, default=d.ga_gens)
    p.add_argument("--svm-C", dest="svm_C", type=float, default=d.svm_C)
    p.add_argument("--svm-epsilon", type=float, default=d.svm_epsilon)
    p.add_argument("--isvm-cv", type=int, default=d.isvm_cv)
    p.add_argument("--n-restarts", type=int, default=d.n_restarts)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--log-every", type=int, default=d.log_every)
    p.add_argument("--no-figures", dest="figures", action="store_false", default=True)
    return p


def config_from_args(args: argparse.Namespace) -> MLConfig:
    kw = {k: v for k, v in vars(args).items() if k in MLConfig.__dataclass_fields__}
    if isinstance(getattr(args, "hidden_dims", None), str):
        kw["hidden_dims"] = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())
    return MLConfig(**kw)


def selected(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """Parse --methods / --norms into validated ordered lists."""
    def pick(raw, allowed, what):
        want = [s.strip() for s in raw.split(",") if s.strip()]
        bad = [w for w in want if w not in allowed]
        if bad:
            raise SystemExit(f"unknown {what}: {bad}. choose from {list(allowed)}")
        return [a for a in allowed if a in want]      # keep canonical order
    return (pick(getattr(args, "methods", ",".join(METHODS)), METHODS, "method"),
            pick(getattr(args, "norms", ",".join(NORMALISERS)), NORMALISERS, "normaliser"))
