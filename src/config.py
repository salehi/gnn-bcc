"""Configuration for the BCC-score GNN.

Everything that defines a run lives here so that a single object can be
serialised into metrics.json and best_model.pt alongside the results.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field

# --- Dataset schema -------------------------------------------------------
# Column order as it appears in the "Data" sheet.
FEATURES = [
    "owners equity",
    "total operational expences",
    "net profit",
    "operational income",
]
TARGET = "BCC score"

# DEA semantics. These do NOT change what the network consumes (all four
# columns are features); they drive the dominance-edge variant and the way
# results are described in the report.
#
# ASSUMPTION - confirm against the thesis. Standard profitability approach for
# bank DMUs: capital and cost are consumed, profit and income are produced.
DEA_INPUTS = ["owners equity", "total operational expences"]
DEA_OUTPUTS = ["net profit", "operational income"]

# A BCC score of 1.0 means the DMU sits on the efficient frontier. Float
# comparison needs a tolerance; the raw file stores e.g. 0.99999999999.
FRONTIER_EPS = 1e-6

# The constant added to `net profit` during preprocessing (visible in the
# "pre data" sheet). Recorded for provenance only - the Data sheet already
# holds the shifted values and they are used as-is. Legitimate because the
# BCC/VRS model is translation invariant in the outputs (Ali & Seiford 1990).
NET_PROFIT_SHIFT = 9_101_232_179.17


@dataclass
class Config:
    # data
    data_path: str = "/app/data/data_with_bcc_score.xlsx"
    sheet: str = "Data"
    out_dir: str = "/app/outputs"
    tag: str = ""

    # architecture (fixed by the user: two hidden blocks, 24 then 12)
    hidden_dims: tuple[int, ...] = (24, 12)
    model: str = "sage"          # sage | mlp
    dropout: float = 0.0         # see README: off by default at this width

    # graph
    k: int = 8
    edges: str = "knn"           # knn | dominance

    # optimisation
    loss: str = "mse"            # mse | huber
    huber_beta: float = 0.05
    lr: float = 5e-3
    weight_decay: float = 5e-4
    max_epochs: int = 1000
    patience: int = 100          # early stopping, on val RMSE
    sched_patience: int = 25
    sched_factor: float = 0.5
    min_lr: float = 1e-5
    grad_clip: float = 1.0

    # splits
    val_size: float = 0.15
    test_size: float = 0.15
    n_bins: int = 10             # stratification bins over the target
    n_folds: int = 5

    # misc
    seed: int = 42
    live_every: int = 5          # rewrite live_training.png every N epochs
    log_every: int = 10
    shuffle_labels: bool = False  # leakage check: must yield R2 ~ 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden_dims"] = list(self.hidden_dims)
        return d


def build_parser(description: str) -> argparse.ArgumentParser:
    d = Config()
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-path", default=d.data_path)
    p.add_argument("--sheet", default=d.sheet)
    p.add_argument("--out-dir", default=d.out_dir)
    p.add_argument("--tag", default=d.tag,
                   help="suffix for output filenames, to keep runs apart")
    p.add_argument("--hidden-dims", default="24,12",
                   help="comma-separated hidden layer widths (default 24,12)")
    p.add_argument("--model", default=d.model, choices=["sage", "mlp"])
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument("-k", "--k", type=int, default=d.k)
    p.add_argument("--edges", default=d.edges, choices=["knn", "dominance"])
    p.add_argument("--loss", default=d.loss, choices=["mse", "huber"])
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--max-epochs", type=int, default=d.max_epochs)
    p.add_argument("--patience", type=int, default=d.patience)
    p.add_argument("--n-folds", type=int, default=d.n_folds)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--live-every", type=int, default=d.live_every)
    p.add_argument("--log-every", type=int, default=d.log_every)
    p.add_argument("--shuffle-labels", action="store_true", default=d.shuffle_labels)
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    kw = {k: v for k, v in vars(args).items() if k in Config.__dataclass_fields__}
    if isinstance(getattr(args, "hidden_dims", None), str):
        kw["hidden_dims"] = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())
    return Config(**kw)
