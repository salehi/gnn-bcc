"""Self-tests for the things that are easy to get silently wrong.

Run with:  ./run.sh selftest

These are not unit tests for their own sake. Each one checks a property that,
if violated, would quietly inflate the reported scores.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from .config import DEA_INPUTS, DEA_OUTPUTS, FEATURES, Config, config_from_args, build_parser
from .data import (LogStandardScaler, build_eval_graph, build_train_edges,
                   edges_to_tensor, load_dataset, make_folds, make_splits,
                   to_tensor)
from .models import build_model, count_params
from .train import IN_COLS, OUT_COLS, prepare_graphs, set_seed

PASS, FAIL = "  PASS  ", "  FAIL  "
_results = []


def check(name, cond, detail=""):
    _results.append(bool(cond))
    print(f"{PASS if cond else FAIL} {name}" + (f"   {detail}" if detail else ""))


def main():
    cfg = config_from_args(build_parser("Self-tests.").parse_args())
    set_seed(cfg.seed)
    X, y, names = load_dataset(cfg)

    print("=" * 78)
    print("  Self-tests")
    print("=" * 78)

    # --- data ----------------------------------------------------------
    check("dataset shape is (927, 4)", X.shape == (927, 4), f"got {X.shape}")
    check("target within DEA range (0, 1]", y.min() > 0 and y.max() <= 1 + 1e-9,
          f"[{y.min():.4f}, {y.max():.4f}]")
    check("29 DMUs on the frontier", int((y >= 1 - 1e-6).sum()) == 29,
          f"got {int((y >= 1 - 1e-6).sum())}")
    check("feature/DEA role mapping is complete",
          sorted(DEA_INPUTS + DEA_OUTPUTS) == sorted(FEATURES))

    # --- splits --------------------------------------------------------
    tr, va, te = make_splits(y, cfg)
    check("splits are disjoint",
          not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te)))
    check("splits cover the dataset", len(tr) + len(va) + len(te) == len(y),
          f"{len(tr)}+{len(va)}+{len(te)}={len(tr)+len(va)+len(te)}")
    check("stratification keeps frontier DMUs in every split",
          all(int((y[s] >= 1 - 1e-6).sum()) > 0 for s in (tr, va, te)),
          f"frontier per split: {[int((y[s] >= 1-1e-6).sum()) for s in (tr, va, te)]}")
    for fold, a, b_, c in make_folds(y, cfg):
        if not (not (set(a) & set(b_)) and not (set(a) & set(c)) and not (set(b_) & set(c))):
            check(f"CV fold {fold} disjoint", False)
            break
    else:
        check(f"all {cfg.n_folds} CV folds are internally disjoint", True)

    # --- scaler: fitted on train only -----------------------------------
    s_train = LogStandardScaler().fit(X[tr])
    s_all = LogStandardScaler().fit(X)
    check("scaler statistics differ from all-data statistics (no leakage)",
          not np.allclose(s_train.mean_, s_all.mean_),
          f"max |dmean| = {np.abs(s_train.mean_ - s_all.mean_).max():.4f}")

    # --- graph: the inductive contract ----------------------------------
    b = prepare_graphs(cfg, X, y, tr, va, te)
    n_tr = len(tr)
    for split in ("val", "test"):
        ei = b[split]["ei"].numpy()
        check(f"[{split}] no held-out node is ever a message source",
              bool((ei[0] < n_tr).all()))
        held = ei[:, ei[1] >= n_tr]
        check(f"[{split}] held-out nodes receive only from training nodes",
              bool((held[0] < n_tr).all()))
        check(f"[{split}] held-out nodes are not connected to each other",
              not bool(((ei[0] >= n_tr) & (ei[1] >= n_tr)).any()))

    # --- the decisive test ----------------------------------------------
    # A training node's output must be bit-identical whether or not held-out
    # nodes are present in the graph. If it is not, the held-out data is
    # influencing training and every reported number is inflated.
    model = build_model(cfg).eval()
    with torch.no_grad():
        base = model(b["x_tr"], b["ei_tr"]).numpy()
        g = b["test"]
        withheld = model(g["x"], g["ei"]).numpy()[:n_tr]
    check("training-node outputs are unchanged by the presence of held-out nodes",
          np.allclose(base, withheld, atol=1e-6),
          f"max |diff| = {np.abs(base - withheld).max():.2e}")

    # A held-out node must be scored the same alone as in a batch of held-out
    # nodes - i.e. each one really is evaluated as a single unseen DMU.
    Z = b["Z"]
    one_idx = te[:1]
    Xc1, ei1, pos1 = build_eval_graph(b["Z_tr"], Z[one_idx], cfg, IN_COLS, OUT_COLS,
                                      b["train_edges"])
    with torch.no_grad():
        alone = model(to_tensor(Xc1), edges_to_tensor(ei1))[pos1[0]].item()
        batch = model(g["x"], g["ei"])[g["pos"][0]].item()
    check("a held-out DMU scores identically alone and in a batch",
          abs(alone - batch) < 1e-5, f"|diff| = {abs(alone - batch):.2e}")

    # --- model ----------------------------------------------------------
    check("hidden widths are 24 then 12", tuple(cfg.hidden_dims) == (24, 12),
          str(tuple(cfg.hidden_dims)))
    np_ = count_params(model)
    # Reported, not asserted against an arbitrary ratio: the 24/12 trunk comes
    # out at ~1.4 parameters per training DMU. That is a tight budget by deep
    # learning standards but NOT an automatic guarantee against overfitting,
    # which is why weight decay and early stopping are both on and the
    # train/test gap is charted.
    check("architecture is the specified small trunk (<2k params)", np_ < 2000,
          f"{np_} params, {np_/len(tr):.2f} per training DMU ({len(tr)} DMUs)")
    out = model(b["x_tr"], b["ei_tr"])
    check("forward pass returns one scalar per node", out.shape == (n_tr,),
          str(tuple(out.shape)))

    # --- dominance edges -------------------------------------------------
    dcfg = Config(**{**cfg.to_dict(), "hidden_dims": tuple(cfg.hidden_dims),
                     "edges": "dominance"})
    de = build_train_edges(b["Z_tr"], dcfg, IN_COLS, OUT_COLS)
    check("dominance mode adds edges on top of k-NN",
          de.shape[1] >= b["train_edges"].shape[1],
          f"{b['train_edges'].shape[1]} -> {de.shape[1]} edges")

    print("=" * 78)
    n_ok = sum(_results)
    print(f"  {n_ok}/{len(_results)} checks passed")
    print("=" * 78)
    sys.exit(0 if n_ok == len(_results) else 1)


if __name__ == "__main__":
    main()
