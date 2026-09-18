"""Assemble outputs/report.md from whatever result files exist."""
from __future__ import annotations

import json
import os

from .config import DEA_INPUTS, DEA_OUTPUTS, build_parser, config_from_args


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _fig(out_dir, name, caption):
    if not os.path.exists(os.path.join(out_dir, "figures", name)):
        return ""
    return f"![{caption}](figures/{name})\n\n*{caption}*\n\n"


def main():
    p = build_parser("Assemble report.md from the result JSON files.")
    cfg = config_from_args(p.parse_args())
    o = cfg.out_dir

    show = _load(f"{o}/metrics.json")
    cv = _load(f"{o}/cv_results.json")
    abl = _load(f"{o}/ablation.json")

    L = []
    A = L.append
    A("# Predicting DEA BCC efficiency scores with a Graph Neural Network\n")
    A("Node regression over a k-nearest-neighbour peer graph of DMUs, "
      "evaluated strictly inductively.\n")

    A("## Why a graph\n")
    A("A BCC score is not a property of a DMU in isolation: it is that DMU's "
      "distance to an efficient frontier spanned by its peer DMUs. Message "
      "passing over a k-NN graph in input/output space lets each prediction "
      "depend on the DMU's peers, which mirrors how DEA computes the score in "
      "the first place. Whether this actually beats a plain MLP on this dataset "
      "is measured below, not assumed.\n")

    A("## DEA setup\n")
    A(f"- **Inputs** (minimised): {', '.join(DEA_INPUTS)}")
    A(f"- **Outputs** (maximised): {', '.join(DEA_OUTPUTS)}")
    A("- `net profit` is the pre-shifted column from the source file "
      "(+9,101,232,179.17). That shift is legitimate precisely because the "
      "BCC/VRS model is translation invariant in the outputs (Ali & Seiford, "
      "1990) — CCR/CRS would not be — so the scores are unaffected.")
    A("- Efficient = score ≥ 1 − 1e-6.\n")
    A("> The input/output assignment above is an **assumption**. It changes the "
      "dominance-edge variant and the wording of this report, not the four "
      "features the network consumes.\n")

    if show:
        c, tr, m = show["config"], show["training"], show["metrics"]
        A("## Model\n")
        A(f"```\n4 features → SAGEConv(4,{c['hidden_dims'][0]}) → LayerNorm → ReLU\n"
          f"           → SAGEConv({c['hidden_dims'][0]},{c['hidden_dims'][1]}) → LayerNorm → ReLU\n"
          f"           → Linear({c['hidden_dims'][1]},1)\n```\n")
        A(f"{tr['n_params']:,} trainable parameters, {show['splits']['train']} "
          f"training DMUs. Loss {c['loss']}, AdamW lr {c['lr']}, weight decay "
          f"{c['weight_decay']}, dropout {c['dropout']}, k={c['k']}, "
          f"edges={c['edges']}. Best epoch {tr['best_epoch']} of "
          f"{tr['epochs_run']} ({tr['seconds']}s).\n")

        A("## Showcase split results\n")
        A("| split | n | RMSE | MAE | MAPE | R² | Spearman ρ | Kendall τ |")
        A("|---|---|---|---|---|---|---|---|")
        for k in ("train", "val", "test"):
            s = m[k]
            A(f"| {k} | {s['n']} | {s['rmse']:.4f} | {s['mae']:.4f} | "
              f"{s['mape']:.2f}% | {s['r2']:+.4f} | {s['spearman']:+.4f} | "
              f"{s['kendall']:+.4f} |")
        A("")
        t = m["test"]
        A(f"Frontier calibration on the test set: {t['frontier_n']} DMUs truly "
          f"sit at 1.0 and the model places them at {t['frontier_pred_mean']:.4f} "
          f"on average — an under-prediction gap of {t['frontier_gap']:+.4f}. "
          f"Near-frontier detection at ≥0.95: precision {t['p95_precision']:.3f}, "
          f"recall {t['p95_recall']:.3f}, F1 {t['p95_f1']:.3f}.\n")

    if cv:
        m, s = cv["cv"]["mean"], cv["cv"]["std"]
        A(f"## {len(cv['cv']['folds'])}-fold strict-inductive cross-validation\n")
        A("These are the numbers to quote.\n")
        A("| metric | mean ± sd |")
        A("|---|---|")
        for k, lbl in (("rmse", "RMSE"), ("mae", "MAE"), ("mape", "MAPE (%)"),
                       ("r2", "R²"), ("spearman", "Spearman ρ"),
                       ("kendall", "Kendall τ"),
                       ("frontier_gap", "frontier under-prediction gap")):
            A(f"| {lbl} | {m[k]:.4f} ± {s[k]:.4f} |")
        A("")

    if abl:
        A("## Does the graph earn its place?\n")
        A("| model | R² | RMSE | Spearman ρ |")
        A("|---|---|---|---|")
        for name, r in abl["graph_vs_no_graph"].items():
            A(f"| {name} | {r['mean']['r2']:+.4f} ± {r['std']['r2']:.4f} | "
              f"{r['mean']['rmse']:.4f} ± {r['std']['rmse']:.4f} | "
              f"{r['mean']['spearman']:+.4f} ± {r['std']['spearman']:.4f} |")
        A("")
        pt = abl.get("paired_test")
        if pt:
            A("Both models were scored on the **same folds**, so the comparison "
              "is paired and is tested as such rather than by eyeballing "
              "overlapping error bars.\n")
            A(f"- Per-fold ΔR² (SAGE − MLP): "
              f"{', '.join(f'{d:+.4f}' for d in pt['per_fold_delta_r2'])}")
            A(f"- Mean ΔR² = **{pt['mean_delta_r2']:+.4f}**, "
              f"paired t = {pt['paired_t']:+.3f}, **p = {pt['p_value']:.3f}** "
              f"(Wilcoxon p = {pt['wilcoxon_p']:.3f})")
            A(f"- Fold-to-fold sd of R²: GraphSAGE {pt['sd_r2_sage']:.4f} vs "
              f"MLP {pt['sd_r2_mlp']:.4f}")
            A("")
        A(f"**Verdict: {abl['verdict']}.**\n")
        if "best_k" in abl:
            A(f"Best neighbourhood size by mean R²: **k = {abl['best_k']}**.\n")

    A("## Charts\n")
    for name, cap in [
        ("01_data_overview.png", "Dataset overview: raw vs log1p feature distributions and the target"),
        ("02_correlations.png", "Correlation structure — weak linear signal, hence a nonlinear model"),
        ("03_graph_structure.png", "The k-NN peer graph, its degree distribution, and neighbourhood homophily"),
        ("04_training_curves.png", "Training history with the early-stopping epoch marked"),
        ("05_pred_vs_actual.png", "Predicted vs actual BCC score per split"),
        ("06_residuals.png", "Residual diagnostics on the held-out test set"),
        ("07_error_by_decile.png", "Where the error lives across the efficiency range"),
        ("08_cv_results.png", "Per-fold cross-validation results"),
        ("09_gnn_vs_mlp.png", "GraphSAGE vs the identical trunk with edges removed"),
        ("10_k_sensitivity.png", "Sensitivity to the number of DEA peers per DMU"),
        ("11_embeddings.png", "Learned 12-d embedding of held-out DMUs"),
        ("12_architecture.png", "Network architecture and parameter budget"),
    ]:
        L.append(_fig(o, name, cap))

    A("## Limitations\n")
    A("- **The labels are themselves frontier-relative.** Every BCC score in "
      "the source file was produced by solving the DEA program over all 927 "
      "DMUs, including the ones held out here. So a training DMU's label "
      "already encodes information about the test DMUs. The strict inductive "
      "protocol guarantees the *model* never sees held-out features or labels, "
      "but it cannot undo how the labels were generated. This is inherent to "
      "any pre-computed DEA target and applies equally to the published "
      "DEA-ML literature; it is stated here rather than glossed over.")
    A("- Only 29 of 927 DMUs sit on the frontier, so the model has little "
      "signal about the frontier itself and under-predicts it. See the decile "
      "chart.")
    A("- The input/output orientation is assumed, not read from the file.")
    A("- Standard GraphSAGE mean aggregation is unweighted, so geometric "
      "distance to a peer sets *whether* an edge exists but not how much that "
      "peer counts.\n")

    path = os.path.join(o, "report.md")
    body = "\n".join(L)
    with open(path, "w") as f:
        f.write(body)
    print(f"wrote {path} ({len(body)} bytes)")


if __name__ == "__main__":
    main()
