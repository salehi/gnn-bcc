"""The GraphSAGE regressor and its edges-removed MLP twin.

Architecture (fixed by the user): two hidden blocks, 24 units then 12.

    x (4) -> SAGEConv(4,24) -> LayerNorm -> ReLU
          -> SAGEConv(24,12) -> LayerNorm -> ReLU
          -> Linear(12,1)

Design notes
------------
* LayerNorm rather than BatchNorm. Under the strict inductive protocol the
  training graph and the inference graph hold different node sets, so batch
  statistics differ between them. LayerNorm normalises per node and is
  independent of batch composition, which removes that inconsistency. It is
  also better behaved than BatchNorm on 12-wide layers.
* No dropout by default. On a 12-unit layer, p=0.2 zeroes ~2 units at a time -
  coarse, high-variance noise rather than smooth regularisation - and with
  ~900 parameters against ~650 training DMUs the model is not in an
  overfitting regime. Weight decay and early stopping carry the load. The
  flag remains for sweeps.
* Linear output, not sigmoid. A sigmoid saturates on the 29 DMUs sitting at
  exactly 1.0 and starves their gradients; predictions are instead clamped to
  the DEA-valid range at evaluation time.
* Two blocks give a 2-hop receptive field: a DMU is compared against its peers
  and its peers' peers, mirroring how a DEA reference set spans the frontier.
* MLPRegressor is the identical trunk with SAGEConv swapped for Linear, so the
  ablation isolates the graph's contribution and nothing else.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GraphSAGERegressor(nn.Module):
    uses_edges = True

    def __init__(self, in_dim: int = 4, hidden_dims=(24, 12), dropout: float = 0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        d = in_dim
        for h in hidden_dims:
            self.convs.append(SAGEConv(d, h, aggr="mean"))
            self.norms.append(nn.LayerNorm(h))
            d = h
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(d, 1)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = self.drop(self.act(norm(conv(x, edge_index))))
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x, edge_index)).squeeze(-1)


class MLPRegressor(nn.Module):
    """Same shape, same normalisation, no message passing. `edge_index` is
    accepted and ignored so the two models are drop-in interchangeable."""
    uses_edges = False

    def __init__(self, in_dim: int = 4, hidden_dims=(24, 12), dropout: float = 0.0):
        super().__init__()
        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()
        d = in_dim
        for h in hidden_dims:
            self.lins.append(nn.Linear(d, h))
            self.norms.append(nn.LayerNorm(h))
            d = h
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(d, 1)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        for lin, norm in zip(self.lins, self.norms):
            x = self.drop(self.act(norm(lin(x))))
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.embed(x)).squeeze(-1)


def build_model(cfg) -> nn.Module:
    cls = GraphSAGERegressor if cfg.model == "sage" else MLPRegressor
    return cls(in_dim=4, hidden_dims=tuple(cfg.hidden_dims), dropout=cfg.dropout)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def layer_summary(model: nn.Module) -> list[tuple[str, str, int]]:
    """(name, shape string, parameter count) per layer, for the diagram."""
    rows = []
    for name, mod in model.named_children():
        if isinstance(mod, nn.ModuleList):
            for i, sub in enumerate(mod):
                n = sum(p.numel() for p in sub.parameters())
                rows.append((f"{name}[{i}] {sub.__class__.__name__}", _shape(sub), n))
        else:
            n = sum(p.numel() for p in mod.parameters())
            if n or isinstance(mod, (nn.ReLU, nn.Dropout)):
                rows.append((f"{name} {mod.__class__.__name__}", _shape(mod), n))
    return rows


def _shape(mod: nn.Module) -> str:
    if isinstance(mod, nn.Linear):
        return f"{mod.in_features} -> {mod.out_features}"
    if isinstance(mod, SAGEConv):
        return f"{mod.in_channels} -> {mod.out_channels}"
    if isinstance(mod, nn.LayerNorm):
        return f"{mod.normalized_shape[0]}"
    return ""
