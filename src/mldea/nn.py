"""One MLP, two optimisers.

BPNN-DEA and GANN-DEA differ in exactly one respect: how the weights are
found. They share this module's architecture, initialisation, activation and
loss, so any gap between them is attributable to the optimiser and not to the
model. The network is written out by hand in NumPy for that reason - it lets
the genetic algorithm treat the whole parameter set as a single flat genome
while backpropagation walks the same graph analytically.

Architecture: 4 -> 24 -> 12 -> 1, tanh hidden units, linear output.

The output is linear rather than a sigmoid squashed into (0, 1]. A BCC score
is bounded, but so is an SVR's target, and forcing a bound into one family of
models and not the other would confound the comparison. Predictions are
clipped into the DEA-valid range at scoring time instead, by the same
`clamp_predictions` the rest of this repository uses.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# Parameterisation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MLPSpec:
    dims: tuple[int, ...]              # (n_in, h1, h2, ..., 1)
    activation: str = "tanh"

    @property
    def shapes(self) -> list[tuple[tuple[int, int], tuple[int]]]:
        return [((self.dims[i], self.dims[i + 1]), (self.dims[i + 1],))
                for i in range(len(self.dims) - 1)]

    @property
    def n_params(self) -> int:
        return sum(a * b + b for (a, b), _ in self.shapes)

    def layer_rows(self) -> list[tuple[str, str, int]]:
        rows = []
        for i, ((a, b), _) in enumerate(self.shapes):
            act = "linear" if i == len(self.shapes) - 1 else self.activation
            rows.append((f"Dense {a}->{b}", act, a * b + b))
        return rows


def _act(z, kind):
    if kind == "tanh":
        return np.tanh(z)
    if kind == "relu":
        return np.maximum(z, 0.0)
    raise ValueError(f"unknown activation {kind!r}")


def _act_grad_from_output(a, kind):
    """d act / d z expressed through the activation's own output."""
    if kind == "tanh":
        return 1.0 - a * a
    if kind == "relu":
        return (a > 0.0).astype(a.dtype)
    raise ValueError(f"unknown activation {kind!r}")


def init_theta(spec: MLPSpec, rng: np.random.Generator, size: int | None = None):
    """Glorot-uniform initialisation, flattened.

    `size` draws a whole population at once (shape (size, n_params)), which is
    how the genetic algorithm seeds generation zero. Every individual is a
    legitimate network under the same scheme backpropagation starts from, so
    neither optimiser gets a better starting point than the other.
    """
    parts = []
    for (fan_in, fan_out), _ in spec.shapes:
        lim = np.sqrt(6.0 / (fan_in + fan_out))
        shp = (fan_in * fan_out,) if size is None else (size, fan_in * fan_out)
        parts.append(rng.uniform(-lim, lim, size=shp))
        parts.append(np.zeros((fan_out,) if size is None else (size, fan_out)))
    return np.concatenate(parts, axis=-1)


def weight_mask(spec: MLPSpec) -> np.ndarray:
    """1 on weight entries, 0 on biases - so L2 never penalises a bias."""
    parts = []
    for (a, b), _ in spec.shapes:
        parts.append(np.ones(a * b))
        parts.append(np.zeros(b))
    return np.concatenate(parts)


def unpack(theta: np.ndarray, spec: MLPSpec):
    """Flat genome -> [(W, b), ...]. A leading population axis is preserved,
    so the same call serves one network or a whole generation."""
    out, off = [], 0
    lead = theta.shape[:-1]
    for (a, b), _ in spec.shapes:
        W = theta[..., off:off + a * b].reshape(*lead, a, b); off += a * b
        bb = theta[..., off:off + b]; off += b
        out.append((W, bb))
    if off != theta.shape[-1]:
        raise ValueError(f"genome length {theta.shape[-1]} != spec {off}")
    return out


# --------------------------------------------------------------------------
# Forward / backward
# --------------------------------------------------------------------------
def forward(theta: np.ndarray, X: np.ndarray, spec: MLPSpec, cache: bool = False):
    """Predictions for `theta` of shape (n_params,) -> (n,), or a population
    (P, n_params) -> (P, n). NumPy's matmul broadcasting covers both."""
    layers = unpack(theta, spec)
    a = X
    acts = [a]
    for i, (W, b) in enumerate(layers):
        z = np.matmul(a, W) + b[..., None, :]
        a = z if i == len(layers) - 1 else _act(z, spec.activation)
        acts.append(a)
    y = a[..., 0]
    return (y, acts) if cache else y


def mse(theta, X, y, spec) -> np.ndarray:
    """Mean squared error; vectorised over a population if theta is 2-D."""
    pred = forward(theta, X, spec)
    return np.mean((pred - y) ** 2, axis=-1)


def loss_grad(theta: np.ndarray, X: np.ndarray, y: np.ndarray, spec: MLPSpec,
              l2: float = 0.0, wmask: np.ndarray | None = None):
    """MSE loss and its exact gradient for a single network.

    Verified against central finite differences in src/mldea/selftest.py - if
    this is wrong, BPNN quietly trains on noise while GANN does not, and the
    comparison between them becomes meaningless.
    """
    pred, acts = forward(theta, X, spec, cache=True)
    n = X.shape[0]
    err = pred - y
    loss = float(np.mean(err ** 2))

    layers = unpack(theta, spec)
    delta = (2.0 / n) * err[:, None]                   # dL/dz_out, (n, 1)
    grads = [None] * len(layers)
    for i in range(len(layers) - 1, -1, -1):
        W, _ = layers[i]
        a_in = acts[i]
        gW = a_in.T @ delta
        gb = delta.sum(axis=0)
        grads[i] = (gW, gb)
        if i > 0:
            a_prev = acts[i]                            # output of layer i-1
            delta = (delta @ W.T) * _act_grad_from_output(a_prev, spec.activation)

    flat = np.concatenate([np.concatenate([gW.ravel(), gb.ravel()])
                           for gW, gb in grads])
    if l2 > 0.0:
        if wmask is None:
            wmask = weight_mask(spec)
        flat = flat + l2 * theta * wmask
        loss += 0.5 * l2 * float(np.sum((theta * wmask) ** 2))
    return loss, flat


# --------------------------------------------------------------------------
# BPNN-DEA: backpropagation with Adam
# --------------------------------------------------------------------------
def train_bpnn(cfg, Ztr, ytr, Zva, yva, seed: int) -> dict:
    spec = MLPSpec(tuple([Ztr.shape[1], *cfg.hidden_dims, 1]), cfg.activation)
    rng = np.random.default_rng(seed)
    theta = init_theta(spec, rng)
    wmask = weight_mask(spec)

    m = np.zeros_like(theta); v = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    n = len(ytr)
    batch = min(cfg.bp_batch, n)

    hist = {"epoch": [], "train_mse": [], "val_mse": [], "grad_norm": []}
    best = {"val": np.inf, "theta": theta.copy(), "epoch": 0}
    bad = 0
    t0 = time.time()

    for epoch in range(1, cfg.bp_epochs + 1):
        perm = rng.permutation(n)
        gnorm = 0.0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            _, g = loss_grad(theta, Ztr[idx], ytr[idx], spec, cfg.bp_l2, wmask)
            t += 1
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * (g * g)
            mh = m / (1 - b1 ** t)
            vh = v / (1 - b2 ** t)
            theta = theta - cfg.bp_lr * mh / (np.sqrt(vh) + eps)
            gnorm = max(gnorm, float(np.linalg.norm(g)))

        tr_mse = float(mse(theta, Ztr, ytr, spec))
        va_mse = float(mse(theta, Zva, yva, spec))
        hist["epoch"].append(epoch)
        hist["train_mse"].append(tr_mse)
        hist["val_mse"].append(va_mse)
        hist["grad_norm"].append(gnorm)

        if va_mse < best["val"] - 1e-12:
            best = {"val": va_mse, "theta": theta.copy(), "epoch": epoch}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.bp_patience:
                break

    return {
        "spec": spec, "theta": best["theta"], "history": hist,
        "best_iter": best["epoch"], "best_val_mse": best["val"],
        "n_iter": len(hist["epoch"]), "seconds": time.time() - t0,
        "n_params": spec.n_params,
        "evaluations": t,                       # gradient steps taken
        "curve_kind": "loss",
    }


# --------------------------------------------------------------------------
# GANN-DEA: the same network, weights evolved
# --------------------------------------------------------------------------
def _tournament(fitness: np.ndarray, k: int, n_out: int, rng) -> np.ndarray:
    """Indices of `n_out` winners; lower fitness (MSE) wins."""
    cand = rng.integers(0, len(fitness), size=(n_out, k))
    return cand[np.arange(n_out), np.argmin(fitness[cand], axis=1)]


def train_gann(cfg, Ztr, ytr, Zva, yva, seed: int) -> dict:
    """Real-coded GA: tournament selection, BLX-alpha crossover, Gaussian
    mutation with an annealed step, elitism.

    The genome is the full flat weight vector, so the search space has one
    dimension per network parameter. Fitness is training MSE. The returned
    network is the generation champion with the best VALIDATION error, which
    is the same checkpointing rule BPNN's early stopping applies - neither
    method is allowed to select on the test set.
    """
    spec = MLPSpec(tuple([Ztr.shape[1], *cfg.hidden_dims, 1]), cfg.activation)
    rng = np.random.default_rng(seed)
    P, D = cfg.ga_pop, spec.n_params
    E = max(0, min(cfg.ga_elite, P - 2))

    pop = init_theta(spec, rng, size=P)
    fit = mse(pop, Ztr, ytr, spec)                    # (P,) vectorised
    evals = P
    sigma = cfg.ga_mut_sigma
    anneal = (cfg.ga_mut_sigma_min / sigma) ** (1.0 / max(1, cfg.ga_gens - 1))

    patience = max(50, int(cfg.ga_patience_frac * cfg.ga_gens))

    hist = {"gen": [], "best_train_mse": [], "mean_train_mse": [],
            "val_mse": [], "sigma": [], "diversity": []}
    best = {"val": np.inf, "theta": None, "gen": 0}
    bad = 0
    t0 = time.time()

    for gen in range(1, cfg.ga_gens + 1):
        order = np.argsort(fit)
        elite = pop[order[:E]] if E else np.empty((0, D))

        n_child = P - E
        pa = pop[_tournament(fit, cfg.ga_tournament, n_child, rng)]
        pb = pop[_tournament(fit, cfg.ga_tournament, n_child, rng)]

        # BLX-alpha: sample each gene from the interval the parents span,
        # widened by alpha on both sides so the search can leave the box.
        lo = np.minimum(pa, pb); hi = np.maximum(pa, pb)
        d = (hi - lo) * cfg.ga_alpha
        child = rng.uniform(lo - d, hi + d)
        keep = rng.random(n_child) >= cfg.ga_cx_rate      # no crossover -> clone
        child[keep] = pa[keep]

        mut = rng.random((n_child, D)) < cfg.ga_mut_rate
        child = child + mut * rng.normal(0.0, sigma, size=(n_child, D))
        if cfg.ga_clip:
            np.clip(child, -cfg.ga_clip, cfg.ga_clip, out=child)

        pop = np.concatenate([elite, child], axis=0)
        fit = mse(pop, Ztr, ytr, spec)
        evals += n_child
        sigma = max(sigma * anneal, cfg.ga_mut_sigma_min)

        champ = int(np.argmin(fit))
        va = float(mse(pop[champ], Zva, yva, spec))
        hist["gen"].append(gen)
        hist["best_train_mse"].append(float(fit[champ]))
        hist["mean_train_mse"].append(float(np.mean(fit)))
        hist["val_mse"].append(va)
        hist["sigma"].append(float(sigma))
        hist["diversity"].append(float(np.mean(np.std(pop, axis=0))))

        if va < best["val"] - 1e-12:
            best = {"val": va, "theta": pop[champ].copy(), "gen": gen}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best["theta"] is None:                            # pathological guard
        best["theta"] = pop[int(np.argmin(fit))].copy()

    return {
        "spec": spec, "theta": best["theta"], "history": hist,
        "best_iter": best["gen"], "best_val_mse": best["val"],
        "n_iter": len(hist["gen"]), "seconds": time.time() - t0,
        "n_params": spec.n_params,
        "evaluations": int(evals),                       # fitness evaluations
        "curve_kind": "loss",
    }


def predict(fitted: dict, X: np.ndarray) -> np.ndarray:
    return forward(fitted["theta"], X, fitted["spec"])
