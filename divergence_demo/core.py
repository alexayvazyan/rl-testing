"""
DQN divergence demo — core simulator.

Setup:
    S1 --(r=0)--> S2 --(r=0)--> terminal
    Network: one-hot state (2) -> hidden (n) -> scalar V(s). No biases.
    3n parameters total: W in R^(n x 2) has 2n, v in R^n has n.

We only ever sample the (S1, r=0, S2) transition. The true value of both
states is 0, so the "correct" fixed point of the Bellman operator is V(S1) =
V(S2) = 0. Shared weights (v) couple V(S1) and V(S2): updating V(S1) toward
gamma * V(S2) also moves V(S2), which can create a positive feedback loop
and cause the semi-gradient TD update to diverge.

A frozen target network breaks that loop by holding V_theta-(S2) fixed
while we fit V_theta(S1).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


# ── Activations ──────────────────────────────────────────────────────────────

def _act(name):
    if name == "linear":
        return (lambda x: x), (lambda x: np.ones_like(x))
    if name == "relu":
        return (lambda x: np.maximum(0.0, x)), (lambda x: (x > 0).astype(x.dtype))
    if name == "tanh":
        return (lambda x: np.tanh(x)), (lambda x: 1.0 - np.tanh(x) ** 2)
    raise ValueError(f"unknown activation: {name}")


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class Config:
    n_hidden: int = 4
    gamma: float = 0.99
    lr: float = 0.1
    reward: float = 0.0
    activation: str = "linear"      # "linear" | "relu" | "tanh"
    init_scale: float = 1.0         # std of initial weights
    init_correlation: float = 0.0   # in [-1, 1]. Sets corr(W[:,0], W[:,1]).
                                    # -1 → w2 = -w1 (anti-aligned, worst for bootstrap).
                                    # +1 → w2 = w1 (both columns identical).
                                    #  0 → independent Gaussian (default).
    n_steps: int = 2000
    use_target_net: bool = False
    target_update_interval: int = 50
    divergence_threshold: float = 1e6   # |V| above this counts as diverged
    seed: int = 0


# ── Simulator ────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    cfg: Config
    v_s1: np.ndarray            # (n_steps+1,) value of S1 before each step, plus final
    v_s2: np.ndarray            # (n_steps+1,) value of S2 before each step, plus final
    td_error: np.ndarray        # (n_steps,)
    weight_norm: np.ndarray     # (n_steps+1,) L2 norm of all params
    diverged: bool
    diverge_step: int           # step at which |V| first exceeded threshold, else -1
    W_final: np.ndarray
    v_final: np.ndarray
    W_init: np.ndarray = field(default=None)
    v_init: np.ndarray = field(default=None)


def run(cfg: Config) -> RunResult:
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_hidden

    # init
    rho = float(np.clip(cfg.init_correlation, -1.0, 1.0))
    w_col0 = rng.normal(scale=cfg.init_scale, size=(n,))
    w_col0_perp = rng.normal(scale=cfg.init_scale, size=(n,))
    w_col1 = rho * w_col0 + np.sqrt(max(0.0, 1.0 - rho ** 2)) * w_col0_perp
    W = np.stack([w_col0, w_col1], axis=1)
    v = rng.normal(scale=cfg.init_scale, size=(n,))
    W_init, v_init = W.copy(), v.copy()

    # target-net copy
    W_tgt, v_tgt = W.copy(), v.copy()

    phi, phi_prime = _act(cfg.activation)

    v_s1_hist = np.empty(cfg.n_steps + 1)
    v_s2_hist = np.empty(cfg.n_steps + 1)
    td_hist = np.empty(cfg.n_steps)
    wn_hist = np.empty(cfg.n_steps + 1)

    diverged = False
    diverge_step = -1

    for t in range(cfg.n_steps):
        # Forward on S1 (online net) and S2 (target net if enabled)
        w1_on = W[:, 0]
        h1_on = phi(w1_on)
        V_s1 = float(v @ h1_on)

        if cfg.use_target_net:
            w2_t = W_tgt[:, 1]
            h2_t = phi(w2_t)
            V_s2_target = float(v_tgt @ h2_t)
        else:
            w2_on = W[:, 1]
            h2_on = phi(w2_on)
            V_s2_target = float(v @ h2_on)

        # Also track V_s2 under the online net for plotting
        w2_on = W[:, 1]
        h2_on = phi(w2_on)
        V_s2 = float(v @ h2_on)

        v_s1_hist[t] = V_s1
        v_s2_hist[t] = V_s2
        wn_hist[t] = float(np.sqrt((W ** 2).sum() + (v ** 2).sum()))

        # TD error (semi-gradient, target treated as constant)
        target = cfg.reward + cfg.gamma * V_s2_target
        delta = V_s1 - target
        td_hist[t] = delta

        # Divergence check
        if not diverged and (abs(V_s1) > cfg.divergence_threshold or
                             abs(V_s2) > cfg.divergence_threshold or
                             not np.isfinite(V_s1) or not np.isfinite(V_s2)):
            diverged = True
            diverge_step = t
            # Stop updating so later history stays at the blow-up value
            # (easier to plot; further steps would overflow).
            v_s1_hist[t + 1:] = V_s1
            v_s2_hist[t + 1:] = V_s2
            wn_hist[t + 1:] = wn_hist[t]
            td_hist[t + 1:] = delta
            break

        # Gradients of V(S1) w.r.t. params
        grad_v = h1_on
        grad_W_col0 = v * phi_prime(w1_on)
        # grad w.r.t. W[:,1] is zero

        # SGD step
        v = v - cfg.lr * delta * grad_v
        W[:, 0] = W[:, 0] - cfg.lr * delta * grad_W_col0

        # Target-net refresh
        if cfg.use_target_net and (t + 1) % cfg.target_update_interval == 0:
            W_tgt = W.copy()
            v_tgt = v.copy()

    # Final state
    V_s1 = float(v @ phi(W[:, 0]))
    V_s2 = float(v @ phi(W[:, 1]))
    v_s1_hist[cfg.n_steps] = V_s1
    v_s2_hist[cfg.n_steps] = V_s2
    wn_hist[cfg.n_steps] = float(np.sqrt((W ** 2).sum() + (v ** 2).sum()))

    return RunResult(
        cfg=cfg,
        v_s1=v_s1_hist,
        v_s2=v_s2_hist,
        td_error=td_hist,
        weight_norm=wn_hist,
        diverged=diverged,
        diverge_step=diverge_step,
        W_final=W,
        v_final=v,
        W_init=W_init,
        v_init=v_init,
    )


# ── Multi-seed helper ────────────────────────────────────────────────────────

def run_many(cfg: Config, seeds) -> list[RunResult]:
    out = []
    for s in seeds:
        c = Config(**{**cfg.__dict__, "seed": int(s)})
        out.append(run(c))
    return out


def divergence_fraction(results) -> float:
    return float(np.mean([r.diverged for r in results]))
