"""
exp4: phase portrait for n_hidden=1. When n=1 and we only sample the S1→S2
transition, only (v, w1) evolve — w2 never gets a gradient because
∂V(S1)/∂w2 = 0. So the dynamics live in the (v, w1) plane with w2 as a
frozen constant. We can draw the update vector field and overlay trajectories
for both the no-target and target-net variants.

The TD fixed point (no target) is the set where v·(w1 - γ·w2) = 0, i.e.
w1 = γ·w2  OR  v = 0.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from core import Config, run

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def vector_field(w2, gamma, lr, grid=25, span=5.0):
    vs = np.linspace(-span, span, grid)
    w1s = np.linspace(-span, span, grid)
    V, W1 = np.meshgrid(vs, w1s, indexing="xy")
    delta = V * W1 - gamma * V * w2
    dv  = -lr * delta * W1
    dw1 = -lr * delta * V
    return vs, w1s, dv, dw1


def simulate_trajectory(cfg, v0, w10, w2):
    # Inline a mini run for n=1 so we can seed initial condition exactly.
    W = np.array([[w10, w2]], dtype=float)
    v = np.array([v0], dtype=float)
    W_tgt, v_tgt = W.copy(), v.copy()

    traj = [(float(v[0]), float(W[0, 0]))]
    for t in range(cfg.n_steps):
        V_s1 = float(v[0] * W[0, 0])
        if cfg.use_target_net:
            V_s2_target = float(v_tgt[0] * W_tgt[0, 1])
        else:
            V_s2_target = float(v[0] * W[0, 1])
        delta = V_s1 - (cfg.reward + cfg.gamma * V_s2_target)
        if abs(V_s1) > cfg.divergence_threshold or not np.isfinite(V_s1):
            break
        grad_v = W[0, 0]
        grad_w1 = v[0]
        v[0] -= cfg.lr * delta * grad_v
        W[0, 0] -= cfg.lr * delta * grad_w1
        if cfg.use_target_net and (t + 1) % cfg.target_update_interval == 0:
            W_tgt = W.copy()
            v_tgt = v.copy()
        traj.append((float(v[0]), float(W[0, 0])))
    return np.array(traj)


if __name__ == "__main__":
    gamma = 0.99
    lr = 0.3
    w2 = 2.0   # fixed w2 magnitude
    span = 3.5
    cfg_base = Config(n_hidden=1, activation="linear",
                       gamma=gamma, lr=lr, init_scale=w2,
                       n_steps=400, target_update_interval=50)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)

    vs, w1s, dv, dw1 = vector_field(w2, gamma, lr, grid=22, span=span)

    starts = [(-1.5, 1.5), (1.5, 1.5), (-1.5, -1.5), (1.5, -1.5),
              (0.5, 2.5), (-0.5, -2.5), (2.5, 0.5), (-2.5, -0.5)]

    for ax, use_tgt, title in [(axes[0], False, "no target net"),
                                (axes[1], True, "target net (update every 50)")]:
        # Streamplot for flow direction (normalized so arrows readable)
        mag = np.sqrt(dv ** 2 + dw1 ** 2) + 1e-12
        ax.streamplot(vs, w1s, dv, dw1, color=np.log10(mag),
                      cmap="viridis", density=1.1, linewidth=0.8, arrowsize=0.8)

        # TD fixed-point line (no-target): w1 = gamma * w2
        ax.axhline(gamma * w2, color="red", ls="--", lw=1.2, alpha=0.8,
                   label=f"w1 = γ·w2 = {gamma*w2:.2f}")
        ax.axvline(0.0, color="red", ls=":", lw=1.0, alpha=0.6, label="v = 0")

        # Trajectories from several starting points
        for v0, w10 in starts:
            cfg = Config(**{**cfg_base.__dict__, "use_target_net": use_tgt})
            traj = simulate_trajectory(cfg, v0, w10, w2)
            ax.plot(traj[:, 0], traj[:, 1], color="white", lw=1.4, alpha=0.9)
            ax.plot(traj[:, 0], traj[:, 1], color="black", lw=0.8, alpha=0.9)
            ax.plot(traj[0, 0], traj[0, 1], "o", color="cyan", markersize=5,
                    markeredgecolor="k")

        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_xlabel("v")
        ax.set_ylabel("w1")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"n=1 phase portrait  (γ={gamma}, lr={lr}, w2={w2})")
    fig.tight_layout()
    outpath = os.path.join(RESULTS_DIR, "exp4_phase_portrait.png")
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"saved {outpath}")
