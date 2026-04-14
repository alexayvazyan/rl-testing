"""
exp3: effect of target-network update interval on divergence probability.

interval=1 is effectively no target network. Larger intervals should decouple
online params from target, giving stable fixed-point regression inside each
window at the cost of stale targets. We plot divergence fraction vs interval.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from core import Config, run_many, divergence_fraction

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

INTERVALS = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
N_SEEDS = 200


def curve(cfg_base):
    seeds = list(range(N_SEEDS))
    fracs_tgt = []
    for iv in INTERVALS:
        cfg_t = Config(**{**cfg_base.__dict__, "use_target_net": True, "target_update_interval": iv})
        fracs_tgt.append(divergence_fraction(run_many(cfg_t, seeds)))
    cfg_no = Config(**{**cfg_base.__dict__, "use_target_net": False})
    frac_no = divergence_fraction(run_many(cfg_no, seeds))
    return fracs_tgt, frac_no


if __name__ == "__main__":
    # Hold every hyperparam fixed across the two regimes — vary only ρ.
    SHARED = dict(activation="linear", n_hidden=4, gamma=0.99, lr=0.2,
                  init_scale=1.2, n_steps=2000)
    configs = [
        ("anti-correlated (ρ=-1)", Config(**SHARED, init_correlation=-1.0)),
        ("i.i.d. (ρ=0)",           Config(**SHARED, init_correlation=0.0)),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, cfg in configs:
        fracs_tgt, frac_no = curve(cfg)
        ax.plot(INTERVALS, fracs_tgt, "-o", label=f"{label}, target net")
        ax.axhline(frac_no, linestyle="--", color=ax.lines[-1].get_color(),
                   alpha=0.6, label=f"{label}, no target")
    ax.set_xscale("log")
    ax.set_xlabel("target-net update interval (steps)")
    ax.set_ylabel("divergence fraction")
    ax.set_title(
        f"Divergence vs target-net update interval  "
        f"(γ=0.99, lr=0.2, n_hidden=4, init_scale=1.2, n_seeds={N_SEEDS})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    outpath = os.path.join(RESULTS_DIR, "exp3_target_interval.png")
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"saved {outpath}")
