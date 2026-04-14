"""
exp5: clean gamma sweep at fixed lr. Plots divergence fraction vs gamma for
no-target vs target-net, anti-correlated init vs i.i.d. init. Gives a
one-glance summary of "higher gamma + shared weights = more bootstrap risk".
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from core import Config, run_many, divergence_fraction

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMAS = np.linspace(0.0, 0.999, 25)
N_SEEDS = 250


def sweep(cfg_base):
    seeds = list(range(N_SEEDS))
    no_fracs, tgt_fracs = [], []
    for gamma in GAMMAS:
        cfg_no = Config(**{**cfg_base.__dict__, "gamma": float(gamma), "use_target_net": False})
        cfg_t  = Config(**{**cfg_base.__dict__, "gamma": float(gamma), "use_target_net": True})
        no_fracs.append(divergence_fraction(run_many(cfg_no, seeds)))
        tgt_fracs.append(divergence_fraction(run_many(cfg_t, seeds)))
    return np.array(no_fracs), np.array(tgt_fracs)


if __name__ == "__main__":
    # Hold every hyperparam fixed across the two panels — vary only the
    # init correlation. This is what makes the anti-correlated vs i.i.d.
    # comparison apples-to-apples.
    SHARED = dict(activation="linear", n_hidden=4, lr=0.2,
                  init_scale=1.2, n_steps=1500, target_update_interval=50)
    cfg_anti = Config(**SHARED, init_correlation=-1.0)
    cfg_iid  = Config(**SHARED, init_correlation=0.0)

    print("sweep anti-correlated...")
    anti_no, anti_tgt = sweep(cfg_anti)
    print("sweep i.i.d....")
    iid_no, iid_tgt = sweep(cfg_iid)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, no, tgt, title in [
        (axes[0], anti_no, anti_tgt, "anti-correlated init (rho=-1)"),
        (axes[1], iid_no, iid_tgt,  "i.i.d. init (rho=0)"),
    ]:
        ax.plot(GAMMAS, no, "-o", label="no target net", color="C3", markersize=4)
        ax.plot(GAMMAS, tgt, "-s", label="target net (every 50)", color="C0", markersize=4)
        ax.set_xlabel("gamma")
        ax.set_ylabel("fraction diverged")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(
        f"Divergence probability vs gamma  "
        f"(lr=0.2, n_hidden=4, init_scale=1.2, n_seeds={N_SEEDS})"
    )
    fig.tight_layout()
    outpath = os.path.join(RESULTS_DIR, "exp5_gamma_sweep.png")
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"saved {outpath}")
