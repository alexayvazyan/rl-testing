"""
exp2: divergence probability over (gamma, lr). Side-by-side heatmaps for
no-target vs target network.

Runs one multi-seed sweep with anti-correlated init (init_correlation=-1)
to highlight the bootstrap-feedback regime, and a separate sweep with
i.i.d. init (init_correlation=0) to contrast.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

from core import Config, run, divergence_fraction, run_many

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMAS = np.linspace(0.0, 0.999, 14)
LRS    = np.geomspace(0.05, 0.5, 14)
N_SEEDS = 80


def sweep(cfg_base: Config, seeds):
    no_grid = np.zeros((len(GAMMAS), len(LRS)))
    tgt_grid = np.zeros((len(GAMMAS), len(LRS)))
    for i, gamma in enumerate(GAMMAS):
        for j, lr in enumerate(LRS):
            cfg_no = Config(**{**cfg_base.__dict__, "gamma": float(gamma), "lr": float(lr), "use_target_net": False})
            cfg_t  = Config(**{**cfg_base.__dict__, "gamma": float(gamma), "lr": float(lr), "use_target_net": True})
            no_grid[i, j]  = divergence_fraction(run_many(cfg_no, seeds))
            tgt_grid[i, j] = divergence_fraction(run_many(cfg_t, seeds))
        print(f"  gamma={gamma:.3f} done")
    return no_grid, tgt_grid


def _log_edges(centres):
    """Cell edges for a geometrically-spaced array of centres."""
    log_c = np.log(centres)
    log_e = np.empty(len(centres) + 1)
    log_e[1:-1] = 0.5 * (log_c[:-1] + log_c[1:])
    log_e[0]    = log_c[0]  - 0.5 * (log_c[1] - log_c[0])
    log_e[-1]   = log_c[-1] + 0.5 * (log_c[-1] - log_c[-2])
    return np.exp(log_e)


def _lin_edges(centres):
    """Cell edges for a linearly-spaced array of centres."""
    e = np.empty(len(centres) + 1)
    e[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    e[0]    = centres[0]  - 0.5 * (centres[1] - centres[0])
    e[-1]   = centres[-1] + 0.5 * (centres[-1] - centres[-2])
    return e


def plot_heatmaps(no_grid, tgt_grid, title, tag):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    diff_grid = no_grid - tgt_grid
    diff_lim = float(np.max(np.abs(diff_grid)))
    diff_lim = max(diff_lim, 0.05)   # don't collapse colourbar to zero

    lr_edges = _log_edges(LRS)
    g_edges  = _lin_edges(GAMMAS)
    LR_E, G_E = np.meshgrid(lr_edges, g_edges)

    panels = [
        (axes[0], no_grid,   "no target net",     "magma",  0.0,        1.0),
        (axes[1], tgt_grid,  "target net",        "magma",  0.0,        1.0),
        (axes[2], diff_grid, "no-tgt minus tgt",  "RdBu_r", -diff_lim,  diff_lim),
    ]

    # Tick set chosen to be human-readable on a log axis covering [0.05, 0.5].
    lr_ticks = [0.05, 0.1, 0.2, 0.3, 0.5]
    fmt = FuncFormatter(lambda x, _: f"{x:g}")

    for ax, grid, label, cmap, vmin, vmax in panels:
        im = ax.pcolormesh(LR_E, G_E, grid, cmap=cmap, vmin=vmin, vmax=vmax,
                           shading="auto")
        ax.set_xscale("log")
        ax.set_xlim(lr_edges[0], lr_edges[-1])
        ax.set_ylim(g_edges[0], g_edges[-1])
        ax.xaxis.set_major_locator(FixedLocator(lr_ticks))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_major_formatter(fmt)
        ax.set_xlabel("learning rate")
        ax.set_ylabel("gamma")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(title)
    fig.tight_layout()
    outpath = os.path.join(RESULTS_DIR, f"exp2_heatmap_{tag}.png")
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"saved {outpath}  (diff range ±{diff_lim:.3f})")


if __name__ == "__main__":
    seeds = list(range(N_SEEDS))

    print("sweep A: anti-correlated init (rho=-1) — bootstrap-dominated")
    cfg_a = Config(activation="linear", n_hidden=4, init_scale=1.2,
                   init_correlation=-1.0, n_steps=1500,
                   target_update_interval=50)
    no_a, tgt_a = sweep(cfg_a, seeds)
    plot_heatmaps(no_a, tgt_a,
                  "Divergence probability — anti-correlated init (rho=-1, init_scale=1.2)",
                  "anticorr")

    print("sweep B: i.i.d. init (rho=0) — overshoot-dominated")
    cfg_b = Config(activation="linear", n_hidden=4, init_scale=1.2,
                   init_correlation=0.0, n_steps=1500,
                   target_update_interval=50)
    no_b, tgt_b = sweep(cfg_b, seeds)
    plot_heatmaps(no_b, tgt_b,
                  "Divergence probability — i.i.d. init (rho=0)",
                  "iid")

    np.savez(os.path.join(RESULTS_DIR, "exp2_data.npz"),
             gammas=GAMMAS, lrs=LRS,
             anticorr_notgt=no_a, anticorr_tgt=tgt_a,
             iid_notgt=no_b, iid_tgt=tgt_b)
