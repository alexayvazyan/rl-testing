"""
exp6: 3D phase diagram of init weights that diverge.

With n_hidden = 1 and no biases there are exactly three scalar parameters:
    w1 = W[0,0]   weight from state-1 input to the hidden unit
    w2 = W[0,1]   weight from state-2 input to the hidden unit
    w3 = v[0]     hidden-to-output readout weight

V(S1) = w3 · w1,  V(S2) = w3 · w2,  TD error δ = w3·(w1 − γ·w2).

We sweep every (w1, w2, w3) on a grid and run the no-target semi-gradient
update until divergence or n_steps. Result: a binary 3D volume marking
which initialisations blow up at fixed γ=0.9, lr=0.05.

Output:
    exp6_phase3d_scatter.png   — 3D scatter of diverged inits
    exp6_phase3d_slices.png    — 2D cross-sections at fixed w3 values
    exp6_phase3d_time.png      — 3D scatter coloured by log(steps-to-divergence)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

GAMMA = 0.9
LR = 0.05
N_STEPS = 5000
DIV_THRESHOLD = 1e6
GRID = 51                       # per-axis resolution  (51^3 ≈ 133k inits)
AXIS_RANGE = (-8.0, 8.0)
# Divergence condition (linear TD, n=1, no target net):
#     α·(w3² + w1² − γ·w1·w2) > 2,   i.e.   w3² + w1² − γ·w1·w2 > 2/α = 40.
# So at |w|≤3 nothing diverges; we need a larger cube.


def sweep_3d():
    """Vectorised sweep over every (w1, w2, w3) on a 3D grid."""
    axis = np.linspace(*AXIS_RANGE, GRID)
    W1g, W2g, W3g = np.meshgrid(axis, axis, axis, indexing="ij")

    w1 = W1g.ravel().astype(np.float64)
    w2 = W2g.ravel().astype(np.float64)
    w3 = W3g.ravel().astype(np.float64)   # this is v[0]

    n = w1.size
    diverged = np.zeros(n, dtype=bool)
    div_step = np.full(n, N_STEPS, dtype=np.int32)
    final_vs1 = np.zeros(n)
    final_vs2 = np.zeros(n)

    alive = np.ones(n, dtype=bool)

    for t in range(N_STEPS):
        # Forward
        v_s1 = w3 * w1
        v_s2 = w3 * w2
        delta = v_s1 - GAMMA * v_s2

        # Divergence detection on still-alive inits
        blown = alive & ((np.abs(v_s1) > DIV_THRESHOLD) |
                         (np.abs(v_s2) > DIV_THRESHOLD) |
                         ~np.isfinite(v_s1) | ~np.isfinite(v_s2))
        if blown.any():
            diverged |= blown
            div_step[blown] = t
            alive &= ~blown

        if not alive.any():
            break

        # Semi-gradient update (no target net) on alive inits only.
        # grad_{w3} V(s1) = w1 ;  grad_{w1} V(s1) = w3
        w3_new = w3 - LR * delta * w1
        w1_new = w1 - LR * delta * w3
        # Apply only where alive
        w3 = np.where(alive, w3_new, w3)
        w1 = np.where(alive, w1_new, w1)
        # w2 is never updated (only sample S1→S2 transitions)

    # One final forward pass to record final values for plotting
    final_vs1[:] = w3 * w1
    final_vs2[:] = w3 * w2

    div_volume = diverged.reshape(GRID, GRID, GRID)
    step_volume = div_step.reshape(GRID, GRID, GRID)
    return axis, div_volume, step_volume


def _boundary_mask(vol):
    """Points on the boundary between diverged and non-diverged cells —
    i.e., diverged points that have at least one non-diverged 6-neighbour."""
    b = np.zeros_like(vol)
    b[1:, :, :]  |= vol[1:, :, :]  & ~vol[:-1, :, :]
    b[:-1, :, :] |= vol[:-1, :, :] & ~vol[1:, :, :]
    b[:, 1:, :]  |= vol[:, 1:, :]  & ~vol[:, :-1, :]
    b[:, :-1, :] |= vol[:, :-1, :] & ~vol[:, 1:, :]
    b[:, :, 1:]  |= vol[:, :, 1:]  & ~vol[:, :, :-1]
    b[:, :, :-1] |= vol[:, :, :-1] & ~vol[:, :, 1:]
    return b


def plot_scatter(axis, div_volume, outpath):
    w1g, w2g, w3g = np.meshgrid(axis, axis, axis, indexing="ij")

    # Plot only the boundary shell — otherwise the interior hides the shape.
    shell = _boundary_mask(div_volume)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    # Colour the shell by w3 so the stability tube through the cube is visible.
    sc = ax.scatter(
        w1g[shell], w2g[shell], w3g[shell],
        c=w3g[shell], cmap="coolwarm", s=9, alpha=0.55,
        marker="o", edgecolors="none",
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.08)
    cb.set_label("w3")
    ax.set_xlabel("w1  (state1 → hidden)")
    ax.set_ylabel("w2  (state2 → hidden)")
    ax.set_zlabel("w3  (hidden → output, = v)")
    ax.set_xlim(*AXIS_RANGE); ax.set_ylim(*AXIS_RANGE); ax.set_zlim(*AXIS_RANGE)
    ax.set_title(
        f"Divergence boundary in weight space  "
        f"(γ={GAMMA}, lr={LR}, n=1, no target net)\n"
        f"{int(div_volume.sum())}/{div_volume.size} inits diverge "
        f"({div_volume.mean()*100:.1f}%).  "
        f"Plot shows shell: red outside → stable tube inside."
    )
    ax.view_init(elev=22, azim=-58)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"saved {outpath}")


def plot_slices(axis, div_volume, outpath):
    """2D cross sections at a handful of w3 values."""
    # Pick representative w3 indices
    idxs = np.linspace(0, GRID - 1, 9).round().astype(int)
    fig, axes = plt.subplots(3, 3, figsize=(11, 11), sharex=True, sharey=True)
    for ax, k in zip(axes.ravel(), idxs):
        slab = div_volume[:, :, k].T   # so w1 is x-axis, w2 is y-axis
        ax.imshow(slab, origin="lower", extent=[*AXIS_RANGE, *AXIS_RANGE],
                  cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
        ax.axhline(0, color="k", lw=0.4, alpha=0.5)
        ax.axvline(0, color="k", lw=0.4, alpha=0.5)
        # overlay w1 = γ w2 line (the TD fixed set for fixed w3 ≠ 0)
        ax.plot(axis, GAMMA * axis, color="blue", ls="--", lw=1.0, alpha=0.7,
                label=f"w1 = γ·w2" if k == idxs[0] else None)
        ax.set_title(f"w3 = {axis[k]:+.2f}", fontsize=10)
    for ax in axes[-1, :]:
        ax.set_xlabel("w2")
    for ax in axes[:, 0]:
        ax.set_ylabel("w1")
    fig.suptitle(f"Divergence slices by w3  (red = diverges;  γ={GAMMA}, lr={LR})")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"saved {outpath}")


def plot_time_to_divergence(axis, div_volume, step_volume, outpath):
    w1g, w2g, w3g = np.meshgrid(axis, axis, axis, indexing="ij")

    # Plot only the shell, coloured by time-to-divergence.
    shell = _boundary_mask(div_volume)
    steps = step_volume[shell].astype(float)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(w1g[shell], w2g[shell], w3g[shell],
                    c=np.log10(steps + 1.0), s=9, alpha=0.6,
                    cmap="viridis", marker="o", edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.08)
    cb.set_label("log10(steps to divergence)")
    ax.set_xlabel("w1"); ax.set_ylabel("w2"); ax.set_zlabel("w3")
    ax.set_xlim(*AXIS_RANGE); ax.set_ylim(*AXIS_RANGE); ax.set_zlim(*AXIS_RANGE)
    ax.set_title(f"Time to divergence on the shell  (γ={GAMMA}, lr={LR})")
    ax.view_init(elev=22, azim=-58)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    print(f"saved {outpath}")


if __name__ == "__main__":
    print(f"sweeping {GRID}^3 = {GRID**3} initialisations...")
    axis, div_volume, step_volume = sweep_3d()
    frac = div_volume.mean()
    print(f"overall divergence fraction: {frac:.3f} "
          f"({int(div_volume.sum())}/{div_volume.size})")

    plot_scatter(axis, div_volume,
                 os.path.join(RESULTS_DIR, "exp6_phase3d_scatter.png"))
    plot_slices(axis, div_volume,
                os.path.join(RESULTS_DIR, "exp6_phase3d_slices.png"))
    plot_time_to_divergence(axis, div_volume, step_volume,
                            os.path.join(RESULTS_DIR, "exp6_phase3d_time.png"))

    np.savez(os.path.join(RESULTS_DIR, "exp6_phase3d.npz"),
             axis=axis, div_volume=div_volume, step_volume=step_volume,
             gamma=GAMMA, lr=LR)
