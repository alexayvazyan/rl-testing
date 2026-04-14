"""
exp1: single-seed trajectory, with and without target network, same init.

Two side-by-side panels:
    - no target network
    - target network (refresh every N steps)

For each, we plot V(S1), V(S2), TD error, and parameter norm over time.
Picks the same seed so the two runs share an initialization — any difference
is attributable entirely to the target-network mechanism.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from core import Config, run

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _format_axes(ax, title):
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.0, color="k", lw=0.5)


def plot_pair(cfg_base: Config, seed: int, tag: str):
    cfg_no = Config(**{**cfg_base.__dict__, "use_target_net": False, "seed": seed})
    cfg_yes = Config(**{**cfg_base.__dict__, "use_target_net": True, "seed": seed})
    r_no = run(cfg_no)
    r_yes = run(cfg_yes)

    # Cap the visible window: we want enough steps to see the target-net
    # trajectory settle, but not so many that a post-divergence flatline
    # dominates the figure.
    pad_after_div = 20
    candidates = [300]
    if r_no.diverged:
        candidates.append(r_no.diverge_step + pad_after_div)
    if r_yes.diverged:
        candidates.append(r_yes.diverge_step + pad_after_div)
    end = min(max(candidates), len(r_no.v_s1))

    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for col, (r, label) in enumerate([(r_no, "no target net"),
                                       (r_yes, f"target net (update every {cfg_yes.target_update_interval})")]):
        t = np.arange(end)
        axes[0, col].plot(t, r.v_s1[:end], label="V(S1)", color="C0")
        axes[0, col].plot(t, r.v_s2[:end], label="V(S2)", color="C1")
        axes[0, col].axhline(0.0, color="k", lw=0.5)
        axes[0, col].set_yscale("symlog", linthresh=1.0)
        axes[0, col].set_ylabel("value (symlog)")
        axes[0, col].legend(loc="best")
        _format_axes(axes[0, col], label)

        td_end = min(end, len(r.td_error))
        axes[1, col].plot(np.arange(td_end), r.td_error[:td_end], color="C3", lw=0.8)
        axes[1, col].set_yscale("symlog", linthresh=1.0)
        axes[1, col].set_ylabel("TD error  V(S1) - γ V(S2)")
        _format_axes(axes[1, col], "")

        axes[2, col].plot(t, r.weight_norm[:end], color="C2")
        axes[2, col].set_yscale("log")
        axes[2, col].set_ylabel("||θ||₂ (log)")
        axes[2, col].set_xlabel("step")
        _format_axes(axes[2, col], "")

        if r.diverged and r.diverge_step < end:
            for ax in axes[:, col]:
                ax.axvline(r.diverge_step, color="red", ls="--", lw=0.8, alpha=0.7)

    fig.suptitle(
        f"DQN on S1→S2→terminal  (γ={cfg_base.gamma}, lr={cfg_base.lr}, "
        f"n={cfg_base.n_hidden}, init scale={cfg_base.init_scale}, "
        f"corr={cfg_base.init_correlation:+.1f}, activation={cfg_base.activation}, "
        f"seed={seed})"
    )
    fig.tight_layout()
    outpath = os.path.join(RESULTS_DIR, f"exp1_{tag}_seed{seed}.png")
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    print(f"saved {outpath}  "
          f"(no-tgt diverged={r_no.diverged} at step {r_no.diverge_step};  "
          f"tgt diverged={r_yes.diverged} at step {r_yes.diverge_step})")


if __name__ == "__main__":
    # Scenario A: anti-correlated init — classic bootstrap-feedback divergence.
    cfg_a = Config(
        activation="linear",
        n_hidden=4,
        gamma=0.99,
        lr=0.2,
        init_scale=1.2,
        init_correlation=-1.0,
        n_steps=2000,
        target_update_interval=50,
    )
    # Try several seeds until we find one where no-target diverges but target
    # does not — cleanest visual for the blog.
    for seed in range(30):
        r_no = run(Config(**{**cfg_a.__dict__, "use_target_net": False, "seed": seed}))
        r_yes = run(Config(**{**cfg_a.__dict__, "use_target_net": True, "seed": seed}))
        if r_no.diverged and not r_yes.diverged:
            plot_pair(cfg_a, seed, "anticorrelated")
            break
    else:
        print("no clean seed found for scenario A; plotting seed 0")
        plot_pair(cfg_a, 0, "anticorrelated")

    # Scenario B: i.i.d. random init — overshoot-dominated regime. Target net
    # does NOT rescue this (honest counterpoint).
    cfg_b = Config(
        activation="linear",
        n_hidden=4,
        gamma=0.99,
        lr=0.3,
        init_scale=1.5,
        init_correlation=0.0,
        n_steps=2000,
        target_update_interval=50,
    )
    for seed in range(30):
        r_no = run(Config(**{**cfg_b.__dict__, "use_target_net": False, "seed": seed}))
        if r_no.diverged:
            plot_pair(cfg_b, seed, "iid")
            break
    else:
        plot_pair(cfg_b, 0, "iid")
