"""
Experiment 8b — Action Persistence (clean version)

Per-fork measurement: for each starting state, which action leads to longer
survival? How long does that advantage persist?

Instead of aggregating survival_0 and survival_1 across all forks (which
cancels out due to CartPole symmetry), we measure per fork:
  - did action 0 or action 1 survive longer at step k?
  - advantage(k) = |survive_0(k) - survive_1(k)| per fork, then averaged

This answers: given that one action was better than the other at k=1,
how likely is it still better at k=10, 20, ...?
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_FORKS  = 1000
MAX_K    = 80

env = gym.make("CartPole-v1")

# per-fork survival sequences
per_fork_abs_diff = np.zeros(MAX_K + 1)   # E[|s0(k) - s1(k)|]
per_fork_counts   = np.zeros(MAX_K + 1)

# also track which action "won" (survived longer) per fork
winner_consistent = np.zeros(MAX_K + 1)   # fraction of forks where winner at k=1 still wins at k
winner_counts     = np.zeros(MAX_K + 1)

n_forks = 0

for _ in range(N_FORKS):
    obs, _ = env.reset()
    walk = np.random.randint(1, 10)
    valid = True
    for _ in range(walk):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            valid = False
            break
    if not valid:
        continue

    start_state = obs.copy()

    # simulate both forks
    survival = [None, None]
    for fork_action in [0, 1]:
        env2 = gym.make("CartPole-v1")
        env2.reset()
        env2.unwrapped.state = start_state.copy()
        _, _, term, trunc, _ = env2.step(fork_action)
        done = term or trunc
        k = 1
        while not done and k < MAX_K:
            _, _, term, trunc, _ = env2.step(env2.action_space.sample())
            done = term or trunc
            k += 1
        survival[fork_action] = k  # survived this many steps after fork
        env2.close()

    s0, s1 = survival[0], survival[1]

    # per-fork: at each lag k, did each action survive? (binary)
    for k in range(1, MAX_K + 1):
        alive0 = 1 if s0 >= k else 0
        alive1 = 1 if s1 >= k else 0
        per_fork_abs_diff[k] += abs(alive0 - alive1)
        per_fork_counts[k]   += 1

    # winner at k=1
    if s0 != s1:  # one action was clearly better
        first_winner = 0 if s0 > s1 else 1
        for k in range(1, MAX_K + 1):
            alive0 = 1 if s0 >= k else 0
            alive1 = 1 if s1 >= k else 0
            # does the winner at k=1 still lead at k?
            winner_at_k = 0 if alive0 > alive1 else (1 if alive1 > alive0 else -1)
            if winner_at_k != -1:  # not tied
                winner_consistent[k] += 1 if winner_at_k == first_winner else 0
                winner_counts[k]     += 1

    n_forks += 1

env.close()

mean_abs_diff = np.divide(per_fork_abs_diff, per_fork_counts,
                          where=per_fork_counts > 0, out=np.zeros(MAX_K + 1))
winner_frac   = np.divide(winner_consistent, winner_counts,
                          where=winner_counts > 0, out=np.full(MAX_K + 1, np.nan))

print(f"Completed {n_forks} forks")
print(f"\nMean |survive_0 - survive_1| per fork at each lag:")
for k in [1, 5, 10, 20, 30, 40, 50]:
    print(f"  k={k:2d}: {mean_abs_diff[k]:.3f}")

print(f"\nFraction of forks where first-step winner still leads at lag k:")
for k in [1, 5, 10, 20, 30, 40, 50]:
    print(f"  k={k:2d}: {winner_frac[k]:.3f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ks = np.arange(1, MAX_K + 1)

# Left: mean abs diff
axes[0].plot(ks, mean_abs_diff[1:], color='steelblue', linewidth=2)
axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_xlabel("k (steps after fork action)")
axes[0].set_ylabel("Mean |P(survive k | action 0) - P(survive k | action 1)| per fork")
axes[0].set_title("Action Persistence — Mean Absolute Survival Difference\n"
                  "(per-fork, then averaged — removes directional bias)")
axes[0].set_xlim(0, MAX_K)

# Right: winner consistency
valid = ~np.isnan(winner_frac[1:])
axes[1].plot(ks[valid], winner_frac[1:][valid], color='darkorange', linewidth=2,
             label="P(same action still winning at k)")
axes[1].axhline(0.5, color='red', linestyle=':', alpha=0.7, label='0.5 = random chance')
axes[1].set_xlabel("k (steps after fork action)")
axes[1].set_ylabel("Fraction of forks where initial winner still leads")
axes[1].set_title("Action Persistence — Winner Consistency\n"
                  "(does the better action at k=1 stay better at k?)")
axes[1].legend(fontsize=9)
axes[1].set_xlim(0, MAX_K)
axes[1].set_ylim(0, 1.05)

plt.suptitle("Experiment 8b: Action Persistence (per-fork measurement)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "exp8b_action_persistence.png"), dpi=200)
plt.close()
print("\nSaved to results/exp8b_action_persistence.png")
