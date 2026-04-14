"""
Experiment 8c — Action Persistence across Starting States

For each of N_STATES distinct starting states:
  - Run N_REPEATS forks (action 0 vs action 1, then random policy)
  - Compute the per-state signed survival difference: P(survive k | a=0) - P(survive k | a=1)
  - Take absolute value of the signed mean per state (not mean of absolute values)
    to avoid positive bias from sampling noise

Plot:
  - All per-state |mean signed difference| curves (faint) + mean curve (bold)
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_STATES   = 500   # distinct starting states
N_REPEATS  = 200   # forks per starting state (to estimate P(survive k | action))
MAX_K      = 80

env = gym.make("CartPole-v1")

all_curves      = []   # shape: (N_STATES, MAX_K+1)
peak_heights    = []
peak_locations  = []

for state_idx in range(N_STATES):
    obs, _ = env.reset()
    walk = np.random.randint(1, 15)
    valid = True
    for _ in range(walk):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            valid = False
            break
    if not valid:
        continue

    start_state = obs.copy()

    # for this starting state, estimate P(survive k | action) via N_REPEATS rolls
    survive_counts = np.zeros((2, MAX_K + 1))  # [action, k]

    for _ in range(N_REPEATS):
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
            survived = k  # survived this many steps
            for kk in range(1, survived + 1):
                survive_counts[fork_action, kk] += 1
            env2.close()

    p_survive = survive_counts / N_REPEATS  # P(survive k | action, start_state)
    # signed difference first (noise cancels), then absolute value per state
    signed_diff = p_survive[0] - p_survive[1]
    abs_signed_diff = np.abs(signed_diff)

    all_curves.append(abs_signed_diff)
    peak_k    = np.argmax(abs_signed_diff[1:]) + 1  # ignore k=0
    peak_val  = abs_signed_diff[peak_k]
    peak_heights.append(peak_val)
    peak_locations.append(peak_k)

env.close()

all_curves     = np.array(all_curves)
peak_heights   = np.array(peak_heights)
peak_locations = np.array(peak_locations)

print(f"Computed {len(all_curves)} starting states")
print(f"\nPeak |signed diff|: mean={peak_heights.mean():.3f}  median={np.median(peak_heights):.3f}  "
      f"std={peak_heights.std():.3f}  max={peak_heights.max():.3f}")
print(f"Peak location:      mean={peak_locations.mean():.1f}  median={np.median(peak_locations):.0f}  "
      f"std={peak_locations.std():.1f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

ks = np.arange(MAX_K + 1)

for curve in all_curves:
    ax.plot(ks[1:], curve[1:], color='steelblue', alpha=0.08, linewidth=0.8)
mean_curve = all_curves.mean(axis=0)
ax.plot(ks[1:], mean_curve[1:], color='black', linewidth=2.5, label='Mean across states')
p25_curve  = np.percentile(all_curves, 25, axis=0)
p75_curve  = np.percentile(all_curves, 75, axis=0)
ax.fill_between(ks[1:], p25_curve[1:], p75_curve[1:], alpha=0.25, color='steelblue',
                label='p25\u2013p75 band')
ax.set_xlabel("k (steps after fork action)")
ax.set_ylabel("|E[P(survive k | a=0) - P(survive k | a=1)]|")
ax.set_title(f"Action Persistence Curves\n({len(all_curves)} starting states, {N_REPEATS} forks each)")
ax.legend(fontsize=9)
ax.set_xlim(0, MAX_K)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "exp8c_persistence_by_state.png"), dpi=200)
plt.close()
print("Saved to results/exp8c_persistence_by_state.png")
