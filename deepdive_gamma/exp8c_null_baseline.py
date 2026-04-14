"""
Null baseline for exp8c — what does the corrected estimator
(|mean signed difference|) look like when both actions are identical?

Method: from each starting state, run 400 forks all with a random first action
(same distribution), randomly split into two groups of 200, compute
signed difference per step, then take |signed diff| per state.

Uses same methodology as updated exp8c (500 states, 200 forks).
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_STATES = 500
N_TOTAL = 400  # total forks per state, split into two groups of 200
MAX_K = 80

env = gym.make("CartPole-v1")

null_curves = []
null_peak_heights = []
null_peak_locations = []

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

    # run N_TOTAL forks, all with a random first action (same distribution)
    survivals = []
    for _ in range(N_TOTAL):
        env2 = gym.make("CartPole-v1")
        env2.reset()
        env2.unwrapped.state = start_state.copy()
        _, _, term, trunc, _ = env2.step(env2.action_space.sample())
        done = term or trunc
        k = 1
        while not done and k < MAX_K:
            _, _, term, trunc, _ = env2.step(env2.action_space.sample())
            done = term or trunc
            k += 1
        survivals.append(k)
        env2.close()

    # randomly split into two groups of 200
    survivals = np.array(survivals)
    perm = np.random.permutation(N_TOTAL)
    half = N_TOTAL // 2
    group0 = survivals[perm[:half]]
    group1 = survivals[perm[half:]]

    # compute P(survive k) for each group
    survive_counts = np.zeros((2, MAX_K + 1))
    for s in group0:
        for kk in range(1, s + 1):
            survive_counts[0, kk] += 1
    for s in group1:
        for kk in range(1, s + 1):
            survive_counts[1, kk] += 1

    p_survive = survive_counts / half
    # same corrected methodology: signed diff, then absolute value per state
    signed_diff = p_survive[0] - p_survive[1]
    abs_signed_diff = np.abs(signed_diff)

    null_curves.append(abs_signed_diff)
    peak_k = np.argmax(abs_signed_diff[1:]) + 1
    null_peak_heights.append(abs_signed_diff[peak_k])
    null_peak_locations.append(peak_k)

env.close()

null_curves = np.array(null_curves)
null_peak_heights = np.array(null_peak_heights)
null_peak_locations = np.array(null_peak_locations)
null_mean = null_curves.mean(axis=0)

print(f"Computed {len(null_curves)} null states")
print(f"Null peak |signed diff|: mean={null_peak_heights.mean():.3f}  median={np.median(null_peak_heights):.3f}")
print(f"Null mean curve peak: {null_mean[1:].max():.3f} at k={np.argmax(null_mean[1:]) + 1}")

# ── Plot: just curves, matching exp8c style ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

ks = np.arange(MAX_K + 1)

for curve in null_curves:
    ax.plot(ks[1:], curve[1:], color='red', alpha=0.05, linewidth=0.8)
ax.plot(ks[1:], null_mean[1:], color='darkred', linewidth=2.5, label='Null mean (no real effect)')

p25_curve = np.percentile(null_curves, 25, axis=0)
p75_curve = np.percentile(null_curves, 75, axis=0)
ax.fill_between(ks[1:], p25_curve[1:], p75_curve[1:], alpha=0.2, color='red',
                label='p25\u2013p75 band')

ax.set_xlabel("k (steps after fork action)")
ax.set_ylabel("|E[P(survive k | group 0) - P(survive k | group 1)]|")
ax.set_title(f"Null Baseline: Same Distribution, Random Split\n"
             f"({len(null_curves)} starting states, {N_TOTAL // 2} forks per group)")
ax.legend(fontsize=9)
ax.set_xlim(0, MAX_K)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "exp8c_null_baseline.png"), dpi=200)
plt.close()
print("Saved to results/exp8c_null_baseline.png")
