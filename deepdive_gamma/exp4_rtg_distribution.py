"""
Experiment 4 — RTG Distribution by Gamma

Key question: does low gamma compress RTGs into a narrow band,
making advantages near-zero and the actor's gradient signal uninformative?

Method: random policy rollouts (decouples from policy quality),
compute RTG distribution at each gamma, measure variance.

Analytical prediction: for a trajectory of length T with reward=1 every step,
RTG at step t = (1 - gamma^(T-t)) / (1 - gamma)
=> RTG range: [1, 1/(1-gamma)]
=> Variance scales as ~1/(1-gamma)^2
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
GAMMAS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
N_EPISODES = 50  # random policy rollouts

env = gym.make("CartPole-v1")

# ── Collect random rollouts ───────────────────────────────────────────────────
print("Collecting random policy rollouts...")
all_trajectories = []  # list of reward sequences
for ep in range(N_EPISODES):
    obs, _ = env.reset()
    rewards = []
    done = False
    while not done:
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        rewards.append(r)
        done = term or trunc
    all_trajectories.append(rewards)

ep_lengths = [len(t) for t in all_trajectories]
print(f"Episode lengths: mean={np.mean(ep_lengths):.1f}, min={min(ep_lengths)}, max={max(ep_lengths)}")
env.close()

# ── Compute RTGs for each gamma ───────────────────────────────────────────────
def compute_rtgs(rewards, gamma):
    rtgs = np.zeros(len(rewards))
    for i in range(len(rewards)-1, -1, -1):
        if i == len(rewards)-1:
            rtgs[i] = rewards[i]
        else:
            rtgs[i] = rewards[i] + gamma * rtgs[i+1]
    return rtgs

rtg_data = {}
for gamma in GAMMAS:
    all_rtgs = []
    for traj in all_trajectories:
        all_rtgs.extend(compute_rtgs(traj, gamma))
    rtg_data[gamma] = np.array(all_rtgs)

# ── Analytical variance prediction ───────────────────────────────────────────
# For constant reward=1, trajectory length T:
# RTG_t = (1 - gamma^(T-t)) / (1 - gamma)
# Mean RTG ≈ 1/(1-gamma) * (1 - gamma^(T/2)) -- rough
# Var(RTG) grows as step position varies across trajectory
# Simple measure: range = max - min RTG within an episode
def analytical_range(T, gamma):
    if gamma == 1.0:
        return T
    rtg_start = (1 - gamma**T) / (1 - gamma)
    rtg_end   = 1.0
    return rtg_start - rtg_end

analytical_ranges = []
mean_T = np.mean(ep_lengths)
for gamma in GAMMAS:
    analytical_ranges.append(analytical_range(mean_T, gamma))

# ── Metrics ───────────────────────────────────────────────────────────────────
variances   = [np.var(rtg_data[g])   for g in GAMMAS]
ranges      = [np.max(rtg_data[g]) - np.min(rtg_data[g]) for g in GAMMAS]
eff_horizons = [1/(1-g) for g in GAMMAS]

print("\nRTG statistics by gamma:")
print(f"{'Gamma':>7} {'Eff_H':>8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Var':>10}")
for g in GAMMAS:
    d = rtg_data[g]
    print(f"{g:>7} {1/(1-g):>8.1f} {np.mean(d):>8.2f} {np.std(d):>8.2f} "
          f"{np.min(d):>8.2f} {np.max(d):>8.2f} {np.var(d):>10.2f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: RTG distributions as violin plots
parts = axes[0,0].violinplot(
    [rtg_data[g] for g in GAMMAS],
    positions=range(len(GAMMAS)),
    showmeans=True, showmedians=True
)
axes[0,0].set_xticks(range(len(GAMMAS)))
axes[0,0].set_xticklabels([str(g) for g in GAMMAS], rotation=30)
axes[0,0].set_xlabel("Gamma")
axes[0,0].set_ylabel("RTG value")
axes[0,0].set_title("RTG Distribution by Gamma\n(random policy, CartPole)")

# Plot 2: RTG variance vs gamma
axes[0,1].plot(GAMMAS, variances, 'o-', color='steelblue')
axes[0,1].set_xlabel("Gamma")
axes[0,1].set_ylabel("Var(RTG)")
axes[0,1].set_title("RTG Variance vs Gamma")
axes[0,1].set_yscale('log')

# Plot 3: RTG variance vs effective horizon (log-log)
axes[0,2].loglog(eff_horizons, variances, 'o-', color='darkorange', label='empirical')
# Analytical prediction: Var ~ eff_horizon^2
predicted = [(e**2) * variances[0] / eff_horizons[0]**2 for e in eff_horizons]
axes[0,2].loglog(eff_horizons, predicted, '--', color='gray', label='~H² scaling')
axes[0,2].set_xlabel("Effective Horizon (log)")
axes[0,2].set_ylabel("Var(RTG) (log)")
axes[0,2].set_title("RTG Variance vs Effective Horizon\n(log-log)")
axes[0,2].legend()

# Plot 4: RTG histograms for a few gammas
colors = ['blue', 'orange', 'green', 'red']
selected = [0.5, 0.9, 0.99, 0.999]
for i, g in enumerate(selected):
    axes[1,0].hist(rtg_data[g], bins=50, alpha=0.5, label=f'γ={g}',
                   color=colors[i], density=True)
axes[1,0].set_xlabel("RTG value")
axes[1,0].set_ylabel("Density")
axes[1,0].set_title("RTG Histograms (selected gammas)")
axes[1,0].legend()

# Plot 5: Within-episode RTG curves (show how RTG varies within one episode)
sample_traj = max(all_trajectories, key=len)  # longest episode
steps = np.arange(len(sample_traj))
for g in [0.5, 0.9, 0.99, 0.999]:
    rtgs = compute_rtgs(sample_traj, g)
    axes[1,1].plot(steps, rtgs, label=f'γ={g}')
axes[1,1].set_xlabel("Step within episode")
axes[1,1].set_ylabel("RTG")
axes[1,1].set_title(f"RTG Shape Within One Episode (len={len(sample_traj)})")
axes[1,1].legend()

# Plot 6: Normalized RTG std (std / mean) — relative spread
rel_std = [np.std(rtg_data[g]) / np.mean(rtg_data[g]) for g in GAMMAS]
axes[1,2].plot(GAMMAS, rel_std, 'o-', color='purple')
axes[1,2].set_xlabel("Gamma")
axes[1,2].set_ylabel("Std(RTG) / Mean(RTG)")
axes[1,2].set_title("Relative RTG Spread\n(= signal-to-noise proxy for advantage)")

plt.suptitle("Experiment 4: RTG Distribution Analysis", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "exp4_rtg_distribution.png"), dpi=200)
plt.close()

print(f"\nKey result: RTG variance at gamma=0.99 is {variances[-2]/variances[0]:.0f}x larger than at gamma=0.5")
print(f"Relative spread (std/mean): {dict(zip([str(g) for g in GAMMAS], [f'{r:.3f}' for r in rel_std]))}")
print("\nImplication: at low gamma, all RTGs look similar -> small advantages -> weak actor gradient signal")
