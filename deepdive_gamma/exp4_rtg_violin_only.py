"""
Experiment 4 (standalone) — RTG Distribution violin plot only.

Shows how gamma compresses the RTG distribution at low values
and expands it at high values.
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
GAMMAS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
N_EPISODES = 50

env = gym.make("CartPole-v1")

print("Collecting random policy rollouts...")
all_trajectories = []
for ep in range(N_EPISODES):
    obs, _ = env.reset()
    rewards = []
    done = False
    while not done:
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        rewards.append(r)
        done = term or trunc
    all_trajectories.append(rewards)
env.close()

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

fig, ax = plt.subplots(figsize=(8, 6))

parts = ax.violinplot(
    [rtg_data[g] for g in GAMMAS],
    positions=range(len(GAMMAS)),
    showmeans=True, showmedians=True
)

ax.set_xticks(range(len(GAMMAS)))
ax.set_xticklabels([str(g) for g in GAMMAS])
ax.set_xlabel("Gamma", fontsize=12)
ax.set_ylabel("RTG value", fontsize=12)
ax.set_title("RTG Distribution by Gamma\n(random policy, CartPole)", fontsize=14)

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "exp4_rtg_violin_only.png")
plt.savefig(out_path, dpi=200)
plt.close()

print(f"Saved to {out_path}")
