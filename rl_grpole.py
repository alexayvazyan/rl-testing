"""
GRPO (Group Relative Policy Optimization) for CartPole

Key difference from PPO: no critic network. Instead, for each state we sample
G trajectories from the current policy, compute discounted returns for each,
and use the group mean as the baseline. Advantages = return - group_mean.

Then we do clipped policy gradient updates (same as PPO) using these advantages.
"""

import numpy as np
import torch as pt
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
import os

dir = os.path.dirname(os.path.abspath(__file__))

class Actor(nn.Module):
    def __init__(self, nstates, nactions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nstates, 64),
            nn.ReLU(),
            nn.Linear(64, nactions)
        )
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.net(x)


def rollout(env, actor, gamma, max_steps=500):
    """Run one episode, return list of (obs, action, logprob) and discounted return."""
    obs, _ = env.reset()
    transitions = []
    rewards = []
    done = False
    while not done and len(rewards) < max_steps:
        with pt.no_grad():
            logits = actor(pt.tensor(obs, dtype=pt.float32))
            probs = pt.softmax(logits, dim=-1)
            action = pt.multinomial(probs, 1).item()
            logprob = pt.log(probs)[action]
        new_obs, reward, term, trunc, _ = env.step(action)
        transitions.append((obs, action, logprob.item()))
        rewards.append(reward)
        obs = new_obs
        done = term or trunc

    # compute per-step RTG
    rtgs = np.zeros(len(rewards))
    for i in range(len(rewards) - 1, -1, -1):
        if i == len(rewards) - 1:
            rtgs[i] = rewards[i]
        else:
            rtgs[i] = rewards[i] + gamma * rtgs[i + 1]

    return transitions, rtgs, len(rewards)


def train_grpo(gamma, n_episodes=2000, group_size=8, batch_epochs=10,
               batch_size=64, lr=0.001, clip=0.2, verbose=True):
    """Train GRPO on CartPole with given gamma. Return run_lengths."""
    env = gym.make("CartPole-v1")
    actor = Actor(4, 2)
    optimizer = pt.optim.Adam(actor.parameters(), lr=lr)
    run_lengths = []

    ep = 0
    while ep < n_episodes:
        # collect a group of trajectories
        group_transitions = []
        group_rtgs = []
        group_lengths = []

        for _ in range(group_size):
            transitions, rtgs, length = rollout(env, actor, gamma)
            group_transitions.append(transitions)
            group_rtgs.append(rtgs)
            group_lengths.append(length)
            run_lengths.append(length)

        ep += group_size

        # flatten all transitions with their RTGs
        all_obs = []
        all_actions = []
        all_old_logprobs = []
        all_rtgs = []

        # compute group-level baseline: mean return across the group
        group_returns = [rtgs[0] for rtgs in group_rtgs]  # return = RTG at step 0
        group_mean = np.mean(group_returns)
        group_std = np.std(group_returns) + 1e-8

        for transitions, rtgs in zip(group_transitions, group_rtgs):
            for t, (obs, action, logprob) in enumerate(transitions):
                all_obs.append(obs)
                all_actions.append(action)
                all_old_logprobs.append(logprob)
                # advantage: this step's RTG minus group mean, normalized
                all_rtgs.append(rtgs[t])

        all_obs = pt.tensor(np.array(all_obs), dtype=pt.float32)
        all_actions = pt.tensor(all_actions, dtype=pt.long)
        all_old_logprobs = pt.tensor(all_old_logprobs, dtype=pt.float32)
        all_rtgs = pt.tensor(np.array(all_rtgs), dtype=pt.float32)

        # advantages: RTG - group mean return, normalized by group std
        advantages = (all_rtgs - group_mean) / group_std

        # PPO-style clipped updates
        n_samples = len(all_obs)
        for _ in range(batch_epochs):
            idx = pt.randperm(n_samples)[:min(batch_size, n_samples)]
            s = all_obs[idx]
            a = all_actions[idx]
            old_lp = all_old_logprobs[idx]
            adv = advantages[idx]

            new_logits = actor(s)
            new_probs = pt.softmax(new_logits, dim=-1)
            new_lp = pt.log(new_probs)[range(len(idx)), a]

            ratio = pt.exp(new_lp - old_lp)
            loss = -pt.mean(pt.min(ratio * adv, pt.clamp(ratio, 1 - clip, 1 + clip) * adv))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose and ep % 200 == 0:
            recent = run_lengths[-min(50, len(run_lengths)):]
            print(f"  ep {ep:4d}: mean last 50 = {np.mean(recent):.1f}")

    env.close()
    return run_lengths, actor


# ── Gamma sweep + violin plot ─────────────────────────────────────────────────
if __name__ == "__main__":
    GAMMAS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
    N_EPISODES = 50  # random rollouts for RTG violin
    results = {}

    # first: train at each gamma and record performance
    print("=== GRPO Gamma Sweep ===")
    for gamma in GAMMAS:
        print(f"\nGamma = {gamma}")
        lengths, actor = train_grpo(gamma, n_episodes=2000)
        results[gamma] = {
            'lengths': lengths,
            'actor': actor,
        }

    # RTG violin plot (random policy, same as exp4 — for comparison)
    print("\nCollecting random policy RTG distributions...")
    env = gym.make("CartPole-v1")
    all_trajectories = []
    for _ in range(N_EPISODES):
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
        for i in range(len(rewards) - 1, -1, -1):
            if i == len(rewards) - 1:
                rtgs[i] = rewards[i]
            else:
                rtgs[i] = rewards[i] + gamma * rtgs[i + 1]
        return rtgs

    rtg_data = {}
    for gamma in GAMMAS:
        all_rtgs = []
        for traj in all_trajectories:
            all_rtgs.extend(compute_rtgs(traj, gamma))
        rtg_data[gamma] = np.array(all_rtgs)

    # ── Plot 1: Training curves ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    window = 20
    for gamma in GAMMAS:
        lengths = results[gamma]['lengths']
        if len(lengths) > window:
            ma = np.convolve(lengths, np.ones(window) / window, mode='valid')
            ax.plot(range(window - 1, len(lengths)), ma, label=f'γ={gamma}', linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.set_title("GRPO Training Curves by Gamma (CartPole)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "results", "grpo_gamma_sweep.png"), dpi=200)
    plt.close()
    print("Saved grpo_gamma_sweep.png")

    # ── Plot 2: RTG violin (random policy) ───────────────────────────────────
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
    ax.set_title("RTG Distribution by Gamma\n(random policy, CartPole — GRPO baseline comparison)",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "results", "grpo_rtg_violin.png"), dpi=200)
    plt.close()
    print("Saved grpo_rtg_violin.png")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n=== Final Performance (mean last 100 episodes) ===")
    for gamma in GAMMAS:
        lengths = results[gamma]['lengths']
        last100 = lengths[-min(100, len(lengths)):]
        print(f"  gamma={gamma}: {np.mean(last100):.1f} ± {np.std(last100):.1f}")
