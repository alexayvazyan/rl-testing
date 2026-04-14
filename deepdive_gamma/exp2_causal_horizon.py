"""
Experiment 2 - Empirical Causal Horizon of CartPole

Method:
- Train a good policy (gamma=0.99) to convergence
- Run episodes with the trained policy
- At a random timestep t, perturb the action (flip it) for ONE step, then resume policy
- Measure survival length after perturbation vs unperturbed baseline
- delta_survival(k) tells us: how much does a perturbation at t affect survival k steps later?
- The k where this effect decays to 0 is the causal horizon
"""

import numpy as np
import torch as pt
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ── Model ─────────────────────────────────────────────────────────────────────
class NN(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, 64), nn.ReLU(),
            nn.Linear(64, 64),  nn.ReLU(),
            nn.Linear(64, nout)
        )
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

class Buffer:
    def __init__(self, maxsize):
        self.buf, self.idx, self.maxsize = [], 0, maxsize

    def push(self, x):
        if len(self.buf) < self.maxsize:
            self.buf.append(None)
        self.buf[self.idx] = x
        self.idx = (self.idx + 1) % self.maxsize

    def sample(self, n):
        import random
        return random.sample(self.buf, n)

    def merge_rtg(self, rtg):
        for i, r in enumerate(rtg):
            self.buf[i] = self.buf[i] + (r,)

    def reset(self):
        self.buf, self.idx = [], 0

    def __len__(self):
        return len(self.buf)

    def __getitem__(self, i):
        return self.buf[i]

# ── Train policy ──────────────────────────────────────────────────────────────
def train_policy(n_episodes=1500, gamma=0.99, seed=42):
    pt.manual_seed(seed)
    env = gym.make("CartPole-v1")
    actor  = NN(4, 2)
    critic = NN(4, 1)
    buf    = Buffer(4000)
    actor_opt  = pt.optim.Adam(actor.parameters(),  lr=0.001)
    critic_opt = pt.optim.Adam(critic.parameters(), lr=0.001)
    criterion  = nn.MSELoss()
    run_lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = terminated = False
        steps = 0

        while not done:
            with pt.no_grad():
                logits = actor(pt.tensor(obs, dtype=pt.float32))
            probs  = pt.softmax(logits, dim=-1)
            action = pt.multinomial(probs, 1).item()
            logprob = pt.log(probs)[action].detach()
            new_obs, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            done = terminated or truncated
            buf.push((obs, action, reward, done, logprob))
            obs = new_obs

            if len(buf) > 800 and done:
                rtg = np.zeros(len(buf))
                for i in range(len(buf) - 1, -1, -1):
                    rtg[i] = buf[i][2] if buf[i][3] else gamma * rtg[i+1] + buf[i][2]
                buf.merge_rtg(rtg)
                for _ in range(10):
                    samples = buf.sample(50)
                    s, a, r, d, lp, rtg_b = zip(*samples)
                    s = pt.stack([pt.tensor(x, dtype=pt.float32) for x in s])
                    rtg_b = pt.tensor(rtg_b, dtype=pt.float32)
                    lp = pt.stack(list(lp))
                    v = critic(s).squeeze()
                    critic_opt.zero_grad(); criterion(v, rtg_b).backward(); critic_opt.step()
                    adv = rtg_b - v.detach()
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    new_lp = pt.log(pt.softmax(actor(s), dim=-1))[range(50), a]
                    ratio = pt.exp(new_lp - lp)
                    loss = -pt.mean(pt.min(ratio*adv, pt.clamp(ratio,0.8,1.2)*adv))
                    actor_opt.zero_grad(); loss.backward(); actor_opt.step()
                buf.reset()

        run_lengths.append(steps)
        if (ep+1) % 100 == 0:
            print(f"  Training ep {ep+1}/{n_episodes}, mean_last50={np.mean(run_lengths[-50:]):.1f}")

    env.close()
    return actor, run_lengths

# ── Perturbation analysis ──────────────────────────────────────────────────────
def run_perturbed(actor, env, perturb_at_step, seed=None):
    """Run one episode, perturbing action at perturb_at_step. Return survival length."""
    if seed is not None:
        env.reset(seed=seed)
    obs, _ = env.reset()
    done = False
    step = 0
    while not done:
        with pt.no_grad():
            logits = actor(pt.tensor(obs, dtype=pt.float32))
        probs = pt.softmax(logits, dim=-1)
        action = pt.argmax(probs).item()  # greedy
        if step == perturb_at_step:
            action = 1 - action  # flip
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1
    return step

def run_baseline(actor, env, seed=None):
    """Run one episode greedy, no perturbation. Return survival length."""
    obs, _ = env.reset()
    done = False
    step = 0
    while not done:
        with pt.no_grad():
            logits = actor(pt.tensor(obs, dtype=pt.float32))
        action = pt.argmax(pt.softmax(logits, dim=-1)).item()
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1
    return step

# ── Main ──────────────────────────────────────────────────────────────────────
print("Training policy to convergence (gamma=0.99)...")
actor, train_lengths = train_policy(n_episodes=1500)
print(f"Training done. Mean last 100 eps: {np.mean(train_lengths[-100:]):.1f}")

env = gym.make("CartPole-v1")

# Measure baseline performance
N_BASELINE = 200
baseline_lengths = [run_baseline(actor, env) for _ in range(N_BASELINE)]
baseline_mean = np.mean(baseline_lengths)
print(f"Baseline mean survival: {baseline_mean:.1f} steps")

# Perturbation sweep
# Perturb at step t=10, measure how much shorter the episode is
# We perturb at a fixed early step and measure survival delta
N_TRIALS  = 200
perturb_steps = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]

survival_means = []
survival_stds  = []

print("\nRunning perturbation analysis...")
for t in perturb_steps:
    lengths = []
    for _ in range(N_TRIALS):
        # only perturb if episode lasts at least t steps (run normally to t, then perturb)
        length = run_perturbed(actor, env, perturb_at_step=t)
        lengths.append(length)
    mean_len = np.mean(lengths)
    delta = mean_len - baseline_mean
    survival_means.append(mean_len)
    survival_stds.append(np.std(lengths))
    print(f"  perturb at t={t:3d}: mean_survival={mean_len:.1f}  delta={delta:+.1f}")

env.close()

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: survival vs perturbation step
axes[0].plot(perturb_steps, survival_means, 'o-', color='steelblue', label='perturbed')
axes[0].axhline(baseline_mean, color='gray', linestyle='--', label=f'baseline ({baseline_mean:.0f})')
axes[0].fill_between(perturb_steps,
                      np.array(survival_means) - np.array(survival_stds),
                      np.array(survival_means) + np.array(survival_stds),
                      alpha=0.2, color='steelblue')
axes[0].set_xlabel("Step at which action was perturbed")
axes[0].set_ylabel("Mean survival length")
axes[0].set_title("Effect of Single Action Perturbation on Survival")
axes[0].legend()

# Plot 2: delta (harm of perturbation) vs perturbation step
deltas = [m - baseline_mean for m in survival_means]
axes[1].plot(perturb_steps, deltas, 'o-', color='darkorange')
axes[1].axhline(0, color='gray', linestyle='--')
axes[1].set_xlabel("Step at which action was perturbed")
axes[1].set_ylabel("Delta survival (perturbed - baseline)")
axes[1].set_title("Causal Influence Decay: Delta Survival vs Perturbation Step")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "exp2_causal_horizon.png"), dpi=200)
plt.close()

# training curve
plt.figure(figsize=(10, 4))
window = 20
smoothed = np.convolve(train_lengths, np.ones(window)/window, mode='valid')
plt.plot(range(window-1, len(train_lengths)), smoothed)
plt.xlabel("Episode"); plt.ylabel("Episode length")
plt.title("Policy Training Curve (gamma=0.99)")
plt.savefig(os.path.join(RESULTS_DIR, "exp2_training_curve.png"), dpi=200)
plt.close()

print("\nDone. Plots saved to results/")
print(f"\nKey result: baseline survival = {baseline_mean:.1f} steps")
print("Perturbation effects:")
for t, m, s in zip(perturb_steps, survival_means, survival_stds):
    print(f"  t={t:3d}: mean={m:.1f} ± {s:.1f}  delta={m-baseline_mean:+.1f}")
