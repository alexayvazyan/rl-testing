"""
Experiment 3 - Synthetic Environment with Tunable Causal Horizon

Environment design:
- 1D position x, starts at 0
- Each step: small random drift applied, plus the lagged effect of agent's action
- Agent action: push left (-1) or push right (+1)
- Effect of action at time t arrives at time t+L (pure lag)
- Failure: |x| > threshold
- Reward: +1 per step survived

This cleanly decouples the causal horizon (L) from other environment properties,
letting us test whether gamma_min ≈ 1 - 1/L.
"""

import numpy as np
import torch as pt
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ── Custom Lag Environment ────────────────────────────────────────────────────
class LagEnv(gym.Env):
    """
    1D environment where agent's actions affect position after a lag of L steps.
    State: [position, velocity_estimate] - 2D to give agent some info
    Action: 0 (push left) or 1 (push right)
    """
    def __init__(self, lag=10, threshold=2.0, max_steps=500, drift_scale=0.05):
        super().__init__()
        self.lag         = lag
        self.threshold   = threshold
        self.max_steps   = max_steps
        self.drift_scale = drift_scale
        self.action_space      = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-threshold*2, high=threshold*2, shape=(2,), dtype=np.float32
        )

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.x            = 0.0
        self.velocity     = 0.0
        self.t            = 0
        self.action_queue = [0.0] * self.lag  # neutral history
        return self._obs(), {}

    def step(self, action):
        force = 1.0 if action == 1 else -1.0
        self.action_queue.append(force)
        lagged_force = self.action_queue.pop(0)

        drift = self.np_random.normal(0, self.drift_scale)
        self.velocity = 0.9 * self.velocity + lagged_force * 0.1 + drift
        self.x       += self.velocity
        self.t       += 1

        terminated = bool(abs(self.x) > self.threshold)
        truncated  = self.t >= self.max_steps
        reward     = 0.0 if terminated else 1.0

        return self._obs(), reward, terminated, truncated, {}

    def _obs(self):
        return np.array([self.x, self.velocity], dtype=np.float32)

# ── Model / Buffer (same as before) ──────────────────────────────────────────
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

# ── PPO train on any env ──────────────────────────────────────────────────────
def train(env, n_states, gamma, n_episodes=800, seed=42):
    pt.manual_seed(seed)
    np.random.seed(seed)

    actor  = NN(n_states, 2)
    critic = NN(n_states, 1)
    buf    = Buffer(4000)
    actor_opt  = pt.optim.Adam(actor.parameters(),  lr=0.001)
    critic_opt = pt.optim.Adam(critic.parameters(), lr=0.001)
    criterion  = nn.MSELoss()
    run_lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
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
                for i in range(len(buf)-1, -1, -1):
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
                    loss = -pt.mean(pt.min(ratio*adv, pt.clamp(ratio, 0.8, 1.2)*adv))
                    actor_opt.zero_grad(); loss.backward(); actor_opt.step()
                buf.reset()

        run_lengths.append(steps)

    return run_lengths

# ── Sweep ─────────────────────────────────────────────────────────────────────
LAGS   = [1, 5, 10, 20, 50]
GAMMAS = [0.7, 0.8, 0.9, 0.95, 0.99]

# Theoretical prediction: gamma_min = 1 - 1/lag (effective_horizon = lag)
theoretical_gamma_min = {lag: 1 - 1/lag for lag in LAGS}

print("Running synthetic environment sweep (lag x gamma)...")
results = {}  # results[lag][gamma] = mean_last100

for lag in LAGS:
    results[lag] = {}
    eff_horizon_needed = lag
    print(f"\n  Lag={lag} (theoretical gamma_min={theoretical_gamma_min[lag]:.3f}, eff_horizon needed={lag})")
    for gamma in GAMMAS:
        eff_horizon = round(1/(1-gamma), 1)
        env = LagEnv(lag=lag)
        run_lengths = train(env, n_states=2, gamma=gamma, n_episodes=600)
        mean_last100 = np.mean(run_lengths[-100:])
        results[lag][gamma] = mean_last100
        converged = "✓" if mean_last100 > 400 else "✗"
        print(f"    gamma={gamma} (eff_horizon={eff_horizon:6.1f}): mean_last100={mean_last100:5.1f} {converged}")
        env.close()

# ── Plot: heatmap of performance ──────────────────────────────────────────────
perf_matrix = np.array([[results[lag][g] for g in GAMMAS] for lag in LAGS])

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

im = axes[0].imshow(perf_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=500)
axes[0].set_xticks(range(len(GAMMAS))); axes[0].set_xticklabels(GAMMAS)
axes[0].set_yticks(range(len(LAGS)));  axes[0].set_yticklabels(LAGS)
axes[0].set_xlabel("Gamma"); axes[0].set_ylabel("Lag (L)")
axes[0].set_title("Mean Last 100 Episode Length\n(green=converged, red=failed)")
plt.colorbar(im, ax=axes[0])
for i in range(len(LAGS)):
    for j in range(len(GAMMAS)):
        axes[0].text(j, i, f"{perf_matrix[i,j]:.0f}", ha='center', va='center', fontsize=8)

# Plot 2: for each lag, performance vs effective horizon, with theoretical cutoff
for lag in LAGS:
    eff_horizons = [1/(1-g) for g in GAMMAS]
    perfs = [results[lag][g] for g in GAMMAS]
    axes[1].plot(eff_horizons, perfs, 'o-', label=f'L={lag}')
    axes[1].axvline(lag, color='gray', linestyle=':', alpha=0.5)

axes[1].set_xlabel("Effective Horizon (1/(1-gamma))")
axes[1].set_ylabel("Mean episode length (last 100)")
axes[1].set_title("Performance vs Effective Horizon by Lag\n(vertical lines = theoretical gamma_min)")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "exp3_synthetic_sweep.png"), dpi=200)
plt.close()

print("\n\nFull results table:")
print(f"{'Lag':>6} | " + " | ".join(f"γ={g}" for g in GAMMAS))
print("-" * 60)
for lag in LAGS:
    row = f"{lag:>6} | " + " | ".join(f"{results[lag][g]:7.1f}" for g in GAMMAS)
    print(row)

print(f"\nTheoretical gamma_min predictions:")
for lag in LAGS:
    print(f"  L={lag:3d}: gamma_min={theoretical_gamma_min[lag]:.3f}  (eff_horizon={lag})")
