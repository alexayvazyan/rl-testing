import numpy as np
import torch as pt
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import json

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ── Hyperparams (fixed across all runs) ──────────────────────────────────────
LR          = 0.001
N_EPISODES  = 600
BATCH_SIZE  = 50
BATCH_EPOCHS= 10
CLIP        = 0.2
BUFFER_SIZE = 4000
MIN_BUFFER  = 800
GAMMAS      = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]

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

# ── Buffer ────────────────────────────────────────────────────────────────────
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

# ── PPO train ─────────────────────────────────────────────────────────────────
def train(gamma, seed=42):
    pt.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("CartPole-v1")
    actor  = NN(4, 2)
    critic = NN(4, 1)
    buf    = Buffer(BUFFER_SIZE)
    actor_opt  = pt.optim.Adam(actor.parameters(),  lr=LR)
    critic_opt = pt.optim.Adam(critic.parameters(), lr=LR)
    criterion  = nn.MSELoss()

    run_lengths = []

    for ep in range(N_EPISODES):
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

            if len(buf) > MIN_BUFFER and done:
                # compute RTG
                rtg = np.zeros(len(buf))
                for i in range(len(buf) - 1, -1, -1):
                    if buf[i][3]:  # done
                        rtg[i] = buf[i][2]
                    else:
                        rtg[i] = gamma * rtg[i + 1] + buf[i][2]
                buf.merge_rtg(rtg)

                for _ in range(BATCH_EPOCHS):
                    samples = buf.sample(BATCH_SIZE)
                    s, a, r, d, lp, rtg_b = zip(*samples)
                    s   = pt.stack([pt.tensor(x, dtype=pt.float32) for x in s])
                    rtg_b = pt.tensor(rtg_b, dtype=pt.float32)
                    lp  = pt.stack(list(lp))

                    v = critic(s).squeeze()
                    critic_loss = criterion(v, rtg_b)
                    critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

                    adv = rtg_b - v.detach()
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # normalise advantages

                    new_logits  = actor(s)
                    new_probs   = pt.softmax(new_logits, dim=-1)
                    new_logprobs = pt.log(new_probs)[range(BATCH_SIZE), a]
                    ratio = pt.exp(new_logprobs - lp)
                    actor_loss = -pt.mean(pt.min(
                        ratio * adv,
                        pt.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv
                    ))
                    actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

                buf.reset()

        run_lengths.append(steps)

    env.close()
    return run_lengths

# ── Run sweep ─────────────────────────────────────────────────────────────────
print("Running gamma sweep...")
results = {}
for gamma in GAMMAS:
    eff_horizon = round(1 / (1 - gamma), 1)
    print(f"  gamma={gamma} (eff_horizon={eff_horizon})...", end="", flush=True)
    run_lengths = train(gamma)
    mean_last100 = np.mean(run_lengths[-100:])
    results[gamma] = {"run_lengths": run_lengths, "mean_last100": mean_last100}
    print(f" mean_last100={mean_last100:.1f}")

# save raw results
with open(os.path.join(RESULTS_DIR, "exp1_results.json"), "w") as f:
    json.dump({str(k): v["mean_last100"] for k, v in results.items()}, f, indent=2)

# ── Plot 1: performance vs gamma ──────────────────────────────────────────────
gammas     = list(results.keys())
means      = [results[g]["mean_last100"] for g in gammas]
eff_horizons = [1 / (1 - g) for g in gammas]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(gammas, means, 'o-', color='steelblue')
axes[0].axhline(490, color='gray', linestyle='--', label='~max performance')
axes[0].set_xlabel("Gamma")
axes[0].set_ylabel("Mean episode length (last 100 eps)")
axes[0].set_title("Performance vs Gamma")
axes[0].legend()

axes[1].plot(eff_horizons, means, 'o-', color='darkorange')
axes[1].axhline(490, color='gray', linestyle='--', label='~max performance')
axes[1].set_xlabel("Effective Horizon (1 / (1-gamma))")
axes[1].set_ylabel("Mean episode length (last 100 eps)")
axes[1].set_title("Performance vs Effective Horizon")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "exp1_gamma_sweep.png"), dpi=200)
plt.close()

# ── Plot 2: training curves for all gammas ────────────────────────────────────
plt.figure(figsize=(12, 6))
window = 20
for gamma in GAMMAS:
    rl = results[gamma]["run_lengths"]
    smoothed = np.convolve(rl, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(rl)), smoothed, label=f"γ={gamma}")
plt.xlabel("Episode")
plt.ylabel("Episode length (smoothed)")
plt.title("Training Curves by Gamma")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "exp1_training_curves.png"), dpi=200)
plt.close()

print("\nDone. Results saved to results/")
print("\nSummary:")
for g in GAMMAS:
    print(f"  gamma={g:5}  eff_horizon={1/(1-g):7.1f}  mean_last100={results[g]['mean_last100']:.1f}")
