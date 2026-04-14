"""
Experiment 7 — RTG Variance as a Predictor of Learnability

Key question: can we predict final training performance from a cheap
random-policy RTG variance measurement, before committing to full training?

If RTG variance (from a random rollout) correlates with final performance,
it's a universal cheap diagnostic for gamma selection.

Tests across: CartPole (dense), sparse CartPole, Markovian lag env (exp3b).
"""

import numpy as np
import torch as pt
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
GAMMAS      = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
N_ROLLOUT   = 2000  # steps for RTG variance estimation
N_EPISODES  = 600   # training episodes per run
N_SEEDS     = 3

# ── Shared model/buffer ───────────────────────────────────────────────────────
class NN(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nin,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,nout))
        self.apply(lambda m: (nn.init.kaiming_uniform_(m.weight,nonlinearity='relu'), nn.init.constant_(m.bias,0)) if isinstance(m,nn.Linear) else None)
    def forward(self, x): return self.net(x)

class Buffer:
    def __init__(self, mx): self.buf,self.idx,self.mx=[],0,mx
    def push(self,x):
        if len(self.buf)<self.mx: self.buf.append(None)
        self.buf[self.idx]=x; self.idx=(self.idx+1)%self.mx
    def sample(self,n):
        import random; return random.sample(self.buf,n)
    def merge_rtg(self,rtg):
        for i,r in enumerate(rtg): self.buf[i]=self.buf[i]+(r,)
    def reset(self): self.buf,self.idx=[],0
    def __len__(self): return len(self.buf)
    def __getitem__(self,i): return self.buf[i]

# ── Sparse CartPole wrapper ───────────────────────────────────────────────────
class SparseCartPole(gym.Wrapper):
    """Reward=1 only if you survive the full episode (500 steps), else 0."""
    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        sparse_r = 1.0 if trunc else 0.0  # reward only at truncation (max steps)
        return obs, sparse_r, term, trunc, info

# ── Markovian lag env (from exp3b) ────────────────────────────────────────────
class MarkovLagEnv(gym.Env):
    def __init__(self, lag=5, threshold=2.0, max_steps=500, drift_scale=0.05):
        super().__init__()
        self.lag, self.threshold = lag, threshold
        self.max_steps, self.drift_scale = max_steps, drift_scale
        self.action_space = spaces.Discrete(2)
        self.obs_dim = 2 + lag
        self.observation_space = spaces.Box(low=-10, high=10, shape=(self.obs_dim,), dtype=np.float32)
    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.x, self.velocity, self.t = 0.0, 0.0, 0
        self.action_queue = [0.0] * self.lag
        return self._obs(), {}
    def step(self, action):
        force = 1.0 if action == 1 else -1.0
        self.action_queue.append(force)
        lagged_force = self.action_queue.pop(0)
        drift = self.np_random.normal(0, self.drift_scale)
        self.velocity = 0.9 * self.velocity + lagged_force * 0.1 + drift
        self.x += self.velocity
        self.t += 1
        terminated = bool(abs(self.x) > self.threshold)
        truncated  = self.t >= self.max_steps
        reward = 0.0 if terminated else 1.0
        return self._obs(), reward, terminated, truncated, {}
    def _obs(self):
        return np.array([self.x, self.velocity] + list(self.action_queue), dtype=np.float32)

# ── RTG variance from random rollout ─────────────────────────────────────────
def measure_rtg_variance(env_fn, gamma, n_steps=N_ROLLOUT):
    env = env_fn()
    obs, _ = env.reset()
    trajs = []
    current = []
    for _ in range(n_steps):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        current.append(r)
        if term or trunc:
            trajs.append(current)
            current = []
            obs, _ = env.reset()
    if current:
        trajs.append(current)
    env.close()

    all_rtgs = []
    for traj in trajs:
        rtg = 0.0
        for r in reversed(traj):
            rtg = r + gamma * rtg
            all_rtgs.append(rtg)
    return np.var(all_rtgs), np.mean(all_rtgs)

# ── PPO train ─────────────────────────────────────────────────────────────────
def train(env_fn, n_states, gamma, seed=0):
    pt.manual_seed(seed); np.random.seed(seed)
    env    = env_fn()
    actor  = NN(n_states, 2)
    critic = NN(n_states, 1)
    buf    = Buffer(4000)
    ao     = pt.optim.Adam(actor.parameters(),  lr=0.001)
    co     = pt.optim.Adam(critic.parameters(), lr=0.001)
    crit   = nn.MSELoss()
    run_lengths = []

    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        done = False; steps = 0
        while not done:
            with pt.no_grad(): logits = actor(pt.tensor(obs, dtype=pt.float32))
            probs  = pt.softmax(logits, dim=-1)
            action = pt.multinomial(probs, 1).item()
            lp     = pt.log(probs)[action].detach()
            new_obs, reward, term, trunc, _ = env.step(action)
            steps += 1; done = term or trunc
            buf.push((obs, action, reward, done, lp)); obs = new_obs

        run_lengths.append(steps)

        if len(buf) > 800 and done:
            rtg_arr = np.zeros(len(buf))
            for i in range(len(buf)-1,-1,-1):
                rtg_arr[i] = buf[i][2] if buf[i][3] else gamma*rtg_arr[i+1]+buf[i][2]
            buf.merge_rtg(rtg_arr)
            for _ in range(10):
                s,a,r,d,lp_b,rtg_b = zip(*buf.sample(50))
                s=pt.stack([pt.tensor(x,dtype=pt.float32) for x in s])
                rtg_b=pt.tensor(rtg_b,dtype=pt.float32); lp_b=pt.stack(list(lp_b))
                v=critic(s).squeeze()
                co.zero_grad(); crit(v,rtg_b).backward(); co.step()
                adv=rtg_b-v.detach(); adv=(adv-adv.mean())/(adv.std()+1e-8)
                nlp=pt.log(pt.softmax(actor(s),dim=-1))[range(50),a]
                ratio=pt.exp(nlp-lp_b)
                loss=-pt.mean(pt.min(ratio*adv,pt.clamp(ratio,0.8,1.2)*adv))
                ao.zero_grad(); loss.backward(); ao.step()
            buf.reset()

    env.close()
    return np.mean(run_lengths[-100:])

# ── Environments to test ──────────────────────────────────────────────────────
envs = {
    'CartPole (dense)':  (lambda: gym.make("CartPole-v1"), 4),
    'CartPole (sparse)': (lambda: SparseCartPole(gym.make("CartPole-v1", max_episode_steps=500)), 4),
    'LagEnv L=5 (Markov)': (lambda: MarkovLagEnv(lag=5), 7),
}

# ── Run everything ────────────────────────────────────────────────────────────
results = {}

for env_name, (env_fn, n_states) in envs.items():
    print(f"\n{'='*60}")
    print(f"Environment: {env_name}")
    results[env_name] = {'rtg_var': [], 'rtg_mean': [], 'perf_mean': [], 'perf_std': []}

    for gamma in GAMMAS:
        # measure RTG variance (cheap, random policy)
        var, mean_rtg = measure_rtg_variance(env_fn, gamma)
        results[env_name]['rtg_var'].append(var)
        results[env_name]['rtg_mean'].append(mean_rtg)
        print(f"  gamma={gamma}: RTG_var={var:.4f}, RTG_mean={mean_rtg:.4f}", end="")

        # train (3 seeds)
        perfs = [train(env_fn, n_states, gamma, seed=s) for s in range(N_SEEDS)]
        pm, ps = np.mean(perfs), np.std(perfs)
        results[env_name]['perf_mean'].append(pm)
        results[env_name]['perf_std'].append(ps)
        print(f"  -> perf={pm:.1f} ± {ps:.1f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(envs), 2, figsize=(16, 5*len(envs)))

for row, (env_name, data) in enumerate(results.items()):
    vars_  = data['rtg_var']
    perfs  = data['perf_mean']
    stds   = data['perf_std']

    # left: perf vs gamma with error bars
    axes[row,0].errorbar(GAMMAS, perfs, yerr=stds, fmt='o-', capsize=5, color='steelblue')
    axes[row,0].set_xlabel("Gamma")
    axes[row,0].set_ylabel("Mean last 100 ep length")
    axes[row,0].set_title(f"{env_name}\nPerformance vs Gamma")

    # right: perf vs RTG variance (the predictor test)
    axes[row,1].errorbar(vars_, perfs, yerr=stds, fmt='o', capsize=5, color='darkorange')
    for i, g in enumerate(GAMMAS):
        axes[row,1].annotate(f'γ={g}', (vars_[i], perfs[i]),
                              textcoords="offset points", xytext=(5,5), fontsize=8)
    # fit a line
    if len(set(vars_)) > 1:
        coeffs = np.polyfit(np.log(np.array(vars_)+1e-8), perfs, 1)
        x_fit  = np.linspace(min(vars_), max(vars_), 100)
        y_fit  = coeffs[0] * np.log(x_fit+1e-8) + coeffs[1]
        axes[row,1].plot(x_fit, y_fit, '--', color='gray', alpha=0.7, label='log-linear fit')
    corr = np.corrcoef(np.log(np.array(vars_)+1e-8), perfs)[0,1]
    axes[row,1].set_xlabel("RTG Variance (random policy)")
    axes[row,1].set_ylabel("Mean last 100 ep length")
    axes[row,1].set_title(f"{env_name}\nPerf vs RTG Variance  (r={corr:.3f})")
    axes[row,1].legend()

plt.suptitle("Experiment 7: RTG Variance as Learnability Predictor", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "exp7_rtg_variance_predictor.png"), dpi=200)
plt.close()

print("\n\n=== CORRELATION SUMMARY ===")
for env_name, data in results.items():
    corr = np.corrcoef(np.log(np.array(data['rtg_var'])+1e-8), data['perf_mean'])[0,1]
    print(f"  {env_name}: r(log RTG_var, perf) = {corr:.3f}")
print("\nIf |r| > 0.9: RTG variance is a useful predictor of learnability.")
