"""
Experiment 3b - Synthetic Lag Environment, now with Markovian state

The key finding from Exp3 was that the lag environment is a POMDP:
the state [x, velocity] doesn't include the action queue.
The agent can't know what corrections are already "in flight."
This confounds the gamma test because the environment is unsolvable
regardless of gamma — it's an observability problem, not a credit assignment problem.

Fix: include the full action queue in the state.
State = [x, velocity, a_{t-1}, ..., a_{t-L}]
Now the MDP is fully Markovian and the gamma hypothesis can be tested cleanly.

Prediction: with Markovian state, gamma_min ≈ 1 - 1/L
"""

import numpy as np
import torch as pt
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

class MarkovLagEnv(gym.Env):
    """
    Lag environment with action history in state — fully Markovian.
    State: [x, velocity, a_{t-1}, a_{t-2}, ..., a_{t-L}]
    """
    def __init__(self, lag=10, threshold=2.0, max_steps=500, drift_scale=0.05):
        super().__init__()
        self.lag, self.threshold = lag, threshold
        self.max_steps, self.drift_scale = max_steps, drift_scale
        self.action_space = spaces.Discrete(2)
        # state dim: x + velocity + lag action slots
        self.obs_dim = 2 + lag
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(self.obs_dim,), dtype=np.float32
        )

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

def train(env, n_states, gamma, n_episodes=800, seed=42):
    pt.manual_seed(seed); np.random.seed(seed)
    actor=NN(n_states,2); critic=NN(n_states,1); buf=Buffer(4000)
    ao=pt.optim.Adam(actor.parameters(),lr=0.001)
    co=pt.optim.Adam(critic.parameters(),lr=0.001)
    crit=nn.MSELoss(); run_lengths=[]
    for ep in range(n_episodes):
        obs,_=env.reset(); done=False; steps=0
        while not done:
            with pt.no_grad(): logits=actor(pt.tensor(obs,dtype=pt.float32))
            probs=pt.softmax(logits,dim=-1)
            action=pt.multinomial(probs,1).item()
            lp=pt.log(probs)[action].detach()
            new_obs,reward,terminated,truncated,_=env.step(action)
            steps+=1; done=terminated or truncated
            buf.push((obs,action,reward,done,lp)); obs=new_obs
            if len(buf)>800 and done:
                rtg=np.zeros(len(buf))
                for i in range(len(buf)-1,-1,-1):
                    rtg[i]=buf[i][2] if buf[i][3] else gamma*rtg[i+1]+buf[i][2]
                buf.merge_rtg(rtg)
                for _ in range(10):
                    s,a,r,d,lp_b,rtg_b=zip(*buf.sample(50))
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
        run_lengths.append(steps)
    return run_lengths

LAGS   = [1, 5, 10, 20]
GAMMAS = [0.7, 0.8, 0.9, 0.95, 0.99]
theoretical_gamma_min = {lag: round(1 - 1/lag, 3) for lag in LAGS}

print("Exp3b: Markovian lag environment sweep...")
results = {}
for lag in LAGS:
    results[lag] = {}
    n_states = 2 + lag
    print(f"\n  Lag={lag} (theoretical gamma_min={theoretical_gamma_min[lag]}, state_dim={n_states})")
    for gamma in GAMMAS:
        eff = round(1/(1-gamma),1)
        env = MarkovLagEnv(lag=lag)
        rl = train(env, n_states=n_states, gamma=gamma, n_episodes=800)
        mean_last100 = np.mean(rl[-100:])
        results[lag][gamma] = {"mean": mean_last100, "run_lengths": rl}
        converged = "✓" if mean_last100 > 400 else "✗"
        print(f"    gamma={gamma} (eff_horizon={eff:6.1f}): mean_last100={mean_last100:5.1f} {converged}")
        env.close()

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap
perf_matrix = np.array([[results[lag][g]["mean"] for g in GAMMAS] for lag in LAGS])
im = axes[0].imshow(perf_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=500)
axes[0].set_xticks(range(len(GAMMAS))); axes[0].set_xticklabels(GAMMAS)
axes[0].set_yticks(range(len(LAGS)));  axes[0].set_yticklabels(LAGS)
axes[0].set_xlabel("Gamma"); axes[0].set_ylabel("Lag (L)")
axes[0].set_title("Mean Last 100 Episode Length — Markovian State\n(green=converged, red=failed)")
plt.colorbar(im, ax=axes[0])
for i in range(len(LAGS)):
    for j in range(len(GAMMAS)):
        axes[0].text(j, i, f"{perf_matrix[i,j]:.0f}", ha='center', va='center', fontsize=9)

# Performance vs effective horizon, by lag
for lag in LAGS:
    effs = [1/(1-g) for g in GAMMAS]
    perfs = [results[lag][g]["mean"] for g in GAMMAS]
    axes[1].plot(effs, perfs, 'o-', label=f'L={lag}')
    axes[1].axvline(lag, linestyle=':', alpha=0.4)

axes[1].set_xlabel("Effective Horizon (1/(1-gamma))")
axes[1].set_ylabel("Mean episode length (last 100)")
axes[1].set_title("Performance vs Effective Horizon — Markovian State\n(vertical lines = theoretical gamma_min)")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "exp3b_markov_sweep.png"), dpi=200)
plt.close()

# Training curves
fig, axes = plt.subplots(len(LAGS), 1, figsize=(12, 4*len(LAGS)))
window = 20
for idx, lag in enumerate(LAGS):
    for gamma in GAMMAS:
        rl = results[lag][gamma]["run_lengths"]
        smoothed = np.convolve(rl, np.ones(window)/window, mode='valid')
        axes[idx].plot(range(window-1, len(rl)), smoothed, label=f"γ={gamma}")
    axes[idx].set_title(f"Lag={lag} (theoretical gamma_min={theoretical_gamma_min[lag]})")
    axes[idx].set_xlabel("Episode"); axes[idx].set_ylabel("Length")
    axes[idx].legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "exp3b_training_curves.png"), dpi=200)
plt.close()

print("\n\nFull results:")
print(f"{'Lag':>5} | gamma_min_theory | " + " | ".join(f"γ={g}" for g in GAMMAS))
print("-"*80)
for lag in LAGS:
    row = f"{lag:>5} | {theoretical_gamma_min[lag]:>16} | " + " | ".join(f"{results[lag][g]['mean']:7.1f}" for g in GAMMAS)
    print(row)
