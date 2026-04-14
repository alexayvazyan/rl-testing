"""
Experiments 5 & 6 — Advantage Magnitude and Critic Accuracy During Training

Exp 5: Track per-update diagnostics during PPO training:
  - mean(|advantage|)
  - critic_loss
  - actor_loss
  - policy_entropy

Exp 6: Track critic R² on held-out data over training.

Hypothesis: low gamma causes critic to converge fast, collapsing advantages,
killing the actor's gradient signal early — before the policy is good.
"""

import numpy as np
import torch as pt
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LR, BATCH_SIZE, BATCH_EPOCHS, CLIP = 0.001, 50, 10, 0.2
N_EPISODES = 800
GAMMAS = [0.5, 0.9, 0.99]

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

def collect_holdout(env, n_steps=500):
    """Collect a fixed holdout set of (obs) for R² tracking."""
    obs, _ = env.reset()
    holdout_obs = [obs]
    done = False
    steps = 0
    while steps < n_steps:
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        holdout_obs.append(obs)
        done = term or trunc
        if done:
            obs, _ = env.reset()
        steps += 1
    return pt.stack([pt.tensor(o, dtype=pt.float32) for o in holdout_obs])

def compute_r2(critic, holdout_obs, holdout_rtgs):
    """Compute R² of critic predictions vs RTGs on holdout set."""
    with pt.no_grad():
        preds = critic(holdout_obs).squeeze().numpy()
    targets = holdout_rtgs.numpy()
    ss_res = np.sum((targets - preds)**2)
    ss_tot = np.sum((targets - np.mean(targets))**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

def train_with_diagnostics(gamma, seed=42):
    pt.manual_seed(seed); np.random.seed(seed)
    env = gym.make("CartPole-v1")

    actor  = NN(4, 2)
    critic = NN(4, 1)
    buf    = Buffer(4000)
    ao     = pt.optim.Adam(actor.parameters(),  lr=LR)
    co     = pt.optim.Adam(critic.parameters(), lr=LR)
    crit   = nn.MSELoss()

    # Holdout set for R² (fixed random observations)
    holdout_obs = collect_holdout(env, n_steps=500)

    # Diagnostics storage
    diag = {
        'update_step': [],
        'episode':     [],
        'mean_abs_adv': [],
        'critic_loss':  [],
        'actor_loss':   [],
        'entropy':      [],
        'r2':           [],
        'run_length':   [],
    }

    update_step = 0
    run_lengths = []

    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done:
            with pt.no_grad():
                logits = actor(pt.tensor(obs, dtype=pt.float32))
            probs  = pt.softmax(logits, dim=-1)
            action = pt.multinomial(probs, 1).item()
            lp     = pt.log(probs)[action].detach()
            new_obs, reward, term, trunc, _ = env.step(action)
            steps += 1
            done = term or trunc
            buf.push((obs, action, reward, done, lp))
            obs = new_obs

        run_lengths.append(steps)

        if len(buf) > 800 and ep % 5 == 0:  # train every 5 episodes
            # compute RTGs
            rtg = np.zeros(len(buf))
            for i in range(len(buf)-1, -1, -1):
                rtg[i] = buf[i][2] if buf[i][3] else gamma * rtg[i+1] + buf[i][2]
            buf.merge_rtg(rtg)

            # compute holdout RTGs for R² (approximate: use current buffer RTGs for obs in holdout)
            # simpler: compute R² on buffer samples
            buf_states = pt.stack([pt.tensor(buf[i][0], dtype=pt.float32) for i in range(len(buf))])
            buf_rtgs   = pt.tensor([buf[i][-1] for i in range(len(buf))], dtype=pt.float32)

            ep_critic_losses, ep_actor_losses, ep_advs, ep_entropies = [], [], [], []

            for _ in range(BATCH_EPOCHS):
                samples = buf.sample(BATCH_SIZE)
                s, a, r, d, lp_b, rtg_b = zip(*samples)
                s     = pt.stack([pt.tensor(x, dtype=pt.float32) for x in s])
                rtg_b = pt.tensor(rtg_b, dtype=pt.float32)
                lp_b  = pt.stack(list(lp_b))

                # critic update
                v = critic(s).squeeze()
                c_loss = crit(v, rtg_b)
                co.zero_grad(); c_loss.backward(); co.step()

                # advantages
                adv = rtg_b - v.detach()
                adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

                # actor update
                new_lp = pt.log(pt.softmax(actor(s), dim=-1))[range(BATCH_SIZE), a]
                ratio  = pt.exp(new_lp - lp_b)
                a_loss = -pt.mean(pt.min(ratio*adv_norm, pt.clamp(ratio, 1-CLIP, 1+CLIP)*adv_norm))
                ao.zero_grad(); a_loss.backward(); ao.step()

                # entropy
                probs_b = pt.softmax(actor(s).detach(), dim=-1)
                entropy = -pt.sum(probs_b * pt.log(probs_b + 1e-8), dim=-1).mean()

                ep_critic_losses.append(c_loss.item())
                ep_actor_losses.append(a_loss.item())
                ep_advs.append(adv.abs().mean().item())
                ep_entropies.append(entropy.item())

            # R² on full buffer
            r2 = compute_r2(critic, buf_states, buf_rtgs)

            diag['update_step'].append(update_step)
            diag['episode'].append(ep)
            diag['mean_abs_adv'].append(np.mean(ep_advs))
            diag['critic_loss'].append(np.mean(ep_critic_losses))
            diag['actor_loss'].append(np.mean(ep_actor_losses))
            diag['entropy'].append(np.mean(ep_entropies))
            diag['r2'].append(r2)
            diag['run_length'].append(np.mean(run_lengths[-20:]))

            update_step += 1
            buf.reset()

        if (ep+1) % 100 == 0:
            print(f"    ep={ep+1} run_length={np.mean(run_lengths[-20:]):.1f}")

    env.close()
    return diag, run_lengths

# ── Run ───────────────────────────────────────────────────────────────────────
all_diag = {}
for gamma in GAMMAS:
    print(f"\nTraining gamma={gamma}...")
    diag, rl = train_with_diagnostics(gamma)
    all_diag[gamma] = diag
    print(f"  Final mean last 100: {np.mean(rl[-100:]):.1f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
colors = {0.5: 'red', 0.9: 'orange', 0.99: 'steelblue'}
metrics = ['mean_abs_adv', 'critic_loss', 'actor_loss', 'entropy', 'r2', 'run_length']
titles  = ['Mean |Advantage|', 'Critic Loss', 'Actor Loss',
           'Policy Entropy', 'Critic R²', 'Episode Length (20-ep avg)']
ylabels = ['|Adv|', 'MSE Loss', 'PPO Loss', 'Entropy (nats)', 'R²', 'Steps']

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
    for gamma in GAMMAS:
        d = all_diag[gamma]
        axes[i].plot(d['episode'], d[metric], color=colors[gamma],
                     label=f'γ={gamma}', alpha=0.8)
    axes[i].set_xlabel("Episode")
    axes[i].set_ylabel(ylabel)
    axes[i].set_title(title)
    axes[i].legend()

plt.suptitle("Experiments 5 & 6: Training Diagnostics by Gamma", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "exp5_6_training_diagnostics.png"), dpi=200)
plt.close()

# ── Key summary ───────────────────────────────────────────────────────────────
print("\n\n=== KEY RESULTS ===")
for gamma in GAMMAS:
    d = all_diag[gamma]
    print(f"\ngamma={gamma}:")
    print(f"  Initial mean|adv|:  {d['mean_abs_adv'][0]:.4f}")
    print(f"  Final   mean|adv|:  {d['mean_abs_adv'][-1]:.4f}")
    print(f"  Initial critic R²:  {d['r2'][0]:.4f}")
    print(f"  Final   critic R²:  {d['r2'][-1]:.4f}")
    print(f"  Initial entropy:    {d['entropy'][0]:.4f}")
    print(f"  Final   entropy:    {d['entropy'][-1]:.4f}")
    # how quickly does R² exceed 0.9?
    r2_arr = np.array(d['r2'])
    ep_arr = np.array(d['episode'])
    above  = ep_arr[r2_arr > 0.9]
    print(f"  R²>0.9 first at ep: {above[0] if len(above) else 'never'}")
