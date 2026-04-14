"""
Exp 1b - Multi-seed gamma sweep to quantify variance and confirm/deny non-monotonicity
"""
import numpy as np
import torch as pt
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LR, N_EPISODES, BATCH_SIZE, BATCH_EPOCHS, CLIP = 0.001, 600, 50, 10, 0.2
GAMMAS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
N_SEEDS = 5
SEEDS   = [0, 1, 2, 3, 4]

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

def train(gamma, seed):
    pt.manual_seed(seed); np.random.seed(seed)
    env=gym.make("CartPole-v1")
    actor=NN(4,2); critic=NN(4,1); buf=Buffer(4000)
    ao=pt.optim.Adam(actor.parameters(),lr=LR)
    co=pt.optim.Adam(critic.parameters(),lr=LR)
    crit=nn.MSELoss(); run_lengths=[]
    for ep in range(N_EPISODES):
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
                for _ in range(BATCH_EPOCHS):
                    s,a,r,d,lp_b,rtg_b=zip(*buf.sample(BATCH_SIZE))
                    s=pt.stack([pt.tensor(x,dtype=pt.float32) for x in s])
                    rtg_b=pt.tensor(rtg_b,dtype=pt.float32); lp_b=pt.stack(list(lp_b))
                    v=critic(s).squeeze()
                    co.zero_grad(); crit(v,rtg_b).backward(); co.step()
                    adv=rtg_b-v.detach(); adv=(adv-adv.mean())/(adv.std()+1e-8)
                    nlp=pt.log(pt.softmax(actor(s),dim=-1))[range(BATCH_SIZE),a]
                    ratio=pt.exp(nlp-lp_b)
                    loss=-pt.mean(pt.min(ratio*adv,pt.clamp(ratio,1-CLIP,1+CLIP)*adv))
                    ao.zero_grad(); loss.backward(); ao.step()
                buf.reset()
        run_lengths.append(steps)
    env.close()
    return np.mean(run_lengths[-100:])

print("Multi-seed gamma sweep...")
all_results = {g: [] for g in GAMMAS}
for gamma in GAMMAS:
    eff = round(1/(1-gamma),1)
    for seed in SEEDS:
        m = train(gamma, seed)
        all_results[gamma].append(m)
        print(f"  gamma={gamma} seed={seed}: {m:.1f}")
    mu, sd = np.mean(all_results[gamma]), np.std(all_results[gamma])
    print(f"  --> gamma={gamma} (eff={eff}): mean={mu:.1f} ± {sd:.1f}\n")

# Plot with error bars
gammas = GAMMAS
means  = [np.mean(all_results[g]) for g in gammas]
stds   = [np.std(all_results[g])  for g in gammas]
effs   = [1/(1-g) for g in gammas]

fig, axes = plt.subplots(1,2,figsize=(14,5))
axes[0].errorbar(gammas, means, yerr=stds, fmt='o-', color='steelblue', capsize=5)
axes[0].axhline(490,color='gray',linestyle='--',label='~max')
axes[0].set_xlabel("Gamma"); axes[0].set_ylabel("Mean last 100 ep length")
axes[0].set_title(f"Performance vs Gamma ({N_SEEDS} seeds)")
axes[0].legend()

axes[1].errorbar(effs, means, yerr=stds, fmt='o-', color='darkorange', capsize=5)
axes[1].axhline(490,color='gray',linestyle='--',label='~max')
axes[1].set_xscale('log')
axes[1].set_xlabel("Effective Horizon (log scale)"); axes[1].set_ylabel("Mean last 100 ep length")
axes[1].set_title(f"Performance vs Effective Horizon ({N_SEEDS} seeds)")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR,"exp1b_multiseed.png"),dpi=200)
plt.close()

print("\nFinal summary:")
for g in gammas:
    mu,sd=np.mean(all_results[g]),np.std(all_results[g])
    print(f"  gamma={g:5} eff={1/(1-g):7.1f}  {mu:.1f} ± {sd:.1f}")
