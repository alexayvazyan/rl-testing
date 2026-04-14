# Gamma and Credit Assignment — Part 2: Looking Inside Training

## The Gap in Part 1

Part 1 established that higher gamma improves CartPole performance, and the relationship is smooth rather than a sharp phase transition. But this was purely an outcome measurement. We never looked inside the training process. We didn't know *why* gamma=0.99 works and gamma=0.9 struggles.

The deeper question: **what does gamma actually change about the gradient signal during training?**

---

## The Mechanism — Answer First

**Gamma determines the variance of the RTG distribution. RTG variance determines the size of the advantage signal. The advantage signal is the only gradient the actor receives. Small advantages = small actor updates = slow or no learning.**

This is not the "critic saturation" story we initially hypothesised. The critic doesn't learn well and then destroy the signal. The signal is intrinsically small at low gamma regardless of what the critic does. The RTG values are simply too similar for any advantage to be meaningful.

---

## Experiment 4 — RTG Distribution by Gamma

**Script**: `exp4_rtg_distribution.py` | **Method**: 50 random policy rollouts, compute RTG distribution at each gamma.

| Gamma | Eff. Horizon | Mean RTG | Std RTG | RTG Variance | Rel. Spread (std/mean) |
|-------|-------------|---------|---------|-------------|----------------------|
| 0.5   | 2.0         | 1.91    | 0.23    | 0.05        | 0.122                |
| 0.7   | 3.3         | 2.97    | 0.61    | 0.37        | 0.205                |
| 0.8   | 5.0         | 4.08    | 1.12    | 1.25        | 0.274                |
| 0.9   | 10.0        | 6.39    | 2.54    | 6.43        | 0.397                |
| 0.95  | 20.0        | 8.73    | 4.50    | 20.24       | 0.516                |
| 0.99  | 100.0       | 12.04   | 8.24    | 67.93       | 0.684                |
| 0.999 | 1000.0      | 13.11   | 9.68    | 93.72       | 0.738                |

**RTG variance at gamma=0.99 is 1,260× larger than at gamma=0.5.**

At gamma=0.5, RTGs range from 1.0 to 2.0 — every step looks nearly identical. At gamma=0.99, RTGs range from 1.0 to 40.7 — massive differentiation between early and late steps.

**Analytical interpretation**: for a trajectory of length T with reward=1 every step, RTG at step t = `(1 - gamma^(T-t)) / (1-gamma)`. RTG variance scales approximately as `1/(1-gamma)^2 = effective_horizon^2`. This is a structural property of the discount function, independent of the policy or network.

**Implication**: at low gamma, the signal space is intrinsically compressed. Even a perfect critic predicting the mean RTG would produce near-zero advantages. The problem is not the critic — it's the geometry of the reward signal.

---

## Experiments 5 & 6 — Advantage Magnitude and Critic Accuracy During Training

**Script**: `exp5_6_advantage_critic.py` | **Gammas**: 0.5, 0.9, 0.99 | **800 episodes each**

### Performance

| Gamma | Final mean last 100 eps |
|-------|------------------------|
| 0.5   | 163.6                  |
| 0.9   | 302.7                  |
| 0.99  | 469.0                  |

### Key diagnostics

| Gamma | Initial \|Adv\| | Final \|Adv\| | Collapse? | Initial R² | Final R² | R²>0.9 at ep |
|-------|---------------|-------------|-----------|-----------|---------|------------|
| 0.5   | 1.28          | 0.079       | 94% drop  | -38.5     | -1.31   | never      |
| 0.9   | 6.74          | 0.560       | 92% drop  | -6.71     | 0.47    | never      |
| 0.99  | 18.23         | 20.35       | **none**  | -2.17     | 0.13    | ep 465     |

### The core finding

**At gamma=0.99, the advantage signal never collapses — it stays at ~20 throughout all 800 episodes.** At gamma=0.5 and 0.9, advantages collapse by 92-94%.

This is not because the critic becomes accurate and destroys the advantages (we hypothesised this). The critic R² for gamma=0.5 is **-1.3** at the end — it never learned the state-value function at all. The advantages collapsed despite the critic failing to learn, because the RTG values are so similar that even a broken critic produces near-zero advantages.

For gamma=0.99, the critic R² eventually hits 0.9 at episode 465 (late in training, after the policy has mostly converged) — but advantages don't collapse because the policy's improvement has simultaneously expanded the RTG range (longer episodes = larger RTG differences).

### Revised mechanism — a positive feedback loop

**High gamma**:
- Large RTG variance → large advantages → strong actor gradient → policy improves
- Better policy → longer episodes → even larger RTG range → advantages stay large
- Positive feedback: learning begets more learning

**Low gamma**:
- Small RTG variance → tiny advantages → weak actor gradient → slow policy improvement
- Policy improves slightly but RTG variance is fundamentally bounded by 1/(1-gamma)²
- No feedback loop: the ceiling is the effective horizon

### The critic saturation hypothesis was wrong

The initial hypothesis was: "low gamma makes the critic converge fast, collapsing advantages." This is incorrect. For low gamma, the critic **never converges** (R²=-1.3 for gamma=0.5) — but advantages still collapse because the signal is intrinsically flat. The critic is irrelevant to the collapse. The RTG geometry is everything.

---

## Experiment 7 — RTG Variance as a Learnability Predictor

**Script**: `exp7_rtg_variance_predictor.py` | **Question**: can a cheap random-policy RTG variance measurement predict final training performance?

### Results across 3 environments

**CartPole (dense reward)**

| Gamma | RTG Var | Perf (mean ± std) |
|-------|---------|------------------|
| 0.5   | 0.05    | 30.1 ± 13.0      |
| 0.7   | 0.35    | 209.1 ± 24.9     |
| 0.8   | 1.19    | 258.6 ± 36.6     |
| 0.9   | 6.63    | 259.3 ± 49.5     |
| 0.95  | 21.3    | 290.8 ± 42.9     |
| 0.99  | 75.4    | 360.6 ± 60.6     |
| 0.999 | 176.8   | 400.5 ± 63.1     |

**r(log RTG_var, performance) = 0.945** — excellent predictor.

**CartPole (sparse reward — +1 only if survived 500 steps)**

RTG variance = 0.000 for ALL gamma values. A random policy never survives 500 steps, so reward signal is always 0. RTG variance is useless as a predictor here (r = 0.000). Performance still improves with gamma (16.7 → 39.6) but the mechanism must differ — even a tiny rare signal propagates differently at high gamma.

**Lag Environment L=5 (Markovian)**

| Gamma | RTG Var | Perf |
|-------|---------|------|
| 0.5   | 0.15    | 15.9 |
| 0.99  | 137.5   | 65.6 |

**r(log RTG_var, performance) = 0.910** — also strong. But absolute performance is capped (~65) regardless of gamma, showing the environment difficulty is the binding constraint, not the signal.

### Conclusion on RTG variance as predictor

For **dense reward environments**: RTG variance from a random rollout predicts training performance with r≈0.94. This is a cheap, training-free diagnostic. Before running a 600-episode training run, compute RTG variance across gamma candidates in seconds. Pick the one with the highest variance.

For **sparse reward environments**: RTG variance from random rollouts is identically zero and useless — the random policy never reaches the reward. Need a different diagnostic (e.g. estimate from partially-trained policy, or use domain knowledge about reward location).

---

## Unified Story

**What gamma actually does**:

1. Gamma determines how far into the future rewards are aggregated into the RTG
2. Larger gamma → larger RTG range across an episode → larger variance in the advantage signal
3. Advantages are the only gradient the actor receives — small advantages = small updates
4. For dense rewards, this creates a positive feedback loop: high gamma enables fast learning which enables even higher gamma effectiveness
5. The mechanism is purely about **information geometry of the RTG distribution**, not about gradient propagation distance or credit assignment in the temporal sense

**What gamma does NOT do** (at least in CartPole):
- It does not determine whether gradients can "reach far enough back" in time — the whole trajectory contributes to each RTG
- The "causal horizon" framing from Part 1 is real but incomplete — the mechanism isn't signal absence, it's signal compression

---

## Open Questions

1. **Sparse rewards**: the predictor fails and mechanism must differ. How does gamma help in sparse reward settings? Is it that high gamma amplifies the rare non-zero reward signal across the full episode, vs low gamma concentrating it near the reward?

2. **The positive feedback loop**: can we directly measure the RTG variance growth over training at gamma=0.99? If RTG variance expands as the policy improves (longer episodes), this positive feedback should be directly visible.

3. **Does normalising advantages remove the effect?** Advantage normalisation (`A = (A - mean) / std`) artificially restores variance regardless of gamma. Does a model trained with normalised advantages show the same gamma sensitivity? If not, RTG variance is confirmed as the mechanism.

4. **Cross-environment generality**: the r=0.91-0.94 correlation holds for dense-reward CartPole and LagEnv. Does it hold for more complex environments (LunarLander, MuJoCo)?

---

## Implementation Notes

- Exp 4: `exp4_rtg_distribution.py` — random rollout only, very fast
- Exp 5/6: `exp5_6_advantage_critic.py` — instrumented PPO training, ~800 eps × 3 gammas
- Exp 7: `exp7_rtg_variance_predictor.py` — 3 environments × 7 gammas × 3 seeds
- All plots in `results/`
- Key finding: **RTG variance is the right quantity to measure, not effective horizon**
