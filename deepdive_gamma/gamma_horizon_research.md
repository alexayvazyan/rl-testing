# Gamma, Effective Horizon, and Causal Range of Influence

## The Core Idea

In CartPole, the agent receives +1 reward every timestep it survives. Failure is caused by accumulated angular momentum over many steps — no single action causes termination. The causal chain between an action and terminal failure spans many timesteps. Call this the **causal horizon** of the environment.

Gamma determines the **effective discount horizon** — how far back the return signal meaningfully reaches:

```
effective_horizon ≈ 1 / (1 - gamma)
```

- gamma=0.9  → horizon ~10 steps
- gamma=0.99 → horizon ~100 steps

**Initial hypothesis**: if the environment's causal horizon exceeds the effective discount horizon, the agent cannot learn to avoid failure — the gradient signal doesn't reach far enough back to assign credit to the responsible action.

---

## The Right Analytical Process (How to Know Before Running Experiments)

Rather than tuning gamma empirically, the right process is:

**Step 1 — characterize the reward structure**
CartPole gives +1 every step. Termination is caused by the pole angle exceeding a threshold — a physical process with momentum and inertia, not a single action.

**Step 2 — estimate the causal horizon from first principles**
CartPole dynamics compound over time. Physics-based reasoning: a small angular deviation builds over many steps before becoming unrecoverable. Causal horizon is on the order of 20-50 steps.

**Step 3 — check consistency with gamma**
With gamma=0.9, effective horizon = 10 steps — falls short of the estimated causal horizon. With gamma=0.99, effective horizon ≈ 100 steps — comfortably covers it.

**The principle**: characterize the environment's temporal structure first, then choose hyperparameters to match. This is analytical reasoning, not grid search.

---

## Experiments and Results

### Experiment 1 — Gamma Sweep on CartPole (Single Seed)
**Script**: `exp1_gamma_sweep.py`

| Gamma | Eff. Horizon | Mean Last 100 Eps |
|-------|-------------|-------------------|
| 0.5   | 2.0         | 72.0              |
| 0.7   | 3.3         | 122.1             |
| 0.8   | 5.0         | 319.6             |
| 0.9   | 10.0        | 253.5             |
| 0.95  | 20.0        | 266.5             |
| 0.99  | 100.0       | 482.7             |
| 0.999 | 1000.0      | 437.2             |

**Apparent finding**: non-monotonic dip at gamma=0.9 (lower than 0.8). Suspected variance from single seed.

---

### Experiment 1b — Multi-Seed Gamma Sweep (5 seeds)
**Script**: `exp1b_multiseed.py`

| Gamma | Eff. Horizon | Mean ± Std         |
|-------|-------------|--------------------|
| 0.5   | 2.0         | 31.8 ± 10.0        |
| 0.7   | 3.3         | 120.9 ± 43.9       |
| 0.8   | 5.0         | 226.9 ± 61.1       |
| 0.9   | 10.0        | 248.9 ± 77.1       |
| 0.95  | 20.0        | 311.2 ± 52.4       |
| 0.99  | 100.0       | 343.6 ± 59.0       |
| 0.999 | 1000.0      | 347.5 ± 53.6       |

**Key findings**:

1. **The relationship IS monotonic** — the single-seed non-monotonicity was noise. Confirmed increasing trend across all gammas.

2. **No sharp phase transition** — performance increases smoothly and gradually with gamma. There is no cliff where it suddenly fails. This contradicts the simple "effective horizon must exceed causal horizon" hypothesis.

3. **gamma=0.999 ≈ gamma=0.99** — the "too high gamma" failure mode is real but mild for CartPole. Returns become very similar across timesteps (everything near-equally valuable), making advantage estimates noisy, but the effect is small.

4. **Very high variance** — std of 60-80 even at gamma=0.99. 600 episodes is insufficient for reliable convergence. Single-seed results are misleading.

**Why no sharp phase transition?**

CartPole has **dense rewards** (+1 every step). Even with gamma=0.5, the agent receives gradient signal — just a weaker, shorter-sighted one. It can still learn to balance for ~30 steps on average. A sharp phase transition would require the signal to completely vanish below a threshold, which only happens with **sparse rewards** (e.g. Pong: reward only at game end). In sparse-reward environments, the effective horizon vs causal horizon relationship should be much sharper.

---

### Experiment 2 — Empirical Causal Horizon of CartPole (Perturbation Analysis)
**Script**: `exp2_causal_horizon.py`

Baseline (trained policy, no perturbation): **499.7 steps** (near-perfect play).

| Perturb at step | Mean survival | Delta |
|----------------|--------------|-------|
| 5              | 499.7        | -0.0  |
| 10             | 499.4        | -0.3  |
| 15             | 499.7        | +0.0  |
| 20             | 499.7        | -0.0  |
| 30             | 499.6        | -0.2  |
| 40             | 498.8        | -0.9  |
| 50             | 498.9        | -0.8  |
| 75             | 499.7        | +0.0  |
| 100            | 500.0        | +0.3  |
| 200            | 499.5        | -0.2  |

**Key finding**: Single action perturbations have essentially zero effect on survival of a trained policy. Deltas are <1 step across all perturbation times.

**Critical methodological insight**: This experiment measured the wrong thing. Perturbation analysis on a **trained policy** measures **policy robustness** — how quickly the policy can recover from a single bad action. A good policy recovers in a few steps, so causal influence decays almost immediately. This is NOT the same as the **learning horizon** — the number of steps of future returns that need to be credited during training to assign blame correctly.

Two distinct concepts:
- **Recovery horizon**: steps for policy to recover from one bad action (~5 steps for CartPole). Measured by perturbation analysis on a trained policy.
- **Learning horizon**: how far back gradient signal must propagate during training to correctly credit actions. This is what gamma actually affects.

To measure the learning horizon directly, we would need to analyze the gradient flow during training, not the policy behavior after training. The perturbation approach fundamentally cannot capture this.

---

### Experiment 3 — Synthetic Lag Environment (Non-Markovian)
**Script**: `exp3_synthetic_env.py`

Built a 1D environment where the agent's action at time t affects position at time t+L. State: `[x, velocity]`.

**Result**: Complete failure across ALL gamma values for ALL lags > 1.

| Lag | γ=0.7 | γ=0.8 | γ=0.9 | γ=0.95 | γ=0.99 |
|-----|-------|-------|-------|--------|--------|
| 1   | 444.6 | 470.3 | 478.2 | 434.3  | 450.3  |
| 5   | 22.1  | 22.5  | 29.2  | 23.1   | 31.5   |
| 10  | 26.9  | 28.0  | 27.4  | 26.9   | 29.8   |
| 20  | 31.3  | 29.0  | 35.3  | 33.2   | 35.6   |
| 50  | 45.9  | 46.1  | 43.4  | 43.2   | 42.6   |

**Critical insight**: The environment is a **POMDP** (Partially Observable MDP). The state `[x, velocity]` does not include the action queue — the agent cannot know what corrections are already "in flight." The Markov property is violated: future position depends on queued actions not in the state. This makes the environment effectively unsolvable regardless of gamma. The failure was not a credit assignment failure — it was an **observability failure**.

---

### Experiment 3b — Synthetic Lag Environment (Markovian State)
**Script**: `exp3b_synthetic_markov.py`

Fixed state representation: `[x, velocity, a_{t-1}, ..., a_{t-L}]`. Now fully Markovian.

| Lag | γ_min (theory) | γ=0.7 | γ=0.8 | γ=0.9 | γ=0.95 | γ=0.99 |
|-----|---------------|-------|-------|-------|--------|--------|
| 1   | 0.0           | 140.8 | **485.8** | 233.9 | 67.1 | **407.1** |
| 5   | 0.8           | 29.4  | 44.5  | 57.6  | 70.6   | 63.9   |
| 10  | 0.9           | 28.9  | 40.8  | 40.2  | 47.5   | 39.2   |
| 20  | 0.95          | 28.0  | 25.9  | 27.8  | 28.8   | 29.4   |

**Findings**:

1. **Lag=1 shows convergence is possible** — gamma=0.8 and gamma=0.99 both converge. The non-monotonic result (gamma=0.95 gets 67.1) is likely variance from a single seed, consistent with exp1b findings.

2. **Lag=5, 10, 20 still fail** even with Markovian state and gamma=0.99. The theoretical prediction does not hold.

3. **Why?** The lag environment is a harder control problem than expected. With random drift and velocity dynamics, the agent must predict its future state L steps ahead and apply corrections that account for drift accumulating during the lag period. This requires learning a forward model of the environment dynamics, which is a significantly harder task than simply balancing a pole. The failure is likely a **learning difficulty** problem (task too hard for this network and episode count) rather than a credit assignment problem.

4. **State dimension grows with lag** — lag=20 requires a 22-dimensional state (2 + 20). The network may need more capacity for larger lags.

---

## Summary of Key Lessons

### Lesson 1: Dense vs Sparse Rewards Determine Phase Transition Sharpness
For dense rewards (CartPole), the gamma-performance relationship is smooth and monotonic — there is no sharp phase transition. Any gamma gives some gradient signal. The phase transition hypothesis applies most cleanly to sparse reward environments where signal completely vanishes below a threshold.

### Lesson 2: Recovery Horizon ≠ Learning Horizon
Perturbation analysis on trained policies measures recovery speed, not causal depth of learning. These are fundamentally different quantities. To measure learning horizon, one would need to analyze gradient flow during training.

### Lesson 3: State Representation and Observability Are Prior to Gamma
The Exp3 failure showed that if the MDP is not Markovian, no amount of gamma tuning helps. Ensuring the state is sufficient is logically prior to hyperparameter selection. Gamma is a credit assignment tool — it can only help if the environment is observable in the first place.

### Lesson 4: Single-Seed Results Are Unreliable
CartPole PPO with 600 episodes shows standard deviations of 60-80 episodes length. Single-seed experiments can produce spurious non-monotonic relationships. At minimum 5 seeds are needed.

### Lesson 5: The Analytical Framework Is Still Valid — But the Story Is Richer
The intuition that "gamma must be long enough to capture the causal horizon" is correct directionally. But the mechanism is not a hard cliff — it's a smooth gradient. And the causal horizon itself is harder to measure than expected (Exp2 showed the naive measurement approach doesn't work).

---

## Open Questions and Next Steps

1. **How to actually measure learning horizon?** Analyze gradient magnitudes flowing back through time during training. Do gradients decay as gamma^t? Does increasing gamma meaningfully extend gradient reach?

2. **Sparse reward test**: Run the gamma sweep on a sparse reward version of CartPole (reward only given if survived to step 500, 0 otherwise). Expect a much sharper phase transition.

3. **Why does the Markovian lag environment still fail?** Is it the drift randomness, the velocity dynamics, insufficient capacity, or insufficient training? Ablate each.

4. **RLHF connection**: In RLHF, reward is given once per conversation (sparse). The causal horizon of the conversation (how early in a response sequence can a bad decision influence the final score?) is the relevant quantity. Gamma must be chosen to cover this. The sparse reward argument suggests this threshold matters sharply, unlike CartPole.

---

## Implementation Notes
- Base PPO: `rl_ppole.py` (gamma=0.99, Adam lr=0.001, batch_size=50, batch_epochs=10, clip=0.2)
- Experiment scripts: `exp1_gamma_sweep.py`, `exp1b_multiseed.py`, `exp2_causal_horizon.py`, `exp3_synthetic_env.py`, `exp3b_synthetic_markov.py`
- Plots in `results/`
