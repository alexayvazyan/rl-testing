"""
Experiment 8 — Environment Temporal Fingerprint

Goal: identify intrinsic properties of CartPole that tell you,
WITHOUT any training or gamma selection, that long-horizon reasoning is necessary.

Three measurements:

  A. Episode length distribution under random policy.
     The median episode length L_50 is the causal horizon.
     For gamma to credit the first action for what happens at the median episode end:
       gamma^L_50 = fraction of future value still credited after L steps
     At gamma=0.9:  0.9^20  = 0.12  (88% of future value discarded by typical episode end)
     At gamma=0.99: 0.99^20 = 0.82  (18% discarded)
     Direct prediction: gamma_min such that gamma^L_50 > 0.5
                     => gamma_min = 0.5^(1/L_50)

  B. Action persistence: take action 0 vs action 1 from the same state,
     then continue with random policy. Measure P(survive k more steps | action_a vs action_b).
     How long does a single decision affect survival probability?
     This is measured as the KL divergence or difference in survival curves between the two forks.

  C. Gamma credit curve vs episode length distribution.
     For each candidate gamma, compute gamma^k as a function of k.
     Overlay the episode length survival function P(episode >= k).
     The "credit mass" is int_0^inf gamma^k * P(episode >= k) dk (discrete version).
     This directly shows how much of the episode's causal structure each gamma can see.
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N_EPISODES  = 2000   # large sample for stable statistics
MAX_K       = 100    # look ahead this many steps

env = gym.make("CartPole-v1")

# ── Collect random policy episode lengths ─────────────────────────────────────
print("Collecting random policy rollouts...")
ep_lengths = []
for _ in range(N_EPISODES):
    obs, _ = env.reset()
    steps = 0
    done = False
    while not done:
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        steps += 1
        done = term or trunc
    ep_lengths.append(steps)

ep_lengths = np.array(ep_lengths)
L_25  = np.percentile(ep_lengths, 25)
L_50  = np.percentile(ep_lengths, 50)
L_75  = np.percentile(ep_lengths, 75)
L_90  = np.percentile(ep_lengths, 90)
L_mean = np.mean(ep_lengths)

print(f"\nA. Episode length statistics (random policy, N={N_EPISODES}):")
print(f"   mean={L_mean:.1f}  p25={L_25:.0f}  median={L_50:.0f}  p75={L_75:.0f}  p90={L_90:.0f}  max={max(ep_lengths)}")

# Survival function S(k) = P(episode length >= k)
S = np.array([(ep_lengths >= k).mean() for k in range(MAX_K + 1)])

# Gamma predictions from episode length
# gamma_min such that gamma^L_50 = 0.5 → gamma_min = 0.5^(1/L_50)
gamma_from_median = 0.5 ** (1 / L_50)
gamma_from_p90    = 0.5 ** (1 / L_90)

print(f"\n   Causal horizon (median): {L_50:.0f} steps")
print(f"   gamma_min (from median, gamma^L50=0.5): {gamma_from_median:.3f}")
print(f"   gamma_min (from p90,    gamma^L90=0.5): {gamma_from_p90:.3f}")
print(f"\n   Interpretation:")
print(f"   At gamma=0.90, credit after {L_50:.0f} steps: {0.90**L_50:.3f}  (discards {(1-0.90**L_50)*100:.0f}%)")
print(f"   At gamma=0.99, credit after {L_50:.0f} steps: {0.99**L_50:.3f}  (discards {(1-0.99**L_50)*100:.0f}%)")

# ── B. Action persistence — fork survival experiment ──────────────────────────
print("\nB. Computing action persistence (fork survival)...")

N_FORKS = 500
# For each fork, we record whether action 0 vs action 1 led to longer survival
# survival_k[action] = fraction of forks where agent survived k more steps
survival_0 = np.zeros(MAX_K + 1)
survival_1 = np.zeros(MAX_K + 1)
n_forks = 0

for _ in range(N_FORKS):
    # Get a starting state
    obs, _ = env.reset()
    # Walk a random number of steps to get a "natural" state
    walk = np.random.randint(1, 10)
    valid = True
    for _ in range(walk):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            valid = False
            break
    if not valid:
        continue

    start_state = obs.copy()

    # Fork: run each action, then random policy, record survival length
    fork_survivals = []
    for fork_action in [0, 1]:
        env2 = gym.make("CartPole-v1")
        env2.reset()
        env2.unwrapped.state = start_state.copy()
        obs2, _, term, trunc, _ = env2.step(fork_action)
        done = term or trunc
        k = 0
        while not done and k < MAX_K:
            obs2, _, term, trunc, _ = env2.step(env2.action_space.sample())
            done = term or trunc
            k += 1
        # survived k steps after the fork action (0..MAX_K)
        fork_survivals.append(k)
        env2.close()

    # Record survival curves: did each fork survive >= k more steps?
    for k in range(MAX_K + 1):
        survival_0[k] += 1 if fork_survivals[0] >= k else 0
        survival_1[k] += 1 if fork_survivals[1] >= k else 0
    n_forks += 1

survival_0 /= n_forks
survival_1 /= n_forks

# Difference in survival probability between the two fork actions
surv_diff = np.abs(survival_0 - survival_1)

# How far does the action choice matter? Find where diff drops to 25% of initial diff
init_diff = surv_diff[1]
persistence_k = None
for k in range(1, MAX_K + 1):
    if surv_diff[k] < 0.25 * init_diff:
        persistence_k = k
        break

print(f"   Completed {n_forks} forks")
print(f"   Initial survival difference |P0-P1| at k=1: {init_diff:.3f}")
if persistence_k:
    print(f"   Action persistence (diff drops to 25%): {persistence_k} steps")
else:
    print(f"   Action persistence: still significant at {MAX_K} steps")

# ── C. Credit mass — how much causal structure can each gamma see? ────────────
print("\nC. Computing credit mass for candidate gammas...")

# Credit mass for gamma = sum_k gamma^k * S(k)  (how much of the survival distribution is seen)
# Normalised by sum_k 1 * S(k) (maximum possible)
GAMMAS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
ks = np.arange(MAX_K + 1)
max_credit = np.sum(S)  # if gamma were 1.0

credit_masses = []
for g in GAMMAS:
    mass = np.sum((g ** ks) * S)
    credit_masses.append(mass / max_credit)
    print(f"   gamma={g}: credit_mass={mass/max_credit:.3f}  "
          f"(gamma^L50={g**L_50:.3f}, gamma^L90={g**L_90:.3f})")

env.close()

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# A: Episode length distribution + survival function
ax = axes[0, 0]
ax.hist(ep_lengths, bins=40, color='steelblue', alpha=0.7, density=True, label='Episode length pdf')
ax.axvline(L_50, color='orange', linestyle='--', linewidth=2,
           label=f'Median={L_50:.0f} steps → γ_min≈{gamma_from_median:.3f}')
ax.axvline(L_90, color='red', linestyle=':', linewidth=2,
           label=f'p90={L_90:.0f} steps → γ_min≈{gamma_from_p90:.3f}')
ax.set_xlabel("Episode length (steps)")
ax.set_ylabel("Density")
ax.set_title("A. Random Policy Episode Length Distribution\n(= causal horizon without any training)")
ax.legend(fontsize=8)

# A2: Survival function + gamma credit curves
ax = axes[0, 1]
ax.plot(ks, S, 'k-', linewidth=2.5, label='S(k) = P(episode ≥ k)')
ax.axvline(L_50, color='orange', linestyle='--', alpha=0.7, label=f'Median={L_50:.0f}')
colors_g = ['red', 'brown', 'darkorange', 'purple', 'blue', 'steelblue', 'green']
for g, c in zip([0.5, 0.9, 0.99], ['red', 'darkorange', 'steelblue']):
    credit_curve = (g ** ks) * S
    ax.fill_between(ks, 0, credit_curve, alpha=0.15, color=c, label=f'γ={g} credit×S(k)')
ax.set_xlabel("k (steps ahead)")
ax.set_ylabel("Probability / Weight")
ax.set_title("A2. Episode Survival Function vs Gamma Credit Decay\n"
             "(shaded area = discounted future value accessible to each gamma)")
ax.legend(fontsize=8)
ax.set_xlim(0, 80)

# B: Action persistence (survival difference)
ax = axes[1, 0]
ax.plot(range(MAX_K + 1), survival_0, 'steelblue', linewidth=2, label='P(survive k | action 0)')
ax.plot(range(MAX_K + 1), survival_1, 'darkorange', linewidth=2, label='P(survive k | action 1)')
ax.fill_between(range(MAX_K + 1), survival_0, survival_1, alpha=0.3, color='gray',
                label='|difference|')
if persistence_k:
    ax.axvline(persistence_k, color='red', linestyle=':', label=f'Persistence ≈ {persistence_k} steps')
ax.set_xlabel("k (steps after fork action)")
ax.set_ylabel("P(survive k more steps)")
ax.set_title("B. Action Persistence\n(How long does a single decision affect survival?)")
ax.legend(fontsize=8)
ax.set_xlim(0, MAX_K)
ax.set_ylim(0, 1.05)

# C: Credit mass vs gamma
ax = axes[1, 1]
bar_colors = ['red', 'brown', 'darkorange', 'purple', 'blue', 'steelblue', 'green']
bars = ax.bar([str(g) for g in GAMMAS], credit_masses, color=bar_colors, alpha=0.8)
ax.axhline(0.5, color='black', linestyle='--', alpha=0.5, label='50% threshold')
for bar, mass in zip(bars, credit_masses):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{mass:.2f}', ha='center', va='bottom', fontsize=9)
ax.set_xlabel("Gamma")
ax.set_ylabel("Normalised Credit Mass")
ax.set_title("C. Credit Mass by Gamma\n"
             "(fraction of causal structure visible = ∑ γᵏ·S(k) / ∑ S(k))")
ax.legend(fontsize=8)
ax.set_ylim(0, 1.1)

plt.suptitle("Experiment 8: Environment Temporal Fingerprint\n"
             "(All from random policy — no training, no gamma assumption needed)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "exp8_temporal_fingerprint.png"), dpi=200)
plt.close()

print("\n\n=== SUMMARY ===")
print(f"Random policy median episode length: {L_50:.0f} steps")
print(f"Predicted gamma_min (gamma^median = 0.5): {gamma_from_median:.3f}")
print(f"\nGamma credit at the median episode horizon ({L_50:.0f} steps):")
for g in [0.5, 0.7, 0.9, 0.95, 0.99, 0.999]:
    print(f"  gamma={g}: {g**L_50:.3f} ({(1-g**L_50)*100:.0f}% discarded)")
print(f"\nCredit mass (fraction of causal structure each gamma can see):")
for g, cm in zip(GAMMAS, credit_masses):
    print(f"  gamma={g}: {cm:.3f}")
print(f"\nConclusion: the environment's causal horizon is ~{L_50:.0f} steps (median episode under random policy).")
print(f"Any gamma below {gamma_from_median:.2f} discards >50% of the causal structure at the median horizon.")
print(f"gamma=0.9 sees only {0.9**L_50*100:.0f}% of the median horizon; gamma=0.99 sees {0.99**L_50*100:.0f}%.")
