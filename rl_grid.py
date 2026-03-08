import numpy as np

# 2x2 grid:
#  0 | 1
# -------
#  2 | 3  <-- goal

# Actions: 0=up, 1=down, 2=left, 3=right
ACTION_NAMES = ["up", "down", "left", "right"]

# Transition table: transitions[state][action] = next_state
# Moving off the grid stays in same state
transitions = [
    # state 0: up->0, down->2, left->0, right->1
    [0, 2, 0, 1],
    # state 1: up->1, down->3, left->0, right->1
    [1, 3, 0, 1],
    # state 2: up->0, down->2, left->2, right->3
    [0, 2, 2, 3],
    # state 3: terminal (goal)
    [3, 3, 3, 3],
]

GOAL = 3
REWARD_GOAL = 10
REWARD_STEP = -1

# Hyperparameters
alpha     = 0.5   # learning rate
gamma     = 0.9   # discount factor
epsilon   = 1.0   # starting exploration rate
eps_min   = 0.05
eps_decay = 0.99  # decay per episode
episodes  = 300

Q = np.zeros((4, 4))

def print_q_table(Q, episode, epsilon):
    print(f"\n--- Episode {episode:>4}  (epsilon={epsilon:.3f}) ---")
    print(f"{'State':<8}", end="")
    for a in ACTION_NAMES:
        print(f"{a:>10}", end="")
    print()
    for s in range(4):
        label = f"{s}(goal)" if s == GOAL else str(s)
        print(f"{label:<8}", end="")
        for a in range(4):
            print(f"{Q[s,a]:>10.3f}", end="")
        print()

def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(4)
    return np.argmax(Q[state])

def run_episode(epsilon):
    state = 0
    steps = 0
    while state != GOAL and steps < 100:
        action = choose_action(state, epsilon)
        next_state = transitions[state][action]
        reward = REWARD_GOAL if next_state == GOAL else REWARD_STEP

        # Q-learning update
        best_next = np.max(Q[next_state])
        Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

        state = next_state
        steps += 1
    return steps

# Episodes at which we print the Q-table
show_at = {1, 5, 10, 25, 50, 100, 150, 200, 250, 300}

print("Q-Learning on a 2x2 grid")
print("Grid layout:  0 | 1")
print("              -----")
print("              2 | 3  (goal)")
print(f"\nHyperparams: alpha={alpha}, gamma={gamma}, epsilon {epsilon}→{eps_min} (decay={eps_decay}/ep)")
print_q_table(Q, 0, epsilon)

for ep in range(1, episodes + 1):
    steps = run_episode(epsilon)
    epsilon = max(eps_min, epsilon * eps_decay)
    if ep in show_at:
        print_q_table(Q, ep, epsilon)
        print(f"  solved in {steps} steps this episode")

print("\n=== Final Q-table ===")
print_q_table(Q, episodes, epsilon)
print("\nBest actions per state:")
for s in range(4):
    best = np.argmax(Q[s])
    print(f"  State {s}: {ACTION_NAMES[best]}")
