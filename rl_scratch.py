import numpy as np
import random

# 4x4 grid, states 0-15:
#  0  1  2  3
#  4  5  6  7
#  8  9 10 11
# 12 13 14 15
# Goal: 15 (bottom-right). Potholes: 5, 10, 13 (teleport to origin)

gamma = 0.9
lr = 0.5
n_episodes = 2000
epsilon = 0.01
q_matrix = np.zeros((16, 4))

POTHOLES = {5, 10, 13}
GOAL = 15

#              L   R   U   D
MOVES      = [-1, +1, -4, +4]
BOUNDARIES = [lambda s: s % 4 > 0,   # can go left
              lambda s: s % 4 < 3,   # can go right
              lambda s: s // 4 > 0,  # can go up
              lambda s: s // 4 < 3]  # can go down

def step(state, action):
    if BOUNDARIES[action](state):
        next_state = state + MOVES[action]
    else:
        next_state = state

    if next_state in POTHOLES:
        return 0, 0   # teleport to origin
    elif next_state == GOAL:
        return GOAL, 1
    else:
        return next_state, 0

def run_iteration():
    truncated = False
    terminated = False
    state = 0
    i = 0
    while not truncated and not terminated:
        action = pick_action(epsilon, q_matrix, state)
        new_state, reward = step(state, action)

        q_matrix[state, action] += lr * (reward + gamma * max(q_matrix[new_state, :]) - q_matrix[state, action])

        if i > 50:
            truncated = True
        else:
            i += 1
        if reward == 1:
            terminated = True
        state = new_state

def pick_action(epsilon, q_matrix, state):
    if epsilon < random.random():
        action = int(random.random() * 4)
    else:
        action = np.argmax(q_matrix[state, :])
    return action


for i in range(n_episodes):
    run_iteration()
    epsilon += 0.001

print("Final Q-matrix:")
print(q_matrix.reshape(16, 4))
print("\nGreedy policy (0=L,1=R,2=U,3=D):")
print(np.argmax(q_matrix, axis=1).reshape(4, 4))
