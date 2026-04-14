import numpy as np
import random
from sympy.polys.polytools import I
import torch as pt
import torch.nn as nn
import matplotlib.pyplot as plt

# 4x4 grid, states 0-15:
#  0  1  2  3
#  4  5  6  7
#  8  9 10 11
# 12 13 14 15
# Goal: 15 (bottom-right). Potholes: 5, 10, 13 (teleport to origin)

gamma = 0.9
lr = 0.01
n_episodes = 2000
epsilon = 0.001
#q_matrix = np.zeros((16, 4))
steps = []
weights_norms = []
POTHOLES = {5, 10, 13}
GOAL = 15

class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 10),
            nn.ReLU(),
            nn.Linear(10, n_actions)
        )
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity= 'relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.net(x)

solver = DQN(16, 4)
optimizer = pt.optim.SGD(solver.parameters(), lr = lr)
criterion = nn.MSELoss()

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
        currstate_q = solver.forward(pt.tensor(np.eye(16)[state], dtype = pt.float32))
        action = pick_action(epsilon, currstate_q)
        new_state, reward = step(state, action)
        with pt.no_grad():
            newstate_q = solver.forward(pt.tensor(np.eye(16)[new_state], dtype = pt.float32))
        target = reward + gamma * max(newstate_q)
        targetv = currstate_q.clone().detach()
        targetv[action] = target
        loss = criterion(currstate_q, targetv)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if i > 50:
            steps.append(i)
            l2 = pt.cat([p.data.flatten() for p in solver.parameters()]).norm(2).item()
            weights_norms.append(l2)
            truncated = True
        else:
            i += 1
        if reward == 1:
            steps.append(i)
            l2 = pt.cat([p.data.flatten() for p in solver.parameters()]).norm(2).item()
            weights_norms.append(l2)
            terminated = True
        state = new_state

def pick_action(epsilon, currstate_q):
    if epsilon < random.random():
        action = int(random.random() * 4)
    else:
        action = pt.argmax(currstate_q).item()
    return action


for i in range(n_episodes):
    run_iteration()
    epsilon += 0.001

#print("Final Q-matrix:")
#print(q_matrix.reshape(16, 4))
#print("\nGreedy policy (0=L,1=R,2=U,3=D):")
#print(np.argmax(q_matrix, axis=1).reshape(4, 4))
data = np.array(steps)
window = 10
moving_avg_steps = np.convolve(steps, np.ones(window)/window, mode = 'valid')
moving_avg_weights = np.convolve(weights_norms, np.ones(window)/window, mode = 'valid')
plt.figure(figsize=(8,5))
plt.plot(range(window-1, len(steps)), moving_avg_steps, label="moving avg")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.title("Training Progress")
plt.legend()
plt.savefig("training_curve.png", dpi=200)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(range(window-1, len(weights_norms)), moving_avg_weights, label="moving avg")
plt.xlabel("Episode")
plt.ylabel("Weights Norms")
plt.title("Training Progress")
plt.legend()
plt.savefig("weights_norms.png", dpi=200)
plt.close()