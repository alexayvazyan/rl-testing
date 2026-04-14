from typing import Any
import numpy as np
import random
import torch as pt
from torch._inductor.config import max_fusion_size
import torch.nn as nn
import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", max_episode_steps=500)
env_test = gym.make("ALE/Pong-v5", render_mode="human")
gamma = 0.9
lr = 0.01
n_episodes = 105
epsilon = 0.01

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 6)
        )
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity= 'relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.net(x)

class ReplayBuffer():
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.idx = 0

    def push(self, s, a, r, ns):
        if len(self.buffer)<self.max_size:
            self.buffer.append(None)
        self.buffer[self.idx] = (s, a, r, ns)
        self.idx = (self.idx+1) % self.max_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def preprocess(obs):
    greyscale = np.dot(obs, [0.2989, 0.5870, 0.1140])
    resize = greyscale[34:194]
    t = pt.tensor(resize, dtype=pt.float32) / 255.0
    return nn.functional.interpolate(t.unsqueeze(0).unsqueeze(0), size=(84, 84)).squeeze()

solver = CNN()
optimizer = pt.optim.Adam(solver.parameters(), lr=lr)
criterion = nn.MSELoss()
normal_buffer = ReplayBuffer(100000)
reward_buffer = ReplayBuffer(10000)

def run_iteration(mode):
    truncated = False
    terminated = False
    i = 0
    frames = pt.zeros(5,84,84)
    if mode == "TRAIN":
        obs, info = env.reset()
        while not truncated and not terminated:
            if not i<5:
                qvals = solver(frames[1:5].unsqueeze(0))

            if (epsilon < random.random()) or (i<5):
                action = env.action_space.sample()
            else:
                action = pt.argmax(qvals).item()

            obs, reward, terminated, truncated, info = env.step(action)
            frames = pt.roll(frames, shifts = -1, dims = 0)
            frames[-1] = preprocess(obs)
            if i<5:
                i+=1
                continue
            else:
                normal_buffer.push(frames[0:4], action, reward, frames[1:5])
                if reward != 0:
                    reward_buffer.push(frames[0:4], action, reward, frames[1:5])

                if len(normal_buffer) > 10000:
                    if len(reward_buffer) >= 2:
                        sample_batch = reward_buffer.sample(2) + normal_buffer.sample(2)
                    else:
                        sample_batch = normal_buffer.sample(4)
                    states, actions, rewards, next_states = zip(*sample_batch)
                    states = pt.stack(states)
                    actions = pt.tensor(list(actions))
                    rewards = pt.tensor(list(rewards))
                    next_states = pt.stack(next_states)
                    with pt.no_grad():
                        nextstate_qmax = solver(next_states).max(dim=1).values
                    oldstate_qvals = solver(states)
                    target = gamma*nextstate_qmax + rewards
                    target_vec = oldstate_qvals.clone().detach()
                    target_vec[range(4),actions] = target
                    loss = criterion(oldstate_qvals, target_vec)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                i+=1

    elif mode == "TEST":
        obs, info = env_test.reset()
        while not truncated and not terminated:
            if i<5:
                action = env_test.action_space.sample()
            else:
                qvals = solver(frames[1:5].unsqueeze(0))
                action = pt.argmax(qvals).item()
            obs, reward, terminated, truncated, info = env_test.step(action)
            frames = pt.roll(frames, shifts = -1, dims = 0)
            frames[-1] = preprocess(obs)
            i+=1

j=0
for i in range(1, n_episodes):
    run_iteration("TRAIN")
    epsilon += 0.02
    j+=1
    if j % 5 == 0:
        print(j)
pt.save(solver.state_dict(), 'pong_weights_prio.pt')

solver.load_state_dict(pt.load("pong_weights_prio.pt"))

solver.eval()
run_iteration("TEST")
