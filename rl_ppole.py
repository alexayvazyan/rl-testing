import numpy as np
import random
import torch as pt
import torch.nn as nn
import gymnasium as gym
import matplotlib.pyplot as plt
import os                                                                                                                                                  
dir = os.path.dirname(os.path.abspath(__file__))

env = gym.make("CartPole-v1")
env_test = gym.make("CartPole-v1", render_mode = 'human')

class NN(nn.Module):
    def __init__(self, nstates, nactions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nstates, 64),
            nn.ReLU(),
            nn.Linear(64, nactions)
        )
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity = 'relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.net(x)

class Buffer():
    def __init__(self, maxsize):
        self.buffer = []
        self.idx = 0
        self.max_size = maxsize
    
    def push(self, x):
        if len(self.buffer)<self.max_size:
            self.buffer.append(None)
        self.buffer[self.idx] = x
        self.idx = (self.idx+1) % self.max_size

    def sample(self, n):
        return random.sample(self.buffer, n)
    
    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]
    
    def reset(self):
        self.buffer = []
        self.idx = 0

    def merge_rtg(self, rtg):
        for i, r in enumerate(rtg):
            self.buffer[i] = self.buffer[i] + (r,)


actor = NN(4,2)
critic = NN(4,1)
TrajectoryBuffer = Buffer(4000)

gamma = 0.9
lr = 0.001
n_episodes = 5000
batch_size = 50
batch_epochs = 10
clip = 0.2

actor_optimizer = pt.optim.Adam(actor.parameters(),lr = lr)
critic_optimizer = pt.optim.Adam(critic.parameters(),lr = lr)
critic_criterion = nn.MSELoss()
run_lengths = []

def run_iteration(mode):
    truncated = False
    terminated = False
    i = 0
    if mode == "TRAIN":
        obs, info = env.reset()
        while not truncated and not terminated:
            with pt.no_grad():
                logits = actor(pt.tensor(obs, dtype = pt.float32))
            probs = pt.softmax(logits, dim = -1)
            action = pt.multinomial(probs, 1).item()    
            logprob = pt.log(probs)[action]
            new_obs, reward, truncated, terminated, info = env.step(action)
            i+=1           
            done = truncated or terminated
            if done:
                run_lengths.append(i)
            TrajectoryBuffer.push(
                (obs, action, reward, done, logprob)
                )
            obs = new_obs
            if len(TrajectoryBuffer) > 800 and done:
                rewards_to_go = np.zeros(len(TrajectoryBuffer))

                for i in range(len(TrajectoryBuffer) -1, -1, -1):
                    if TrajectoryBuffer[i][-2] == True: #if done, reward, else discount
                        rewards_to_go[i] = (TrajectoryBuffer[i][2])
                    else:
                        rewards_to_go[i] = (gamma*rewards_to_go[i+1] + TrajectoryBuffer[i][2])

                TrajectoryBuffer.merge_rtg(rewards_to_go)

                for i in range(0, batch_epochs):
                    samples = TrajectoryBuffer.sample(batch_size)
                    s1, a, r, d, lp, rtg = zip(*samples)
                    s1 = pt.stack([pt.tensor(s, dtype = pt.float32) for s in s1])
                    rtg = pt.tensor(rtg, dtype = pt.float32)
                    lp = pt.stack(lp)
                    #train critic
                    v = critic(s1).squeeze()
                    critic_loss = critic_criterion(v, rtg)
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()

                    #train actor
                    adv = rtg - v.detach()
                    new_logits = actor(s1)
                    new_probs = pt.softmax(new_logits, dim = -1)
                    new_logprobs = pt.log(new_probs)[range(batch_size), a]

                    ratio = pt.exp(new_logprobs - lp)
                    actor_loss = -pt.mean(pt.min(ratio*adv, pt.clamp(ratio, 1-clip, 1+clip)*adv))
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                TrajectoryBuffer.reset()
    elif mode == "TEST":
        obs, info = env_test.reset()
        while not truncated and not terminated:
            with pt.no_grad():
                logits = actor(pt.tensor(obs, dtype = pt.float32))
                probs = pt.softmax(logits, dim = -1)
                action = pt.argmax(probs).item()    
                new_obs, reward, truncated, terminated, info = env_test.step(action)       
                obs = new_obs  


for i in range(0, n_episodes):
    run_iteration("TRAIN")

run_iteration("TEST")  

            
window = 10
moving_avg_steps = np.convolve(run_lengths, np.ones(window)/window, mode = 'valid')
plt.figure(figsize=(8,5))
plt.plot(range(window-1, len(run_lengths)), moving_avg_steps, label="moving avg")
plt.xlabel("Episode")
plt.ylabel("Duration")
plt.title("Training Progress")
plt.legend()
plt.savefig(os.path.join(dir,"average_game_length_cartpole_ppo_tuned_lowgamma.png"), dpi=200)
plt.close()