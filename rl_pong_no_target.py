"""DQN variant without a target network — bootstrap target is computed with
`solver` itself instead of a frozen copy. Intended as an A/B against rl_pong.py
to motivate why the target network matters. Expect Q-values to diverge."""

from typing import Any
import argparse
import csv
import time
import numpy as np
import random
import torch as pt
from torch._inductor.config import max_fusion_size
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import ale_py

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", type=str, default="no_target_" + time.strftime("%Y%m%d-%H%M%S"))
args, _ = parser.parse_known_args()

run_name = args.run_name
tb_logdir = f"runs/{run_name}"
writer = SummaryWriter(log_dir=tb_logdir)
print(f"TensorBoard logs -> {tb_logdir} (NO target net — bootstrap uses solver)")

gym.register_envs(ale_py)
num_envs = 16
env = gym.make_vec(
    "ALE/Pong-v5",
    num_envs=num_envs,
    vectorization_mode="async",
    max_episode_steps=500,
)
env_test = None
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print(f"Using device: {device}")
gamma = 0.99
lr = 0.001
n_episodes = 205
steps_per_episode = 500
batch_size = 512
grad_steps_per_tick = 2
replay_capacity = 100_000
learning_starts = 10_000

epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_env_steps = 500_000

def current_epsilon():
    frac = min(1.0, total_env_steps / epsilon_decay_env_steps)
    return epsilon_start + (epsilon_end - epsilon_start) * frac

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
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, max_size, frame_shape=(4, 84, 84)):
        self.max_size = max_size
        self.size = 0
        self.idx = 0
        self.states = pt.zeros((max_size, *frame_shape), dtype=pt.uint8, device=device)
        self.next_states = pt.zeros((max_size, *frame_shape), dtype=pt.uint8, device=device)
        self.actions = pt.zeros((max_size,), dtype=pt.long, device=device)
        self.rewards = pt.zeros((max_size,), dtype=pt.float32, device=device)
        self.dones = pt.zeros((max_size,), dtype=pt.float32, device=device)

    def push_batch(self, states, actions, rewards, next_states, dones):
        n = states.shape[0]
        end = self.idx + n
        if end <= self.max_size:
            sl = slice(self.idx, end)
            self.states[sl] = (states * 255).to(pt.uint8)
            self.next_states[sl] = (next_states * 255).to(pt.uint8)
            self.actions[sl] = actions
            self.rewards[sl] = rewards
            self.dones[sl] = dones
        else:
            first = self.max_size - self.idx
            self.states[self.idx:] = (states[:first] * 255).to(pt.uint8)
            self.next_states[self.idx:] = (next_states[:first] * 255).to(pt.uint8)
            self.actions[self.idx:] = actions[:first]
            self.rewards[self.idx:] = rewards[:first]
            self.dones[self.idx:] = dones[:first]
            second = n - first
            self.states[:second] = (states[first:] * 255).to(pt.uint8)
            self.next_states[:second] = (next_states[first:] * 255).to(pt.uint8)
            self.actions[:second] = actions[first:]
            self.rewards[:second] = rewards[first:]
            self.dones[:second] = dones[first:]
        self.size = min(self.size + n, self.max_size)
        self.idx = end % self.max_size

    def sample(self, batch_size):
        idx = pt.randint(0, self.size, (batch_size,), device=device)
        s = self.states[idx].float() / 255.0
        ns = self.next_states[idx].float() / 255.0
        return s, self.actions[idx], self.rewards[idx], ns, self.dones[idx]

    def __len__(self):
        return self.size


_grey_weights = pt.tensor([0.2989, 0.5870, 0.1140], device=device)

def preprocess_batch(obs):
    t = pt.from_numpy(obs).to(device)
    t = t.float()
    grey = (t * _grey_weights).sum(dim=-1)
    crop = grey[:, 34:194] / 255.0
    out = nn.functional.interpolate(crop.unsqueeze(1), size=(84, 84), mode="area")
    return out.squeeze(1)

def preprocess(obs):
    return preprocess_batch(obs[None])[0]

solver = CNN().to(device)
optimizer = pt.optim.Adam(solver.parameters(), lr=lr)
criterion = nn.SmoothL1Loss()
replay_experience = ReplayBuffer(replay_capacity)

log_interval_s = 10.0
total_env_steps = 0
total_grad_steps = 0
total_episodes = 0
ep_returns = np.zeros(num_envs, dtype=np.float64)
recent_returns: list[float] = []
t_start = time.perf_counter()
t_last_log = t_start
steps_at_last_log = 0
probe_states: pt.Tensor | None = None
probe_log_every = 100
grads_at_last_log = 0

log_csv_path = f"train_log_{run_name}.csv"
log_csv_file = open(log_csv_path, "w", newline="")
log_csv = csv.writer(log_csv_file)
log_csv.writerow([
    "elapsed_s", "env_steps", "grad_steps", "episodes",
    "env_per_s", "grad_per_s", "mean_return_50",
])
log_csv_file.flush()

def run_iteration(mode):
    if mode == "TRAIN":
        frames = pt.zeros(num_envs, 5, 84, 84, device=device)
        obs, info = env.reset()
        frames[:, -1] = preprocess_batch(obs)
        for i in range(steps_per_episode):
            if i < 5:
                actions_np = env.action_space.sample()
            else:
                with pt.no_grad():
                    qvals = solver(frames[:, 1:5])
                greedy = qvals.argmax(dim=1).cpu().numpy()
                random_actions = env.action_space.sample()
                eps = current_epsilon()
                random_mask = np.random.rand(num_envs) < eps
                actions_np = np.where(random_mask, random_actions, greedy)

            obs, rewards_np, terminated, truncated, info = env.step(actions_np)
            states_gpu = frames[:, 1:5].clone()
            frames = pt.roll(frames, shifts=-1, dims=1)
            frames[:, -1] = preprocess_batch(obs)
            next_states_gpu = frames[:, 1:5]

            global total_env_steps, total_grad_steps, total_episodes
            global ep_returns, t_last_log, steps_at_last_log, grads_at_last_log
            total_env_steps += num_envs
            ep_returns += rewards_np
            dones = np.logical_or(terminated, truncated)
            if dones.any():
                for e in np.where(dones)[0]:
                    recent_returns.append(float(ep_returns[e]))
                    ep_returns[e] = 0.0
                total_episodes += int(dones.sum())

            if i >= 5:
                actions_gpu = pt.from_numpy(actions_np).long().to(device)
                rewards_gpu = pt.from_numpy(rewards_np).float().to(device)
                dones_gpu = pt.from_numpy(dones.astype(np.float32)).to(device)
                replay_experience.push_batch(states_gpu, actions_gpu, rewards_gpu, next_states_gpu, dones_gpu)

                if len(replay_experience) > learning_starts:
                    global probe_states
                    if probe_states is None:
                        probe_states, _, _, _, _ = replay_experience.sample(32)
                        probe_states = probe_states.clone()
                    for _ in range(grad_steps_per_tick):
                        states, actions, rewards, next_states, done_flags = replay_experience.sample(batch_size)
                        # NO TARGET NET — bootstrap with solver itself. This is the "moving target"
                        # pathology the target network is designed to prevent.
                        with pt.no_grad():
                            nextstate_qmax = solver(next_states).max(dim=1).values
                        oldstate_qvals = solver(states)
                        target = rewards + gamma * (1.0 - done_flags) * nextstate_qmax
                        target_vec = oldstate_qvals.detach().clone()
                        target_vec[pt.arange(batch_size, device=device), actions] = target
                        loss = criterion(oldstate_qvals, target_vec)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_grad_steps += 1

                        writer.add_scalar("train/loss", loss.item(), total_grad_steps)
                        with pt.no_grad():
                            q_max = oldstate_qvals.max().item()
                            q_mean = oldstate_qvals.mean().item()
                            q_abs_max = oldstate_qvals.abs().max().item()
                        writer.add_scalar("q/max", q_max, total_grad_steps)
                        writer.add_scalar("q/mean", q_mean, total_grad_steps)
                        writer.add_scalar("q/abs_max", q_abs_max, total_grad_steps)

                        if total_grad_steps % probe_log_every == 0:
                            with pt.no_grad():
                                probe_q = solver(probe_states).max(dim=1).values
                            writer.add_scalar("q/probe_mean_max", probe_q.mean().item(), total_grad_steps)
                            writer.add_scalar("q/probe_abs_max", probe_q.abs().max().item(), total_grad_steps)

            now = time.perf_counter()
            if now - t_last_log >= log_interval_s:
                dt = now - t_last_log
                env_rate = (total_env_steps - steps_at_last_log) / dt
                grad_rate = (total_grad_steps - grads_at_last_log) / dt
                mean_ret = float(np.mean(recent_returns[-50:])) if recent_returns else float("nan")
                elapsed = now - t_start
                print(
                    f"[{elapsed:6.1f}s] "
                    f"env_steps={total_env_steps:>9}  "
                    f"grad_steps={total_grad_steps:>7}  "
                    f"eps={total_episodes:>4}  "
                    f"env/s={env_rate:6.0f}  "
                    f"grad/s={grad_rate*batch_size:6.1f}  "
                    f"mean_ret(50)={mean_ret:+.2f}  "
                    f"eps_remaining={n_episodes - 1 - i_outer}  "
                    f"buf={len(replay_experience):>6}  "
                    f"eps_val={current_epsilon():.3f}"
                )
                log_csv.writerow([
                    f"{elapsed:.2f}", total_env_steps, total_grad_steps, total_episodes,
                    f"{env_rate:.1f}", f"{grad_rate:.2f}", f"{mean_ret:.3f}",
                ])
                log_csv_file.flush()
                writer.add_scalar("throughput/env_per_s", env_rate, total_env_steps)
                writer.add_scalar("throughput/grad_per_s", grad_rate, total_env_steps)
                writer.add_scalar("throughput/samples_per_s", grad_rate * batch_size, total_env_steps)
                writer.add_scalar("episode/count", total_episodes, total_env_steps)
                writer.add_scalar("episode/mean_return_50", mean_ret, total_env_steps)
                writer.add_scalar("schedule/epsilon", current_epsilon(), total_env_steps)
                writer.add_scalar("replay/fill", len(replay_experience), total_env_steps)
                t_last_log = now
                steps_at_last_log = total_env_steps
                grads_at_last_log = total_grad_steps

    elif mode == "TEST":
        truncated = False
        terminated = False
        i = 0
        frames = pt.zeros(5, 84, 84, device=device)
        global env_test
        if env_test is None:
            env_test = gym.make("ALE/Pong-v5", render_mode="human")
        obs, info = env_test.reset()
        while not truncated and not terminated:
            if i < 5:
                action = env_test.action_space.sample()
            else:
                with pt.no_grad():
                    qvals = solver(frames[1:5].unsqueeze(0))
                action = pt.argmax(qvals).item()
            obs, reward, terminated, truncated, info = env_test.step(action)
            frames = pt.roll(frames, shifts=-1, dims=0)
            frames[-1] = preprocess(obs)
            i += 1

j = 0
i_outer = 0
for i in range(1, n_episodes):
    i_outer = i
    run_iteration("TRAIN")
    j += 1
    if j % 5 == 0:
        print(f"--- outer episode {j}/{n_episodes - 1} done ---")
log_csv_file.close()
writer.close()
pt.save(solver.state_dict(), f'pong_weights_{run_name}.pt')
