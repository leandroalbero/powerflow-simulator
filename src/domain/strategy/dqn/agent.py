"""DQN agent with experience replay and target network."""

import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DqnAgent:
    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 9,
        lr: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        target_update_freq: int = 1000,
        device: str = "cpu",
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer: deque = deque(maxlen=buffer_size)
        self.train_steps = 0

    def select_action(self, state: np.ndarray, epsilon: float = 0.05) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        q = self.get_q_values(state)
        return int(np.argmax(q))

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(t).cpu().numpy()[0]

    def store_transition(
        self, state: np.ndarray, action: int, reward: float,
        next_state: np.ndarray, done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample_batch(self, batch_size: Optional[int] = None) -> Optional[tuple]:
        bs = batch_size or self.batch_size
        if len(self.buffer) < bs:
            return None
        batch = random.sample(list(self.buffer), bs)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def train_step(self) -> Optional[float]:
        batch = self.sample_batch()
        if batch is None:
            return None

        states, actions, rewards, next_states, dones = batch
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            target = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy_net.load_state_dict(torch.load(path, weights_only=True))
        self.target_net.load_state_dict(self.policy_net.state_dict())
