# DQN Battery Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a DQN reinforcement learning agent to control battery charge/discharge, learning patterns that LP-based strategies can't capture.

**Architecture:** A Gym-like environment wraps the energy simulator. A DQN agent (2×128 MLP) learns a policy mapping state (SoC, time, solar, load, rates, forecasts) to discrete charge/discharge actions. Training uses experience replay and a target network. The trained policy is saved as a `.pt` file and loaded by `DqnStrategy` for simulation.

**Tech Stack:** PyTorch 2.9.1 (MPS backend for Apple Silicon), numpy. No new dependencies.

---

## File Structure

```
src/domain/strategy/dqn/
  __init__.py               # Package marker
  environment.py            # BatteryEnv: Gym-like environment
  agent.py                  # DQN network + DqnAgent (train/act)
  train.py                  # CLI: python -m src.domain.strategy.dqn.train
src/domain/strategy/
  dqn_strategy.py           # DqnStrategy: BaseEnergyStrategy wrapper
web/backend/services/
  simulation_service.py     # Register DQN strategy (modify)
tests/domain/strategy/dqn/
  __init__.py
  test_environment.py       # BatteryEnv tests
  test_agent.py             # DQN agent tests
  test_dqn_strategy.py      # Strategy wrapper tests
```

---

### Task 1: BatteryEnv — Tests + Implementation

**Files:**
- Create: `tests/domain/strategy/dqn/__init__.py`
- Create: `tests/domain/strategy/dqn/test_environment.py`
- Create: `src/domain/strategy/dqn/__init__.py`
- Create: `src/domain/strategy/dqn/environment.py`

BatteryEnv wraps the tariff/battery simulation into a step-based RL environment. Each step: agent observes state, picks action, receives reward (negative energy cost).

- [ ] **Step 1: Write BatteryEnv tests**

Create `tests/domain/strategy/dqn/test_environment.py`:

```python
import numpy as np
import pytest

from src.domain.strategy.dqn.environment import BatteryEnv


class TestBatteryEnv:
    @pytest.fixture
    def env(self):
        """24h of synthetic data: solar bell curve, load with evening peak."""
        n = 24
        solar = np.array([0,0,0,0,0,0, 0.1,0.5,1.5,3.0, 4.0,4.5,4.5,4.0, 3.0,1.5,0.5,0.1, 0,0,0,0,0,0], dtype=np.float32)
        load = np.array([0.3,0.3,0.3,0.3,0.3,0.5, 0.8,1.2,1.5,1.0, 0.8,0.8,1.0,0.8, 0.8,1.0,1.5,2.0, 2.5,2.0,1.5,1.0,0.5,0.3], dtype=np.float32)
        hours = np.arange(24, dtype=np.int32)
        import_rates = np.array([0.085]*8 + [0.134]*2 + [0.182]*4 + [0.134]*4 + [0.182]*4 + [0.134]*2, dtype=np.float32)
        export_rate = 0.08
        return BatteryEnv(
            solar=solar, load=load, hours=hours,
            import_rates=import_rates, export_rate=export_rate,
            battery_capacity=15.0, max_charge_rate=4.8,
            max_discharge_rate=4.8, efficiency=0.95,
            initial_soc=0.1, min_soc_frac=0.1,
            dt=1.0,
        )

    def test_reset_returns_state(self, env):
        state = env.reset()
        assert isinstance(state, np.ndarray)
        assert state.dtype == np.float32
        assert len(state) == env.state_dim

    def test_step_returns_tuple(self, env):
        env.reset()
        state, reward, done, info = env.step(4)  # action 4 = no-op (middle)
        assert isinstance(state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_episode_terminates(self, env):
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(4)
            steps += 1
        assert steps == 24  # one step per hour

    def test_soc_stays_in_bounds(self, env):
        env.reset()
        for _ in range(24):
            state, _, done, _ = env.step(0)  # max discharge
            if done:
                break
            soc = state[0]  # first element is normalized SoC
            assert 0.0 <= soc <= 1.0

    def test_charging_increases_soc(self, env):
        state = env.reset()
        initial_soc = state[0]
        next_state, _, _, _ = env.step(8)  # max charge
        assert next_state[0] >= initial_soc

    def test_state_dim_property(self, env):
        assert env.state_dim == 16
        assert env.action_dim == 9
```

- [ ] **Step 2: Implement BatteryEnv**

Create `src/domain/strategy/dqn/__init__.py` (empty) and `src/domain/strategy/dqn/environment.py`:

```python
"""Gym-like battery environment for RL training."""

import math

import numpy as np


# 9 discrete actions: -1.0, -0.75, ..., 0, ..., 0.75, 1.0
ACTIONS = np.linspace(-1.0, 1.0, 9, dtype=np.float32)

# Tariff hour boundaries for "hours_to_next" features
VALLEY_HOURS = set(range(0, 8))
PEAK_HOURS = set(range(10, 14)) | set(range(18, 22))


class BatteryEnv:
    """Step-based battery control environment for DQN training."""

    state_dim = 16
    action_dim = 9

    def __init__(
        self,
        solar: np.ndarray,
        load: np.ndarray,
        hours: np.ndarray,
        import_rates: np.ndarray,
        export_rate: float,
        battery_capacity: float,
        max_charge_rate: float,
        max_discharge_rate: float,
        efficiency: float,
        initial_soc: float,
        min_soc_frac: float,
        dt: float,
        solar_forecast_4h: np.ndarray | None = None,
        solar_forecast_12h: np.ndarray | None = None,
    ) -> None:
        self._solar = solar
        self._load = load
        self._hours = hours
        self._import_rates = import_rates
        self._export_rate = export_rate
        self._capacity = battery_capacity
        self._max_charge = max_charge_rate
        self._max_discharge = max_discharge_rate
        self._efficiency = efficiency
        self._initial_soc = initial_soc
        self._min_soc = min_soc_frac * battery_capacity
        self._dt = dt
        self._n = len(solar)

        # Precompute forecast features (avg solar in next 4h/12h windows)
        self._fc_4h = solar_forecast_4h if solar_forecast_4h is not None else self._build_lookahead(4)
        self._fc_12h = solar_forecast_12h if solar_forecast_12h is not None else self._build_lookahead(12)

        # Rolling load average (past 4h)
        self._load_avg_4h = self._build_rolling_avg(4)

        self._soc = 0.0
        self._step_idx = 0

    def _build_lookahead(self, window_hours: int) -> np.ndarray:
        steps = int(window_hours / self._dt)
        result = np.zeros(self._n, dtype=np.float32)
        for i in range(self._n):
            end = min(i + steps, self._n)
            if end > i:
                result[i] = np.mean(self._solar[i:end])
        return result

    def _build_rolling_avg(self, window_hours: int) -> np.ndarray:
        steps = int(window_hours / self._dt)
        result = np.zeros(self._n, dtype=np.float32)
        for i in range(self._n):
            start = max(0, i - steps)
            result[i] = np.mean(self._load[start:i + 1])
        return result

    def reset(self) -> np.ndarray:
        self._soc = self._initial_soc * self._capacity
        self._step_idx = 0
        return self._get_state()

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, dict]:
        t = self._step_idx
        action_val = ACTIONS[action_idx]
        hour = int(self._hours[t]) % 24
        solar_kw = float(self._solar[t])
        load_kw = float(self._load[t])
        rate = float(self._import_rates[t])

        # Convert action to charge/discharge power
        if action_val > 0:
            charge_power = action_val * self._max_charge
            discharge_power = 0.0
        elif action_val < 0:
            charge_power = 0.0
            discharge_power = -action_val * self._max_discharge
        else:
            charge_power = 0.0
            discharge_power = 0.0

        # Apply battery constraints
        space = self._capacity - self._soc
        max_charge_energy = min(charge_power * self._dt, space / self._efficiency)
        actual_charge = max_charge_energy / self._dt if self._dt > 0 else 0.0

        available = (self._soc - self._min_soc) * self._efficiency
        max_discharge_energy = min(discharge_power * self._dt, available)
        actual_discharge = max_discharge_energy / self._dt if self._dt > 0 else 0.0

        # Update SoC
        self._soc += actual_charge * self._dt * self._efficiency
        self._soc -= actual_discharge * self._dt / self._efficiency
        self._soc = np.clip(self._soc, self._min_soc, self._capacity)

        # Energy balance
        direct_solar = min(solar_kw, load_kw)
        remaining_load = load_kw - direct_solar - actual_discharge
        remaining_solar = solar_kw - direct_solar - actual_charge

        grid_import = max(0.0, remaining_load + actual_charge) if remaining_load > 0 else actual_charge
        grid_export = max(0.0, remaining_solar)

        # Reward = negative cost
        cost = grid_import * self._dt * rate - grid_export * self._dt * self._export_rate
        reward = -cost

        self._step_idx += 1
        done = self._step_idx >= self._n

        return self._get_state(), float(reward), done, {}

    def _get_state(self) -> np.ndarray:
        t = min(self._step_idx, self._n - 1)
        hour = int(self._hours[t]) % 24
        dow = 0  # simplified: no day-of-week in hourly training data
        month = 6  # simplified: set per-episode externally if needed

        # Hours to next valley/peak
        hours_to_valley = 0
        for h_offset in range(1, 25):
            if (hour + h_offset) % 24 in VALLEY_HOURS:
                hours_to_valley = h_offset
                break

        hours_to_peak = 0
        for h_offset in range(1, 25):
            if (hour + h_offset) % 24 in PEAK_HOURS:
                hours_to_peak = h_offset
                break

        state = np.array([
            self._soc / self._capacity,                    # 0: normalized SoC
            math.sin(2 * math.pi * hour / 24),             # 1: hour_sin
            math.cos(2 * math.pi * hour / 24),             # 2: hour_cos
            math.sin(2 * math.pi * dow / 7),               # 3: dow_sin
            math.cos(2 * math.pi * dow / 7),               # 4: dow_cos
            math.sin(2 * math.pi * month / 12),            # 5: month_sin
            math.cos(2 * math.pi * month / 12),            # 6: month_cos
            self._solar[t] / 5.0,                          # 7: normalized solar
            self._load[t] / 5.0,                           # 8: normalized load
            self._import_rates[t] / 0.2,                   # 9: normalized rate
            self._fc_4h[t] / 5.0,                          # 10: solar forecast 4h
            self._fc_12h[t] / 5.0,                         # 11: solar forecast 12h
            self._load_avg_4h[t] / 5.0,                    # 12: load avg 4h
            hours_to_valley / 24.0,                        # 13: hours to valley
            hours_to_peak / 24.0,                          # 14: hours to peak
            self._dt,                                      # 15: timestep duration
        ], dtype=np.float32)

        return state
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/domain/strategy/dqn/test_environment.py -v`
Expected: All 6 PASS

- [ ] **Step 4: Commit**

```bash
git add src/domain/strategy/dqn/ tests/domain/strategy/dqn/
git commit -m "feat: add BatteryEnv for DQN training"
```

---

### Task 2: DQN Agent — Tests + Implementation

**Files:**
- Create: `tests/domain/strategy/dqn/test_agent.py`
- Create: `src/domain/strategy/dqn/agent.py`

- [ ] **Step 1: Write DQN agent tests**

Create `tests/domain/strategy/dqn/test_agent.py`:

```python
import numpy as np
import pytest
import torch

from src.domain.strategy.dqn.agent import DqnAgent


class TestDqnAgent:
    @pytest.fixture
    def agent(self):
        return DqnAgent(state_dim=16, action_dim=9)

    def test_select_action_returns_valid_index(self, agent):
        state = np.random.randn(16).astype(np.float32)
        action = agent.select_action(state, epsilon=0.0)
        assert 0 <= action < 9

    def test_select_action_with_full_exploration(self, agent):
        state = np.random.randn(16).astype(np.float32)
        actions = [agent.select_action(state, epsilon=1.0) for _ in range(100)]
        # With epsilon=1.0, should see variety
        assert len(set(actions)) > 1

    def test_store_and_sample(self, agent):
        for _ in range(100):
            s = np.random.randn(16).astype(np.float32)
            a = np.random.randint(9)
            r = np.random.randn()
            s2 = np.random.randn(16).astype(np.float32)
            agent.store_transition(s, a, r, s2, False)
        batch = agent.sample_batch(32)
        assert batch is not None
        assert batch[0].shape == (32, 16)  # states

    def test_train_step_runs(self, agent):
        for _ in range(64):
            s = np.random.randn(16).astype(np.float32)
            a = np.random.randint(9)
            r = np.random.randn()
            s2 = np.random.randn(16).astype(np.float32)
            agent.store_transition(s, a, r, s2, False)
        loss = agent.train_step()
        assert loss is not None
        assert loss >= 0

    def test_save_and_load(self, agent, tmp_path):
        state = np.random.randn(16).astype(np.float32)
        q_before = agent.get_q_values(state)

        path = tmp_path / "test_model.pt"
        agent.save(str(path))

        agent2 = DqnAgent(state_dim=16, action_dim=9)
        agent2.load(str(path))
        q_after = agent2.get_q_values(state)

        np.testing.assert_array_almost_equal(q_before, q_after)
```

- [ ] **Step 2: Implement DQN agent**

Create `src/domain/strategy/dqn/agent.py`:

```python
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
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/domain/strategy/dqn/test_agent.py -v`
Expected: All 5 PASS

- [ ] **Step 4: Commit**

```bash
git add src/domain/strategy/dqn/agent.py tests/domain/strategy/dqn/test_agent.py
git commit -m "feat: add DQN agent with replay buffer and target network"
```

---

### Task 3: Training Script

**Files:**
- Create: `src/domain/strategy/dqn/train.py`

CLI entry point: `python -m src.domain.strategy.dqn.train`. Loads data, creates environment episodes (1 month each), trains the DQN, saves to `output_files/dqn_policy.pt`.

- [ ] **Step 1: Implement training script**

Create `src/domain/strategy/dqn/train.py`:

```python
"""Train DQN battery agent on historical data.

Usage: python -m src.domain.strategy.dqn.train [--episodes 200] [--output output_files/dqn_policy.pt]
"""

import argparse
import sys

import numpy as np
import pandas as pd
import pytz

from src.domain.strategy.dqn.agent import DqnAgent
from src.domain.strategy.dqn.environment import BatteryEnv

LOCAL_TZ = pytz.timezone("Europe/Madrid")

IMPORT_RATES_BY_HOUR = {
    0: 0.085, 1: 0.085, 2: 0.085, 3: 0.085, 4: 0.085, 5: 0.085, 6: 0.085, 7: 0.085,
    8: 0.134, 9: 0.134,
    10: 0.182, 11: 0.182, 12: 0.182, 13: 0.182,
    14: 0.134, 15: 0.134, 16: 0.134, 17: 0.134,
    18: 0.182, 19: 0.182, 20: 0.182, 21: 0.182,
    22: 0.134, 23: 0.134,
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    solar_df = pd.read_csv("data/pv_power.csv", index_col="last_changed")
    solar_df.index = pd.to_datetime(solar_df.index, utc=True).tz_convert(LOCAL_TZ)
    solar_df["state"] = pd.to_numeric(solar_df["state"], errors="coerce")
    solar_df = solar_df[["state"]].dropna().sort_index()

    load_df = pd.read_csv("data/house_consumption.csv", index_col="last_changed")
    load_df.index = pd.to_datetime(load_df.index, utc=True).tz_convert(LOCAL_TZ)
    load_df["state"] = pd.to_numeric(load_df["state"], errors="coerce")
    load_df = load_df[["state"]].dropna().sort_index()

    return solar_df, load_df


def build_monthly_episodes(
    solar_df: pd.DataFrame, load_df: pd.DataFrame,
) -> list[dict]:
    """Resample to hourly and split into monthly episodes."""
    solar_h = solar_df.resample("1h").mean().fillna(0.0)
    load_h = load_df.resample("1h").mean().fillna(0.0)
    common = solar_h.index.intersection(load_h.index)
    solar_h = solar_h.loc[common]
    load_h = load_h.loc[common]

    episodes = []
    for period, group in load_h.groupby(load_h.index.to_period("M")):
        month_idx = group.index
        solar_month = solar_h.loc[month_idx]
        if len(month_idx) < 24:
            continue
        episodes.append({
            "solar": (solar_month["state"].values / 1000.0).astype(np.float32),
            "load": (group["state"].values / 1000.0).astype(np.float32),
            "hours": np.array([ts.hour for ts in month_idx], dtype=np.int32),
            "import_rates": np.array(
                [IMPORT_RATES_BY_HOUR[ts.hour] for ts in month_idx], dtype=np.float32
            ),
            "month": month_idx[0].month,
        })

    return episodes


def train(episodes: int = 200, output: str = "output_files/dqn_policy.pt") -> None:
    print("Loading data...")
    solar_df, load_df = load_data()
    episode_data = build_monthly_episodes(solar_df, load_df)
    print(f"Built {len(episode_data)} monthly episodes")

    agent = DqnAgent(state_dim=16, action_dim=9, device="cpu")

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = (epsilon - epsilon_min) / min(episodes, 100)
    best_reward = -float("inf")

    for ep in range(episodes):
        ep_data = episode_data[ep % len(episode_data)]
        env = BatteryEnv(
            solar=ep_data["solar"], load=ep_data["load"],
            hours=ep_data["hours"], import_rates=ep_data["import_rates"],
            export_rate=0.08, battery_capacity=15.0,
            max_charge_rate=4.8, max_discharge_rate=4.8,
            efficiency=0.95, initial_soc=0.1, min_soc_frac=0.1, dt=1.0,
        )

        state = env.reset()
        total_reward = 0.0
        total_loss = 0.0
        loss_count = 0

        while True:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)

            loss = agent.train_step()
            if loss is not None:
                total_loss += loss
                loss_count += 1

            total_reward += reward
            state = next_state
            if done:
                break

        epsilon = max(epsilon_min, epsilon - epsilon_decay)
        avg_loss = total_loss / max(loss_count, 1)

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(output)

        if (ep + 1) % 10 == 0:
            print(
                f"Episode {ep+1}/{episodes} | "
                f"Reward: {total_reward:.2f} | "
                f"Best: {best_reward:.2f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    print(f"\nTraining complete. Best reward: {best_reward:.2f}")
    print(f"Model saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN battery agent")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--output", type=str, default="output_files/dqn_policy.pt")
    args = parser.parse_args()
    train(episodes=args.episodes, output=args.output)
```

- [ ] **Step 2: Smoke test training (5 episodes)**

Run: `python -m src.domain.strategy.dqn.train --episodes 5 --output /tmp/test_dqn.pt`
Expected: Prints 1 progress line, creates model file

- [ ] **Step 3: Commit**

```bash
git add src/domain/strategy/dqn/train.py
git commit -m "feat: add DQN training script"
```

---

### Task 4: DqnStrategy Wrapper — Tests + Implementation

**Files:**
- Create: `tests/domain/strategy/dqn/test_dqn_strategy.py`
- Create: `src/domain/strategy/dqn_strategy.py`

- [ ] **Step 1: Write DqnStrategy tests**

Create `tests/domain/strategy/dqn/test_dqn_strategy.py`:

```python
import numpy as np
import pytest
import pytz
import pandas as pd

from src.domain.battery.models import Battery
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import PowerTariff, Rate, EnergyDirection
from src.domain.strategy.model import EnergyFlow
from src.domain.strategy.dqn.agent import DqnAgent
from src.domain.strategy.dqn_strategy import DqnStrategy

LOCAL_TZ = pytz.timezone("Europe/Madrid")


class TestDqnStrategy:
    @pytest.fixture
    def tariff(self):
        schedule = {
            (0, 8): Rate(price=0.085, energy_direction=EnergyDirection.IMPORT),
            (8, 10): Rate(price=0.134, energy_direction=EnergyDirection.IMPORT),
            (10, 14): Rate(price=0.182, energy_direction=EnergyDirection.IMPORT),
            (14, 18): Rate(price=0.134, energy_direction=EnergyDirection.IMPORT),
            (18, 22): Rate(price=0.182, energy_direction=EnergyDirection.IMPORT),
            (22, 24): Rate(price=0.134, energy_direction=EnergyDirection.IMPORT),
            (0, 24): Rate(price=0.08, energy_direction=EnergyDirection.EXPORT),
        }
        return PowerTariff(rate_schedule=schedule)

    @pytest.fixture
    def battery(self):
        return Battery(capacity=15.0, max_charge_rate=4.8, max_discharge_rate=4.8, efficiency=0.95)

    @pytest.fixture
    def grid(self):
        return Grid(max_import=5.0, max_export=5.0)

    @pytest.fixture
    def agent(self):
        return DqnAgent(state_dim=16, action_dim=9)

    def test_returns_valid_flow(self, battery, grid, tariff, agent):
        strategy = DqnStrategy(battery=battery, grid=grid, tariff=tariff, agent=agent)
        ts = pd.Timestamp("2024-06-15 10:00", tz=LOCAL_TZ)
        strategy.set_timestamp(ts)
        flows = strategy.calculate_energy_flows(3.0, 1.5, 10, 1.0 / 60.0)
        assert isinstance(flows, EnergyFlow)

    def test_with_saved_model(self, battery, grid, tariff, agent, tmp_path):
        path = tmp_path / "test.pt"
        agent.save(str(path))

        loaded_agent = DqnAgent(state_dim=16, action_dim=9)
        loaded_agent.load(str(path))

        strategy = DqnStrategy(battery=battery, grid=grid, tariff=tariff, agent=loaded_agent)
        ts = pd.Timestamp("2024-06-15 10:00", tz=LOCAL_TZ)
        strategy.set_timestamp(ts)
        flows = strategy.calculate_energy_flows(3.0, 1.5, 10, 1.0 / 60.0)
        assert isinstance(flows, EnergyFlow)
```

- [ ] **Step 2: Implement DqnStrategy**

Create `src/domain/strategy/dqn_strategy.py`:

```python
"""DQN strategy: learned policy for battery control."""

import math

import numpy as np
import pandas as pd

from src.domain.battery.models import Battery
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import PowerTariff
from src.domain.strategy.dqn.agent import DqnAgent
from src.domain.strategy.dqn.environment import ACTIONS, PEAK_HOURS, VALLEY_HOURS
from src.domain.strategy.model import BaseEnergyStrategy, EnergyFlow


class DqnStrategy(BaseEnergyStrategy):
    """Strategy that uses a trained DQN policy for charge/discharge decisions."""

    def __init__(
        self,
        battery: Battery,
        grid: Grid,
        tariff: PowerTariff,
        agent: DqnAgent,
    ) -> None:
        super().__init__(battery, grid, tariff)
        self._agent = agent
        self._current_ts: pd.Timestamp | None = None
        self._load_avg_buffer: list[float] = []

    def set_timestamp(self, ts: pd.Timestamp) -> None:
        self._current_ts = ts

    def _build_state(self, solar_power: float, load_power: float, hour: int, duration: float) -> np.ndarray:
        ts = self._current_ts
        dow = ts.weekday() if ts is not None else 0
        month = ts.month if ts is not None else 6
        rate = self.tariff.get_import_rate(hour % 24)

        self._load_avg_buffer.append(load_power)
        if len(self._load_avg_buffer) > 240:  # ~4h at 1-min resolution
            self._load_avg_buffer = self._load_avg_buffer[-240:]
        load_avg_4h = np.mean(self._load_avg_buffer) if self._load_avg_buffer else 0.0

        hours_to_valley = 0
        for h_offset in range(1, 25):
            if (hour + h_offset) % 24 in VALLEY_HOURS:
                hours_to_valley = h_offset
                break

        hours_to_peak = 0
        for h_offset in range(1, 25):
            if (hour + h_offset) % 24 in PEAK_HOURS:
                hours_to_peak = h_offset
                break

        return np.array([
            self.battery.current_charge / self.battery.capacity,
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * dow / 7),
            math.cos(2 * math.pi * dow / 7),
            math.sin(2 * math.pi * month / 12),
            math.cos(2 * math.pi * month / 12),
            solar_power / 5.0,
            load_power / 5.0,
            rate / 0.2,
            solar_power / 5.0,      # solar_forecast_4h (use current as proxy)
            solar_power / 5.0,      # solar_forecast_12h (use current as proxy)
            load_avg_4h / 5.0,
            hours_to_valley / 24.0,
            hours_to_peak / 24.0,
            duration,
        ], dtype=np.float32)

    def calculate_energy_flows(
        self, solar_power: float, load_power: float, hour: int, duration: float,
    ) -> EnergyFlow:
        if duration == 0:
            raise ZeroDivisionError("Duration cannot be zero")

        state = self._build_state(solar_power, load_power, hour, duration)
        action_idx = self._agent.select_action(state, epsilon=0.0)
        action_val = float(ACTIONS[action_idx])

        flows = EnergyFlow()
        solar_energy = solar_power * duration
        load_energy = load_power * duration
        flows.direct_solar = min(solar_energy, load_energy)

        if action_val > 0:
            charge_power = action_val * self.battery.max_charge_rate
            actual = float(self.battery.charge(charge_power, duration))
            flows.battery_charge = actual
        elif action_val < 0:
            discharge_power = -action_val * self.battery.max_discharge_rate
            actual = float(self.battery.discharge(discharge_power, duration))
            flows.battery_discharge = actual

        remaining_load = load_energy - flows.direct_solar - flows.battery_discharge * duration
        if remaining_load > 1e-6:
            imported = float(self.grid.import_power(remaining_load / duration, duration))
            flows.grid_import += imported

        # Grid import for battery charging
        if flows.battery_charge > 0:
            charge_from_solar = min(solar_energy - flows.direct_solar, flows.battery_charge * duration)
            charge_from_grid = flows.battery_charge * duration - max(0, charge_from_solar)
            if charge_from_grid > 1e-6:
                imported = float(self.grid.import_power(charge_from_grid / duration, duration))
                flows.grid_import += imported

        remaining_solar = solar_energy - flows.direct_solar - flows.battery_charge * duration
        if remaining_solar > 1e-6:
            exported = float(self.grid.export_power(remaining_solar / duration, duration))
            flows.grid_export = exported

        return flows
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/domain/strategy/dqn/ -v`
Expected: All 13 PASS

- [ ] **Step 4: Commit**

```bash
git add src/domain/strategy/dqn_strategy.py tests/domain/strategy/dqn/test_dqn_strategy.py
git commit -m "feat: add DqnStrategy wrapper for trained DQN policy"
```

---

### Task 5: Register DQN + Train + Run

**Files:**
- Modify: `web/backend/services/simulation_service.py`

- [ ] **Step 1: Register DQN in simulation service**

Add import:
```python
from src.domain.strategy.dqn.agent import DqnAgent
from src.domain.strategy.dqn_strategy import DqnStrategy
```

Add to STRATEGY_REGISTRY:
```python
    StrategyInfo(
        id="dqn_agent",
        name="DQN Agent",
        description="Deep Q-Network reinforcement learning agent trained on historical data. "
        "Uses learned policy for charge/discharge decisions.",
    ),
```

Add to _STRATEGY_CLASSES:
```python
    "dqn_agent": DqnStrategy,
```

Add DQN branch in _run_strategy:
```python
            elif strategy_id == "dqn_agent":
                agent = DqnAgent(state_dim=16, action_dim=9)
                model_path = "output_files/dqn_policy.pt"
                import os
                if os.path.exists(model_path):
                    agent.load(model_path)
                strategy = strategy_cls(battery, grid, tariff, agent=agent)
```

Add `set_timestamp` support (already wired from MPC task).

- [ ] **Step 2: Train the model (200 episodes)**

Run: `python -m src.domain.strategy.dqn.train --episodes 200`

- [ ] **Step 3: Run DQN vs MPC vs Smart Discharge via API**

```bash
curl -s -X POST http://127.0.0.1:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"strategies": ["dqn_agent", "mpc", "smart_discharge"]}'
```

- [ ] **Step 4: Commit**

```bash
git add web/backend/services/simulation_service.py
git commit -m "feat: register DQN agent strategy in simulation service"
```

---

### Task 6: Lint + Full Test Suite

- [ ] **Step 1: Run linter and fix issues**

Run: `make lint`

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -v`

- [ ] **Step 3: Commit fixes**

```bash
git add -u
git commit -m "style: fix lint issues in DQN strategy"
```
