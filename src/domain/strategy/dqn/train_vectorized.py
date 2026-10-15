"""Vectorized DQN training — run N environments in parallel to saturate GPU.

Usage: python3 -u -m src.domain.strategy.dqn.train_vectorized --device cuda:0
"""

import argparse
import random
from collections import deque
from typing import Optional, cast

import numpy as np
import pandas as pd
import pytz
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.store import read_entity

LOCAL_TZ = pytz.timezone("Europe/Madrid")

IMPORT_RATES_BY_HOUR = {
    0: 0.085, 1: 0.085, 2: 0.085, 3: 0.085, 4: 0.085, 5: 0.085, 6: 0.085, 7: 0.085,
    8: 0.134, 9: 0.134, 10: 0.182, 11: 0.182, 12: 0.182, 13: 0.182,
    14: 0.134, 15: 0.134, 16: 0.134, 17: 0.134, 18: 0.182, 19: 0.182, 20: 0.182, 21: 0.182,
    22: 0.134, 23: 0.134,
}


def _power_frame(key: str) -> pd.DataFrame:
    frame = read_entity(key)[["mean"]].rename(columns={"mean": "state"})
    frame.index = frame.index.tz_convert(LOCAL_TZ)
    return frame.dropna().sort_index()


def load_episodes() -> list[dict[str, np.ndarray]]:
    solar_df = _power_frame("pv")
    load_df = _power_frame("load")

    solar_h = solar_df.resample("1h").mean().fillna(0.0)
    load_h = load_df.resample("1h").mean().fillna(0.0)
    common = solar_h.index.intersection(load_h.index)

    episodes = []
    for _, group in load_h.loc[common].groupby(load_h.loc[common].index.to_period("M")):
        idx = group.index
        s = solar_h.loc[idx]
        if len(idx) < 24:
            continue
        episodes.append({
            "solar": (np.asarray(s["state"], dtype=float) / 1000.0).astype(np.float32),
            "load": (np.asarray(group["state"], dtype=float) / 1000.0).astype(np.float32),
            "hours": np.array([t.hour for t in idx], dtype=np.int32),
            "rates": np.array([IMPORT_RATES_BY_HOUR[t.hour] for t in idx], dtype=np.float32),
        })
    return episodes


class VectorizedBatteryEnv:
    """Run N battery environments in parallel using numpy arrays."""

    ACTIONS = np.linspace(-1.0, 1.0, 9, dtype=np.float32)
    STATE_DIM = 16
    ACTION_DIM = 9

    def __init__(self, n_envs: int, episodes: list[dict[str, np.ndarray]], capacity: float = 15.0,
                 max_charge: float = 4.8, max_discharge: float = 4.8, eff: float = 0.95,
                 min_soc_frac: float = 0.1) -> None:
        self.n = n_envs
        self.episodes = episodes
        self.cap = capacity
        self.max_ch = max_charge
        self.max_dis = max_discharge
        self.eff = eff
        self.min_soc = min_soc_frac * capacity

        # Per-env state
        self.soc = np.zeros(n_envs, dtype=np.float32)
        self.step_idx = np.zeros(n_envs, dtype=np.int32)
        self.ep_data: list[Optional[dict[str, np.ndarray]]] = [None] * n_envs
        self.ep_len = np.zeros(n_envs, dtype=np.int32)

    def reset_all(self) -> np.ndarray:
        for i in range(self.n):
            self._reset_env(i)
        return self._get_states()

    def _reset_env(self, i: int) -> None:
        ep = random.choice(self.episodes)
        self.ep_data[i] = ep
        self.ep_len[i] = len(ep["solar"])
        self.soc[i] = random.uniform(0.1, 0.5) * self.cap
        self.step_idx[i] = 0

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step all envs. actions: (n_envs,) int array of action indices."""
        rewards = np.zeros(self.n, dtype=np.float32)
        dones = np.zeros(self.n, dtype=bool)

        for i in range(self.n):
            t = self.step_idx[i]
            if t >= self.ep_len[i]:
                dones[i] = True
                self._reset_env(i)
                continue

            ep = self.ep_data[i]
            assert ep is not None
            action_val = self.ACTIONS[actions[i]]
            solar = ep["solar"][t]
            load = ep["load"][t]
            rate = ep["rates"][t]

            # Charge/discharge
            if action_val > 0:
                ch = action_val * self.max_ch
                space = self.cap - self.soc[i]
                actual_ch = min(ch, space / self.eff)
                self.soc[i] += actual_ch * self.eff
                dis = 0.0
            elif action_val < 0:
                dis = -action_val * self.max_dis
                avail = (self.soc[i] - self.min_soc) * self.eff
                actual_dis = min(dis, max(0, avail))
                self.soc[i] -= actual_dis / self.eff
                dis = actual_dis
                actual_ch = 0.0
            else:
                actual_ch = 0.0
                dis = 0.0

            self.soc[i] = np.clip(self.soc[i], self.min_soc, self.cap)

            # Energy balance
            direct = min(solar, load)
            remaining_load = load - direct - dis
            grid_import = max(0, remaining_load) + actual_ch
            grid_export = max(0, solar - direct - actual_ch)

            cost = grid_import * rate - grid_export * 0.08
            rewards[i] = -cost

            self.step_idx[i] += 1
            if self.step_idx[i] >= self.ep_len[i]:
                dones[i] = True

        states = self._get_states()

        # Auto-reset done envs
        for i in range(self.n):
            if dones[i]:
                self._reset_env(i)

        return states, rewards, dones

    def _get_states(self) -> np.ndarray:
        import math
        states = np.zeros((self.n, self.STATE_DIM), dtype=np.float32)
        for i in range(self.n):
            t = min(self.step_idx[i], self.ep_len[i] - 1)
            ep = self.ep_data[i]
            assert ep is not None
            h = int(ep["hours"][t]) % 24
            states[i, 0] = self.soc[i] / self.cap
            states[i, 1] = math.sin(2 * math.pi * h / 24)
            states[i, 2] = math.cos(2 * math.pi * h / 24)
            states[i, 7] = ep["solar"][t] / 5.0
            states[i, 8] = ep["load"][t] / 5.0
            states[i, 9] = ep["rates"][t] / 0.2
            # Simplified: skip dow/month/forecast features for speed
        return states


class BigQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int, layers: int) -> None:
        super().__init__()
        mods = []
        d = state_dim
        for _ in range(layers):
            mods.extend([nn.Linear(d, hidden), nn.ReLU()])
            d = hidden
        mods.append(nn.Linear(d, action_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))


def train(device: str, episodes: list[dict[str, np.ndarray]], n_envs: int, total_steps: int, hidden: int,
          layers: int, lr: float, batch_size: int, name: str, output: str) -> float:
    print(f"\n[{name}] {n_envs} parallel envs, {hidden}x{layers} net, "
          f"batch={batch_size}, lr={lr}, device={device}", flush=True)

    dev = torch.device(device)
    policy = BigQNetwork(16, 9, hidden, layers).to(dev)
    target = BigQNetwork(16, 9, hidden, layers).to(dev)
    target.load_state_dict(policy.state_dict())
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=2_000_000)
    env = VectorizedBatteryEnv(n_envs, episodes)
    states = env.reset_all()

    epsilon = 1.0
    eps_decay = 0.9999
    eps_min = 0.05
    best_avg = -float("inf")
    ep_rewards = np.zeros(n_envs, dtype=np.float32)
    finished_rewards: deque[float] = deque(maxlen=100)
    step = 0

    while step < total_steps:
        # Select actions
        if random.random() < epsilon:
            actions = np.random.randint(0, 9, size=n_envs)
        else:
            with torch.no_grad():
                s_t = torch.FloatTensor(states).to(dev)
                q = policy(s_t).cpu().numpy()
                actions = np.argmax(q, axis=1)

        next_states, rewards, dones = env.step(actions)

        # Store transitions (batch)
        for i in range(n_envs):
            buffer.append((states[i], actions[i], rewards[i] * 10.0, next_states[i], dones[i]))
            ep_rewards[i] += rewards[i]
            if dones[i]:
                finished_rewards.append(ep_rewards[i])
                ep_rewards[i] = 0.0

        states = next_states
        step += n_envs

        # Train on GPU
        if len(buffer) >= batch_size:
            batch = random.sample(list(buffer), batch_size)
            s, a, r, ns, d = zip(*batch)
            s_t = torch.FloatTensor(np.array(s)).to(dev)
            a_t = torch.LongTensor(np.array(a)).to(dev)
            r_t = torch.FloatTensor(np.array(r)).to(dev)
            ns_t = torch.FloatTensor(np.array(ns)).to(dev)
            d_t = torch.FloatTensor(np.array(d, dtype=np.float32)).to(dev)

            q_vals = policy(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target(ns_t).max(1)[0]
                targets = r_t + 0.99 * next_q * (1 - d_t)
            loss = nn.MSELoss()(q_vals, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target
        if step % (1000 * n_envs) == 0:
            target.load_state_dict(policy.state_dict())

        epsilon = max(eps_min, epsilon * eps_decay)

        # Log
        if step % (5000 * n_envs) == 0 and len(finished_rewards) > 0:
            avg = float(np.mean(list(finished_rewards)))
            if avg > best_avg:
                best_avg = avg
                torch.save(policy.state_dict(), output)
            print(f"[{name}] step={step:,} | eps={epsilon:.3f} | "
                  f"avg100={avg:.1f} | best={best_avg:.1f} | "
                  f"buf={len(buffer):,}", flush=True)

    torch.save(policy.state_dict(), output.replace(".pt", "_final.pt"))
    print(f"[{name}] DONE. Best avg: {best_avg:.1f}", flush=True)
    return best_avg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print(f"Loading data on {args.device}...", flush=True)
    episodes = load_episodes()
    print(f"Built {len(episodes)} episodes", flush=True)

    gpu_tag = args.device.replace(":", "")

    if "0" in gpu_tag:
        # RTX 3090 (24GB) + strong CPU: 256-512 parallel envs, big batches
        configs = [
            # name, n_envs, total_steps, hidden, layers, lr, batch_size
            ("vec_512x3_256env", 256, 5_000_000, 512, 3, 5e-4, 16384),
            ("vec_1024x2_256env", 256, 5_000_000, 1024, 2, 3e-4, 16384),
            ("vec_256x4_512env", 512, 8_000_000, 256, 4, 1e-4, 8192),
        ]
    else:
        # RTX 2060 (6GB): 128-256 parallel envs
        configs = [
            ("vec_256x3_128env", 128, 5_000_000, 256, 3, 5e-4, 4096),
            ("vec_128x2_256env", 256, 8_000_000, 128, 2, 1e-3, 4096),
            ("vec_384x2_128env", 128, 5_000_000, 384, 2, 5e-4, 4096),
        ]

    results = {}
    for name, n_envs, total_steps, hidden, layers, lr, batch_size in configs:
        output = f"output_files/dqn_{name}_{gpu_tag}.pt"
        best = train(args.device, episodes, n_envs, total_steps, hidden, layers, lr, batch_size, name, output)
        results[name] = best

    print(f"\n{'='*60}\nRESULTS on {args.device}:", flush=True)
    for name, best in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:30s} best_avg={best:.1f}", flush=True)


if __name__ == "__main__":
    main()
