"""Train multiple DQN architectures on GPU in parallel.

Usage: python3 -m src.domain.strategy.dqn.train_gpu
Runs on CUDA GPUs: architecture variants on GPU 0 (RTX 3090) and GPU 1 (RTX 2060).
"""

import os
import random
import sys
from multiprocessing import Process

import numpy as np
import pandas as pd
import pytz
import torch

from src.domain.strategy.dqn.agent import DqnAgent, QNetwork
from src.domain.strategy.dqn.environment import BatteryEnv

LOCAL_TZ = pytz.timezone("Europe/Madrid")

IMPORT_RATES_BY_HOUR = {
    0: 0.085, 1: 0.085, 2: 0.085, 3: 0.085, 4: 0.085, 5: 0.085, 6: 0.085, 7: 0.085,
    8: 0.134, 9: 0.134, 10: 0.182, 11: 0.182, 12: 0.182, 13: 0.182,
    14: 0.134, 15: 0.134, 16: 0.134, 17: 0.134, 18: 0.182, 19: 0.182, 20: 0.182, 21: 0.182,
    22: 0.134, 23: 0.134,
}


def load_episodes() -> list[dict]:
    solar_df = pd.read_csv("data/pv_power.csv", index_col="last_changed")
    solar_df.index = pd.to_datetime(solar_df.index, utc=True).tz_convert(LOCAL_TZ)
    solar_df["state"] = pd.to_numeric(solar_df["state"], errors="coerce")
    solar_df = solar_df[["state"]].dropna().sort_index()

    load_df = pd.read_csv("data/house_consumption.csv", index_col="last_changed")
    load_df.index = pd.to_datetime(load_df.index, utc=True).tz_convert(LOCAL_TZ)
    load_df["state"] = pd.to_numeric(load_df["state"], errors="coerce")
    load_df = load_df[["state"]].dropna().sort_index()

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
        })
    return episodes


def train_variant(
    name: str,
    device: str,
    episodes_data: list[dict],
    num_episodes: int,
    lr: float,
    batch_size: int,
    hidden_size: int,
    num_layers: int,
    gamma: float,
    reward_scale: float,
    train_steps_per_env: int,
    output_path: str,
) -> None:
    """Train a single DQN variant."""
    print(f"[{name}] Starting on {device}, lr={lr}, hidden={hidden_size}x{num_layers}, "
          f"gamma={gamma}, reward_scale={reward_scale}", flush=True)

    agent = DqnAgent(
        state_dim=16, action_dim=9,
        lr=lr, batch_size=batch_size,
        buffer_size=500_000,
        target_update_freq=1000,
        device=device,
    )

    # Replace network with custom architecture
    if hidden_size != 128 or num_layers != 2:
        import torch.nn as nn
        layers = []
        in_dim = 16
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_size), nn.ReLU()])
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, 9))
        agent.policy_net = nn.Sequential(*layers).to(agent.device)
        agent.target_net = nn.Sequential(*layers).to(agent.device)
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        agent.optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=lr)

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = (epsilon - epsilon_min) / min(num_episodes, 200)
    best_avg = -float("inf")
    recent: list[float] = []

    for ep in range(num_episodes):
        ep_data = random.choice(episodes_data)
        init_soc = random.uniform(0.1, 0.5)

        env = BatteryEnv(
            solar=ep_data["solar"], load=ep_data["load"],
            hours=ep_data["hours"], import_rates=ep_data["import_rates"],
            export_rate=0.08, battery_capacity=15.0,
            max_charge_rate=4.8, max_discharge_rate=4.8,
            efficiency=0.95, initial_soc=init_soc, min_soc_frac=0.1, dt=1.0,
        )

        state = env.reset()
        total_reward = 0.0

        while True:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward * reward_scale, next_state, done)

            for _ in range(train_steps_per_env):
                agent.train_step()

            total_reward += reward
            state = next_state
            if done:
                break

        epsilon = max(epsilon_min, epsilon - epsilon_decay)
        recent.append(total_reward)
        if len(recent) > 20:
            recent.pop(0)
        avg = np.mean(recent)

        if len(recent) >= 20 and avg > best_avg:
            best_avg = avg
            agent.save(output_path)

        if (ep + 1) % 50 == 0:
            print(f"[{name}] Ep {ep+1}/{num_episodes} | "
                  f"Reward: {total_reward:.1f} | Avg20: {avg:.1f} | "
                  f"Best: {best_avg:.1f} | Eps: {epsilon:.3f}", flush=True)

    agent.save(output_path.replace(".pt", "_final.pt"))
    print(f"[{name}] Done! Best avg: {best_avg:.1f}. Saved to {output_path}", flush=True)


def main() -> None:
    print("Loading data...", flush=True)
    episodes = load_episodes()
    print(f"Built {len(episodes)} episodes", flush=True)

    # Define architecture variants
    variants = [
        # GPU 0 (RTX 3090) — larger models, more episodes
        {
            "name": "large_3layer",
            "device": "cuda:0",
            "num_episodes": 2000,
            "lr": 5e-4,
            "batch_size": 256,
            "hidden_size": 256,
            "num_layers": 3,
            "gamma": 0.99,
            "reward_scale": 10.0,
            "train_steps_per_env": 4,
            "output_path": "output_files/dqn_large_3layer.pt",
        },
        {
            "name": "deep_4layer",
            "device": "cuda:0",
            "num_episodes": 2000,
            "lr": 3e-4,
            "batch_size": 256,
            "hidden_size": 128,
            "num_layers": 4,
            "gamma": 0.995,
            "reward_scale": 15.0,
            "train_steps_per_env": 8,
            "output_path": "output_files/dqn_deep_4layer.pt",
        },
        # GPU 1 (RTX 2060) — baseline and low gamma
        {
            "name": "baseline_long",
            "device": "cuda:1",
            "num_episodes": 3000,
            "lr": 1e-3,
            "batch_size": 128,
            "hidden_size": 128,
            "num_layers": 2,
            "gamma": 0.99,
            "reward_scale": 10.0,
            "train_steps_per_env": 4,
            "output_path": "output_files/dqn_baseline_long.pt",
        },
        {
            "name": "high_gamma",
            "device": "cuda:1",
            "num_episodes": 2000,
            "lr": 5e-4,
            "batch_size": 128,
            "hidden_size": 192,
            "num_layers": 2,
            "gamma": 0.999,
            "reward_scale": 20.0,
            "train_steps_per_env": 4,
            "output_path": "output_files/dqn_high_gamma.pt",
        },
    ]

    processes = []
    for v in variants:
        p = Process(target=train_variant, kwargs={**v, "episodes_data": episodes})
        p.start()
        processes.append((v["name"], p))
        print(f"Started {v['name']} on {v['device']}", flush=True)

    for name, p in processes:
        p.join()
        print(f"{name} finished with exit code {p.exitcode}", flush=True)

    print("\nAll training complete!", flush=True)


if __name__ == "__main__":
    main()
