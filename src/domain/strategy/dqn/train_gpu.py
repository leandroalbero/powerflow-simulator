"""Train DQN on GPU — sequential variants.

Usage: python3 -u -m src.domain.strategy.dqn.train_gpu [--device cuda:0]
"""

import argparse
import random
import sys

import numpy as np
import pandas as pd
import pytz
import torch
import torch.nn as nn

LOCAL_TZ = pytz.timezone("Europe/Madrid")

IMPORT_RATES_BY_HOUR = {
    0: 0.085, 1: 0.085, 2: 0.085, 3: 0.085, 4: 0.085, 5: 0.085, 6: 0.085, 7: 0.085,
    8: 0.134, 9: 0.134, 10: 0.182, 11: 0.182, 12: 0.182, 13: 0.182,
    14: 0.134, 15: 0.134, 16: 0.134, 17: 0.134, 18: 0.182, 19: 0.182, 20: 0.182, 21: 0.182,
    22: 0.134, 23: 0.134,
}


def load_episodes():
    from src.domain.strategy.dqn.environment import BatteryEnv
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

    episodes = []
    for period, group in load_h.loc[common].groupby(load_h.loc[common].index.to_period("M")):
        idx = group.index
        s = solar_h.loc[idx]
        if len(idx) < 24:
            continue
        episodes.append({
            "solar": (s["state"].values / 1000.0).astype(np.float32),
            "load": (group["state"].values / 1000.0).astype(np.float32),
            "hours": np.array([t.hour for t in idx], dtype=np.int32),
            "rates": np.array([IMPORT_RATES_BY_HOUR[t.hour] for t in idx], dtype=np.float32),
        })
    return episodes


def make_env(ep_data, init_soc=0.1):
    from src.domain.strategy.dqn.environment import BatteryEnv
    return BatteryEnv(
        solar=ep_data["solar"], load=ep_data["load"],
        hours=ep_data["hours"], import_rates=ep_data["rates"],
        export_rate=0.08, battery_capacity=15.0,
        max_charge_rate=4.8, max_discharge_rate=4.8,
        efficiency=0.95, initial_soc=init_soc, min_soc_frac=0.1, dt=1.0,
    )


def train_variant(name, device, episodes, num_ep, lr, hidden, layers, gamma, reward_scale, train_per_step, output):
    print(f"\n{'='*60}", flush=True)
    print(f"[{name}] device={device} lr={lr} hidden={hidden}x{layers} gamma={gamma} "
          f"reward_scale={reward_scale} episodes={num_ep}", flush=True)
    print(f"{'='*60}", flush=True)

    from src.domain.strategy.dqn.agent import DqnAgent

    # Large batch to saturate GPU — 4096 for 3090, 1024 for 2060
    gpu_batch = 4096 if "cuda:0" in device else 1024
    agent = DqnAgent(state_dim=16, action_dim=9, lr=lr, batch_size=gpu_batch,
                     buffer_size=1_000_000, target_update_freq=500, device=device)

    # Replace network if non-default architecture
    if hidden != 128 or layers != 2:
        net_layers = []
        in_d = 16
        for _ in range(layers):
            net_layers.extend([nn.Linear(in_d, hidden), nn.ReLU()])
            in_d = hidden
        net_layers.append(nn.Linear(in_d, 9))
        agent.policy_net = nn.Sequential(*net_layers).to(agent.device)
        agent.target_net = nn.Sequential(*net_layers).to(agent.device)
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        agent.optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=lr)

    eps = 1.0
    eps_min = 0.05
    eps_decay = (eps - eps_min) / min(num_ep, 200)
    best_avg = -float("inf")
    recent = []

    for ep in range(num_ep):
        ep_data = random.choice(episodes)
        env = make_env(ep_data, init_soc=random.uniform(0.1, 0.5))
        state = env.reset()
        total_reward = 0.0

        while True:
            action = agent.select_action(state, eps)
            ns, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward * reward_scale, ns, done)
            for _ in range(train_per_step):
                agent.train_step()
            total_reward += reward
            state = ns
            if done:
                break

        eps = max(eps_min, eps - eps_decay)
        recent.append(total_reward)
        if len(recent) > 20:
            recent.pop(0)
        avg = np.mean(recent)

        if len(recent) >= 20 and avg > best_avg:
            best_avg = avg
            agent.save(output)

        if (ep + 1) % 50 == 0:
            print(f"[{name}] Ep {ep+1}/{num_ep} | R={total_reward:.1f} | "
                  f"Avg20={avg:.1f} | Best={best_avg:.1f} | eps={eps:.3f}", flush=True)

    agent.save(output.replace(".pt", "_final.pt"))
    print(f"[{name}] DONE. Best avg: {best_avg:.1f}", flush=True)
    return best_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print("Loading data...", flush=True)
    episodes = load_episodes()
    print(f"Built {len(episodes)} episodes on {args.device}", flush=True)
    print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}", flush=True)

    gpu_tag = args.device.replace(":", "")  # e.g. "cuda0"

    # GPU 0 (RTX 3090, 24GB): big batches, big networks
    # GPU 1 (RTX 2060, 6GB): moderate batches, focused hyperparams
    if "0" in gpu_tag:
        variants = [
            # name, episodes, lr, hidden, layers, gamma, reward_scale, train_per_step
            ("big_512x3_3k", 3000, 5e-4, 512, 3, 0.99, 10.0, 32),
            ("huge_1024x2_2k", 2000, 3e-4, 1024, 2, 0.995, 15.0, 32),
            ("deep_256x5_2k", 2000, 1e-4, 256, 5, 0.999, 20.0, 32),
            ("wide_768x2_2k", 2000, 5e-4, 768, 2, 0.99, 10.0, 16),
        ]
    else:
        variants = [
            ("baseline_3k", 3000, 1e-3, 128, 2, 0.99, 10.0, 16),
            ("med_256x3_2k", 2000, 5e-4, 256, 3, 0.99, 10.0, 16),
            ("high_gamma_2k", 2000, 5e-4, 192, 2, 0.999, 20.0, 16),
            ("fast_lr_2k", 2000, 2e-3, 128, 2, 0.99, 15.0, 16),
        ]

    results = {}
    for name, num_ep, lr, hidden, layers, gamma, rs, tps in variants:
        output = f"output_files/dqn_{name}_{gpu_tag}.pt"
        best = train_variant(name, args.device, episodes, num_ep, lr, hidden, layers, gamma, rs, tps, output)
        results[name] = best

    print(f"\n{'='*60}", flush=True)
    print("RESULTS SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for name, best in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:30s} best_avg={best:.1f}", flush=True)


if __name__ == "__main__":
    main()
