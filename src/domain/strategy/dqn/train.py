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


REWARD_SCALE = 10.0  # Amplify reward signal (hourly costs are tiny: 0.01-0.18 EUR)


def train(episodes: int = 200, output: str = "output_files/dqn_policy.pt") -> None:
    import random

    print("Loading data...", flush=True)
    solar_df, load_df = load_data()
    episode_data = build_monthly_episodes(solar_df, load_df)
    print(f"Built {len(episode_data)} monthly episodes", flush=True)

    agent = DqnAgent(
        state_dim=16, action_dim=9, device="cpu",
        lr=1e-3, batch_size=128, buffer_size=200_000,
        target_update_freq=500,
    )

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = (epsilon - epsilon_min) / min(episodes, 150)
    best_avg_reward = -float("inf")

    # Track rolling average over last 10 episodes for stable model selection
    recent_rewards: list[float] = []

    for ep in range(episodes):
        ep_data = random.choice(episode_data)

        # Randomize initial SoC for generalization
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
        total_loss = 0.0
        loss_count = 0

        while True:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            # Scale reward for better gradient signal
            agent.store_transition(state, action, reward * REWARD_SCALE, next_state, done)

            # Multiple training steps per env step for faster learning
            for _ in range(4):
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

        # Rolling average for stable model selection
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 10:
            recent_rewards.pop(0)
        avg_recent = np.mean(recent_rewards)

        if len(recent_rewards) >= 10 and avg_recent > best_avg_reward:
            best_avg_reward = avg_recent
            agent.save(output)

        if (ep + 1) % 20 == 0:
            print(
                f"Episode {ep+1}/{episodes} | "
                f"Reward: {total_reward:.2f} | "
                f"Avg10: {avg_recent:.2f} | "
                f"Best Avg: {best_avg_reward:.2f} | "
                f"Loss: {avg_loss:.6f} | "
                f"Eps: {epsilon:.3f}",
                flush=True,
            )

    # Final save
    agent.save(output)
    print(f"\nTraining complete. Best avg reward: {best_avg_reward:.2f}", flush=True)
    print(f"Model saved to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN battery agent")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--output", type=str, default="output_files/dqn_policy.pt")
    args = parser.parse_args()
    train(episodes=args.episodes, output=args.output)
