import math
import numpy as np
from typing import List, Tuple
from src.domain.strategy.model import BaseEnergyStrategy, EnergyFlow
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class BatteryEnvironment:
    """Environment for the battery control problem"""

    def __init__(self, battery, grid, tariff):
        self.battery = battery
        self.grid = grid
        self.tariff = tariff

        # Normalization constants based on actual data ranges
        self.MAX_SOLAR_POWER = 5.0  # kW
        self.MAX_LOAD_POWER = 5.0  # kW
        self.MAX_RATE = 0.2  # €/kWh

        # Track moving averages for reward normalization
        self.cost_ema = 0  # Exponential moving average of costs
        self.alpha = 0.01  # EMA decay factor

    def get_state(self, hour: int, solar_power: float, load_power: float) -> np.ndarray:
        """Create state representation"""
        battery_level = self.battery.current_charge / self.battery.capacity
        import_rate = self.tariff.get_rate(hour % 24, EnergyDirection.IMPORT).price
        export_rate = self.tariff.get_rate(hour % 24, EnergyDirection.EXPORT).price

        return np.array([
            np.sin(2 * np.pi * hour / 24),  # Cyclical hour representation
            np.cos(2 * np.pi * hour / 24),  # Cyclical hour representation
            battery_level,
            solar_power / self.MAX_SOLAR_POWER,
            load_power / self.MAX_LOAD_POWER,
            import_rate / self.MAX_RATE,
            export_rate / self.MAX_RATE,
        ], dtype=np.float32)

    def step(self, action: float, solar_power: float, load_power: float,
             hour: int, duration: float) -> float:
        """Execute action and return reward"""
        if duration == 0:
            raise ZeroDivisionError("Duration cannot be zero")

        flows = EnergyFlow()

        # Calculate energy values
        solar_energy = solar_power * duration
        load_energy = load_power * duration

        # Direct solar consumption
        flows.direct_solar = min(solar_energy, load_energy)
        flows.remaining_solar = solar_energy - flows.direct_solar
        flows.remaining_load = load_energy - flows.direct_solar

        # Battery charging/discharging based on action
        if action > 0:  # Charge
            max_charge = min(
                action * self.battery.max_charge_rate,
                self.battery.max_charge_rate,
                (self.battery.capacity - self.battery.current_charge) / duration
            )
            # First try to charge from remaining solar
            if flows.remaining_solar > 0:
                solar_charge = float(self.battery.charge(
                    min(max_charge, flows.remaining_solar / duration),
                    duration
                ))
                flows.battery_charge = solar_charge
                flows.remaining_solar -= solar_charge * duration

            # If we still want to charge more, use grid
            remaining_charge = max_charge - flows.battery_charge
            if remaining_charge > 0:
                grid_charge = float(self.battery.charge(remaining_charge, duration))
                flows.battery_charge += grid_charge
                flows.grid_import += grid_charge

        elif action < 0:  # Discharge
            max_discharge = min(
                -action * self.battery.max_discharge_rate,
                self.battery.max_discharge_rate,
                self.battery.current_charge / duration
            )
            discharged = float(self.battery.discharge(max_discharge, duration))
            flows.battery_discharge = discharged
            flows.remaining_load -= discharged * duration

        # Handle remaining solar/load
        if flows.remaining_solar > 0:
            exported = float(self.grid.export_power(flows.remaining_solar / duration, duration))
            flows.grid_export = exported
            flows.remaining_solar -= exported * duration

        if flows.remaining_load > 0:
            imported = float(self.grid.import_power(flows.remaining_load / duration, duration))
            flows.grid_import += imported
            flows.remaining_load -= imported * duration

            # Calculate reward components with careful scaling
        import_cost = flows.grid_import * duration * self.tariff.get_import_rate(hour)
        export_revenue = flows.grid_export * duration * self.tariff.get_export_rate(hour)
        net_cost = import_cost - export_revenue

        # Update cost EMA for normalization
        self.cost_ema = (1 - self.alpha) * self.cost_ema + self.alpha * abs(net_cost)
        normalized_cost = net_cost / (self.cost_ema + 1e-6)  # Avoid division by zero

        # Calculate additional reward components
        solar_utilization = flows.direct_solar / (solar_energy + 1e-6)
        battery_efficiency = (flows.battery_discharge * duration) / (flows.battery_charge * duration + 1e-6)
        peak_hour_penalty = 1.0 if self.tariff.get_import_rate(hour) > 0.15 else 0.0

        # Immediate reward for good actions
        reward_components = {
            'cost': -normalized_cost * 2.0,  # Negative cost is positive reward
            'solar': solar_utilization * 0.5,  # Reward for using solar directly
            'battery': battery_efficiency * 0.3,  # Reward for efficient battery use
            'peak_saving': -flows.grid_import * peak_hour_penalty * 0.4  # Penalty for peak imports
        }

        # Combine rewards with priority on cost reduction
        reward = sum(reward_components.values())

        return reward


class DQN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DQN, self).__init__()

        # Using layer normalization instead of batch normalization
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        # Initialize weights using Xavier initialization
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class RLBatteryStrategy(BaseEnergyStrategy):
    def __init__(self, battery, grid, tariff):
        super().__init__(battery, grid, tariff)
        self.env = BatteryEnvironment(battery, grid, tariff)

        # RL components
        self.state_size = 7  # [sin(hour), cos(hour), battery_level, solar_power, load_power, import_rate, export_rate]
        self.action_size = 11  # Discretize actions from -1 to 1
        self.hidden_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.hidden_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Improved training parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0003)
        self.memory = ReplayBuffer(50000)  # Larger replay buffer
        self.batch_size = 64
        self.gamma = 0.95  # Slightly lower discount factor for more immediate rewards

        # Epsilon decay for better exploration
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 5000  # Steps for decay
        self.steps_done = 0

    def get_epsilon(self):
        """Calculate current epsilon value with decay"""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        return epsilon

    def calculate_energy_flows(self, solar_power: float, load_power: float,
                               hour: int, duration: float) -> EnergyFlow:
        if duration == 0:
            raise ZeroDivisionError("Duration cannot be zero")

        state = self.env.get_state(hour, solar_power, load_power)

        # Use decaying epsilon for exploration
        epsilon = self.get_epsilon()

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action_idx = random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.max(1)[1].item()

        # Convert discrete action to continuous
        action = (action_idx / (self.action_size - 1)) * 2 - 1

        # Initialize flows
        flows = self._calculate_initial_flows(solar_power * duration, load_power * duration)

        # Handle battery charging/discharging based on action
        if action > 0:  # Charge
            max_charge = min(
                action * self.battery.max_charge_rate,
                self.battery.max_charge_rate,
                (self.battery.capacity - self.battery.current_charge) / duration
            )
            # First try to charge from remaining solar
            if flows.remaining_solar > 0:
                solar_charge = float(self.battery.charge(
                    min(max_charge, flows.remaining_solar / duration),
                    duration
                ))
                flows.battery_charge = solar_charge
                flows.remaining_solar -= solar_charge * duration

            # If we still want to charge more, use grid
            remaining_charge = max_charge - flows.battery_charge
            if remaining_charge > 0:
                grid_charge = float(self.battery.charge(remaining_charge, duration))
                flows.battery_charge += grid_charge
                flows.grid_import += grid_charge

        elif action < 0:  # Discharge
            max_discharge = min(
                -action * self.battery.max_discharge_rate,
                self.battery.max_discharge_rate,
                self.battery.current_charge / duration
            )
            discharged = float(self.battery.discharge(max_discharge, duration))
            flows.battery_discharge = discharged
            flows.remaining_load -= discharged * duration

        # Handle remaining solar/load
        if flows.remaining_solar > 0:
            exported = float(self.grid.export_power(flows.remaining_solar / duration, duration))
            flows.grid_export = exported
            flows.remaining_solar -= exported * duration

        if flows.remaining_load > 0:
            imported = float(self.grid.import_power(flows.remaining_load / duration, duration))
            flows.grid_import += imported
            flows.remaining_load -= imported * duration

        # Calculate reward (negative cost)
        cost = (flows.grid_import * duration * self.tariff.get_import_rate(hour) -
                flows.grid_export * duration * self.tariff.get_export_rate(hour))
        reward = -cost

        # Store transition in replay buffer
        next_state = self.env.get_state(hour, solar_power, load_power)
        self.memory.push(state, action_idx, reward, next_state)

        # Train if enough samples
        if len(self.memory) >= self.batch_size:
            self.train_step()

        return flows

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))

        # Convert to numpy arrays first for better performance
        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.device)
        action_batch = torch.LongTensor(np.array(batch[1])).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch[2])).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.device)

        # Compute Q(s_t, a)
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]

        # Compute expected Q values
        expected_q_values = reward_batch + self.gamma * next_q_values

        # Compute loss and optimize
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * 0.01 + target_net_state_dict[key] * 0.99
        self.target_net.load_state_dict(target_net_state_dict)

    def save_model(self, path: str):
        import os
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


from datetime import datetime, timedelta
import pandas as pd
import pytz
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.domain.battery.models import Battery
from src.domain.grid.model import Grid
from src.domain.energy_load.model import EnergyLoad
from src.domain.power_tariff.model import PowerTariff, Rate, EnergyDirection
from src.domain.solar_generator.solar_generator import SolarGenerator
from src.domain.energy_simulator.models import EnergySimulator
from src.domain.strategy.model import SelfConsumeStrategy


class TrainingManager:
    def __init__(self, start_date, end_date, solar_data, load_data, local_tz):
        self.start_date = start_date
        self.end_date = end_date
        self.solar_data = solar_data
        self.load_data = load_data
        self.local_tz = local_tz

        # Initialize components
        self.battery = Battery(capacity=5.4, max_charge_rate=2.1, max_discharge_rate=2.1)
        self.grid = Grid(max_import=5.0, max_export=5.0)
        self.tariff = PowerTariff(
            rate_schedule={
                (0, 8): Rate(price=0.085, energy_direction=EnergyDirection.IMPORT),
                (8, 10): Rate(price=0.134, energy_direction=EnergyDirection.IMPORT),
                (10, 14): Rate(price=0.182, energy_direction=EnergyDirection.IMPORT),
                (14, 18): Rate(price=0.134, energy_direction=EnergyDirection.IMPORT),
                (18, 22): Rate(price=0.182, energy_direction=EnergyDirection.IMPORT),
                (22, 24): Rate(price=0.134, energy_direction=EnergyDirection.IMPORT),
                (0, 24): Rate(price=0.08, energy_direction=EnergyDirection.EXPORT)
            }
        )

        # Set aside validation days for consistent evaluation
        self.validation_days = [15, 45, 75, 105]  # Every 30 days plus some offset

        # Initialize training metrics
        self.training_improvements = []  # Store relative improvements over baseline
        self.validation_metrics = []  # Store performance on validation days
        self.training_costs = []  # Store RL strategy costs
        self.baseline_costs = []  # Store baseline strategy costs
        self.best_improvement = -float('inf')
        self.episodes_completed = 0

    def run_episode(self, rl_strategy, day_offset):
        """Run one training episode with detailed logging"""
        episode_start = self.start_date + timedelta(days=day_offset)
        episode_end = episode_start + timedelta(days=1)

        print(f"\nAnalyzing period: {episode_start} to {episode_end}")

        # Prepare data for this episode
        episode_solar = self.solar_data[
            (self.solar_data.index >= episode_start) &
            (self.solar_data.index <= episode_end)
            ]
        episode_load = self.load_data[
            (self.load_data.index >= episode_start) &
            (self.load_data.index <= episode_end)
            ]

        print("\nData Summary:")
        print("Solar data columns:", list(episode_solar.columns))
        print("Load data columns:", list(episode_load.columns))

        total_solar = episode_solar.iloc[:, 0].sum() / 1000  # Assuming first column is power
        total_load = episode_load.iloc[:, 0].sum() / 1000  # Assuming first column is power
        print(f"Total solar generation: {total_solar:.2f} kWh")
        print(f"Total load consumption: {total_load:.2f} kWh")

        # Run RL strategy
        print("\nRunning RL Strategy:")
        self.battery.current_charge = 0.54  # Reset battery
        load = EnergyLoad(episode_load)
        solar = SolarGenerator(episode_solar)
        rl_sim = EnergySimulator(self.battery, load, self.grid, self.tariff, solar, strategy=rl_strategy)

        prev_timestamp = None
        hourly_costs_rl = []
        for timestamp in episode_load.index:
            hour = timestamp.hour
            if prev_timestamp is None:
                duration = 1.0 / 60.0
            else:
                duration = (timestamp - prev_timestamp).total_seconds() / 3600.0

            solar_power = solar.get_generation_safe(timestamp, default=0.0) / 1000.0
            load_power = load.get_load_safe(timestamp, default=0.0) / 1000.0

            if hour != (prev_timestamp.hour if prev_timestamp else -1):
                rate = self.tariff.get_rate(hour, EnergyDirection.IMPORT).price
                # print(f"\nHour {hour:02d} - Import Rate: €{rate:.3f}/kWh")
                # print(f"Solar: {solar_power:.3f} kW, Load: {load_power:.3f} kW")

            flows = rl_sim.step(timestamp, prev_timestamp)

            if hour != (prev_timestamp.hour if prev_timestamp else -1):
                hourly_costs_rl.append(rl_sim.total_cost)

            prev_timestamp = timestamp

        rl_metrics = rl_sim.get_metrics()
        print("\nRL Strategy Summary:")
        print(f"Total cost: €{rl_metrics['total_cost']:.2f}")
        print(f"Grid imported: {rl_metrics['total_grid_imported']:.2f} kWh")
        print(f"Solar generated: {rl_metrics['total_solar_generated']:.2f} kWh")
        print(f"Solar consumed: {rl_metrics['total_solar_consumed']:.2f} kWh")
        print(f"Solar exported: {rl_metrics['total_solar_exported']:.2f} kWh")
        print(
            f"Battery throughput: {rl_metrics['total_battery_in']:.2f} kWh in, {rl_metrics['total_battery_out']:.2f} kWh out")

        # Run baseline strategy
        print("\nRunning Baseline Strategy:")
        self.battery.current_charge = 0.54  # Reset battery
        baseline_strategy = SelfConsumeStrategy(self.battery, self.grid, self.tariff)
        baseline_sim = EnergySimulator(
            self.battery, load, self.grid, self.tariff, solar,
            strategy=baseline_strategy
        )

        prev_timestamp = None
        hourly_costs_baseline = []
        for timestamp in episode_load.index:
            flows = baseline_sim.step(timestamp, prev_timestamp)
            if prev_timestamp and timestamp.hour != prev_timestamp.hour:
                hourly_costs_baseline.append(baseline_sim.total_cost)
            prev_timestamp = timestamp

        baseline_metrics = baseline_sim.get_metrics()
        print("\nBaseline Strategy Summary:")
        print(f"Total cost: €{baseline_metrics['total_cost']:.2f}")
        print(f"Grid imported: {baseline_metrics['total_grid_imported']:.2f} kWh")
        print(f"Solar generated: {baseline_metrics['total_solar_generated']:.2f} kWh")
        print(f"Solar consumed: {baseline_metrics['total_solar_consumed']:.2f} kWh")
        print(f"Solar exported: {baseline_metrics['total_solar_exported']:.2f} kWh")
        print(
            f"Battery throughput: {baseline_metrics['total_battery_in']:.2f} kWh in, {baseline_metrics['total_battery_out']:.2f} kWh out")

        # Calculate improvement
        baseline_cost = baseline_metrics['total_cost']
        rl_cost = rl_metrics['total_cost']
        improvement = ((baseline_cost - rl_cost) / abs(baseline_cost)) * 100 if baseline_cost != 0 else 0

        print(f"\nImprovement over baseline: {improvement:.1f}%")

        return rl_metrics, baseline_metrics, improvement

    def evaluate_validation_days(self, rl_strategy):
        """Evaluate the strategy on consistent validation days"""
        validation_results = []

        for day in self.validation_days:
            _, _, improvement = self.run_episode(rl_strategy, int(day))
            validation_results.append(improvement)

        avg_validation_improvement = np.mean(validation_results)
        self.validation_metrics.append(avg_validation_improvement)

        return avg_validation_improvement

    def train(self, rl_strategy, num_episodes=100, save_interval=10):
        """Train the RL strategy for specified number of episodes"""
        total_days = (self.end_date - self.start_date).days
        available_days = [d for d in range(total_days) if d not in self.validation_days]

        # Track costs for each episode
        self.training_costs = []
        self.baseline_costs = []

        progress_bar = tqdm(range(num_episodes), desc="Training Episodes")
        for episode in progress_bar:
            # Choose random day for this episode (excluding validation days)
            # Convert numpy.int64 to Python int
            start_offset = int(np.random.choice(available_days))

            # Run episode
            rl_metrics, baseline_metrics, improvement = self.run_episode(rl_strategy, start_offset)

            # Store costs
            self.training_costs.append(rl_metrics['total_cost'])
            self.baseline_costs.append(baseline_metrics['total_cost'])
            self.training_improvements.append(improvement)

            # Evaluate on validation days every 50 episodes
            if episode % 50 == 0:
                validation_improvement = self.evaluate_validation_days(rl_strategy)

                # Save model if validation performance improves
                if validation_improvement > self.best_improvement:
                    self.best_improvement = validation_improvement
                    rl_strategy.save_model('models/best_rl_strategy.pth')

            # Update progress bar
            avg_last_10 = np.mean(self.training_improvements[-10:]) if len(self.training_improvements) >= 10 else 0
            progress_bar.set_postfix({
                'Avg Improvement (Last 10)': f'{avg_last_10:.1f}%',
                'Best Validation': f'{self.best_improvement:.1f}%'
            })

            # Regular checkpoint save
            if episode % save_interval == 0:
                rl_strategy.save_model(f'models/rl_strategy_episode_{episode}.pth')

        # Print final summary
        print("\nTraining completed!")
        print(f"Initial baseline cost: €{self.baseline_costs[0]:.2f}")
        print(f"Initial RL cost: €{self.training_costs[0]:.2f}")
        print(f"Final baseline cost: €{self.baseline_costs[-1]:.2f}")
        print(f"Final RL cost: €{self.training_costs[-1]:.2f}")
        print(f"Best RL cost: €{min(self.training_costs):.2f}")
        print(
            f"Average improvement over baseline: {np.mean([(b - r) / abs(b) * 100 for b, r in zip(self.baseline_costs, self.training_costs)]):.1f}%")

        return self.training_improvements, self.validation_metrics

    def plot_training_progress(self):
        """Plot training progress and validation performance"""
        plt.figure(figsize=(15, 5))

        # Plot training improvements
        plt.subplot(1, 3, 1)
        plt.plot(self.training_improvements, label='Training Improvements')
        plt.axhline(y=0, color='r', linestyle='--', label='Baseline')
        plt.xlabel('Episode')
        plt.ylabel('Improvement over Baseline (%)')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)

        # Plot validation improvements
        plt.subplot(1, 3, 2)
        plt.plot(range(0, len(self.validation_metrics) * 50, 50),
                 self.validation_metrics,
                 label='Validation Improvements',
                 marker='o')
        plt.axhline(y=0, color='r', linestyle='--', label='Baseline')
        plt.xlabel('Episode')
        plt.ylabel('Improvement over Baseline (%)')
        plt.title('Validation Performance')
        plt.legend()
        plt.grid(True)

        # Plot moving average of improvements
        window = min(10, len(self.training_improvements))
        if window > 0:
            moving_avg = np.convolve(self.training_improvements,
                                     np.ones(window) / window,
                                     mode='valid')
            plt.subplot(1, 3, 3)
            plt.plot(moving_avg, label=f'{window}-Episode Moving Average')
            plt.axhline(y=0, color='r', linestyle='--', label='Baseline')
            plt.xlabel('Episode')
            plt.ylabel('Improvement over Baseline (%)')
            plt.title('Smoothed Training Progress')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print("\nTraining Summary:")
        print(f"Best validation improvement: {self.best_improvement:.1f}%")
        print(f"Final validation improvement: {self.validation_metrics[-1]:.1f}%")
        print(f"Average improvement (last 10): {np.mean(self.training_improvements[-10:]):.1f}%")
        print(f"Best improvement: {max(self.training_improvements):.1f}%")


def main():
    # Create necessary directories
    import os
    os.makedirs('models', exist_ok=True)

    # Load data
    solar_data = pd.read_csv(
        'data/pv_power_highres.csv',
        index_col='last_changed',
    )

    load_data = pd.read_csv(
        'data/house_consumption_highres.csv',
        index_col='last_changed',
    )

    print("\nDataset Info:")
    print("\nSolar Data:")
    print(solar_data.head())
    print("\nColumns:", list(solar_data.columns))

    print("\nLoad Data:")
    print(load_data.head())
    print("\nColumns:", list(load_data.columns))

    # Setup timezone and dates
    local_tz = pytz.timezone('Europe/Madrid')
    solar_data.index = pd.to_datetime(solar_data.index, utc=True).tz_convert(local_tz)
    load_data.index = pd.to_datetime(load_data.index, utc=True).tz_convert(local_tz)

    # Define training period - just a few days for testing
    train_start = datetime(2024, 1, 1, tzinfo=local_tz)
    train_end = datetime(2024, 1, 10, tzinfo=local_tz)

    print(f"\nAnalyzing period from {train_start} to {train_end}")

    # Initialize components
    battery = Battery(capacity=5.4, max_charge_rate=2.1, max_discharge_rate=2.1)
    grid = Grid(max_import=5.0, max_export=5.0)
    tariff = PowerTariff(
        rate_schedule={
            (0, 8): Rate(price=0.085, energy_direction=EnergyDirection.IMPORT),
            (8, 10): Rate(price=0.134, energy_direction=EnergyDirection.IMPORT),
            (10, 14): Rate(price=0.182, energy_direction=EnergyDirection.IMPORT),
            (14, 18): Rate(price=0.134, energy_direction=EnergyDirection.IMPORT),
            (18, 22): Rate(price=0.182, energy_direction=EnergyDirection.IMPORT),
            (22, 24): Rate(price=0.134, energy_direction=EnergyDirection.IMPORT),
            (0, 24): Rate(price=0.08, energy_direction=EnergyDirection.EXPORT)
        }
    )

    print("\nTariff structure:")
    for (start, end), rate in tariff.rate_schedule.items():
        if rate.energy_direction == EnergyDirection.IMPORT:
            print(f"Hours {start:02d}-{end:02d}: €{rate.price:.3f}/kWh (Import)")
    print(f"Export rate: €{tariff.rate_schedule[(0, 24)].price:.3f}/kWh")

    # Initialize RL strategy
    rl_strategy = RLBatteryStrategy(battery, grid, tariff)

    # Initialize training manager
    trainer = TrainingManager(train_start, train_end, solar_data, load_data, local_tz)

    # Train model with fewer episodes for debugging
    print("\nStarting training...")
    training_improvements, validation_metrics = trainer.train(rl_strategy, num_episodes=5000)

    # Plot results
    trainer.plot_training_progress()

    # Final evaluation - now using trainer instance variables
    print("\nTraining completed!")
    print(f"Initial baseline cost: €{trainer.baseline_costs[0]:.2f}")
    print(f"Initial RL cost: €{trainer.training_costs[0]:.2f}")
    print(f"Final baseline cost: €{trainer.baseline_costs[-1]:.2f}")
    print(f"Final RL cost: €{trainer.training_costs[-1]:.2f}")
    print(f"Best RL cost: €{min(trainer.training_costs):.2f}")
    print(
        f"Average improvement over baseline: {np.mean([(b - r) / abs(b) * 100 for b, r in zip(trainer.baseline_costs, trainer.training_costs)]):.1f}%"
    )


if __name__ == "__main__":
    main()