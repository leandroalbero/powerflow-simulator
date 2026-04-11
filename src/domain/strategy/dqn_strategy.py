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
