"""DQN strategy: learned policy for battery control."""

import math

import numpy as np

from src.domain.battery.models import Battery
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import EnergyDirection, PowerTariff, local_date_hour
from src.domain.strategy.dqn.agent import DqnAgent
from src.domain.strategy.dqn.environment import ACTIONS, PEAK_HOURS, VALLEY_HOURS
from src.domain.strategy.model import BaseEnergyStrategy, Decision


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
        self._load_avg_buffer: list[float] = []

    def _build_state(self, solar_power: float, load_power: float, duration: float) -> np.ndarray:
        ts = self.now
        if ts is not None:
            local_day, hour = local_date_hour(ts)
            dow, month = local_day.weekday(), local_day.month
            rate = self.tariff.rate_at(ts, EnergyDirection.IMPORT).price
        else:
            hour, dow, month = 0, 0, 6
            rate = self.tariff.get_import_rate(0)

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

    def _decide(self, solar_kw: float, load_kw: float, dt_h: float) -> Decision:
        state = self._build_state(solar_kw, load_kw, dt_h)
        action_val = float(ACTIONS[self._agent.select_action(state, epsilon=0.0)])
        if action_val > 0:
            return Decision(action_val * self.battery.max_charge_rate, allow_grid_charge=True)
        return Decision(action_val * self.battery.max_discharge_rate)
