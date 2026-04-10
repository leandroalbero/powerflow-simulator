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
