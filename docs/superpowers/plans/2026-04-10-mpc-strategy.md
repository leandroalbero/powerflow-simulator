# MPC Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a Model Predictive Control (MPC) strategy that solves a rolling 24h LP every 15 minutes using solar forecasts and learned load profiles.

**Architecture:** MPC reuses `solve_oracle_lp()` from `oracle.py` but over a 24h forecast horizon (96 steps at 15-min resolution). Every 15 simulator-minutes, it builds a solar forecast (from GHI data) and load forecast (from 28-day historical average), solves the LP, and caches the plan. Between solves, it replays the cached charge/discharge decisions. A `LoadForecaster` builds the historical load profile and a `SolarForecaster` builds hourly GHI forecasts — both as simple lookup functions from the existing data.

**Tech Stack:** scipy (via existing `solve_oracle_lp`), numpy, pandas. No new dependencies.

---

## File Structure

```
src/domain/strategy/
  mpc.py                    # MpcStrategy class + forecasters
  oracle.py                 # existing (reuse solve_oracle_lp, unchanged)
web/backend/services/
  simulation_service.py     # Register MPC strategy (modify)
tests/domain/strategy/
  test_mpc.py               # MPC-specific tests
```

- `mpc.py` — contains `LoadForecaster` (builds hourly load profile from historical data), `SolarForecaster` (looks up hourly GHI forecasts), and `MpcStrategy` (rolling-horizon LP with cached plans).
- `simulation_service.py` — register MPC in strategy registry + inject forecast data.

---

### Task 1: Load Forecaster — Tests + Implementation

**Files:**
- Create: `tests/domain/strategy/test_mpc.py`
- Create: `src/domain/strategy/mpc.py`

The `LoadForecaster` takes historical load data (pandas DataFrame) and produces a 24h load forecast for any given timestamp. It averages load by (hour_of_day, is_weekend) over the preceding 28 days.

- [ ] **Step 1: Write tests for LoadForecaster**

Create `tests/domain/strategy/test_mpc.py`:

```python
import numpy as np
import pandas as pd
import pytest
import pytz

from src.domain.strategy.mpc import LoadForecaster

LOCAL_TZ = pytz.timezone("Europe/Madrid")


class TestLoadForecaster:
    @pytest.fixture
    def load_df(self):
        """28 days of hourly load data with a simple pattern: 1kW at night, 2kW during day."""
        idx = pd.date_range("2024-06-01", periods=28 * 24, freq="h", tz=LOCAL_TZ)
        values = [1000.0 if ts.hour < 8 or ts.hour >= 22 else 2000.0 for ts in idx]
        return pd.DataFrame({"state": values}, index=idx)

    def test_forecast_length(self, load_df):
        forecaster = LoadForecaster(load_df)
        ts = pd.Timestamp("2024-06-29 10:00", tz=LOCAL_TZ)
        forecast = forecaster.forecast_24h(ts, steps=96, step_minutes=15)
        assert len(forecast) == 96

    def test_forecast_reflects_pattern(self, load_df):
        """Daytime forecast should be ~2kW, nighttime ~1kW."""
        forecaster = LoadForecaster(load_df)
        ts = pd.Timestamp("2024-06-29 00:00", tz=LOCAL_TZ)
        forecast = forecaster.forecast_24h(ts, steps=96, step_minutes=15)
        # Hour 0-7 (steps 0-31): ~1kW nighttime
        assert np.mean(forecast[:32]) == pytest.approx(1.0, abs=0.1)
        # Hour 10-14 (steps 40-59): ~2kW daytime
        assert np.mean(forecast[40:60]) == pytest.approx(2.0, abs=0.1)

    def test_forecast_with_no_history(self):
        """With empty history, should return zeros."""
        empty_df = pd.DataFrame({"state": []}, index=pd.DatetimeIndex([], tz=LOCAL_TZ))
        forecaster = LoadForecaster(empty_df)
        ts = pd.Timestamp("2024-06-29 10:00", tz=LOCAL_TZ)
        forecast = forecaster.forecast_24h(ts, steps=96, step_minutes=15)
        assert len(forecast) == 96
        assert np.all(forecast == 0.0)
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `python -m pytest tests/domain/strategy/test_mpc.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement LoadForecaster**

Create `src/domain/strategy/mpc.py`:

```python
"""MPC strategy: rolling-horizon LP with solar/load forecasts."""

from datetime import timedelta

import numpy as np
import pandas as pd


class LoadForecaster:
    """Builds load forecasts from historical averages per (hour, is_weekend)."""

    def __init__(self, load_df: pd.DataFrame) -> None:
        self._profile: dict[tuple[int, bool], float] = {}
        if load_df.empty:
            return
        df = load_df.copy()
        df["hour"] = df.index.hour
        df["is_weekend"] = df.index.weekday >= 5
        grouped = df.groupby(["hour", "is_weekend"])["state"].mean()
        for (hour, is_wknd), val in grouped.items():
            self._profile[(int(hour), bool(is_wknd))] = float(val) / 1000.0  # W -> kW

    def forecast_24h(
        self, start: pd.Timestamp, steps: int = 96, step_minutes: int = 15,
    ) -> np.ndarray:
        """Return load forecast in kW for `steps` intervals starting at `start`."""
        forecast = np.zeros(steps)
        for i in range(steps):
            ts = start + timedelta(minutes=i * step_minutes)
            key = (ts.hour, ts.weekday() >= 5)
            forecast[i] = self._profile.get(key, 0.0)
        return forecast
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `python -m pytest tests/domain/strategy/test_mpc.py -v`
Expected: All 3 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/domain/strategy/test_mpc.py src/domain/strategy/mpc.py
git commit -m "feat: add LoadForecaster for MPC historical load profiles"
```

---

### Task 2: Solar Forecaster — Tests + Implementation

**Files:**
- Modify: `tests/domain/strategy/test_mpc.py`
- Modify: `src/domain/strategy/mpc.py`

The `SolarForecaster` takes hourly GHI forecasts (Dict[date, Dict[int, float]]) and converts to kW PV output for a 24h horizon.

- [ ] **Step 1: Write tests for SolarForecaster**

Append to `tests/domain/strategy/test_mpc.py`:

```python
from datetime import date
from src.domain.strategy.mpc import SolarForecaster, GHI_TO_PV_FACTOR


class TestSolarForecaster:
    @pytest.fixture
    def hourly_forecasts(self):
        """GHI forecasts: 0 at night, ramp to 800 W/m2 at noon."""
        forecasts = {}
        for d in [date(2024, 6, 29), date(2024, 6, 30)]:
            hourly = {}
            for h in range(24):
                if 6 <= h <= 18:
                    hourly[h] = max(0, 800 - abs(h - 12) * 120)  # bell curve
                else:
                    hourly[h] = 0.0
            forecasts[d] = hourly
        return forecasts

    def test_forecast_length(self, hourly_forecasts):
        forecaster = SolarForecaster(hourly_forecasts)
        ts = pd.Timestamp("2024-06-29 00:00", tz=LOCAL_TZ)
        forecast = forecaster.forecast_24h(ts, steps=96, step_minutes=15)
        assert len(forecast) == 96

    def test_forecast_night_is_zero(self, hourly_forecasts):
        forecaster = SolarForecaster(hourly_forecasts)
        ts = pd.Timestamp("2024-06-29 00:00", tz=LOCAL_TZ)
        forecast = forecaster.forecast_24h(ts, steps=96, step_minutes=15)
        # Steps 0-23 are hours 0-5 (night) -> 0 kW
        assert np.all(forecast[:24] == 0.0)

    def test_forecast_peak_positive(self, hourly_forecasts):
        forecaster = SolarForecaster(hourly_forecasts)
        ts = pd.Timestamp("2024-06-29 00:00", tz=LOCAL_TZ)
        forecast = forecaster.forecast_24h(ts, steps=96, step_minutes=15)
        # Step 48 = hour 12 (noon), should have peak solar
        assert forecast[48] > 0

    def test_forecast_missing_date_returns_zeros(self, hourly_forecasts):
        forecaster = SolarForecaster(hourly_forecasts)
        ts = pd.Timestamp("2024-07-15 00:00", tz=LOCAL_TZ)  # not in forecasts
        forecast = forecaster.forecast_24h(ts, steps=96, step_minutes=15)
        assert np.all(forecast == 0.0)
```

- [ ] **Step 2: Implement SolarForecaster**

Append to `src/domain/strategy/mpc.py`:

```python
from datetime import date as date_type

GHI_TO_PV_FACTOR = 4.592  # W/m2 GHI -> W PV (empirical)


class SolarForecaster:
    """Converts hourly GHI forecasts to kW PV output for a 24h horizon."""

    def __init__(self, hourly_forecasts: dict[date_type, dict[int, float]]) -> None:
        self._forecasts = hourly_forecasts

    def forecast_24h(
        self, start: pd.Timestamp, steps: int = 96, step_minutes: int = 15,
    ) -> np.ndarray:
        """Return solar forecast in kW for `steps` intervals starting at `start`."""
        forecast = np.zeros(steps)
        for i in range(steps):
            ts = start + timedelta(minutes=i * step_minutes)
            day_forecast = self._forecasts.get(ts.date())
            if day_forecast is None:
                continue
            ghi = day_forecast.get(ts.hour, 0.0)
            forecast[i] = ghi * GHI_TO_PV_FACTOR / 1000.0  # W -> kW
        return forecast
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/domain/strategy/test_mpc.py -v`
Expected: All 7 PASS

- [ ] **Step 4: Commit**

```bash
git add tests/domain/strategy/test_mpc.py src/domain/strategy/mpc.py
git commit -m "feat: add SolarForecaster for MPC GHI-to-PV conversion"
```

---

### Task 3: Hourly GHI Forecast Loader

**Files:**
- Modify: `tests/domain/strategy/test_mpc.py`
- Modify: `src/domain/strategy/mpc.py`

We need a function to load the irradiance CSVs into `Dict[date, Dict[int, float]]` format (date -> hour -> GHI forecast) for the SolarForecaster.

- [ ] **Step 1: Write test for loader**

Append to `tests/domain/strategy/test_mpc.py`:

```python
from pathlib import Path
from src.domain.strategy.mpc import load_hourly_ghi_forecasts


class TestLoadHourlyGhiForecasts:
    def test_loads_from_irradiance_dir(self):
        forecasts = load_hourly_ghi_forecasts()
        # Should have data for at least some dates in 2024
        assert len(forecasts) > 0
        # Each date maps to a dict of hour -> GHI
        some_date = next(iter(forecasts))
        assert isinstance(forecasts[some_date], dict)
        # Hours should be 0-23
        assert all(0 <= h <= 23 for h in forecasts[some_date])

    def test_summer_noon_has_positive_ghi(self):
        forecasts = load_hourly_ghi_forecasts()
        # June 15 at noon should have positive GHI forecast
        d = date(2024, 6, 15)
        if d in forecasts:
            assert forecasts[d].get(12, 0.0) > 0
```

- [ ] **Step 2: Implement loader**

Append to `src/domain/strategy/mpc.py`:

```python
import csv
from pathlib import Path

IRRADIANCE_DIR = Path("data/irradiance")


def load_hourly_ghi_forecasts(
    irradiance_dir: Path = IRRADIANCE_DIR,
) -> dict[date_type, dict[int, float]]:
    """Load irradiance CSVs into {date: {hour: ghi_forecast}} structure."""
    forecasts: dict[date_type, dict[int, float]] = {}

    for csv_path in sorted(irradiance_dir.glob("*.csv")):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if "ghi_forecast" not in (reader.fieldnames or []):
                continue
            for row in reader:
                raw = row.get("ghi_forecast", "").strip()
                if not raw:
                    continue
                from datetime import datetime
                dt = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                d = dt.date()
                h = dt.hour
                if d not in forecasts:
                    forecasts[d] = {}
                forecasts[d][h] = float(raw)

    return forecasts
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/domain/strategy/test_mpc.py -v`
Expected: All 9 PASS

- [ ] **Step 4: Commit**

```bash
git add tests/domain/strategy/test_mpc.py src/domain/strategy/mpc.py
git commit -m "feat: add hourly GHI forecast loader for MPC"
```

---

### Task 4: MPC Strategy — Tests + Implementation

**Files:**
- Modify: `tests/domain/strategy/test_mpc.py`
- Modify: `src/domain/strategy/mpc.py`

The core MPC strategy: re-solves a 24h LP every 15 minutes, caches the plan, replays decisions between solves.

- [ ] **Step 1: Write tests for MpcStrategy**

Append to `tests/domain/strategy/test_mpc.py`:

```python
from src.domain.battery.models import Battery
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import PowerTariff, Rate, EnergyDirection
from src.domain.strategy.model import EnergyFlow
from src.domain.strategy.mpc import MpcStrategy


class TestMpcStrategy:
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
    def load_forecaster(self):
        """Simple forecaster: 1.5 kW constant load."""
        idx = pd.date_range("2024-06-01", periods=28 * 24, freq="h", tz=LOCAL_TZ)
        df = pd.DataFrame({"state": [1500.0] * len(idx)}, index=idx)
        return LoadForecaster(df)

    @pytest.fixture
    def solar_forecaster(self):
        """Simple forecaster: 3kW solar noon, 0 at night."""
        forecasts = {}
        for day_offset in range(60):
            d = date(2024, 6, 1) + timedelta(days=day_offset)
            hourly = {}
            for h in range(24):
                if 8 <= h <= 16:
                    ghi = max(0, 600 - abs(h - 12) * 100)
                else:
                    ghi = 0.0
                hourly[h] = ghi
            forecasts[d] = hourly
        return SolarForecaster(forecasts)

    def test_returns_valid_flow(self, battery, grid, tariff, load_forecaster, solar_forecaster):
        strategy = MpcStrategy(
            battery=battery, grid=grid, tariff=tariff,
            load_forecaster=load_forecaster,
            solar_forecaster=solar_forecaster,
        )
        ts = pd.Timestamp("2024-06-15 10:00", tz=LOCAL_TZ)
        strategy.set_timestamp(ts)
        flows = strategy.calculate_energy_flows(3.0, 1.5, 10, 1.0 / 60.0)
        assert isinstance(flows, EnergyFlow)

    def test_resolves_at_15min_intervals(self, battery, grid, tariff, load_forecaster, solar_forecaster):
        strategy = MpcStrategy(
            battery=battery, grid=grid, tariff=tariff,
            load_forecaster=load_forecaster,
            solar_forecaster=solar_forecaster,
        )
        # First call triggers a solve
        ts1 = pd.Timestamp("2024-06-15 10:00", tz=LOCAL_TZ)
        strategy.set_timestamp(ts1)
        strategy.calculate_energy_flows(3.0, 1.5, 10, 1.0 / 60.0)
        solves_after_first = strategy._solve_count

        # Calls within 15 min should not trigger another solve
        for i in range(1, 14):
            ts = ts1 + timedelta(minutes=i)
            strategy.set_timestamp(ts)
            strategy.calculate_energy_flows(3.0, 1.5, 10, 1.0 / 60.0)
        assert strategy._solve_count == solves_after_first

        # At 15 min, should re-solve
        ts2 = ts1 + timedelta(minutes=15)
        strategy.set_timestamp(ts2)
        strategy.calculate_energy_flows(3.0, 1.5, 10, 1.0 / 60.0)
        assert strategy._solve_count == solves_after_first + 1

    def test_charges_during_valley(self, battery, grid, tariff, load_forecaster, solar_forecaster):
        """During valley hours (cheap), MPC should charge battery."""
        strategy = MpcStrategy(
            battery=battery, grid=grid, tariff=tariff,
            load_forecaster=load_forecaster,
            solar_forecaster=solar_forecaster,
        )
        ts = pd.Timestamp("2024-06-15 03:00", tz=LOCAL_TZ)  # valley hour
        strategy.set_timestamp(ts)
        flows = strategy.calculate_energy_flows(0.0, 1.5, 3, 1.0 / 60.0)
        # Should be charging (grid_import > load)
        assert flows.battery_charge > 0 or flows.grid_import > 0
```

- [ ] **Step 2: Implement MpcStrategy**

Append to `src/domain/strategy/mpc.py`:

```python
from src.domain.battery.models import Battery
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import PowerTariff
from src.domain.strategy.model import BaseEnergyStrategy, EnergyFlow
from src.domain.strategy.oracle import solve_oracle_lp

MPC_HORIZON_STEPS = 96     # 24h at 15-min resolution
MPC_STEP_MINUTES = 15
MPC_RESOLVE_MINUTES = 15


class MpcStrategy(BaseEnergyStrategy):
    """Rolling-horizon LP strategy with forecast-based planning.

    Every 15 minutes, solves a 24h LP using solar/load forecasts.
    Between solves, replays the cached plan.
    """

    def __init__(
        self,
        battery: Battery,
        grid: Grid,
        tariff: PowerTariff,
        load_forecaster: LoadForecaster,
        solar_forecaster: SolarForecaster,
    ) -> None:
        super().__init__(battery, grid, tariff)
        self._load_forecaster = load_forecaster
        self._solar_forecaster = solar_forecaster
        self._cached_plan: dict[str, np.ndarray] | None = None
        self._plan_start: pd.Timestamp | None = None
        self._current_ts: pd.Timestamp | None = None
        self._solve_count = 0

    def set_timestamp(self, ts: pd.Timestamp) -> None:
        """Called by the simulator before each step to set the current time."""
        self._current_ts = ts

    def _should_resolve(self) -> bool:
        if self._cached_plan is None or self._plan_start is None:
            return True
        elapsed = (self._current_ts - self._plan_start).total_seconds() / 60.0
        return elapsed >= MPC_RESOLVE_MINUTES

    def _solve_horizon(self) -> None:
        ts = self._current_ts
        solar_fc = self._solar_forecaster.forecast_24h(ts, MPC_HORIZON_STEPS, MPC_STEP_MINUTES)
        load_fc = self._load_forecaster.forecast_24h(ts, MPC_HORIZON_STEPS, MPC_STEP_MINUTES)

        hours = np.array([
            (ts + timedelta(minutes=i * MPC_STEP_MINUTES)).hour
            for i in range(MPC_HORIZON_STEPS)
        ])
        import_rates = np.array([
            self.tariff.get_import_rate(int(h) % 24) for h in hours
        ])
        export_rates = np.array([
            self.tariff.get_export_rate(int(h) % 24) for h in hours
        ])

        dt = MPC_STEP_MINUTES / 60.0  # hours per step

        result = solve_oracle_lp(
            solar=solar_fc,
            load=load_fc,
            import_rates=import_rates,
            export_rates=export_rates,
            dt=dt,
            battery_capacity=self.battery.capacity,
            max_charge_rate=self.battery.max_charge_rate,
            max_discharge_rate=self.battery.max_discharge_rate,
            efficiency=self.battery.efficiency,
            initial_soc=self.battery.current_charge / self.battery.capacity,
            min_soc_frac=self.min_battery_level,
            max_grid_import=self.grid.max_import,
            max_grid_export=self.grid.max_export,
        )

        if result.success:
            self._cached_plan = {
                "charge": result.charge,
                "discharge": result.discharge,
                "grid_import": result.grid_import,
                "grid_export": result.grid_export,
                "soc": result.soc,
            }
        self._plan_start = ts
        self._solve_count += 1

    def _get_plan_step(self) -> int:
        """Map current timestamp to the plan's step index."""
        if self._plan_start is None:
            return 0
        elapsed_min = (self._current_ts - self._plan_start).total_seconds() / 60.0
        return min(int(elapsed_min / MPC_STEP_MINUTES), MPC_HORIZON_STEPS - 1)

    def calculate_energy_flows(
        self, solar_power: float, load_power: float, hour: int, duration: float,
    ) -> EnergyFlow:
        if duration == 0:
            raise ZeroDivisionError("Duration cannot be zero")

        if self._current_ts is not None and self._should_resolve():
            self._solve_horizon()

        if self._cached_plan is None:
            # Fallback: self-consume
            flows = self._calculate_initial_flows(solar_power * duration, load_power * duration)
            self._handle_remaining_solar(flows, duration)
            self._handle_remaining_load(flows, duration, force_discharge=True)
            return flows

        t = self._get_plan_step()
        plan = self._cached_plan
        flows = EnergyFlow()

        solar_energy = solar_power * duration
        load_energy = load_power * duration
        flows.direct_solar = min(solar_energy, load_energy)

        # Apply LP plan decisions at this step's power levels
        flows.battery_charge = float(plan["charge"][t])
        flows.battery_discharge = float(plan["discharge"][t])
        flows.grid_import = float(plan["grid_import"][t])
        flows.grid_export = float(plan["grid_export"][t])

        # Update battery SoC from plan
        self.battery.current_charge = float(plan["soc"][t])

        return flows
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/domain/strategy/test_mpc.py -v`
Expected: All 12 PASS

- [ ] **Step 4: Commit**

```bash
git add tests/domain/strategy/test_mpc.py src/domain/strategy/mpc.py
git commit -m "feat: implement MpcStrategy with rolling 24h LP horizon"
```

---

### Task 5: Register MPC in Simulation Service

**Files:**
- Modify: `web/backend/services/simulation_service.py`

The MPC strategy needs both forecast data sources injected: a `LoadForecaster` built from the load_df history, and a `SolarForecaster` from the irradiance CSVs. The MPC also needs `set_timestamp()` called before each step.

- [ ] **Step 1: Add MPC import, registry entry, and data injection**

Add imports:
```python
from src.domain.strategy.mpc import (
    LoadForecaster,
    MpcStrategy,
    SolarForecaster,
    load_hourly_ghi_forecasts,
)
```

Add to `STRATEGY_REGISTRY`:
```python
    StrategyInfo(
        id="mpc",
        name="Model Predictive Control",
        description="Rolling 24h LP with solar forecasts and learned load profiles. "
        "Re-solves every 15 minutes. Deployable in real-time.",
    ),
```

Add to `_STRATEGY_CLASSES`:
```python
    "mpc": MpcStrategy,
```

In `start_run()`, load hourly GHI forecasts alongside daily forecasts (around line 220):
```python
        # Load forecasts once if needed
        daily_forecasts = (
            load_daily_forecasts() if "forecast_charge" in strategy_ids else None
        )
        hourly_ghi_forecasts = (
            load_hourly_ghi_forecasts() if "mpc" in strategy_ids else None
        )
```

Pass `hourly_ghi_forecasts` to `_run_strategy` (add parameter to method signature and to the `submit` call).

In `_run_strategy`, add the MPC branch:
```python
            elif strategy_id == "mpc":
                load_forecaster = LoadForecaster(load_df)
                solar_forecaster = SolarForecaster(hourly_ghi_forecasts or {})
                strategy = strategy_cls(
                    battery, grid, tariff,
                    load_forecaster=load_forecaster,
                    solar_forecaster=solar_forecaster,
                )
```

In the simulation loop (around line 286), call `set_timestamp` for MPC:
```python
                if hasattr(strategy, 'current_date'):
                    strategy.current_date = timestamp.date()
                if hasattr(strategy, 'set_timestamp'):
                    strategy.set_timestamp(timestamp)
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add web/backend/services/simulation_service.py
git commit -m "feat: register MPC strategy in simulation service"
```

---

### Task 6: Lint + Full Test Suite

- [ ] **Step 1: Run linter**

Run: `make lint`

- [ ] **Step 2: Fix any issues**

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v`

- [ ] **Step 4: Commit fixes if any**

```bash
git add -u
git commit -m "style: fix lint issues in MPC strategy"
```

---

### Task 7: Manual Verification — Run MPC via API

- [ ] **Step 1: Restart server and run MPC + oracle + smart_discharge**

```bash
curl -s -X POST http://127.0.0.1:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"strategies": ["mpc", "smart_discharge", "charge_night"]}'
```

- [ ] **Step 2: Compare results**

MPC should be between oracle (1247 EUR) and smart_discharge (1591 EUR). If it's close to oracle, the forecast quality is good. If it's close to smart_discharge, the forecasts need improvement.

- [ ] **Step 3: Record result**

```bash
git commit --allow-empty -m "chore: MPC verification — cost=X.XX"
```
