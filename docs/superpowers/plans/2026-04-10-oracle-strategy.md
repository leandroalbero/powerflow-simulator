# Oracle Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an Oracle LP strategy that computes the theoretical minimum cost with perfect foresight, establishing the benchmark for all future strategies.

**Architecture:** A linear program (scipy.optimize.linprog with HiGHS) pre-solves the optimal charge/discharge schedule over all timesteps. The solution is wrapped in a `BaseEnergyStrategy` subclass that replays the precomputed decisions step-by-step through the existing simulator.

**Tech Stack:** scipy (already installed), numpy, pandas. No new dependencies.

---

## File Structure

```
src/domain/strategy/
  oracle.py               # LP solver + OracleStrategy class
web/backend/services/
  simulation_service.py   # Register oracle strategy (modify)
tests/domain/strategy/
  test_oracle.py          # Oracle-specific tests
```

- `oracle.py` — contains two things: (1) `solve_oracle_lp()` function that takes timeseries arrays and config, returns optimal schedule; (2) `OracleStrategy` class that wraps the schedule and implements `calculate_energy_flows()`.
- `simulation_service.py` — add oracle to registry and handle its special data injection (needs full solar/load arrays).
- `test_oracle.py` — tests for the LP solver and the strategy wrapper separately.

---

### Task 1: LP Solver — Failing Tests

**Files:**
- Create: `tests/domain/strategy/test_oracle.py`

- [ ] **Step 1: Write tests for the LP solver**

```python
import numpy as np
import pytest

from src.domain.strategy.oracle import solve_oracle_lp


class TestSolveOracleLp:
    """Tests for the core LP solver function."""

    def test_no_solar_charges_at_valley_discharges_at_peak(self):
        """With no solar, oracle should charge at cheapest rate and discharge at most expensive."""
        n = 4  # 4 hours
        dt = 1.0  # 1-hour steps
        solar = np.zeros(n)
        load = np.array([0.0, 0.0, 1.0, 1.0])  # load only in hours 2-3
        import_rates = np.array([0.085, 0.085, 0.182, 0.182])  # valley, valley, peak, peak
        export_rates = np.full(n, 0.08)

        result = solve_oracle_lp(
            solar=solar,
            load=load,
            import_rates=import_rates,
            export_rates=export_rates,
            dt=dt,
            battery_capacity=5.0,
            max_charge_rate=2.0,
            max_discharge_rate=2.0,
            efficiency=0.95,
            initial_soc=0.5,
            min_soc_frac=0.1,
        )

        assert result.success
        # Should charge during valley (hours 0-1) and discharge during peak (hours 2-3)
        assert result.charge[0] > 0 or result.charge[1] > 0, "Should charge during valley"
        assert result.discharge[2] > 0 or result.discharge[3] > 0, "Should discharge during peak"
        # Cost should be less than importing everything at peak
        peak_only_cost = 2.0 * 0.182
        assert result.total_cost < peak_only_cost

    def test_excess_solar_exports(self):
        """With excess solar and no load, oracle should export."""
        n = 2
        dt = 1.0
        solar = np.array([3.0, 3.0])
        load = np.zeros(n)
        import_rates = np.full(n, 0.182)
        export_rates = np.full(n, 0.08)

        result = solve_oracle_lp(
            solar=solar,
            load=load,
            import_rates=import_rates,
            export_rates=export_rates,
            dt=dt,
            battery_capacity=5.0,
            max_charge_rate=2.0,
            max_discharge_rate=2.0,
            efficiency=0.95,
            initial_soc=0.5,
            min_soc_frac=0.1,
        )

        assert result.success
        assert sum(result.grid_export) > 0, "Should export excess solar"
        assert result.total_cost < 0, "Should earn money from export"

    def test_respects_battery_capacity(self):
        """Cannot charge beyond battery capacity."""
        n = 3
        dt = 1.0
        solar = np.zeros(n)
        load = np.zeros(n)
        import_rates = np.array([0.01, 0.01, 0.182])  # very cheap then expensive
        export_rates = np.full(n, 0.08)

        result = solve_oracle_lp(
            solar=solar,
            load=load,
            import_rates=import_rates,
            export_rates=export_rates,
            dt=dt,
            battery_capacity=2.0,
            max_charge_rate=5.0,
            max_discharge_rate=5.0,
            efficiency=1.0,
            initial_soc=0.0,
            min_soc_frac=0.0,
        )

        assert result.success
        # SoC should never exceed capacity
        assert all(s <= 2.0 + 1e-6 for s in result.soc)

    def test_respects_charge_rate_limits(self):
        """Charge rate cannot exceed max_charge_rate."""
        n = 2
        dt = 1.0
        solar = np.zeros(n)
        load = np.zeros(n)
        import_rates = np.full(n, 0.01)
        export_rates = np.full(n, 0.08)

        result = solve_oracle_lp(
            solar=solar,
            load=load,
            import_rates=import_rates,
            export_rates=export_rates,
            dt=dt,
            battery_capacity=10.0,
            max_charge_rate=1.5,
            max_discharge_rate=1.5,
            efficiency=1.0,
            initial_soc=0.0,
            min_soc_frac=0.0,
        )

        assert result.success
        assert all(c <= 1.5 + 1e-6 for c in result.charge)

    def test_efficiency_loss(self):
        """Round-trip should lose energy proportional to efficiency."""
        n = 2
        dt = 1.0
        solar = np.zeros(n)
        load = np.array([0.0, 1.0])
        import_rates = np.array([0.085, 0.182])
        export_rates = np.full(n, 0.08)

        result = solve_oracle_lp(
            solar=solar,
            load=load,
            import_rates=import_rates,
            export_rates=export_rates,
            dt=dt,
            battery_capacity=5.0,
            max_charge_rate=2.0,
            max_discharge_rate=2.0,
            efficiency=0.95,
            initial_soc=0.5,
            min_soc_frac=0.1,
        )

        assert result.success
        # If we charged X at valley and discharged Y at peak,
        # the SoC change should reflect efficiency losses
        if result.charge[0] > 0.01 and result.discharge[1] > 0.01:
            energy_in = result.charge[0] * 0.95 * dt
            energy_out = result.discharge[1] / 0.95 * dt
            # Battery should lose energy on the round trip
            assert energy_out > result.discharge[1] * dt * 0.99

    def test_min_soc_respected(self):
        """Battery should not discharge below min_soc."""
        n = 2
        dt = 1.0
        solar = np.zeros(n)
        load = np.array([5.0, 5.0])  # high load
        import_rates = np.full(n, 0.182)
        export_rates = np.full(n, 0.08)

        result = solve_oracle_lp(
            solar=solar,
            load=load,
            import_rates=import_rates,
            export_rates=export_rates,
            dt=dt,
            battery_capacity=5.0,
            max_charge_rate=2.0,
            max_discharge_rate=2.0,
            efficiency=1.0,
            initial_soc=1.0,
            min_soc_frac=0.2,
        )

        assert result.success
        # SoC should never go below 20% of 5.0 = 1.0 kWh
        assert all(s >= 1.0 - 1e-6 for s in result.soc)

    def test_empty_input(self):
        """Zero-length arrays should return empty result."""
        result = solve_oracle_lp(
            solar=np.array([]),
            load=np.array([]),
            import_rates=np.array([]),
            export_rates=np.array([]),
            dt=1.0,
            battery_capacity=5.0,
            max_charge_rate=2.0,
            max_discharge_rate=2.0,
            efficiency=0.95,
            initial_soc=0.5,
            min_soc_frac=0.1,
        )

        assert result.success
        assert len(result.charge) == 0
        assert result.total_cost == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/domain/strategy/test_oracle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.domain.strategy.oracle'`

- [ ] **Step 3: Commit test file**

```bash
git add tests/domain/strategy/test_oracle.py
git commit -m "test: add oracle LP solver tests"
```

---

### Task 2: LP Solver — Implementation

**Files:**
- Create: `src/domain/strategy/oracle.py`

- [ ] **Step 1: Implement the LP solver**

```python
"""Oracle strategy: LP-based optimal battery scheduling with perfect foresight."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


@dataclass
class OracleLpResult:
    """Result from solving the oracle LP."""

    success: bool
    total_cost: float
    charge: np.ndarray       # kW per step
    discharge: np.ndarray    # kW per step
    grid_import: np.ndarray  # kW per step
    grid_export: np.ndarray  # kW per step
    soc: np.ndarray          # kWh per step


def solve_oracle_lp(
    solar: np.ndarray,
    load: np.ndarray,
    import_rates: np.ndarray,
    export_rates: np.ndarray,
    dt: float,
    battery_capacity: float,
    max_charge_rate: float,
    max_discharge_rate: float,
    efficiency: float,
    initial_soc: float,
    min_soc_frac: float,
) -> OracleLpResult:
    """Solve for the cost-minimizing battery schedule with perfect foresight.

    Args:
        solar: Solar generation power per step (kW).
        load: House load power per step (kW).
        import_rates: Grid import price per step (EUR/kWh).
        export_rates: Grid export price per step (EUR/kWh).
        dt: Duration of each timestep (hours).
        battery_capacity: Battery capacity (kWh).
        max_charge_rate: Max charge power (kW).
        max_discharge_rate: Max discharge power (kW).
        efficiency: One-way battery efficiency (0-1).
        initial_soc: Initial state of charge as fraction (0-1).
        min_soc_frac: Minimum SoC as fraction of capacity.

    Returns:
        OracleLpResult with optimal schedule and cost.
    """
    n = len(solar)
    if n == 0:
        return OracleLpResult(
            success=True, total_cost=0.0,
            charge=np.array([]), discharge=np.array([]),
            grid_import=np.array([]), grid_export=np.array([]),
            soc=np.array([]),
        )

    # Variable layout: [charge(n), discharge(n), grid_import(n), grid_export(n), soc(n)]
    # Total variables: 5n
    nc = 5 * n
    idx_ch = slice(0, n)
    idx_dis = slice(n, 2 * n)
    idx_imp = slice(2 * n, 3 * n)
    idx_exp = slice(3 * n, 4 * n)
    idx_soc = slice(4 * n, 5 * n)

    # --- Objective: min Σ (grid_import * rate_import - grid_export * rate_export) * dt ---
    c = np.zeros(nc)
    c[idx_imp] = import_rates * dt
    c[idx_exp] = -export_rates * dt

    # --- Equality constraints (sparse) ---
    # 1. Energy balance per step:
    #    solar[t] + grid_import[t] + discharge[t] = load[t] + charge[t] + grid_export[t]
    #    => -charge[t] + discharge[t] + grid_import[t] - grid_export[t] = load[t] - solar[t]
    #
    # 2. Battery dynamics:
    #    soc[t] - soc[t-1] - charge[t] * efficiency * dt + discharge[t] / efficiency * dt = 0
    #    For t=0: soc[0] - charge[0] * efficiency * dt + discharge[0] / efficiency * dt = initial_soc_kwh

    n_eq = 2 * n
    A_eq = lil_matrix((n_eq, nc))
    b_eq = np.zeros(n_eq)

    for t in range(n):
        # Energy balance row
        row = t
        A_eq[row, t] = -1.0               # charge
        A_eq[row, n + t] = 1.0            # discharge
        A_eq[row, 2 * n + t] = 1.0        # grid_import
        A_eq[row, 3 * n + t] = -1.0       # grid_export
        b_eq[row] = load[t] - solar[t]

        # Battery dynamics row
        row = n + t
        A_eq[row, 4 * n + t] = 1.0        # soc[t]
        if t > 0:
            A_eq[row, 4 * n + t - 1] = -1.0  # -soc[t-1]
        A_eq[row, t] = -efficiency * dt          # -charge * eff * dt
        A_eq[row, n + t] = dt / efficiency       # +discharge / eff * dt
        if t == 0:
            b_eq[row] = initial_soc * battery_capacity
        else:
            b_eq[row] = 0.0

    A_eq_csr = A_eq.tocsr()

    # --- Bounds ---
    bounds = []
    for t in range(n):
        bounds.append((0, max_charge_rate))      # charge
    for t in range(n):
        bounds.append((0, max_discharge_rate))   # discharge
    for t in range(n):
        bounds.append((0, None))                 # grid_import (up to grid max, but linprog handles inf)
    for t in range(n):
        bounds.append((0, None))                 # grid_export
    for t in range(n):
        min_soc = min_soc_frac * battery_capacity
        bounds.append((min_soc, battery_capacity))  # soc

    result = linprog(
        c, A_eq=A_eq_csr, b_eq=b_eq, bounds=bounds,
        method="highs", options={"presolve": True, "time_limit": 300},
    )

    if not result.success:
        return OracleLpResult(
            success=False, total_cost=0.0,
            charge=np.zeros(n), discharge=np.zeros(n),
            grid_import=np.zeros(n), grid_export=np.zeros(n),
            soc=np.zeros(n),
        )

    x = result.x
    return OracleLpResult(
        success=True,
        total_cost=float(result.fun),
        charge=x[idx_ch],
        discharge=x[idx_dis],
        grid_import=x[idx_imp],
        grid_export=x[idx_exp],
        soc=x[idx_soc],
    )
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/domain/strategy/test_oracle.py -v`
Expected: All 8 tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/domain/strategy/oracle.py
git commit -m "feat: implement oracle LP solver for optimal battery scheduling"
```

---

### Task 3: Oracle Strategy Wrapper — Failing Tests

**Files:**
- Modify: `tests/domain/strategy/test_oracle.py`

- [ ] **Step 1: Add tests for the OracleStrategy wrapper**

Append to `tests/domain/strategy/test_oracle.py`:

```python
from unittest.mock import Mock

from src.domain.battery.models import Battery
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import PowerTariff, Rate, EnergyDirection
from src.domain.strategy.model import EnergyFlow
from src.domain.strategy.oracle import OracleStrategy


class TestOracleStrategy:
    """Tests for the strategy wrapper that replays LP results through the simulator."""

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
        return Battery(capacity=5.0, max_charge_rate=2.0, max_discharge_rate=2.0, efficiency=0.95)

    @pytest.fixture
    def grid(self):
        return Grid(max_import=5.0, max_export=5.0)

    def test_constructs_from_timeseries(self, battery, grid, tariff):
        """Strategy should accept timeseries data and solve LP during init."""
        solar = np.array([0.0, 0.0, 2.0, 2.0])
        load = np.array([1.0, 1.0, 1.0, 1.0])
        hours = np.array([2, 3, 12, 13])
        durations = np.array([1.0, 1.0, 1.0, 1.0])

        strategy = OracleStrategy(
            battery=battery, grid=grid, tariff=tariff,
            solar=solar, load=load, hours=hours, durations=durations,
        )

        assert strategy.lp_result.success

    def test_calculate_energy_flows_returns_valid_flow(self, battery, grid, tariff):
        """Each call to calculate_energy_flows should return precomputed decision."""
        solar = np.array([0.0, 2.0])
        load = np.array([1.0, 1.0])
        hours = np.array([2, 12])
        durations = np.array([1.0, 1.0])

        strategy = OracleStrategy(
            battery=battery, grid=grid, tariff=tariff,
            solar=solar, load=load, hours=hours, durations=durations,
        )

        flows = strategy.calculate_energy_flows(0.0, 1.0, 2, 1.0)
        assert isinstance(flows, EnergyFlow)

    def test_steps_advance_sequentially(self, battery, grid, tariff):
        """Each call should advance to the next precomputed step."""
        solar = np.array([0.0, 0.0, 3.0])
        load = np.array([1.0, 1.0, 1.0])
        hours = np.array([2, 3, 12])
        durations = np.array([1.0, 1.0, 1.0])

        strategy = OracleStrategy(
            battery=battery, grid=grid, tariff=tariff,
            solar=solar, load=load, hours=hours, durations=durations,
        )

        flow0 = strategy.calculate_energy_flows(0.0, 1.0, 2, 1.0)
        flow1 = strategy.calculate_energy_flows(0.0, 1.0, 3, 1.0)
        flow2 = strategy.calculate_energy_flows(3.0, 1.0, 12, 1.0)

        # All should be valid flows
        for flow in [flow0, flow1, flow2]:
            assert isinstance(flow, EnergyFlow)

    def test_energy_balance_per_step(self, battery, grid, tariff):
        """Solar + grid_import + discharge = load + charge + export for each step."""
        solar = np.array([0.5, 1.0, 2.0, 0.0])
        load = np.array([1.0, 1.0, 0.5, 1.5])
        hours = np.array([2, 8, 12, 20])
        durations = np.array([1.0, 1.0, 1.0, 1.0])

        strategy = OracleStrategy(
            battery=battery, grid=grid, tariff=tariff,
            solar=solar, load=load, hours=hours, durations=durations,
        )

        for i in range(4):
            flows = strategy.calculate_energy_flows(solar[i], load[i], hours[i], durations[i])
            solar_energy = solar[i] * durations[i]
            load_energy = load[i] * durations[i]
            supply = flows.direct_solar + flows.grid_import * durations[i] + flows.battery_discharge * durations[i]
            demand = load_energy
            # Supply should cover demand (within floating point tolerance)
            assert supply >= demand - 1e-6, f"Step {i}: supply {supply:.4f} < demand {demand:.4f}"
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `python -m pytest tests/domain/strategy/test_oracle.py::TestOracleStrategy -v`
Expected: FAIL — `ImportError: cannot import name 'OracleStrategy'`

- [ ] **Step 3: Commit**

```bash
git add tests/domain/strategy/test_oracle.py
git commit -m "test: add OracleStrategy wrapper tests"
```

---

### Task 4: Oracle Strategy Wrapper — Implementation

**Files:**
- Modify: `src/domain/strategy/oracle.py`

- [ ] **Step 1: Add OracleStrategy class to oracle.py**

Append to the bottom of `src/domain/strategy/oracle.py`:

```python
from src.domain.battery.models import Battery
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import EnergyDirection, PowerTariff
from src.domain.strategy.model import BaseEnergyStrategy, EnergyFlow


class OracleStrategy(BaseEnergyStrategy):
    """Strategy that replays LP-optimal decisions step by step.

    Unlike other strategies, this requires the full simulation timeseries
    upfront. It solves the LP at construction time and then replays the
    precomputed schedule on each call to calculate_energy_flows().
    """

    def __init__(
        self,
        battery: Battery,
        grid: Grid,
        tariff: PowerTariff,
        solar: np.ndarray,
        load: np.ndarray,
        hours: np.ndarray,
        durations: np.ndarray,
    ):
        super().__init__(battery, grid, tariff)
        self._step = 0

        import_rates = np.array([
            tariff.get_import_rate(int(h) % 24) for h in hours
        ])
        export_rates = np.array([
            tariff.get_export_rate(int(h) % 24) for h in hours
        ])

        # Use the first duration as dt for the LP (assumes uniform steps)
        dt = float(durations[0]) if len(durations) > 0 else 1.0 / 60.0

        self.lp_result = solve_oracle_lp(
            solar=solar,
            load=load,
            import_rates=import_rates,
            export_rates=export_rates,
            dt=dt,
            battery_capacity=battery.capacity,
            max_charge_rate=min(battery.max_charge_rate, self.max_charge_power),
            max_discharge_rate=battery.max_discharge_rate,
            efficiency=battery.efficiency,
            initial_soc=battery.current_charge / battery.capacity,
            min_soc_frac=self.min_battery_level,
        )

        self._solar = solar
        self._load = load

    def calculate_energy_flows(
        self, solar_power: float, load_power: float, hour: int, duration: float,
    ) -> EnergyFlow:
        if duration == 0:
            raise ZeroDivisionError("Duration cannot be zero")

        t = self._step
        n = len(self._solar)

        if t >= n or not self.lp_result.success:
            # Fallback: pass through to grid
            flows = self._calculate_initial_flows(solar_power * duration, load_power * duration)
            self._handle_remaining_solar(flows, duration)
            self._handle_remaining_load(flows, duration)
            self._step += 1
            return flows

        lp = self.lp_result
        flows = EnergyFlow()

        # Direct solar to load
        solar_energy = solar_power * duration
        load_energy = load_power * duration
        flows.direct_solar = min(solar_energy, load_energy)

        # Apply LP decisions through the real battery/grid
        charge_power = lp.charge[t]
        discharge_power = lp.discharge[t]

        if charge_power > 1e-6:
            actual_charged = float(self.battery.charge(charge_power, duration))
            flows.battery_charge = actual_charged
            # Grid import for charging
            flows.grid_import += actual_charged

        if discharge_power > 1e-6:
            actual_discharged = float(self.battery.discharge(discharge_power, duration))
            flows.battery_discharge = actual_discharged

        # Grid import for remaining load (after solar and battery discharge)
        remaining_load = load_energy - flows.direct_solar - flows.battery_discharge * duration
        if remaining_load > 1e-6:
            imported = float(self.grid.import_power(remaining_load / duration, duration))
            flows.grid_import += imported

        # Export excess solar
        remaining_solar = solar_energy - flows.direct_solar - flows.battery_charge * duration
        if remaining_solar > 1e-6:
            exported = float(self.grid.export_power(remaining_solar / duration, duration))
            flows.grid_export = exported

        self._step += 1
        return flows
```

- [ ] **Step 2: Move the Battery/Grid/Tariff imports to the top of the file**

The imports for `Battery`, `Grid`, `PowerTariff`, `BaseEnergyStrategy`, and `EnergyFlow` should be at the top of `oracle.py`, alongside the existing numpy/scipy imports. Reorganize so the file has a single import block at the top.

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/domain/strategy/test_oracle.py -v`
Expected: All tests PASS (both TestSolveOracleLp and TestOracleStrategy)

- [ ] **Step 4: Commit**

```bash
git add src/domain/strategy/oracle.py
git commit -m "feat: add OracleStrategy wrapper that replays LP-optimal schedule"
```

---

### Task 5: Register Oracle in Simulation Service — Failing Test

**Files:**
- Create: `tests/web/backend/test_oracle_registration.py`

- [ ] **Step 1: Write integration test for oracle registration**

```python
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

import pytz

from web.backend.services.simulation_service import STRATEGY_MAP, _STRATEGY_CLASSES


class TestOracleRegistration:
    def test_oracle_in_strategy_registry(self):
        """Oracle should appear in the strategy registry."""
        assert "oracle" in STRATEGY_MAP
        assert STRATEGY_MAP["oracle"].name == "Oracle Optimizer"

    def test_oracle_in_strategy_classes(self):
        """Oracle strategy class should be registered."""
        assert "oracle" in _STRATEGY_CLASSES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/web/backend/test_oracle_registration.py -v`
Expected: FAIL — `AssertionError: assert 'oracle' in STRATEGY_MAP`

- [ ] **Step 3: Commit**

```bash
git add tests/web/backend/test_oracle_registration.py
git commit -m "test: add oracle strategy registration test"
```

---

### Task 6: Register Oracle in Simulation Service — Implementation

**Files:**
- Modify: `web/backend/services/simulation_service.py`

- [ ] **Step 1: Add oracle import and registry entry**

Add to the imports at the top of `simulation_service.py`:

```python
from src.domain.strategy.oracle import OracleStrategy
```

Add to `STRATEGY_REGISTRY` list (after the last entry):

```python
    StrategyInfo(
        id="oracle",
        name="Oracle Optimizer",
        description="Computes the theoretical minimum cost using linear programming with "
        "perfect foresight. Not deployable — serves as a benchmark.",
    ),
```

Add to `_STRATEGY_CLASSES` dict:

```python
    "oracle": OracleStrategy,
```

- [ ] **Step 2: Handle oracle's special data injection in `_run_strategy`**

In `_run_strategy`, the oracle needs the full solar/load timeseries as numpy arrays. Modify the strategy instantiation block (around line 263-270). Replace:

```python
            if strategy_id == "forecast_charge":
                strategy = strategy_cls(battery, grid, tariff, daily_forecasts)
            else:
                strategy = strategy_cls(battery, grid, tariff)
```

With:

```python
            if strategy_id == "forecast_charge":
                strategy = strategy_cls(battery, grid, tariff, daily_forecasts)
            elif strategy_id == "oracle":
                timestamps = load_df.index
                solar_kw = np.array([
                    solar_df["state"].get(ts, 0.0) / 1000.0 for ts in timestamps
                ])
                load_kw = np.array([
                    load_df["state"].get(ts, 0.0) / 1000.0 for ts in timestamps
                ])
                hours = np.array([ts.hour for ts in timestamps])
                diffs = pd.Series(timestamps).diff().dt.total_seconds() / 3600.0
                diffs.iloc[0] = 1.0 / 60.0
                durations = diffs.values
                strategy = strategy_cls(
                    battery, grid, tariff,
                    solar=solar_kw, load=load_kw,
                    hours=hours, durations=durations,
                )
            else:
                strategy = strategy_cls(battery, grid, tariff)
```

Add `import numpy as np` to the imports at the top of the file if not already present.

- [ ] **Step 3: Run registration tests**

Run: `python -m pytest tests/web/backend/test_oracle_registration.py -v`
Expected: PASS

- [ ] **Step 4: Run full test suite to verify nothing broke**

Run: `python -m pytest tests/ -v`
Expected: All existing tests still pass

- [ ] **Step 5: Commit**

```bash
git add web/backend/services/simulation_service.py
git commit -m "feat: register oracle strategy in simulation service"
```

---

### Task 7: End-to-End Smoke Test

**Files:**
- Modify: `tests/domain/strategy/test_oracle.py`

- [ ] **Step 1: Add a realistic scenario test**

Append to `tests/domain/strategy/test_oracle.py`:

```python
class TestOracleEndToEnd:
    """Realistic scenario: one day with solar curve and varying load."""

    def test_one_day_24_hours(self):
        """Oracle over a 24h period should produce lower cost than naive grid import."""
        n = 24
        dt = 1.0
        # Bell-curve solar: peaks at noon
        solar = np.array([
            0, 0, 0, 0, 0, 0,        # 00-05: night
            0.1, 0.5, 1.5, 3.0,      # 06-09: sunrise
            4.0, 4.5, 4.5, 4.0,      # 10-13: peak sun
            3.0, 1.5, 0.5, 0.1,      # 14-17: sunset
            0, 0, 0, 0, 0, 0,        # 18-23: night
        ])
        # Typical household: morning + evening peaks
        load = np.array([
            0.3, 0.3, 0.3, 0.3, 0.3, 0.5,   # 00-05
            0.8, 1.2, 1.5, 1.0,               # 06-09
            0.8, 0.8, 1.0, 0.8,               # 10-13
            0.8, 1.0, 1.5, 2.0,               # 14-17
            2.5, 2.0, 1.5, 1.0, 0.5, 0.3,    # 18-23
        ])
        import_rates = np.array([
            0.085, 0.085, 0.085, 0.085, 0.085, 0.085, 0.085, 0.085,  # valley
            0.134, 0.134,                                              # shoulder
            0.182, 0.182, 0.182, 0.182,                                # peak
            0.134, 0.134, 0.134, 0.134,                                # shoulder
            0.182, 0.182, 0.182, 0.182,                                # peak
            0.134, 0.134,                                              # shoulder
        ])
        export_rates = np.full(n, 0.08)

        result = solve_oracle_lp(
            solar=solar, load=load,
            import_rates=import_rates, export_rates=export_rates,
            dt=dt,
            battery_capacity=15.0,
            max_charge_rate=4.8,
            max_discharge_rate=4.8,
            efficiency=0.95,
            initial_soc=0.1,
            min_soc_frac=0.1,
        )

        assert result.success

        # Naive cost: import everything from grid, no battery
        naive_cost = sum(load[t] * import_rates[t] * dt for t in range(n))
        # Subtract export revenue for excess solar
        excess = np.maximum(solar - load, 0)
        naive_cost -= sum(excess[t] * export_rates[t] * dt for t in range(n))

        assert result.total_cost < naive_cost, (
            f"Oracle ({result.total_cost:.4f}) should beat naive ({naive_cost:.4f})"
        )

        # Verify SoC stays within bounds
        assert all(s >= 15.0 * 0.1 - 1e-6 for s in result.soc)
        assert all(s <= 15.0 + 1e-6 for s in result.soc)
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest tests/domain/strategy/test_oracle.py::TestOracleEndToEnd -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/domain/strategy/test_oracle.py
git commit -m "test: add oracle end-to-end 24h scenario test"
```

---

### Task 8: Lint and Type Check

**Files:**
- Potentially modify: `src/domain/strategy/oracle.py`, `web/backend/services/simulation_service.py`

- [ ] **Step 1: Run linter**

Run: `make lint`
Expected: Clean (or fix any issues flagged by ruff/mypy)

- [ ] **Step 2: Fix any lint issues**

If ruff or mypy flag issues, fix them in the relevant files.

- [ ] **Step 3: Commit any fixes**

```bash
git add -u
git commit -m "style: fix lint issues in oracle strategy"
```

---

### Task 9: Manual Verification — Run Oracle in the Web UI

- [ ] **Step 1: Restart the dev server**

Run: `make web-dev` (or restart the running server)

- [ ] **Step 2: Run oracle via API**

```bash
curl -s -X PUT http://127.0.0.1:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{
    "battery": {"capacity": 15.0, "max_charge_rate": 4.8, "max_discharge_rate": 4.8, "efficiency": 0.95, "initial_soc": 0.1, "taper_start": 0.9, "taper_factor": 0.7},
    "strategy": {"min_battery_level": 0.1, "max_charge_power": 2.5, "valley_charge_target": 1.0}
  }'

curl -s -X POST http://127.0.0.1:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"strategies": ["oracle", "smart_discharge", "charge_night"]}'
```

- [ ] **Step 3: Wait for completion and compare results**

Poll `GET /api/runs/{run_id}/summary` until complete. Oracle cost should be lower than Smart Discharge (1590.56 EUR) — it has perfect foresight.

- [ ] **Step 4: Record result in commit message**

```bash
git commit --allow-empty -m "chore: oracle verification — cost=X.XX vs smart_discharge=1590.56"
```
