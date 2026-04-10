"""Oracle strategy: LP-based optimal battery scheduling with perfect foresight."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linprog  # type: ignore[import-untyped]
from scipy.sparse import diags, hstack, vstack  # type: ignore[import-untyped]
from scipy.sparse import eye as speye  # type: ignore[import-untyped]

from src.domain.battery.models import Battery
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import PowerTariff
from src.domain.strategy.model import BaseEnergyStrategy, EnergyFlow


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
    dt: "float | np.ndarray",
    battery_capacity: float,
    max_charge_rate: float,
    max_discharge_rate: float,
    efficiency: float,
    initial_soc: float,
    min_soc_frac: float,
    max_grid_import: float = 5.0,
    max_grid_export: float = 5.0,
) -> OracleLpResult:
    """Solve for the cost-minimizing battery schedule with perfect foresight.

    Args:
        solar: Solar generation power per step (kW).
        load: House load power per step (kW).
        import_rates: Grid import price per step (EUR/kWh).
        export_rates: Grid export price per step (EUR/kWh).
        dt: Duration of each timestep (hours). Scalar or per-step array.
        battery_capacity: Battery capacity (kWh).
        max_charge_rate: Max charge power from grid (kW).
        max_discharge_rate: Max discharge power (kW).
        efficiency: One-way battery efficiency (0-1).
        initial_soc: Initial state of charge as fraction (0-1).
        min_soc_frac: Minimum SoC as fraction of capacity.
        max_grid_import: Grid import power limit (kW).
        max_grid_export: Grid export power limit (kW).

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

    # Support both scalar and per-step dt
    dt_arr = np.broadcast_to(np.asarray(dt, dtype=float), n)

    # Variable layout: [charge(n), discharge(n), grid_import(n), grid_export(n), soc(n)]
    nc = 5 * n
    idx_ch = slice(0, n)
    idx_dis = slice(n, 2 * n)
    idx_imp = slice(2 * n, 3 * n)
    idx_exp = slice(3 * n, 4 * n)
    idx_soc = slice(4 * n, 5 * n)

    # Objective: min Σ (grid_import[t] * rate_import[t] - grid_export[t] * rate_export[t]) * dt[t]
    c = np.zeros(nc)
    c[idx_imp] = import_rates * dt_arr
    c[idx_exp] = -export_rates * dt_arr

    # Equality constraints built with sparse diagonal ops (vectorized, no Python loop).
    #
    # Row block 1 (energy balance, n rows):
    #   -charge + discharge + grid_import - grid_export = load - solar
    # Row block 2 (battery dynamics, n rows):
    #   soc[t] - soc[t-1] - charge[t]*eff*dt[t] + discharge[t]/eff*dt[t] = 0
    #   For t=0: soc[0] - charge[0]*eff*dt[0] + discharge[0]/eff*dt[0] = initial_soc_kwh

    I = speye(n, format="csr")  # noqa: E741
    Z = speye(n, format="csr") * 0  # zero block
    D = diags(dt_arr, 0, format="csr")  # per-step duration diagonal

    # Energy balance: [-I, +I, +I, -I, 0] @ [ch, dis, imp, exp, soc] = load - solar
    A_bal = hstack([-I, I, I, -I, Z], format="csr")
    b_bal = load - solar

    # Battery dynamics with per-step durations
    soc_block = speye(n, format="csr") + diags([-1.0], [-1], shape=(n, n), format="csr")
    A_dyn = hstack([
        -efficiency * D,           # charge * eff * dt[t]
        (1.0 / efficiency) * D,    # discharge / eff * dt[t]
        Z, Z,                      # grid_import, grid_export (zero)
        soc_block,                 # soc
    ], format="csr")
    b_dyn = np.zeros(n)
    b_dyn[0] = initial_soc * battery_capacity

    A_eq_csr = vstack([A_bal, A_dyn], format="csr")
    b_eq = np.concatenate([b_bal, b_dyn])

    # Bounds (vectorized) — includes grid import/export caps
    min_soc = min_soc_frac * battery_capacity
    lb = np.concatenate([
        np.zeros(n),                          # charge >= 0
        np.zeros(n),                          # discharge >= 0
        np.zeros(n),                          # grid_import >= 0
        np.zeros(n),                          # grid_export >= 0
        np.full(n, min_soc),                  # soc >= min_soc
    ])
    # Grid import must cover any net load that exceeds battery discharge.
    # Use max(grid_cap, peak_net_load) to ensure feasibility.
    min_grid_needed = np.maximum(load - solar - max_discharge_rate, 0)
    effective_grid_import = np.maximum(max_grid_import, min_grid_needed)

    ub = np.concatenate([
        np.full(n, max_charge_rate),          # charge <= max_charge_rate
        np.full(n, max_discharge_rate),       # discharge <= max_discharge_rate
        effective_grid_import,                # grid_import <= max(cap, needed)
        np.full(n, max_grid_export),          # grid_export <= grid cap
        np.full(n, battery_capacity),         # soc <= capacity
    ])
    bounds = list(zip(lb, ub))

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
        self._elapsed_hours = 0.0
        self._n_lp = len(solar)

        import_rates = np.array([
            tariff.get_import_rate(int(h) % 24) for h in hours
        ])
        export_rates = np.array([
            tariff.get_export_rate(int(h) % 24) for h in hours
        ])

        self.lp_result = solve_oracle_lp(
            solar=solar,
            load=load,
            import_rates=import_rates,
            export_rates=export_rates,
            dt=durations,
            battery_capacity=battery.capacity,
            max_charge_rate=battery.max_charge_rate,
            max_discharge_rate=battery.max_discharge_rate,
            efficiency=battery.efficiency,
            initial_soc=battery.current_charge / battery.capacity,
            min_soc_frac=self.min_battery_level,
            max_grid_import=grid.max_import,
            max_grid_export=grid.max_export,
        )

        # Precompute cumulative hours for LP step lookup
        self._lp_cum_hours = np.cumsum(durations)

    def calculate_energy_flows(
        self, solar_power: float, load_power: float, hour: int, duration: float,
    ) -> EnergyFlow:
        if duration == 0:
            raise ZeroDivisionError("Duration cannot be zero")

        # Map simulator's minute-resolution calls to the correct LP hourly step
        t = int(np.searchsorted(self._lp_cum_hours, self._elapsed_hours, side="right"))
        t = min(t, self._n_lp - 1)
        self._elapsed_hours += duration

        if t >= self._n_lp or not self.lp_result.success:
            flows = self._calculate_initial_flows(solar_power * duration, load_power * duration)
            self._handle_remaining_solar(flows, duration)
            self._handle_remaining_load(flows, duration)
            return flows

        lp = self.lp_result
        flows = EnergyFlow()

        solar_energy = solar_power * duration
        load_energy = load_power * duration
        flows.direct_solar = min(solar_energy, load_energy)

        # Replay LP decisions — power levels from LP, scaled to this sub-step duration.
        # LP decided charge/discharge at power level X (kW) for the full hour.
        # We apply the same power level for this sub-step.
        flows.battery_charge = float(lp.charge[t])
        flows.battery_discharge = float(lp.discharge[t])
        flows.grid_import = float(lp.grid_import[t])
        flows.grid_export = float(lp.grid_export[t])

        # Keep battery SoC in sync for timeseries tracking
        self.battery.current_charge = float(lp.soc[t])

        return flows
