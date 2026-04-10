"""Oracle strategy: LP-based optimal battery scheduling with perfect foresight."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linprog  # type: ignore[import-untyped]
from scipy.sparse import lil_matrix  # type: ignore[import-untyped]

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
    nc = 5 * n
    idx_ch = slice(0, n)
    idx_dis = slice(n, 2 * n)
    idx_imp = slice(2 * n, 3 * n)
    idx_exp = slice(3 * n, 4 * n)
    idx_soc = slice(4 * n, 5 * n)

    # Objective: min Σ (grid_import * rate_import - grid_export * rate_export) * dt
    c = np.zeros(nc)
    c[idx_imp] = import_rates * dt
    c[idx_exp] = -export_rates * dt

    # Equality constraints (sparse):
    # 1. Energy balance: -charge + discharge + grid_import - grid_export = load - solar
    # 2. Battery dynamics: soc[t] - soc[t-1] - charge[t]*eff*dt + discharge[t]/eff*dt = 0
    #    For t=0: soc[0] - charge[0]*eff*dt + discharge[0]/eff*dt = initial_soc_kwh

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

    # Bounds
    bounds: list[tuple[float, Optional[float]]] = []
    for t in range(n):
        bounds.append((0, max_charge_rate))      # charge
    for t in range(n):
        bounds.append((0, max_discharge_rate))   # discharge
    for t in range(n):
        bounds.append((0, None))                 # grid_import
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
            flows = self._calculate_initial_flows(solar_power * duration, load_power * duration)
            self._handle_remaining_solar(flows, duration)
            self._handle_remaining_load(flows, duration)
            self._step += 1
            return flows

        lp = self.lp_result
        flows = EnergyFlow()

        solar_energy = solar_power * duration
        load_energy = load_power * duration
        flows.direct_solar = min(solar_energy, load_energy)

        charge_power = lp.charge[t]
        discharge_power = lp.discharge[t]

        if charge_power > 1e-6:
            actual_charged = float(self.battery.charge(charge_power, duration))
            flows.battery_charge = actual_charged
            flows.grid_import += actual_charged

        if discharge_power > 1e-6:
            actual_discharged = float(self.battery.discharge(discharge_power, duration))
            flows.battery_discharge = actual_discharged

        remaining_load = load_energy - flows.direct_solar - flows.battery_discharge * duration
        if remaining_load > 1e-6:
            imported = float(self.grid.import_power(remaining_load / duration, duration))
            flows.grid_import += imported

        remaining_solar = solar_energy - flows.direct_solar - flows.battery_charge * duration
        if remaining_solar > 1e-6:
            exported = float(self.grid.export_power(remaining_solar / duration, duration))
            flows.grid_export = exported

        self._step += 1
        return flows
