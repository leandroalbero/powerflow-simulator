"""Oracle strategy: LP-based optimal battery scheduling with perfect foresight.

The LP runs on the simulation frame's own grid aggregated to 5-minute slots
(``slot = (ts - lp_start) // 5 min``) and is replayed by timestamp through
``_settle`` against the actual 1-minute data, so its cost is realised exactly
like every other strategy's.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linprog  # type: ignore[import-untyped]
from scipy.sparse import csr_matrix, diags, hstack, vstack  # type: ignore[import-untyped]
from scipy.sparse import eye as speye  # type: ignore[import-untyped]

from src.domain.battery.models import Battery
from src.domain.energy_simulator.models import price_arrays
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import PowerTariff
from src.domain.strategy.model import BaseEnergyStrategy, Decision

SLOT_MINUTES = 5
_EPS = 1e-6


@dataclass
class OracleLpResult:
    """Result from solving the oracle LP (powers in kW per step, soc in kWh at step end)."""

    success: bool
    total_cost: float
    charge: np.ndarray
    discharge: np.ndarray
    grid_import: np.ndarray
    grid_export: np.ndarray
    soc: np.ndarray
    curtail: Optional[np.ndarray] = None


def _empty_result(n: int, success: bool) -> OracleLpResult:
    z = np.zeros(n)
    return OracleLpResult(success=success, total_cost=0.0, charge=z, discharge=z.copy(),
                          grid_import=z.copy(), grid_export=z.copy(), soc=z.copy(), curtail=z.copy())


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
    cycle_cost_eur_kwh: float = 0.0,
    terminal_value_eur_kwh: float = 0.0,
    discharge_efficiency: Optional[float] = None,
    taper_start: Optional[float] = None,
    taper_factor: float = 0.0,
    active: Optional[np.ndarray] = None,
    max_grid_charge: Optional[float] = None,
) -> OracleLpResult:
    """Cost-minimising battery schedule with perfect foresight.

    ``efficiency`` is the ONE-WAY charge efficiency (``discharge_efficiency``
    defaults to it). ``initial_soc``/``min_soc_frac`` are fractions of capacity.
    The import cap binds controllable import only: a step may import up to
    ``max(max_grid_import, load - solar)``. ``terminal_value_eur_kwh`` rewards
    energy stored at the end (not a cash flow). ``taper_start``/``taper_factor``
    add the battery's charge-power taper as a linear constraint on the previous
    step's SoC. ``active`` = False forces charge = discharge = 0 for that step.
    ``max_grid_charge`` caps the grid-sourced part of charging (kW): charge may
    exceed it only by the step's solar surplus.
    """
    n = len(solar)
    if n == 0:
        return _empty_result(0, True)

    dt_arr = np.broadcast_to(np.asarray(dt, dtype=float), n)
    eta_c = efficiency
    eta_d = efficiency if discharge_efficiency is None else discharge_efficiency
    nv = 6 * n
    i_ch, i_dis, i_imp, i_exp, i_cur, i_soc = (slice(k * n, (k + 1) * n) for k in range(6))

    c = np.zeros(nv)
    c[i_imp] = import_rates * dt_arr
    c[i_exp] = -export_rates * dt_arr
    if cycle_cost_eur_kwh > 0:
        c[i_ch] += 0.5 * cycle_cost_eur_kwh * dt_arr
        c[i_dis] += 0.5 * cycle_cost_eur_kwh * dt_arr
    if terminal_value_eur_kwh:
        c[6 * n - 1] -= terminal_value_eur_kwh

    eye = speye(n, format="csr")
    zero = csr_matrix((n, n))
    dmat = diags(dt_arr, 0, format="csr")
    # balance: -ch + dis + imp - exp - curtail = load - solar
    a_bal = hstack([-eye, eye, eye, -eye, -eye, zero], format="csr")
    b_bal = load - solar
    # dynamics: soc[t] - soc[t-1] - ch*eta_c*dt + dis/eta_d*dt = 0
    soc_diff = eye - diags([1.0], [-1], shape=(n, n), format="csr")
    a_dyn = hstack([-eta_c * dmat, (1.0 / eta_d) * dmat, zero, zero, zero, soc_diff], format="csr")
    b_dyn = np.zeros(n)
    b_dyn[0] = initial_soc * battery_capacity
    a_eq = vstack([a_bal, a_dyn], format="csr")
    b_eq = np.concatenate([b_bal, b_dyn])

    a_ub = b_ub = None
    if taper_start is not None and taper_factor > 0 and taper_start < 1.0:
        # ch[t] <= max_c * ((1 - f) + f * (cap - soc[t-1]) / ((1 - ts) * cap))
        k = max_charge_rate * taper_factor / ((1.0 - taper_start) * battery_capacity)
        rhs = max_charge_rate * ((1.0 - taper_factor) + taper_factor / (1.0 - taper_start))
        prev_soc = diags([1.0], [-1], shape=(n, n), format="csr")
        a_ub = hstack([eye, zero, zero, zero, zero, k * prev_soc], format="csr")
        b_ub = np.full(n, rhs)
        b_ub[0] -= k * initial_soc * battery_capacity

    act = np.ones(n, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    ch_ub = np.full(n, float(max_charge_rate))
    if max_grid_charge is not None:
        ch_ub = np.minimum(ch_ub, max(0.0, max_grid_charge) + np.maximum(solar - load, 0.0))
    lb = np.concatenate([np.zeros(5 * n), np.full(n, min_soc_frac * battery_capacity)])
    ub = np.concatenate([
        np.where(act, ch_ub, 0.0),
        np.where(act, max_discharge_rate, 0.0),
        np.maximum(max_grid_import, load - solar),
        np.full(n, max_grid_export),
        np.maximum(solar, 0.0),
        np.full(n, battery_capacity),
    ])
    if initial_soc * battery_capacity < lb[5 * n]:
        lb[5 * n:] = np.minimum(lb[5 * n:], initial_soc * battery_capacity)

    result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=np.column_stack([lb, ub]),
                     method="highs", options={"presolve": True, "time_limit": 600})
    if not result.success:
        return _empty_result(n, False)

    x = result.x
    grid_cost = float(np.sum((x[i_imp] * import_rates - x[i_exp] * export_rates) * dt_arr))
    return OracleLpResult(
        success=True, total_cost=grid_cost,
        charge=x[i_ch], discharge=x[i_dis], grid_import=x[i_imp], grid_export=x[i_exp],
        soc=x[i_soc], curtail=x[i_cur],
    )


@dataclass
class SlotSeries:
    start_ns: int
    slot_ns: int
    solar: np.ndarray
    load: np.ndarray
    dt: np.ndarray
    import_rates: np.ndarray
    export_rates: np.ndarray
    first_row: np.ndarray


def aggregate_slots(frame: pd.DataFrame, import_price: np.ndarray, export_price: np.ndarray,
                    slot_minutes: int = SLOT_MINUTES) -> SlotSeries:
    """Mean pv/load over the valid minutes of each slot; a slot's duration counts valid minutes only."""
    ns = frame.index.asi8
    slot_ns = slot_minutes * 60 * 1_000_000_000
    start_ns = int(ns[0] - ns[0] % slot_ns)
    sidx = (ns - start_ns) // slot_ns
    n = int(sidx[-1]) + 1
    valid = frame["valid"].to_numpy(dtype=bool)
    pv = np.where(valid, frame["pv_kw"].to_numpy(dtype=float), 0.0)
    load = np.where(valid, np.nan_to_num(frame["load_kw"].to_numpy(dtype=float)), 0.0)
    cnt = np.bincount(sidx, weights=valid.astype(float), minlength=n)
    safe = np.maximum(cnt, 1.0)
    _, first = np.unique(sidx, return_index=True)
    return SlotSeries(
        start_ns=start_ns, slot_ns=slot_ns,
        solar=np.bincount(sidx, weights=pv, minlength=n) / safe,
        load=np.bincount(sidx, weights=load, minlength=n) / safe,
        dt=cnt / 60.0,
        import_rates=import_price[first], export_rates=export_price[first], first_row=first,
    )


class OracleStrategy(BaseEnergyStrategy):
    """Replays the perfect-foresight LP plan for the frame, by timestamp.

    Within each 5-minute slot the replay holds the planned grid exchange and
    lets the battery absorb the sub-slot deviations of the actual minutes; the
    battery is only touched through ``_settle``.
    """

    def __init__(
        self,
        battery: Battery,
        grid: Grid,
        tariff: PowerTariff,
        frame: pd.DataFrame,
        min_battery_level: float = 0.1,
        max_charge_power: Optional[float] = None,
        cycle_cost_eur_kwh: float = 0.0,
        chunk_days: int = 31,
        overlap_days: int = 3,
    ):
        super().__init__(battery, grid, tariff, min_battery_level, max_charge_power)
        import_price, export_price = price_arrays(tariff, frame.index)
        valid = frame["valid"].to_numpy(dtype=bool)
        mean_import = float(import_price[valid].mean()) if valid.any() else 0.0
        # Same valuation of stored energy as the simulator's adjusted_cost, so the plan neither dumps nor hoards.
        self.terminal_value_eur_kwh = mean_import * battery.discharge_efficiency
        self.slots = aggregate_slots(frame, import_price, export_price)
        self.lp_result = self._solve(cycle_cost_eur_kwh, chunk_days, overlap_days)

    def _solve(self, cycle_cost: float, chunk_days: int, overlap_days: int) -> OracleLpResult:
        s = self.slots
        n = len(s.solar)
        per_day = 24 * 60 // SLOT_MINUTES
        chunk, overlap = chunk_days * per_day, overlap_days * per_day
        b = self.battery
        out = {k: np.zeros(n) for k in ("charge", "discharge", "grid_import", "grid_export", "soc", "curtail")}
        soc_frac = b.current_charge / b.capacity
        cost = 0.0
        a = 0
        while a < n:
            end = min(n, a + chunk)
            stop = n if n - end < per_day else min(n, end + overlap)
            if stop == n:
                end = n
            sl = slice(a, stop)
            r = solve_oracle_lp(
                solar=s.solar[sl], load=s.load[sl], import_rates=s.import_rates[sl],
                export_rates=s.export_rates[sl], dt=s.dt[sl], battery_capacity=b.capacity,
                max_charge_rate=b.max_charge_rate,
                max_discharge_rate=b.max_discharge_rate, efficiency=b.charge_efficiency,
                discharge_efficiency=b.discharge_efficiency, initial_soc=soc_frac,
                min_soc_frac=self.min_battery_level, max_grid_import=self.grid.max_import,
                max_grid_export=self.grid.max_export, cycle_cost_eur_kwh=cycle_cost,
                terminal_value_eur_kwh=self.terminal_value_eur_kwh, taper_start=b.taper_start,
                taper_factor=b.taper_factor, active=s.dt[sl] > 0, max_grid_charge=self.grid_charge_power,
            )
            if not r.success:
                return _empty_result(n, False)
            keep = end - a
            for k in out:
                out[k][a:end] = getattr(r, k)[:keep]
            cost += float(np.sum((r.grid_import[:keep] * s.import_rates[a:end]
                                  - r.grid_export[:keep] * s.export_rates[a:end]) * s.dt[a:end]))
            soc_frac = r.soc[keep - 1] / b.capacity
            a = end
        return OracleLpResult(success=True, total_cost=cost, **out)

    def _slot(self) -> int:
        if self.now is None:
            raise RuntimeError("OracleStrategy needs strategy.now set before each step")
        return int((pd.Timestamp(self.now).value - self.slots.start_ns) // self.slots.slot_ns)

    def _decide(self, solar_kw: float, load_kw: float, dt_h: float) -> Decision:
        lp = self.lp_result
        i = self._slot()
        if not lp.success or not 0 <= i < len(lp.charge):
            return self._self_consume(solar_kw, load_kw)
        ch, dis = float(lp.charge[i]), float(lp.discharge[i])
        planned_grid = float(lp.grid_import[i] - lp.grid_export[i])
        cmd = planned_grid + solar_kw - load_kw
        if dis <= _EPS and planned_grid > _EPS:
            # The plan imports and keeps the battery for later: do not spend it on sub-slot spikes.
            cmd = max(cmd, 0.0)
        plan_surplus = max(0.0, float(self.slots.solar[i] - self.slots.load[i]))
        plan_deficit = max(0.0, float(self.slots.load[i] - self.slots.solar[i]))
        return Decision(cmd, allow_grid_charge=ch > plan_surplus + _EPS,
                        allow_export_discharge=dis > plan_deficit + _EPS)
