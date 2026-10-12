"""Minute-step simulator over the fixed power frame.

``run`` iterates the arrays of :func:`src.data.frame.build_power_frame`, lets the
strategy decide and settle each valid minute, and prices realised import/export
with ``tariff.rate_at(ts, ...)``. Invalid minutes (load unknown) are skipped
entirely - no flows, no cost, battery untouched - but stay in the timeseries as
NaN so charts show the gap.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.domain.power_tariff.model import EnergyDirection
from src.domain.strategy.model import BaseEnergyStrategy

DT_H = 1.0 / 60.0

ProgressFn = Callable[[int], None]


@dataclass
class SimResult:
    metrics: Dict[str, Any]
    timeseries: Dict[str, np.ndarray] = field(default_factory=dict)


def price_arrays(tariff: Any, index: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row ``rate_at`` prices, looked up once per clock hour.

    Europe/Madrid offsets are whole hours, so every minute of a UTC hour shares
    the local hour and date that ``rate_at`` depends on.
    """
    n = len(index)
    imp = np.empty(n)
    exp = np.empty(n)
    if n == 0:
        return imp, exp
    hour_key = index.asi8 // 3_600_000_000_000
    _, first, inverse = np.unique(hour_key, return_index=True, return_inverse=True)
    imp_h = np.empty(len(first))
    exp_h = np.empty(len(first))
    for j, pos in enumerate(first):
        ts = index[pos]
        imp_h[j] = tariff.rate_at(ts, EnergyDirection.IMPORT).price
        exp_h[j] = tariff.rate_at(ts, EnergyDirection.EXPORT).price
    imp[:] = imp_h[inverse]
    exp[:] = exp_h[inverse]
    return imp, exp


def run(frame: pd.DataFrame, strategy: BaseEnergyStrategy, tariff: Any,
        on_progress: Optional[ProgressFn] = None) -> SimResult:
    """Simulate ``strategy`` over ``frame`` (fixed 1-minute grid) and settle costs against ``tariff``."""
    n = len(frame)
    index = frame.index
    pv = frame["pv_kw"].to_numpy(dtype=float)
    load = frame["load_kw"].to_numpy(dtype=float)
    valid = frame["valid"].to_numpy(dtype=bool)
    imp_price, exp_price = price_arrays(tariff, index)

    battery = strategy.battery
    initial_kwh = battery.current_charge
    nan = np.full(n, np.nan)
    level, imp, exp, ch, dis = nan.copy(), nan.copy(), nan.copy(), nan.copy(), nan.copy()
    gc, curt, unserved, over, direct = nan.copy(), nan.copy(), nan.copy(), nan.copy(), nan.copy()

    stride = max(1, n // 100)
    for i, ts in enumerate(index):
        if valid[i]:
            strategy.now = ts
            f = strategy.calculate_energy_flows(pv[i], load[i], ts.hour, DT_H)
            imp[i] = f.grid_import
            exp[i] = f.grid_export
            ch[i] = f.battery_charge
            dis[i] = f.battery_discharge
            gc[i] = f.battery_charge_from_grid
            curt[i] = f.curtailed
            unserved[i] = f.unserved
            over[i] = f.over_limit
            direct[i] = f.direct_solar
        level[i] = battery.current_charge
        if on_progress is not None and (i + 1) % stride == 0:
            on_progress(min(100, int((i + 1) * 100 / n)))

    step_cost = imp * imp_price - exp * exp_price
    metrics = _metrics(frame, strategy, valid, pv, load, imp_price, exp_price,
                       imp, exp, ch, dis, gc, curt, unserved, over, direct, step_cost, initial_kwh)
    with np.errstate(invalid="ignore"):
        timeseries = {
            "ts": (index.asi8 // 1_000_000).astype(np.int64),
            "battery_level_kwh": level,
            "grid_import_kw": imp / DT_H,
            "grid_export_kw": exp / DT_H,
            "solar_kw": np.where(valid, pv, np.nan),
            "load_kw": np.where(valid, load, np.nan),
            "battery_charge_kw": ch / DT_H,
            "battery_discharge_kw": dis / DT_H,
            "grid_charge_kw": gc / DT_H,
            "curtailed_kw": curt / DT_H,
            "import_price": imp_price,
            "export_price": exp_price,
            "step_cost": step_cost,
        }
    if on_progress is not None:
        on_progress(100)
    return SimResult(metrics=metrics, timeseries=timeseries)


def _metrics(frame, strategy, valid, pv, load, imp_price, exp_price, imp, exp, ch, dis, gc, curt,
             unserved, over, direct, step_cost, initial_kwh) -> Dict[str, Any]:
    battery = strategy.battery
    s = lambda a: float(np.nansum(a))  # noqa: E731
    solar_kwh = float(pv[valid].sum()) * DT_H
    load_kwh = float(load[valid].sum()) * DT_H
    import_kwh, export_kwh = s(imp), s(exp)
    import_cost = s(imp * imp_price)
    export_revenue = s(exp * exp_price)
    total_cost = import_cost - export_revenue
    final_kwh = battery.current_charge
    delta = final_kwh - initial_kwh
    mean_import_rate = float(imp_price[valid].mean()) if valid.any() else 0.0
    # Energy left in (or taken from) the battery is worth what it will displace: an
    # average import kWh, after discharge losses. Ranking uses the adjusted cost.
    terminal_value = delta * mean_import_rate * battery.discharge_efficiency
    usable = battery.capacity * (1.0 - strategy.min_battery_level)
    battery_out = s(dis)
    curtailed_kwh = s(curt)

    m: Dict[str, Any] = {
        "total_cost": total_cost,
        "total_solar_generated": solar_kwh,
        "total_solar_consumed": s(direct),
        "total_solar_exported": export_kwh,
        "total_grid_imported": import_kwh,
        "total_battery_in": s(ch),
        "total_battery_out": battery_out,
        "total_house_consumption": load_kwh,
        "battery_level": final_kwh,
        "battery_capacity": battery.capacity,
        "self_consumption_rate": (
            100.0 * max(0.0, solar_kwh - export_kwh - curtailed_kwh) / solar_kwh if solar_kwh > 0 else 0.0
        ),
        "solar_fraction": (100.0 * max(0.0, load_kwh - import_kwh) / load_kwh) if load_kwh > 0 else 0.0,
        "import_cost": import_cost,
        "export_revenue": export_revenue,
        "unserved_kwh": s(unserved),
        "curtailed_kwh": curtailed_kwh,
        "over_limit_kwh": s(over),
        "grid_charge_kwh": s(gc),
        "initial_battery_kwh": initial_kwh,
        "final_battery_kwh": final_kwh,
        "stored_energy_delta_kwh": delta,
        "mean_import_rate": mean_import_rate,
        "terminal_value_eur": terminal_value,
        "adjusted_cost": total_cost - terminal_value,
        "valid_hours": float(valid.sum()) * DT_H,
        "excluded_hours": float((~valid).sum()) * DT_H,
        "equivalent_full_cycles": battery_out / usable if usable > 0 else 0.0,
    }
    if "ev_kw" in frame.columns:
        m["ev_kwh"] = float(frame["ev_kw"].to_numpy(dtype=float)[valid].sum()) * DT_H
    m["monthly"] = _monthly(frame.index, valid, pv, load, imp, exp, dis, imp_price, exp_price)
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}


def _monthly(index, valid, pv, load, imp, exp, dis, imp_price, exp_price) -> List[Dict[str, Any]]:
    if len(index) == 0:
        return []
    df = pd.DataFrame({
        "import_kwh": imp, "export_kwh": exp,
        "import_cost": imp * imp_price, "export_revenue": exp * exp_price,
        "solar_kwh": np.where(valid, pv * DT_H, 0.0), "load_kwh": np.where(valid, load * DT_H, 0.0),
        "battery_out_kwh": dis,
    }, index=index)
    g = df.groupby(index.strftime("%Y-%m")).sum(min_count=0)
    out = []
    for month, row in g.iterrows():
        out.append({
            "month": month,
            "cost": round(float(row["import_cost"] - row["export_revenue"]), 6),
            **{k: round(float(row[k]), 6) for k in df.columns},
        })
    return out
