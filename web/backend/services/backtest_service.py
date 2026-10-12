"""Synchronous simulated-vs-measured backtest."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.domain.energy_simulator.models import DT_H, price_arrays
from src.domain.energy_simulator.models import run as simulate
from web.backend.models.feature_schemas import (
    BacktestMetrics,
    BacktestResponse,
    DailyBacktest,
    HardwareSegment,
    SocSeries,
)
from web.backend.models.schemas import SystemConfig
from web.backend.services.presets import SWITCH_DATE, config_for_hardware, hardware_by_id
from web.backend.services.simulation_service import SimulationService

_MAX_SPAN = pd.Timedelta(days=366)
_SOC_POINTS = 3000


@dataclass(frozen=True, slots=True)
class _Segment:
    preset_id: str
    start: pd.Timestamp
    end: pd.Timestamp


def run_backtest(
    sim: SimulationService,
    start: str,
    end: str,
    strategy_id: str,
    hardware: str,
) -> BacktestResponse:
    start_ts, end_ts = _bounds(start, end)
    if end_ts - start_ts > _MAX_SPAN:
        raise ValueError("date range exceeds 1 year")
    if hardware != "auto":
        hardware_by_id(hardware)
    frame = sim.data_service.get_frame(start, end, include_measured=True)
    if len(frame) == 0 or "grid_kw" not in frame.columns or not np.isfinite(frame["grid_kw"].to_numpy()).any():
        raise ValueError("no measured data in range")
    segments = _segments(frame.index[0], frame.index[-1] + pd.Timedelta(minutes=1), hardware)
    initial = _initial_soc(frame)
    pieces: list[tuple[pd.DataFrame, dict]] = []
    used: list[HardwareSegment] = []
    soc = initial
    for segment in segments:
        sl = frame[(frame.index >= segment.start) & (frame.index < segment.end)]
        if len(sl) == 0:
            continue
        cfg = _segment_config(sim.config, segment.preset_id, soc)
        result = _run_segment(sim, sl, strategy_id, cfg)
        pieces.append((sl, result.timeseries))
        used.append(HardwareSegment(
            preset_id=segment.preset_id,
            from_=segment.start.strftime("%Y-%m-%d"),
            to=(segment.end - pd.Timedelta(minutes=1)).strftime("%Y-%m-%d"),
        ))
        soc = float(result.metrics["final_battery_kwh"]) / cfg.battery.capacity
    if not pieces:
        raise ValueError("no measured data in range")
    return _assemble(strategy_id, frame, pieces, used, sim)


def _bounds(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = _localize(start)
    end_ts = _localize(end)
    if start_ts > end_ts:
        raise ValueError(f"start ({start}) must not be after end ({end})")
    return start_ts, end_ts


def _localize(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("Europe/Madrid")
    return ts.tz_convert("Europe/Madrid")


def _segments(start: pd.Timestamp, end: pd.Timestamp, hardware: str) -> list[_Segment]:
    if hardware != "auto":
        return [_Segment(hardware, start, end)]
    switch = pd.Timestamp(SWITCH_DATE, tz="Europe/Madrid")
    if end <= switch:
        return [_Segment("pre_2026_04", start, end)]
    if start >= switch:
        return [_Segment("current", start, end)]
    return [_Segment("pre_2026_04", start, switch), _Segment("current", switch, end)]


def _initial_soc(frame: pd.DataFrame) -> float:
    soc = frame["soc_pct"].to_numpy(dtype=float) if "soc_pct" in frame.columns else np.array([])
    finite = soc[np.isfinite(soc)] if len(soc) else soc
    if len(finite) == 0:
        return 0.5
    return float(np.clip(finite[0] / 100.0, 0.0, 1.0))


def _segment_config(base: SystemConfig, preset_id: str, initial_soc: float) -> SystemConfig:
    cfg = config_for_hardware(preset_id)
    return cfg.model_copy(update={
        "tariff": base.tariff,
        "strategy": base.strategy,
        "exclude_ev": base.exclude_ev,
        "battery": cfg.battery.model_copy(update={"initial_soc": initial_soc}),
    })


def _run_segment(sim: SimulationService, frame: pd.DataFrame, strategy_id: str, cfg: SystemConfig):
    from web.backend.services.simulation_service import RunInputs

    inputs = RunInputs(frame=frame)
    tariff = sim._build_tariff(cfg)
    strategy = sim._build_strategy(strategy_id, cfg, inputs, tariff)
    return simulate(frame, strategy, tariff)


def _assemble(strategy_id, frame, pieces, used, sim: SimulationService) -> BacktestResponse:
    sim_imp = np.concatenate([p[1]["grid_import_kw"] for p in pieces])
    sim_exp = np.concatenate([p[1]["grid_export_kw"] for p in pieces])
    sim_level = np.concatenate([p[1]["battery_level_kwh"] for p in pieces])
    caps = np.concatenate([
        np.full(len(p[0]), _segment_config(sim.config, used[i].preset_id, 0.5).battery.capacity)
        for i, p in enumerate(pieces)
    ])
    grid = frame["grid_kw"].to_numpy(dtype=float)
    meas_imp_kw = np.maximum(-grid, 0.0)
    meas_exp_kw = np.maximum(grid, 0.0)
    tariff = sim._build_tariff(sim.config)
    imp_price, exp_price = price_arrays(tariff, frame.index)
    meas_cost = np.nansum(meas_imp_kw * DT_H * imp_price - meas_exp_kw * DT_H * exp_price)
    sim_cost = np.nansum(np.nan_to_num(sim_imp) * DT_H * imp_price - np.nan_to_num(sim_exp) * DT_H * exp_price)
    daily = _daily(frame.index, meas_imp_kw, meas_exp_kw, sim_imp, sim_exp, imp_price, exp_price)
    totals = _totals(meas_imp_kw, meas_exp_kw, sim_imp, sim_exp)
    sim_soc = sim_level / caps * 100.0
    meas_soc = frame["soc_pct"].to_numpy(dtype=float) if "soc_pct" in frame.columns else np.full(len(frame), np.nan)
    return BacktestResponse(
        strategy_id=strategy_id,
        hardware_used=used,
        daily=daily,
        soc=_soc(frame.index, meas_soc, sim_soc),
        metrics=BacktestMetrics(
            import_error_pct=_rel(totals[1], totals[0]),
            export_error_pct=_rel(totals[3], totals[2]),
            cost_error_pct=_rel(sim_cost, meas_cost),
            daily_import_mae_kwh=float(np.mean([abs(row.sim_import_kwh - row.measured_import_kwh) for row in daily] or [0.0])),
            soc_rmse_pct=_rmse(sim_soc, meas_soc),
            measured_cost=float(meas_cost),
            sim_cost=float(sim_cost),
        ),
    )


def _daily(index, meas_imp, meas_exp, sim_imp, sim_exp, imp_price, exp_price) -> list[DailyBacktest]:
    day = index.tz_convert("Europe/Madrid").strftime("%Y-%m-%d")
    rows = []
    for date in pd.unique(day):
        mask = day == date
        rows.append(DailyBacktest(
            date=str(date),
            measured_import_kwh=float(np.nansum(meas_imp[mask]) * DT_H),
            sim_import_kwh=float(np.nansum(sim_imp[mask]) * DT_H),
            measured_export_kwh=float(np.nansum(meas_exp[mask]) * DT_H),
            sim_export_kwh=float(np.nansum(sim_exp[mask]) * DT_H),
            measured_cost=float(np.nansum(meas_imp[mask] * DT_H * imp_price[mask] - meas_exp[mask] * DT_H * exp_price[mask])),
            sim_cost=float(np.nansum(np.nan_to_num(sim_imp[mask]) * DT_H * imp_price[mask]
                                     - np.nan_to_num(sim_exp[mask]) * DT_H * exp_price[mask])),
        ))
    return rows


def _totals(meas_imp, meas_exp, sim_imp, sim_exp) -> tuple[float, float, float, float]:
    return (
        float(np.nansum(meas_imp) * DT_H),
        float(np.nansum(sim_imp) * DT_H),
        float(np.nansum(meas_exp) * DT_H),
        float(np.nansum(sim_exp) * DT_H),
    )


def _rel(sim: float, measured: float) -> float:
    if measured == 0.0:
        return 0.0 if sim == 0.0 else 100.0
    return 100.0 * (sim - measured) / measured


def _rmse(sim: np.ndarray, measured: np.ndarray) -> float:
    ok = np.isfinite(sim) & np.isfinite(measured)
    if not ok.any():
        return 0.0
    return float(np.sqrt(np.mean((sim[ok] - measured[ok]) ** 2)))


def _soc(index: pd.DatetimeIndex, measured: np.ndarray, sim: np.ndarray) -> SocSeries:
    n = len(index)
    take = np.arange(n) if n <= _SOC_POINTS else np.unique(np.linspace(0, n - 1, _SOC_POINTS).astype(int))
    ts = (index.asi8 // 1_000_000).astype(np.int64)[take]
    meas = [None if v != v else float(v) for v in measured[take]]
    sim_s = [None if v != v else float(v) for v in sim[take]]
    return SocSeries(ts=ts.tolist(), measured_pct=meas, sim_pct=sim_s)
