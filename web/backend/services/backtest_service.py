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
from web.backend.services.client_errors import public_error
from web.backend.services.presets import SWITCH_DATE, config_for_hardware, hardware_by_id
from web.backend.services.simulation_service import RunInputs, SimulationService
from web.backend.services.span import inclusive_day_count

_MAX_DAYS = 366
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
    if inclusive_day_count(start, end) > _MAX_DAYS:
        raise ValueError("date range exceeds 1 year")
    if hardware != "auto":
        hardware_by_id(hardware)
    frame = sim.data_service.get_frame(start, end, include_measured=True)
    if len(frame) == 0 or "grid_kw" not in frame.columns or not np.isfinite(frame["grid_kw"].to_numpy()).any():
        raise ValueError("no measured data in range")
    measured = sim.config.model_copy(update={"exclude_ev": False})
    try:
        shared = sim.prepare_inputs(frame, [strategy_id], measured)
    except ValueError:
        raise
    except Exception:
        public_error("backtest failed")
        raise ValueError("backtest inputs unavailable") from None
    segments = _segments(frame.index[0], frame.index[-1] + pd.Timedelta(minutes=1), hardware)
    initial = _initial_soc(frame)
    pieces: list[tuple[pd.DataFrame, dict]] = []
    used: list[HardwareSegment] = []
    soc = initial
    for segment in segments:
        sl = frame[(frame.index >= segment.start) & (frame.index < segment.end)]
        if len(sl) == 0:
            continue
        cfg = _segment_config(measured, segment.preset_id, soc)
        result = _run_segment(sim, sl, strategy_id, cfg, shared)
        pieces.append((sl, result.timeseries))
        used.append(HardwareSegment(
            preset_id=segment.preset_id,
            from_=segment.start.strftime("%Y-%m-%d"),
            to=(segment.end - pd.Timedelta(minutes=1)).strftime("%Y-%m-%d"),
        ))
        soc = float(result.metrics["final_battery_kwh"]) / cfg.battery.capacity
    if not pieces:
        raise ValueError("no measured data in range")
    tariff = sim._build_tariff(sim.config, prices=shared.pvpc_prices)
    return _assemble(strategy_id, frame, pieces, used, sim, tariff)


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


def _run_segment(sim: SimulationService, frame: pd.DataFrame, strategy_id: str, cfg: SystemConfig, shared: RunInputs):
    inputs = RunInputs(
        frame=frame,
        history=shared.history,
        daily_forecasts=shared.daily_forecasts,
        ghi=shared.ghi,
        pvpc_prices=shared.pvpc_prices,
    )
    tariff = sim._build_tariff(cfg, prices=shared.pvpc_prices)
    strategy = sim._build_strategy(strategy_id, cfg, inputs, tariff)
    return simulate(frame, strategy, tariff)


def _assemble(strategy_id, frame, pieces, used, sim: SimulationService, tariff) -> BacktestResponse:
    sim_imp = _place(frame, pieces, "grid_import_kw")
    sim_exp = _place(frame, pieces, "grid_export_kw")
    sim_level = _place(frame, pieces, "battery_level_kwh")
    caps = np.full(len(frame), np.nan)
    for segment, (sl, _ts) in zip(used, pieces):
        cap = _segment_config(sim.config, segment.preset_id, 0.5).battery.capacity
        caps[frame.index.get_indexer(sl.index)] = cap
    grid = frame["grid_kw"].to_numpy(dtype=float)
    meas_imp_kw = np.maximum(-grid, 0.0)
    meas_exp_kw = np.maximum(grid, 0.0)
    both = (
        np.isfinite(meas_imp_kw) & np.isfinite(meas_exp_kw)
        & np.isfinite(sim_imp) & np.isfinite(sim_exp)
    )
    meas_imp_kw = np.where(both, meas_imp_kw, np.nan)
    meas_exp_kw = np.where(both, meas_exp_kw, np.nan)
    sim_imp = np.where(both, sim_imp, np.nan)
    sim_exp = np.where(both, sim_exp, np.nan)
    imp_price, exp_price = price_arrays(tariff, frame.index)
    meas_cost = np.nansum(meas_imp_kw * DT_H * imp_price - meas_exp_kw * DT_H * exp_price)
    sim_cost = np.nansum(sim_imp * DT_H * imp_price - sim_exp * DT_H * exp_price)
    daily = _daily(frame.index, meas_imp_kw, meas_exp_kw, sim_imp, sim_exp, imp_price, exp_price)
    totals = _totals(meas_imp_kw, meas_exp_kw, sim_imp, sim_exp)
    sim_soc = sim_level / caps * 100.0
    meas_soc = frame["soc_pct"].to_numpy(dtype=float) if "soc_pct" in frame.columns else np.full(len(frame), np.nan)
    sim_soc = np.where(both, sim_soc, np.nan)
    meas_soc = np.where(both, meas_soc, np.nan)
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


def _place(frame: pd.DataFrame, pieces: list[tuple[pd.DataFrame, dict]], key: str) -> np.ndarray:
    out = np.full(len(frame), np.nan)
    for sl, timeseries in pieces:
        out[frame.index.get_indexer(sl.index)] = timeseries[key]
    return out


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
            sim_cost=float(np.nansum(sim_imp[mask] * DT_H * imp_price[mask] - sim_exp[mask] * DT_H * exp_price[mask])),
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
