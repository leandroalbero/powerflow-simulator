"""Backtest compares only joint-valid minutes and uses inclusive day bounds."""

import inspect

import numpy as np
import pandas as pd
import pytest

from src.domain.energy_simulator.models import run as simulate
from tests.synthetic import synthetic_frame
from tests.web.backend.test_features import _csv
from web.backend.api.backtest import post_backtest
from web.backend.models.schemas import SystemConfig
from web.backend.services import backtest_service as backtest_mod
from web.backend.services.backtest_service import run_backtest
from web.backend.services.data_service import DataService
from web.backend.services.simulation_service import RunInputs, SimulationService


def test_backtest_handler_does_not_block_the_event_loop():
    assert inspect.iscoroutinefunction(post_backtest)


def test_ignores_minutes_valid_on_only_one_side(tmp_path, monkeypatch):
    monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
    data = DataService()
    sim = SimulationService(data)
    frame = synthetic_frame(days=2, outage=True, spike=False)
    cfg = SystemConfig()
    tariff = sim._build_tariff(cfg)
    result = simulate(frame, sim._build_strategy("self_consume", cfg, RunInputs(frame=frame), tariff), tariff)
    sim_imp = result.timeseries["grid_import_kw"]
    sim_exp = result.timeseries["grid_export_kw"]
    matched = np.where(np.isfinite(sim_imp), np.nan_to_num(sim_exp) - np.nan_to_num(sim_imp), np.nan)
    grid = matched.copy()
    # Load outage: simulation is NaN, measured grid stays up. Must not count.
    grid[~np.isfinite(sim_imp)] = -0.5
    # Grid outage: simulation is finite, measured is missing. Must not count.
    finite = np.flatnonzero(np.isfinite(sim_imp))
    hole = finite[200:260]
    grid[hole] = np.nan
    frame = frame.copy()
    frame["grid_kw"] = grid
    sim_soc = result.timeseries["battery_level_kwh"] / cfg.battery.capacity * 100.0
    soc = np.where(np.isfinite(sim_soc), sim_soc, 0.0)
    soc[hole] = 99.0
    frame["soc_pct"] = soc
    monkeypatch.setattr(data, "get_frame", lambda *args, **kwargs: frame)
    monkeypatch.setattr(
        "web.backend.services.backtest_service._initial_soc", lambda _frame: cfg.battery.initial_soc,
    )
    out = run_backtest(sim, "2026-03-04", "2026-03-05", "self_consume", "current")
    assert out.metrics.import_error_pct == pytest.approx(0.0, abs=1e-6)
    assert out.metrics.export_error_pct == pytest.approx(0.0, abs=1e-6)
    assert out.metrics.cost_error_pct == pytest.approx(0.0, abs=1e-4)
    assert out.metrics.soc_rmse_pct == pytest.approx(0.0, abs=1e-6)


def test_year_limit_uses_inclusive_days(tmp_path, monkeypatch):
    monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
    sim = SimulationService(DataService())
    with pytest.raises(ValueError, match="1 year"):
        run_backtest(sim, "2025-01-01", "2026-01-02", "self_consume", "current")
    monkeypatch.setattr(sim.data_service, "get_frame", lambda *args, **kwargs: pd.DataFrame())
    with pytest.raises(ValueError, match="no measured"):
        run_backtest(sim, "2025-01-01", "2026-01-01", "self_consume", "current")


def test_inclusive_week_covers_seven_days(tmp_path, monkeypatch):
    monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
    idx = pd.date_range("2026-10-05", "2026-10-12", freq="1min", tz="Europe/Madrid", inclusive="left")
    data = DataService()
    data.load_default_data()
    data.upload_solar(_csv(pd.Series(np.full(len(idx), 800.0), index=idx)))
    data.upload_load(_csv(pd.Series(np.full(len(idx), 400.0), index=idx)))
    seen = {}
    real = data.get_frame

    def wrapped(start, end, **kwargs):
        frame = real(start, end, **kwargs).copy()
        seen["n"] = len(frame)
        frame["grid_kw"] = 0.0
        frame["soc_pct"] = 50.0
        frame["battery_kw"] = 0.0
        return frame

    data.get_frame = wrapped
    out = run_backtest(SimulationService(data), "2026-10-05", "2026-10-11", "self_consume", "current")
    assert seen["n"] == 7 * 1440
    assert len(out.daily) == 7
    assert out.daily[0].date == "2026-10-05"
    assert out.daily[-1].date == "2026-10-11"


def test_backtest_uses_pvpc_when_config_does(tmp_path, monkeypatch):
    monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
    frame = synthetic_frame(days=1, outage=False, spike=False)
    frame["grid_kw"] = 0.0
    frame["soc_pct"] = 50.0
    data = DataService()
    monkeypatch.setattr(data, "get_frame", lambda *args, **kwargs: frame)
    sim = SimulationService(data)
    sim.config = sim.config.model_copy(update={
        "tariff": sim.config.tariff.model_copy(update={"use_pvpc": True}),
    })
    seen = {}

    def prices(start, end, strict=False):
        seen["called"] = True
        idx = pd.date_range(start, end, freq="h")
        return {ts.to_pydatetime(): 0.2 for ts in idx}

    sim.pvpc_service.get_prices = prices
    run_backtest(sim, "2026-03-04", "2026-03-04", "self_consume", "current")
    assert seen.get("called") is True


def test_prepare_inputs_includes_ev_even_when_server_config_excludes_it(tmp_path, monkeypatch):
    monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
    frame = synthetic_frame(days=1, outage=False, spike=False)
    frame["grid_kw"] = 0.0
    frame["soc_pct"] = 50.0
    data = DataService()
    monkeypatch.setattr(data, "get_frame", lambda *_args, **_kwargs: frame)
    sim = SimulationService(data)
    sim.config = sim.config.model_copy(update={"exclude_ev": True})
    seen = {}
    real_prepare = sim.prepare_inputs

    def prepare(frame, strategy_ids, config, load_pvpc=True):
        seen["prepare"] = config.exclude_ev
        return real_prepare(frame, strategy_ids, config, load_pvpc=load_pvpc)

    sim.prepare_inputs = prepare
    real_segment = backtest_mod._run_segment

    def run_segment(sim_obj, sl, strategy_id, cfg, shared):
        seen["segment"] = cfg.exclude_ev
        return real_segment(sim_obj, sl, strategy_id, cfg, shared)

    monkeypatch.setattr(backtest_mod, "_run_segment", run_segment)
    run_backtest(sim, "2026-03-04", "2026-03-04", "self_consume", "current")
    assert seen["prepare"] is False
    assert seen["segment"] is False
    assert sim.config.exclude_ev is True
