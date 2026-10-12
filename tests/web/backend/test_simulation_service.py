import json
import logging
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from web.backend.api.simulation import get_timeseries, start_simulation
from web.backend.models.schemas import DateRange, SimulationRequest, SystemConfig
from web.backend.services import data_service as ds_module
from web.backend.services.data_service import DataService
from web.backend.services.downsampler import downsample_indices
from web.backend.services.simulation_service import STRATEGY_MAP, STRATEGY_REGISTRY, SimulationService


def _csv(series: pd.Series) -> bytes:
    lines = ['"last_changed","state"']
    for ts, v in series.items():
        lines.append(f'"{ts.isoformat()}",' + ("" if np.isnan(v) else f'"{v}"'))
    return ("\n".join(lines) + "\n").encode()


@pytest.fixture
def data(tmp_path, monkeypatch):
    """Two local days of 1-min pv/load uploaded as CSV, with a 3 h load outage on day 2."""
    monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
    idx = pd.date_range("2026-03-04", "2026-03-06", freq="1min", tz="Europe/Madrid", inclusive="left")
    hour = idx.hour + idx.minute / 60
    pv = pd.Series(np.clip(np.sin((hour - 7.5) / 11 * np.pi), 0, None) * 4000.0, index=idx)
    load = pd.Series(600.0 + 1500.0 * ((hour >= 19) & (hour < 23)), index=idx)
    load = load[~((idx.day == 5) & (hour >= 10) & (hour < 13))]
    svc = DataService()
    svc.load_default_data()
    svc.upload_solar(_csv(pv))
    svc.upload_load(_csv(load))
    return svc


def _wait(sim: SimulationService, run_id: str, timeout: float = 30.0):
    run = sim.get_run(run_id)
    t0 = time.time()
    while run.status == "running" and time.time() - t0 < timeout:
        time.sleep(0.05)
    return run


def _request(sim):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(sim_service=sim)))


class TestValidation:
    @pytest.mark.parametrize("ids,start,end,msg", [
        ([], None, None, "at least one strategy"),
        (["nope"], None, None, "Unknown strategies"),
        (["self_consume"], "2026-13-01", None, "Invalid start"),
        (["self_consume"], "2026-03-05", "2026-03-04", "must not be after"),
    ])
    def test_bad_requests_raise_and_create_no_run(self, data, ids, start, end, msg):
        sim = SimulationService(data)
        with pytest.raises(ValueError, match=msg):
            sim.start_run(ids, start, end)
        assert sim.runs == {}

    def test_api_maps_value_error_to_422(self, data):
        sim = SimulationService(data)
        with pytest.raises(HTTPException) as exc:
            start_simulation(SimulationRequest(strategies=[]), _request(sim))
        assert exc.value.status_code == 422

    def test_no_data_in_range(self, data):
        with pytest.raises(ValueError, match="No data"):
            SimulationService(data).start_run(["self_consume"], "2027-01-01", "2027-01-02")


class TestRuns:
    def test_config_travels_with_the_run(self, data):
        sim = SimulationService(data)
        cfg = SystemConfig()
        cfg.battery.capacity = 7.0
        cfg.battery.initial_soc = 0.3
        run = _wait(sim, sim.start_run(["self_consume", "no_battery"], "2026-03-04", "2026-03-05", config=cfg))
        assert run.status == "completed"
        m = run.strategies["self_consume"].metrics
        assert m["battery_capacity"] == 7.0
        assert m["initial_battery_kwh"] == pytest.approx(2.1)
        assert sim.config.battery.capacity == SystemConfig().battery.capacity
        assert m["excluded_hours"] == pytest.approx(3.0 - 0.25)
        assert run.strategies["no_battery"].metrics["adjusted_cost"] >= m["adjusted_cost"]

    def test_min_soc_and_target_reach_strategies_at_construction(self, data):
        sim = SimulationService(data)
        cfg = SystemConfig()
        cfg.strategy.min_battery_level = 0.25
        cfg.strategy.valley_charge_target = 0.8
        cfg.battery.initial_soc = 0.5
        frame = data.get_frame("2026-03-04", "2026-03-05")
        from web.backend.services.simulation_service import RunInputs
        inputs = RunInputs(frame=frame)
        tariff = sim._build_tariff(cfg)
        oracle = sim._build_strategy("oracle", cfg, inputs, tariff)
        assert oracle.min_battery_level == 0.25
        assert oracle.lp_result.soc.min() >= 0.25 * cfg.battery.capacity - 1e-6
        night = sim._build_strategy("charge_night", cfg, inputs, tariff)
        assert night.valley_charge_target == 0.8 and night.min_battery_level == 0.25

    def test_failed_strategy_logs_traceback(self, data, caplog, monkeypatch, tmp_path):
        from src.domain.strategy import dqn_strategy
        monkeypatch.setattr(dqn_strategy, "REPO_ROOT", tmp_path)
        sim = SimulationService(data)
        with caplog.at_level(logging.ERROR):
            run = _wait(sim, sim.start_run(["dqn_agent", "self_consume"], "2026-03-04", "2026-03-04"))
        assert run.status == "completed"
        assert run.strategies["dqn_agent"].status == "failed"
        assert "DQN policy not trained" in run.strategies["dqn_agent"].error
        assert any(r.exc_info for r in caplog.records)

    def test_timeseries_api_is_json_safe_and_shows_gap(self, data):
        sim = SimulationService(data)
        run = _wait(sim, sim.start_run(["self_consume"], "2026-03-04", "2026-03-05"))
        raw = sim.get_run_timeseries(run.run_id, "self_consume")
        assert isinstance(raw["grid_import_kw"], np.ndarray)
        assert np.isnan(raw["grid_import_kw"]).any()
        resp = get_timeseries(run.run_id, _request(sim), strategy="self_consume", start=None, end=None,
                              max_points=500)
        payload = json.dumps(resp.model_dump(), allow_nan=False)
        assert "null" in payload
        assert resp.point_count <= 500 + 5
        day = get_timeseries(run.run_id, _request(sim), strategy="self_consume", start="2026-03-05T00:00:00",
                             end="2026-03-05T23:59:00", max_points=2000)
        assert day.point_count == 1440
        assert day.house_consumption[11 * 60] is None
        assert day.timestamps[0].startswith("2026-03-05T00:00:00+01:00")


class TestDataService:
    def test_upload_overrides_and_reload_drops(self, data):
        assert set(data.get_data_info().uploads) == {"pv", "load"}
        data.load_default_data()
        assert data.get_data_info().uploads == []

    def test_data_info_reports_coverage_and_gaps(self, data):
        info = data.get_data_info()
        assert info.data_from.startswith("2026-03-04T00:00:00+01:00")
        assert info.data_until.startswith("2026-03-05T23:59:00+01:00")
        assert 0.9 < info.coverage < 1.0
        assert len(info.gaps) == 1 and info.gaps[0].hours == pytest.approx(2.75)
        assert {"pv", "load"} <= set(info.entities)

    def test_frame_cache_is_lru_of_four(self, data):
        frames = [data.get_frame("2026-03-04", end) for end in ("2026-03-04", "2026-03-05")]
        assert data.get_frame("2026-03-04", "2026-03-04") is frames[0]
        for d in ("2026-03-04", "2026-03-05"):
            data.get_frame(d, d, exclude_ev=True)
            data.get_frame(d, d)
        assert len(data._frames) == ds_module.FRAME_CACHE_SIZE

    def test_bad_upload_rejected(self, data):
        with pytest.raises(ValueError):
            data.upload_load(b"foo,bar\n1,2\n")


def test_registry_groups():
    groups = {s.id: s.group for s in STRATEGY_REGISTRY}
    assert {k for k, v in groups.items() if v == "core"} == {
        "no_battery", "self_consume", "charge_night", "forecast_charge", "mpc", "mpc_perfect", "oracle"}
    assert groups["dqn_agent"] == "experimental"
    assert "no_battery" in STRATEGY_MAP


def test_lttb_skips_nan_and_keeps_gap_marker():
    y = np.sin(np.linspace(0, 20, 10_000))
    y[4000:4500] = np.nan
    idx = downsample_indices({"y": y}, 300)
    assert len(idx) <= 302
    assert 4000 in idx
    assert np.isnan(y[idx]).sum() == 1


def test_pvpc_window_stops_at_end_of_tomorrow(data):
    sim = SimulationService(data)
    frame = data.get_frame("2026-03-04", "2026-03-05")
    start, end = sim._pvpc_window(frame, SystemConfig())
    tomorrow_end = pd.Timestamp.now(tz="Europe/Madrid").normalize() + pd.Timedelta(days=2)
    assert pd.Timestamp(end) <= tomorrow_end
    assert pd.Timestamp(start) == frame.index[0]


def test_daterange_model_accepts_dates():
    assert DateRange(start="2026-03-04", end="2026-03-05").start == "2026-03-04"
