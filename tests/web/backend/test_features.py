"""Feature-phase API: history, envelope, presets, refresh, sweep, backtest."""

import csv
import io
import threading
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data.config import MissingCredentialError
from tests.synthetic import synthetic_frame
from web.backend.models.feature_schemas import SweepDetail
from web.backend.models.schemas import SystemConfig
from web.backend.services.data_service import DataService
from web.backend.services.envelope import envelope_downsample
from web.backend.services.refresh_jobs import RefreshRegistry
from web.backend.services.simulation_service import SimulationService


def _csv(series: pd.Series) -> bytes:
    lines = ['"last_changed","state"']
    for ts, value in series.items():
        lines.append(f'"{ts.isoformat()}",' + ("" if np.isnan(value) else f'"{value}"'))
    return ("\n".join(lines) + "\n").encode()


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
    idx = pd.date_range("2026-03-04", "2026-03-06", freq="1min", tz="Europe/Madrid", inclusive="left")
    hour = idx.hour + idx.minute / 60
    pv = pd.Series(np.clip(np.sin((hour - 7.5) / 11 * np.pi), 0, None) * 4000.0, index=idx)
    load = pd.Series(600.0 + 1500.0 * ((hour >= 19) & (hour < 23)), index=idx)
    svc = DataService()
    svc.load_default_data()
    svc.upload_solar(_csv(pv))
    svc.upload_load(_csv(load))
    return svc


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("POWERFLOW_EAGER_POOL", "0")
    from web.backend.main import app

    with TestClient(app) as test_client:
        yield test_client


def _wait_summary(client: TestClient, run_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    body = {}
    while time.time() < deadline:
        body = client.get(f"/api/runs/{run_id}/summary").json()
        if body.get("status") != "running":
            return body
        time.sleep(0.05)
    return body


class TestEnvelope:
    def test_keeps_single_minute_spike_and_sums_step_cost(self):
        n = 10_000
        ts = np.arange(n, dtype=np.int64) * 60_000
        solar = np.zeros(n)
        solar[4321] = 11.0
        cost = np.ones(n)
        env = envelope_downsample(ts, {"solar_power": solar, "grid_import": np.zeros(n)}, cost, 200)
        assert env.downsampled
        assert 11.0 in env.columns["solar_power"]
        assert len(env.ts_ms) <= 400
        assert np.nansum(env.step_cost) == pytest.approx(np.nansum(cost))

    def test_nan_gap_stays_null(self):
        n = 500
        ts = np.arange(n, dtype=np.int64) * 60_000
        grid = np.ones(n)
        grid[200:260] = np.nan
        env = envelope_downsample(ts, {"grid_import": grid}, np.zeros(n), 20)
        assert np.isnan(env.columns["grid_import"]).any()


class TestStrategiesAndPresets:
    def test_dqn_unavailable_and_presets(self, client: TestClient):
        listed = client.get("/api/strategies").json()
        dqn = next(item for item in listed if item["id"] == "dqn_agent")
        assert dqn["available"] is False
        assert dqn["unavailable_reason"]
        assert next(item for item in listed if item["id"] == "self_consume")["available"] is True
        presets = client.get("/api/presets").json()
        hardware = {item["id"]: item for item in presets["hardware"]}
        assert hardware["current"]["battery"]["capacity"] == 15.36
        assert hardware["current"]["battery"]["max_charge_rate"] == 4.5
        assert hardware["current"]["battery"]["max_discharge_rate"] == 4.6
        assert hardware["current"]["valid_from"] == "2026-04-07"
        assert hardware["pre_2026_04"]["battery"]["max_charge_rate"] == 2.2
        assert hardware["pre_2026_04"]["battery"]["max_discharge_rate"] == 2.1
        assert hardware["pre_2026_04"]["valid_until"] == "2026-04-06"
        assert hardware["double_battery"]["battery"]["capacity"] == 30.72
        tariffs = {item["id"]: item["tariff"] for item in presets["tariffs"]}
        assert tariffs["2_0td"]["use_pvpc"] is False
        assert tariffs["pvpc"]["use_pvpc"] is True
        assert tariffs["pvpc"]["export_follows_pvpc"] is False
        assert tariffs["pvpc_indexed_export"]["export_follows_pvpc"] is True
        assert tariffs["pvpc_indexed_export"]["export_pvpc_discount"] == 0.02

    def test_data_info_flags(self, data, monkeypatch):
        info = data.get_data_info()
        assert info.has_ev is False
        assert info.has_measured is False
        data._info = None
        monkeypatch.setattr(data, "_entities_available", lambda: ["pv", "load", "grid", "battery", "soc", "ev"])
        info = data.get_data_info()
        assert info.has_ev is True
        assert info.has_measured is True


class TestRuns:
    def test_label_summary_cancel_and_unknown(self, client: TestClient, data, monkeypatch):
        client.app.state.data_service = data
        client.app.state.sim_service = SimulationService(data)
        missing = client.post("/api/runs/nope/cancel")
        assert missing.status_code == 404
        assert client.get("/api/runs/nope/summary").status_code == 404
        assert client.delete("/api/runs/nope").status_code == 404

        def slow(frame, strategy, tariff, on_progress=None):
            for pct in range(0, 101, 5):
                if on_progress is not None:
                    on_progress(pct)
                time.sleep(0.03)
            raise AssertionError("cancel did not stop the run")

        import web.backend.services.simulation_service as sim_mod
        monkeypatch.setattr(sim_mod, "simulate", slow)
        started = client.post("/api/simulate", json={
            "strategies": ["self_consume"],
            "date_range": {"start": "2026-03-04", "end": "2026-03-05"},
            "label": "night",
        })
        assert started.status_code == 200
        run_id = started.json()["run_id"]
        time.sleep(0.05)
        cancelled = client.post(f"/api/runs/{run_id}/cancel")
        assert cancelled.status_code == 200
        body = _wait_summary(client, run_id)
        assert body["label"] == "night"
        assert body["created_at"]
        assert body["date_range"]["start"] == "2026-03-04"
        assert body["config"]["battery"]["capacity"] == 15.36
        assert body["strategies"][0]["status"] == "cancelled"
        assert "progress" in body["strategies"][0]

    def test_history_csv_and_eviction(self, client: TestClient, data, tmp_path, monkeypatch):
        client.app.state.data_service = data
        sim = SimulationService(data)
        client.app.state.sim_service = sim
        ids = []
        for _ in range(9):
            run_id = sim.start_run(["no_battery"], "2026-03-04", "2026-03-05", label="hist")
            ids.append(run_id)
            deadline = time.time() + 30
            while sim.get_run(run_id).status == "running" and time.time() < deadline:
                time.sleep(0.02)
        expired = client.get(f"/api/runs/{ids[0]}/timeseries", params={"strategy": "no_battery"})
        assert expired.status_code == 410
        assert expired.json()["detail"] == "timeseries expired, re-run"
        fresh = SimulationService(data)
        listed = fresh.list_runs(50)
        assert any(item["run_id"] == ids[-1] for item in listed)
        assert fresh.fetch_timeseries(ids[-1], "no_battery")[0] == "evicted"
        client.app.state.sim_service = sim
        export = client.get(f"/api/runs/{ids[-1]}/export.csv", params={"kind": "summary"})
        assert export.status_code == 200
        assert "attachment" in export.headers["content-disposition"]
        rows = list(csv.reader(io.StringIO(export.text)))
        assert rows[0][0] == "strategy_id"
        assert rows[1][0] == "no_battery"
        assert client.get(f"/api/runs/{ids[-1]}/export.csv", params={"kind": "timeseries"}).status_code == 422
        gone = client.delete(f"/api/runs/{ids[-1]}")
        assert gone.status_code == 204
        assert client.get(f"/api/runs/{ids[-1]}/summary").status_code == 404


class TestRefresh:
    def test_503_409_and_completion(self, client: TestClient, monkeypatch):
        def boom(env_path=None):
            raise MissingCredentialError("INFLUX_URL")

        monkeypatch.setattr("src.data.config.load_influx_config", boom)
        denied = client.post("/api/data/refresh", json={})
        assert denied.status_code == 503
        assert denied.json()["detail"] == "INFLUX_* not configured"

        monkeypatch.undo()
        monkeypatch.setattr("src.data.config.load_influx_config", lambda env_path=None: object())
        gate = threading.Event()

        def fake_refresh(**kwargs):
            kwargs["progress"]("pv added=1")
            gate.wait(5)
            return {"updated_at": "2026-10-12T00:00:00Z", "entities": {
                "pv": {"rows_added": 1, "rows": 1, "first_ts": None, "last_ts": None,
                       "gaps_over_1h": 0, "fetched": 1},
            }}

        monkeypatch.setattr("src.data.refresh.refresh", fake_refresh)
        reloaded = {"n": 0}
        client.app.state.refresh_registry = RefreshRegistry(lambda: reloaded.__setitem__("n", reloaded["n"] + 1))
        first = client.post("/api/data/refresh", json={"full": False})
        assert first.status_code == 202
        job_id = first.json()["job_id"]
        second = client.post("/api/data/refresh", json={})
        assert second.status_code == 409
        gate.set()
        deadline = time.time() + 5
        body = {}
        while time.time() < deadline:
            body = client.get(f"/api/data/refresh/{job_id}").json()
            if body["status"] != "running":
                break
            time.sleep(0.05)
        assert body["status"] == "completed"
        assert body["summary"]["entities"]["pv"]["rows_added"] == 1
        assert reloaded["n"] == 1
        assert client.get("/api/data/refresh/missing").status_code == 404


class TestBacktest:
    def test_zero_error_when_measured_matches_sim(self, data, monkeypatch):
        sim = SimulationService(data)
        cfg = SystemConfig()
        frame = synthetic_frame(days=2)
        from src.domain.energy_simulator.models import run as simulate
        from web.backend.services.simulation_service import RunInputs

        inputs = RunInputs(frame=frame)
        tariff = sim._build_tariff(cfg)
        strategy = sim._build_strategy("self_consume", cfg, inputs, tariff)
        result = simulate(frame, strategy, tariff)
        grid = np.where(
            np.isfinite(result.timeseries["grid_import_kw"]),
            np.nan_to_num(result.timeseries["grid_export_kw"]) - np.nan_to_num(result.timeseries["grid_import_kw"]),
            np.nan,
        )
        frame = frame.copy()
        frame["grid_kw"] = grid
        frame["battery_kw"] = 0.0
        frame["soc_pct"] = 50.0
        monkeypatch.setattr(data, "get_frame", lambda *args, **kwargs: frame)
        sim.data_service = data
        from web.backend.services.backtest_service import run_backtest

        out = run_backtest(sim, "2026-03-04", "2026-03-06", "self_consume", "current")
        assert out.metrics.import_error_pct == pytest.approx(0.0, abs=1e-6)
        assert out.metrics.export_error_pct == pytest.approx(0.0, abs=1e-6)
        assert out.metrics.cost_error_pct == pytest.approx(0.0, abs=1e-4)
        assert out.hardware_used[0].preset_id == "current"

    def test_auto_splits_and_chains_soc(self, data, monkeypatch):
        sim = SimulationService(data)
        idx = pd.date_range("2026-04-06", "2026-04-08", freq="1min", tz="Europe/Madrid", inclusive="left")
        frame = pd.DataFrame({
            "pv_kw": np.full(len(idx), 0.2),
            "load_kw": np.full(len(idx), 0.2),
            "ev_kw": np.zeros(len(idx)),
            "valid": np.ones(len(idx), dtype=bool),
            "grid_kw": np.zeros(len(idx)),
            "battery_kw": np.zeros(len(idx)),
            "soc_pct": np.full(len(idx), 90.0),
        }, index=idx)
        frame.index.name = "ts"
        monkeypatch.setattr(data, "get_frame", lambda *args, **kwargs: frame)
        sim.data_service = data
        from web.backend.services.backtest_service import run_backtest

        out = run_backtest(sim, "2026-04-06", "2026-04-08", "self_consume", "auto")
        assert [seg.preset_id for seg in out.hardware_used] == ["pre_2026_04", "current"]
        switch = pd.Timestamp("2026-04-07", tz="Europe/Madrid")
        stamps = pd.to_datetime(out.soc.ts, unit="ms", utc=True).tz_convert("Europe/Madrid")
        before = [pct for stamp, pct in zip(stamps, out.soc.sim_pct) if stamp < switch and pct is not None]
        after = [pct for stamp, pct in zip(stamps, out.soc.sim_pct) if stamp >= switch and pct is not None]
        assert before and after
        assert abs(before[-1] - after[0]) < 2.0
        assert after[0] > 80

    def test_rejects_long_range_and_missing_measured(self, client: TestClient, data):
        client.app.state.data_service = data
        client.app.state.sim_service = SimulationService(data)
        too_long = client.post("/api/backtest", json={
            "date_range": {"start": "2024-01-01", "end": "2026-01-01"},
        })
        assert too_long.status_code == 422
        no_measured = client.post("/api/backtest", json={
            "date_range": {"start": "2026-03-04", "end": "2026-03-05"},
        })
        assert no_measured.status_code == 422


def test_sweep_ws_replays_finished_cases(client: TestClient):
    class _Fake:
        def get(self, sweep_id: str):
            if sweep_id != "abc":
                return None
            return SweepDetail(
                sweep_id="abc",
                label=None,
                created_at="2026-10-12T00:00:00+02:00",
                status="completed",
                date_range={"start": "2026-03-04", "end": "2026-03-06"},
                done=1,
                total=1,
                cases=[{
                    "tariff": "2_0td",
                    "capacity_kwh": 0,
                    "strategy_id": "no_battery",
                    "status": "completed",
                    "adjusted_cost": 1.0,
                    "savings_vs_baseline": 0.0,
                }],
            )

    client.app.state.sweep_service = _Fake()
    with client.websocket_connect("/api/ws/sweep/abc") as ws:
        first = ws.receive_json()
        assert first["type"] == "case_complete"
        assert first["case"]["strategy_id"] == "no_battery"
        kinds = {first["type"]}
        while "sweep_complete" not in kinds:
            msg = ws.receive_json()
            kinds.add(msg["type"])
        assert "progress" in kinds
