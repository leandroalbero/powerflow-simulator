"""Run listing, delete, timeseries status, and websocket replay."""

import asyncio
import json
import threading
import time
import uuid

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tests.web.backend.test_features import _csv
from web.backend.models.schemas import SystemConfig
from web.backend.services.data_service import DataService
from web.backend.services.simulation_service import RunStillActive, SimulationRun, SimulationService


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
    idx = pd.date_range("2026-03-04", "2026-03-06", freq="1min", tz="Europe/Madrid", inclusive="left")
    svc = DataService()
    svc.load_default_data()
    svc.upload_solar(_csv(pd.Series(np.full(len(idx), 1000.0), index=idx)))
    svc.upload_load(_csv(pd.Series(np.full(len(idx), 500.0), index=idx)))
    return svc


def test_delete_active_run_is_409(data, monkeypatch):
    gate = threading.Event()

    def block(frame, strategy, tariff, on_progress=None):
        if on_progress is not None:
            on_progress(1)
        gate.wait(5)
        raise AssertionError("deleted run kept simulating")

    monkeypatch.setattr("web.backend.services.simulation_service.simulate", block)
    svc = SimulationService(data)
    run_id = svc.start_run(["self_consume"], "2026-03-04", "2026-03-05")
    deadline = time.time() + 2
    while time.time() < deadline and svc.get_run(run_id).strategies["self_consume"].status != "running":
        time.sleep(0.01)
    with pytest.raises(RunStillActive, match="cancel first"):
        svc.delete_run(run_id)
    assert run_id in svc.runs
    gate.set()
    svc._executor.shutdown(wait=False, cancel_futures=True)


def test_failed_strategy_timeseries_is_missing_not_evicted(data, monkeypatch):
    def boom(frame, strategy, tariff, on_progress=None):
        raise RuntimeError("solver blew up at /secret/path")

    monkeypatch.setattr("web.backend.services.simulation_service.simulate", boom)
    svc = SimulationService(data)
    run_id = svc.start_run(["no_battery"], "2026-03-04", "2026-03-05")
    deadline = time.time() + 3
    while time.time() < deadline and svc.get_run(run_id).status == "running":
        time.sleep(0.01)
    status, _ts = svc.fetch_timeseries(run_id, "no_battery")
    assert status == "missing"
    assert svc.get_run(run_id).strategies["no_battery"].error == "simulation failed"


def test_list_runs_snapshot_under_lock(data):
    svc = SimulationService(data)
    stop = threading.Event()
    errors = []

    def writer() -> None:
        while not stop.is_set():
            with svc._registry_lock:
                svc.runs[uuid.uuid4().hex] = SimulationRun(
                    run_id="x", created_at="t", config=SystemConfig(),
                )

    def reader() -> None:
        while not stop.is_set():
            try:
                svc.list_runs(50)
            except RuntimeError as exc:
                errors.append(exc)
                stop.set()

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for thread in threads:
        thread.start()
    time.sleep(0.2)
    stop.set()
    for thread in threads:
        thread.join(timeout=2)
    assert errors == []


def test_replay_does_not_complete_before_strategy_done(data, monkeypatch):
    svc = SimulationService(data)
    run = SimulationRun(run_id="abc", status="running", created_at="t", config=SystemConfig())
    from web.backend.services.simulation_service import StrategyRunResult

    run.strategies["A"] = StrategyRunResult(strategy_id="A", status="running", progress=50)
    run.strategies["B"] = StrategyRunResult(strategy_id="B", status="completed", metrics={"total_cost": 1.0})
    svc.runs["abc"] = run

    class _WS:
        def __init__(self) -> None:
            self.app = type("A", (), {"state": type("S", (), {"sim_service": svc})()})()
            self.messages = []

        async def accept(self) -> None:
            return None

        async def send_text(self, text: str) -> None:
            msg = json.loads(text)
            self.messages.append(msg)
            if msg.get("strategy") == "A" and "percent" in msg:
                with run.lock:
                    run.strategies["A"].status = "completed"
                    run.strategies["A"].metrics = {"total_cost": 2.0}
                    run.status = "completed"

        async def send_json(self, obj) -> None:
            self.messages.append(obj)

        async def close(self) -> None:
            self.messages.append({"closed": True})

    async def fast_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("web.backend.api.simulation.asyncio.sleep", fast_sleep)
    from web.backend.api.simulation import ws_run_progress

    ws = _WS()
    asyncio.run(ws_run_progress(ws, "abc"))
    kinds = [msg.get("status") or msg.get("strategy") for msg in ws.messages]
    complete_at = next(i for i, msg in enumerate(ws.messages) if msg.get("status") == "run_complete")
    done = [msg for msg in ws.messages[:complete_at] if msg.get("strategy") == "A" and msg.get("status") == "completed"]
    assert done, kinds


def test_delete_route_409(tmp_path, monkeypatch):
    monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("POWERFLOW_EAGER_POOL", "0")
    from web.backend.main import app

    with TestClient(app) as client:
        sim = client.app.state.sim_service
        run = SimulationRun(run_id="live", status="running", created_at="t", config=SystemConfig())
        sim.runs["live"] = run
        denied = client.delete("/api/runs/live")
        assert denied.status_code == 409
        assert denied.json()["detail"] == "run still active, cancel first"
        assert "live" in sim.runs
