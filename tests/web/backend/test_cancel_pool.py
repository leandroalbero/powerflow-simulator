"""Cancel must stop queued work, and a dead pool must not leave ghost runs."""

import queue
import threading
import time
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.synthetic import synthetic_frame
from tests.web.backend.test_features import _csv
from web.backend.models.schemas import SystemConfig
from web.backend.services.cancel import RunCancelled
from web.backend.services.data_service import DataService
from web.backend.services.history_store import HistoryStore
from web.backend.services.pool import SimPool
from web.backend.services.sim_worker import CasePayload, execute_case
from web.backend.services.simulation_service import RunInputs, SimulationService


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
    idx = pd.date_range("2026-03-04", "2026-03-06", freq="1min", tz="Europe/Madrid", inclusive="left")
    pv = pd.Series(np.full(len(idx), 1000.0), index=idx)
    load = pd.Series(np.full(len(idx), 500.0), index=idx)
    svc = DataService()
    svc.load_default_data()
    svc.upload_solar(_csv(pv))
    svc.upload_load(_csv(load))
    return svc


def test_execute_case_skips_build_when_already_cancelled():
    frame = synthetic_frame(days=1, outage=False, spike=False)
    cancel = threading.Event()
    cancel.set()
    payload = CasePayload(
        frame=frame, strategy_id="oracle", config=SystemConfig(), cancel=cancel, job_id="j",
    )
    assert execute_case(payload).status == "cancelled"


def test_oracle_chunk_hook_fires_before_the_lp(data):
    frame = synthetic_frame(days=1, outage=False, spike=False)
    svc = SimulationService(data)
    calls = {"n": 0}

    def on_chunk() -> None:
        calls["n"] += 1
        raise RunCancelled()

    with pytest.raises(RunCancelled):
        svc._build_strategy(
            "oracle", SystemConfig(), RunInputs(frame=frame), svc._build_tariff(), on_chunk=on_chunk,
        )
    assert calls["n"] == 1


def test_cancel_does_not_overwrite_a_queued_strategy(data):
    svc = SimulationService(data)
    gate = threading.Event()
    for _ in range(4):
        svc._executor.submit(gate.wait)
    run_id = svc.start_run(["self_consume"], "2026-03-04", "2026-03-05")
    assert svc.cancel_run(run_id) == "cancelled"
    assert svc.runs[run_id].futures["self_consume"].cancelled()
    gate.set()
    deadline = time.time() + 3
    run = svc.get_run(run_id)
    while time.time() < deadline and run.strategies["self_consume"].status == "running":
        time.sleep(0.02)
    assert run.strategies["self_consume"].status == "cancelled"
    assert run.status == "cancelled"
    stored = svc.history.get_run(run_id)
    assert stored["status"] == "cancelled"
    assert stored["strategies"][0]["status"] == "cancelled"
    svc._executor.shutdown(wait=False, cancel_futures=True)


def test_broken_pool_fails_the_run_and_recreates(data):
    class _Boom:
        def submit(self, *_args, **_kwargs):
            raise BrokenProcessPool("dead")

    class _Pool:
        def __init__(self) -> None:
            self.executor = _Boom()
            self.recreated = 0
            self.manager = SimpleNamespace(Event=threading.Event)

        def recreate(self) -> None:
            self.recreated += 1

    svc = SimulationService(data)
    pool = _Pool()
    svc.pool = pool
    svc._progress_q = queue.Queue()
    run_id = svc.start_run(["no_battery"], "2026-03-04", "2026-03-05")
    run = svc.get_run(run_id)
    assert run.status == "failed"
    assert run.strategies["no_battery"].status == "failed"
    assert run.strategies["no_battery"].error == "worker process crashed"
    assert pool.recreated == 1
    assert "dead" not in (run.strategies["no_battery"].error or "")


def test_restart_marks_persisted_running_jobs_failed(tmp_path):
    path = tmp_path / "runs.db"
    store = HistoryStore(path)
    config = SystemConfig().model_dump(mode="json")
    store.upsert_run(
        run_id="run1", label=None, created_at="2026-10-13T00:00:00+02:00", status="running",
        date_start="2026-10-05", date_end="2026-10-05", config=config,
        strategies=[{"strategy_id": "no_battery", "status": "running", "progress": 10, "metrics": None, "error": None}],
    )
    store.upsert_sweep({
        "sweep_id": "sw1", "label": None, "created_at": "2026-10-13T00:00:00+02:00", "status": "running",
        "date_start": "2026-10-05", "date_end": "2026-10-05", "done": 0, "total": 1,
        "cases": [{"tariff": "2_0td", "capacity_kwh": 5, "strategy_id": "self_consume", "status": "running"}],
        "request": {},
    })
    store.close()
    fresh = HistoryStore(path)
    run = fresh.get_run("run1")
    sweep = fresh.get_sweep("sw1")
    assert run["status"] == "failed"
    assert run["strategies"][0]["error"] == "interrupted by restart"
    assert sweep["status"] == "failed"
    assert sweep["cases"][0]["status"] == "failed"
    assert sweep["cases"][0]["error"] == "interrupted by restart"
    fresh.close()


def test_sqlite_write_after_close_is_a_no_op(tmp_path):
    store = HistoryStore(tmp_path / "runs.db")
    store.close()
    store.upsert_run(
        run_id="x", label=None, created_at="t", status="completed",
        date_start="2026-10-05", date_end="2026-10-05", config={}, strategies=[],
    )


def test_pool_shutdown_does_not_wait_out_a_stuck_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
    pool = SimPool()
    pool.executor.submit(time.sleep, 30)
    started = time.monotonic()
    pool.shutdown()
    assert time.monotonic() - started < 8


