"""Sweep isolation, limits, shared inputs, and metrics-only results."""

import threading
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tests.synthetic import synthetic_frame
from tests.web.backend.test_features import _csv
from web.backend.models.feature_schemas import SweepRequest
from web.backend.models.schemas import SystemConfig
from web.backend.services.data_service import DataService
from web.backend.services.sim_worker import CaseOutput, CasePayload, execute_case
from web.backend.services.simulation_service import SimulationService
from web.backend.services.sweep_service import SweepBusy, SweepService


class _Future:
    def __init__(self, output=None, exc=None):
        self._output = output
        self._exc = exc
        self._cancelled = False

    def cancel(self) -> bool:
        self._cancelled = True
        return True

    def cancelled(self) -> bool:
        return self._cancelled

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._output


class _Pool:
    def __init__(self, submit):
        self.executor = self
        self._submit = submit
        self.manager = type("M", (), {"Event": threading.Event})()
        self.recreated = 0

    def submit(self, fn, payload):
        return self._submit(fn, payload)

    def recreate(self) -> None:
        self.recreated += 1


def _completed(strategy_id: str = "no_battery") -> CaseOutput:
    return CaseOutput(
        "j", strategy_id, "completed",
        metrics={
            "total_cost": 2.0, "adjusted_cost": 1.5, "valid_hours": 48.0,
            "equivalent_full_cycles": 0.1, "self_consumption_rate": 0.5,
        },
    )


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
    idx = pd.date_range("2026-03-04", "2026-03-06", freq="1min", tz="Europe/Madrid", inclusive="left")
    svc = DataService()
    svc.load_default_data()
    svc.upload_solar(_csv(pd.Series(np.full(len(idx), 1000.0), index=idx)))
    svc.upload_load(_csv(pd.Series(np.full(len(idx), 500.0), index=idx)))
    return svc


def _service(data, submit) -> tuple[SweepService, _Pool]:
    sim = SimulationService(data)
    pool = _Pool(submit)
    return SweepService(sim, pool, sim.history), pool


def _wait(sweep: SweepService, sweep_id: str) -> dict:
    deadline = time.time() + 5
    detail = None
    while time.time() < deadline:
        detail = sweep.get(sweep_id)
        if detail is not None and detail.status != "running":
            return detail
        time.sleep(0.02)
    return detail


def test_one_failing_case_does_not_fail_the_sweep(data):
    def submit(_fn, payload: CasePayload):
        if payload.strategy_id == "self_consume":
            return _Future(exc=ValueError("secret path /Users/leandro"))
        return _Future(output=_completed(payload.strategy_id))

    sweep, _pool = _service(data, submit)
    body = SweepRequest(
        date_range={"start": "2026-03-04", "end": "2026-03-05"},
        strategies=["self_consume"],
        tariffs=["2_0td"],
        capacities_kwh=[10],
    )
    sweep_id, total = sweep.start(body)
    assert total == 2
    detail = _wait(sweep, sweep_id)
    assert detail.status == "completed"
    by_id = {case.strategy_id: case for case in detail.cases}
    assert by_id["self_consume"].status == "failed"
    assert by_id["self_consume"].error == "simulation failed"
    assert "leandro" not in (by_id["self_consume"].error or "")
    assert by_id["no_battery"].status == "completed"
    assert all(case.status in ("completed", "failed") for case in detail.cases)
    assert sweep_id not in sweep.jobs


def test_sweep_cases_request_metrics_only(data):
    seen = []

    def submit(_fn, payload: CasePayload):
        seen.append(payload)
        return _Future(output=_completed(payload.strategy_id))

    sweep, _pool = _service(data, submit)
    sweep.start(SweepRequest(
        date_range={"start": "2026-03-04", "end": "2026-03-05"},
        strategies=["self_consume"],
        tariffs=["2_0td"],
        capacities_kwh=[10],
    ))
    deadline = time.time() + 3
    while time.time() < deadline and len(seen) < 2:
        time.sleep(0.02)
    assert seen
    assert all(payload.metrics_only for payload in seen)


def test_sweep_mpc_gets_the_same_inputs_as_simulate(data, monkeypatch):
    history = synthetic_frame(days=1, outage=False, spike=False)
    history.attrs["marker"] = "lookback"
    forecasts = {pd.Timestamp("2026-03-04").date(): 3.5}
    seen = []

    def submit(_fn, payload: CasePayload):
        seen.append(payload)
        return _Future(output=_completed(payload.strategy_id))

    sim = SimulationService(data)
    sim._history_frame = lambda frame, exclude_ev: history
    monkeypatch.setattr(
        "web.backend.services.simulation_service.load_daily_forecasts", lambda: forecasts,
    )
    monkeypatch.setattr(
        "web.backend.services.simulation_service.load_irradiance",
        lambda: pd.DataFrame({"ghi": [], "ghi_forecast": []}),
    )
    sweep = SweepService(sim, _Pool(submit), sim.history)
    sweep.start(SweepRequest(
        date_range={"start": "2026-03-04", "end": "2026-03-05"},
        strategies=["mpc", "forecast_charge"],
        tariffs=["2_0td"],
        capacities_kwh=[10],
    ))
    deadline = time.time() + 3
    while time.time() < deadline and len(seen) < 3:
        time.sleep(0.02)
    assert len(seen) == 3
    assert all(payload.history is history for payload in seen)
    assert all(payload.daily_forecasts == forecasts for payload in seen)
    assert all(payload.ghi is not None for payload in seen)


def test_execute_case_omits_timeseries_when_metrics_only():
    frame = synthetic_frame(days=1, outage=False, spike=False)
    out = execute_case(CasePayload(
        frame=frame, strategy_id="no_battery", config=SystemConfig(), metrics_only=True,
    ))
    assert out.status == "completed"
    assert out.timeseries is None
    assert out.metrics["valid_hours"] > 0


def test_sweep_limits_are_422(data):
    sweep, _pool = _service(data, lambda *_a, **_k: _Future(output=_completed()))
    base = {
        "date_range": {"start": "2026-03-04", "end": "2026-03-05"},
        "strategies": ["self_consume"],
        "tariffs": ["2_0td"],
        "capacities_kwh": [10],
    }
    with pytest.raises(ValueError, match="200"):
        sweep.start(SweepRequest(**{**base, "capacities_kwh": [float("nan")]}))
    with pytest.raises(ValueError, match="200"):
        sweep.start(SweepRequest(**{**base, "capacities_kwh": [float("inf")]}))
    with pytest.raises(ValueError, match="12 capacities"):
        sweep.start(SweepRequest(**{**base, "capacities_kwh": [1.0] * 13}))
    with pytest.raises(ValueError, match="366"):
        sweep.start(SweepRequest(**{
            **base,
            "date_range": {"start": "2025-01-01", "end": "2026-01-02"},
        }))
    with pytest.raises(ValueError, match="limit is 200"):
        sweep.start(SweepRequest(**{
            **base,
            "tariffs": ["2_0td", "pvpc", "pvpc_indexed_export"],
            "capacities_kwh": [1.0] * 12,
            "strategies": ["self_consume", "charge_night", "mpc", "oracle", "smart_discharge", "no_battery"],
        }))


def test_second_sweep_is_409(data):
    gate = threading.Event()

    def submit(_fn, payload):
        gate.wait(2)
        return _Future(output=_completed(payload.strategy_id))

    sweep, _pool = _service(data, submit)
    body = SweepRequest(
        date_range={"start": "2026-03-04", "end": "2026-03-05"},
        strategies=["self_consume"], tariffs=["2_0td"], capacities_kwh=[10],
    )
    sweep.start(body)
    with pytest.raises(SweepBusy, match="already running"):
        sweep.start(body)
    gate.set()


def test_inclusive_week_frame_covers_seven_days(tmp_path, monkeypatch):
    monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
    idx = pd.date_range("2026-10-05", "2026-10-12", freq="1min", tz="Europe/Madrid", inclusive="left")
    data = DataService()
    data.load_default_data()
    data.upload_solar(_csv(pd.Series(np.full(len(idx), 800.0), index=idx)))
    data.upload_load(_csv(pd.Series(np.full(len(idx), 400.0), index=idx)))
    seen = {}
    real = data.get_frame

    def wrapped(start, end, **kwargs):
        frame = real(start, end, **kwargs)
        seen["n"] = len(frame)
        seen["start"] = frame.index[0]
        seen["end"] = frame.index[-1]
        return frame

    data.get_frame = wrapped
    sim = SimulationService(data)
    sweep = SweepService(sim, _Pool(lambda _fn, payload: _Future(output=_completed(payload.strategy_id))), sim.history)
    sweep.start(SweepRequest(
        date_range={"start": "2026-10-05", "end": "2026-10-11"},
        strategies=["self_consume"], tariffs=["2_0td"], capacities_kwh=[5],
    ))
    assert seen["n"] == 7 * 1440
    assert seen["start"] == pd.Timestamp("2026-10-05", tz="Europe/Madrid")
    assert seen["end"] == pd.Timestamp("2026-10-11 23:59", tz="Europe/Madrid")


def test_sweep_limit_route_is_422(tmp_path, monkeypatch):
    monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("POWERFLOW_EAGER_POOL", "0")
    from web.backend.main import app

    with TestClient(app) as client:
        sim = client.app.state.sim_service
        client.app.state.sweep_service = SweepService(
            sim, _Pool(lambda *_a, **_k: _Future(output=_completed())), sim.history,
        )
        denied = client.post("/api/sweep", json={
            "date_range": {"start": "2026-03-04", "end": "2026-03-05"},
            "strategies": ["self_consume"],
            "tariffs": ["2_0td"],
            "capacities_kwh": [250],
        })
        assert denied.status_code == 422
        assert "200" in denied.json()["detail"]
