"""A spring-forward must not let a 367-day range through the 366-day cap."""

import time

import pytest
from fastapi.testclient import TestClient

from tests.synthetic import synthetic_frame
from web.backend.services.span import inclusive_day_count
from web.backend.services.sweep_service import SweepService


class _Future:
    def __init__(self, output):
        self._output = output

    def cancel(self) -> bool:
        return True

    def cancelled(self) -> bool:
        return False

    def result(self, timeout=None):
        return self._output


class _Pool:
    def __init__(self):
        self.executor = self
        self.manager = type("M", (), {"Event": __import__("threading").Event})()

    def submit(self, _fn, payload):
        from tests.web.backend.test_sweep_review import _completed

        return _Future(_completed(payload.strategy_id))

    def recreate(self) -> None:
        return None


def test_plain_dates_count_calendar_days_and_datetimes_ceil():
    assert inclusive_day_count("2026-03-29", "2027-03-30") == 367
    assert inclusive_day_count("2026-03-29", "2027-03-29") == 366
    assert inclusive_day_count("2026-03-29T00:00", "2027-03-30T23:59") == 367
    assert inclusive_day_count("2026-03-29T00:00", "2027-03-29T23:59") == 366


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("POWERFLOW_EAGER_POOL", "0")
    from web.backend.main import app

    frame = synthetic_frame(days=1, outage=False, spike=False)
    frame["grid_kw"] = 0.0
    frame["soc_pct"] = 50.0
    with TestClient(app) as test_client:
        sim = test_client.app.state.sim_service
        monkeypatch.setattr(sim.data_service, "get_frame", lambda *_args, **_kwargs: frame)
        test_client.app.state.sweep_service = SweepService(sim, _Pool(), sim.history)
        yield test_client


def test_sweep_and_backtest_reject_367_days_and_accept_366(client: TestClient):
    denied_sweep = client.post("/api/sweep", json=_sweep_body("2026-03-29", "2027-03-30"))
    assert denied_sweep.status_code == 422
    assert "366" in denied_sweep.json()["detail"]

    denied_backtest = client.post("/api/backtest", json=_backtest_body("2026-03-29", "2027-03-30"))
    assert denied_backtest.status_code == 422
    assert "1 year" in denied_backtest.json()["detail"]

    accepted_sweep = client.post("/api/sweep", json=_sweep_body("2026-03-29", "2027-03-29"))
    assert accepted_sweep.status_code == 200
    assert "sweep_id" in accepted_sweep.json()
    sweep_id = accepted_sweep.json()["sweep_id"]
    deadline = time.time() + 5
    body = {}
    while time.time() < deadline:
        body = client.get(f"/api/sweep/{sweep_id}").json()
        if body["status"] != "running":
            break
        time.sleep(0.02)
    assert body["status"] != "running"

    accepted_backtest = client.post("/api/backtest", json=_backtest_body("2026-03-29", "2027-03-29"))
    assert accepted_backtest.status_code == 200
    assert accepted_backtest.json()["strategy_id"] == "self_consume"


def _sweep_body(start: str, end: str) -> dict:
    return {
        "date_range": {"start": start, "end": end},
        "strategies": ["self_consume"],
        "tariffs": ["2_0td"],
        "capacities_kwh": [10],
    }


def _backtest_body(start: str, end: str) -> dict:
    return {
        "date_range": {"start": start, "end": end},
        "strategy_id": "self_consume",
        "hardware": "current",
    }
