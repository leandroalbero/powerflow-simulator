"""Sweep matrix on a 2-day synthetic frame, through the process pool."""

import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tests.synthetic import synthetic_frame


@pytest.fixture
def sweep_client(tmp_path, monkeypatch):
    monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("POWERFLOW_EAGER_POOL", "1")
    from web.backend.main import app

    frame = synthetic_frame(days=2)
    with TestClient(app) as client:
        client.app.state.sim_service.data_service.get_frame = lambda *args, **kwargs: frame

        def prices(start, end, strict=False):
            idx = pd.date_range(start, end, freq="h")
            return {ts.to_pydatetime(): 0.12 for ts in idx}

        client.app.state.sim_service.pvpc_service.get_prices = prices
        yield client


def test_simulate_strategies_run_in_the_process_pool(sweep_client: TestClient):
    started = sweep_client.post("/api/simulate", json={
        "strategies": ["no_battery", "self_consume"],
        "date_range": {"start": "2026-03-04", "end": "2026-03-06"},
        "label": "pooled",
    })
    assert started.status_code == 200, started.text
    run_id = started.json()["run_id"]
    deadline = time.time() + 60
    body = {}
    while time.time() < deadline:
        body = sweep_client.get(f"/api/runs/{run_id}/summary").json()
        if body["status"] != "running":
            break
        time.sleep(0.1)
    assert body["status"] == "completed", body
    assert body["label"] == "pooled"
    assert {item["strategy_id"] for item in body["strategies"]} == {"no_battery", "self_consume"}
    series = sweep_client.get(f"/api/runs/{run_id}/timeseries", params={"strategy": "self_consume", "max_points": 100})
    assert series.status_code == 200
    payload = series.json()
    assert payload["downsampled"] is True
    assert len(payload["ts"]) <= 200
    assert "battery_charge" in payload and "step_cost" in payload


def test_sweep_savings_sign_and_baseline_count(sweep_client: TestClient):
    started = sweep_client.post("/api/sweep", json={
        "date_range": {"start": "2026-03-04", "end": "2026-03-06"},
        "strategies": ["self_consume"],
        "tariffs": ["2_0td", "pvpc"],
        "capacities_kwh": [10, 15.36],
        "label": "matrix",
    })
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["total_cases"] == 6
    sweep_id = body["sweep_id"]
    deadline = time.time() + 120
    detail = {}
    while time.time() < deadline:
        detail = sweep_client.get(f"/api/sweep/{sweep_id}").json()
        if detail["status"] != "running":
            break
        time.sleep(0.2)
    assert detail["status"] == "completed", detail
    cases = detail["cases"]
    baselines = [case for case in cases if case["capacity_kwh"] == 0 and case["strategy_id"] == "no_battery"]
    assert len(baselines) == 2
    by_tariff = {case["tariff"]: case["adjusted_cost"] for case in baselines}
    for case in cases:
        if case["capacity_kwh"] == 0 and case["strategy_id"] == "no_battery":
            assert case["savings_vs_baseline"] == pytest.approx(0.0)
            continue
        expected = by_tariff[case["tariff"]] - case["adjusted_cost"]
        assert case["savings_vs_baseline"] == pytest.approx(expected)
        assert case["status"] == "completed"
    self_consume = [case for case in cases if case["strategy_id"] == "self_consume"]
    assert self_consume
    assert any(case["savings_vs_baseline"] > 0 for case in self_consume)
    export = sweep_client.get(f"/api/sweep/{sweep_id}/export.csv")
    assert export.status_code == 200
    assert "attachment" in export.headers["content-disposition"]
    assert "savings_vs_baseline" in export.text.splitlines()[0]
    missing = sweep_client.post("/api/sweep/missing/cancel")
    assert missing.status_code == 404
    listed = sweep_client.get("/api/sweeps").json()
    assert any(item["sweep_id"] == sweep_id for item in listed)
