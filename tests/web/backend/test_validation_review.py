"""Labels longer than 120 characters are rejected before they are stored."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("POWERFLOW_EAGER_POOL", "0")
    from web.backend.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_label_over_120_is_422(client: TestClient):
    too_long = "x" * 121
    simulate = client.post("/api/simulate", json={"strategies": ["no_battery"], "label": too_long})
    assert simulate.status_code == 422
    assert "120" in simulate.text
    sweep = client.post("/api/sweep", json={
        "date_range": {"start": "2026-10-05", "end": "2026-10-11"},
        "strategies": ["self_consume"],
        "tariffs": ["2_0td"],
        "capacities_kwh": [10],
        "label": too_long,
    })
    assert sweep.status_code == 422
    assert "120" in sweep.text
    exact = client.post("/api/simulate", json={"strategies": ["no_battery"], "label": "x" * 120})
    assert "at most 120" not in exact.text
