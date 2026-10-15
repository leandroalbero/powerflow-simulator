"""A refresh failure must end the job, and the body is optional."""

import time

import pytest
from fastapi.testclient import TestClient

from src.data.influx import InfluxHttpError


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("POWERFLOW_EAGER_POOL", "0")
    from web.backend.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_influx_error_fails_the_job_and_hides_the_url(client: TestClient, monkeypatch):
    monkeypatch.setattr("src.data.config.load_influx_config", lambda env_path=None: object())

    def raise_influx(**_kwargs):
        raise InfluxHttpError(401)

    monkeypatch.setattr("src.data.refresh.refresh", raise_influx)
    started = client.post("/api/data/refresh")
    assert started.status_code == 202
    job_id = started.json()["job_id"]
    deadline = time.time() + 5
    body = {}
    while time.time() < deadline:
        body = client.get(f"/api/data/refresh/{job_id}").json()
        if body["status"] != "running":
            break
        time.sleep(0.05)
    assert body["status"] == "failed"
    assert body["error"] == "refresh failed"
    assert "influx" not in (body["error"] or "")
    assert "401" not in (body["error"] or "")
    again = client.post("/api/data/refresh", json={})
    assert again.status_code == 202
