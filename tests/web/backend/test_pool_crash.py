"""A real worker crash or shutdown must not deadlock the pool, and late failures never resurrect runs."""

import threading
import time

import pytest

import web.backend.services.sim_worker as sim_worker
from tests.web.backend import crash_cases
from tests.web.backend.test_cancel_pool import data  # noqa: F401 — fixture
from web.backend.services.pool import SimPool
from web.backend.services.sim_worker import CasePayload
from web.backend.services.simulation_service import SimulationService


def _wait(predicate, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


@pytest.fixture
def pool():
    pool = SimPool()
    yield pool
    pool.shutdown()


def test_worker_crash_fails_the_strategy_and_the_pool_recovers(data, pool, monkeypatch):  # noqa: F811
    monkeypatch.setattr(sim_worker, "execute_case", crash_cases.crash_or_run)
    svc = SimulationService(data)
    svc.attach_pool(pool)
    first = pool.executor
    run_id = svc.start_run(["no_battery", "self_consume"], "2026-03-04", "2026-03-04")
    run = svc.runs[run_id]

    assert _wait(lambda: run.status != "running", 20), {k: v.status for k, v in run.strategies.items()}
    assert run.strategies["no_battery"].status == "failed"
    assert run.strategies["no_battery"].error == "worker process crashed"
    assert run.strategies["self_consume"].status in ("completed", "failed")
    assert svc.history.get_run(run_id)["status"] == run.status
    assert pool.executor is not first
    probe = pool.executor.submit(crash_cases.quick, CasePayload(frame=None, strategy_id="x", config=None, job_id="p"))
    assert probe.result(timeout=20).status == "completed"


def test_recreate_only_replaces_a_broken_executor(pool):
    healthy = pool.executor
    pool.recreate()
    assert pool.executor is healthy


def test_shutdown_with_work_in_flight_lets_the_manager_thread_exit(data, pool, monkeypatch):  # noqa: F811
    monkeypatch.setattr(sim_worker, "execute_case", crash_cases.slow)
    svc = SimulationService(data)
    svc.attach_pool(pool)
    executor = pool.executor
    run_id = svc.start_run(["self_consume"], "2026-03-04", "2026-03-04")
    run = svc.runs[run_id]
    assert _wait(lambda: run.futures["self_consume"].running(), 20)

    manager = executor._executor_manager_thread
    pool.shutdown()
    manager.join(timeout=10)
    assert not manager.is_alive()
    assert pool.executor is executor


def test_prepare_failure_after_cancel_and_delete_does_not_resurrect(data, monkeypatch):  # noqa: F811
    svc = SimulationService(data)
    release = threading.Event()

    def slow_fail(*_args, **_kwargs):
        release.wait(5)
        raise RuntimeError("REE fetch failed")

    monkeypatch.setattr(svc, "prepare_inputs", slow_fail)
    errors: list[BaseException] = []

    def start() -> None:
        try:
            svc.start_run(["self_consume"], "2026-03-04", "2026-03-04")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=start)
    thread.start()
    assert _wait(lambda: bool(svc.runs), 5)
    run_id = next(iter(svc.runs))
    assert svc.cancel_run(run_id) == "cancelled"
    assert svc.delete_run(run_id)
    release.set()
    thread.join(5)

    assert svc.history.get_run(run_id) is None
    assert run_id not in {row["run_id"] for row in svc.list_runs()}


def test_persist_after_delete_is_a_no_op(data):  # noqa: F811
    svc = SimulationService(data)
    run_id = svc.start_run(["no_battery"], "2026-03-04", "2026-03-04")
    run = svc.runs[run_id]
    assert _wait(lambda: run.status == "completed", 20)
    assert svc.delete_run(run_id)
    svc._persist_run(run)
    assert svc.history.get_run(run_id) is None
