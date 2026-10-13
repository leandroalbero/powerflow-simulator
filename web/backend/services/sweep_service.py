"""Scenario matrix. One frame, process-pool cases, baseline per tariff."""

import logging
import math
import sqlite3
import threading
import uuid
from concurrent.futures import CancelledError, Future
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from web.backend.models.feature_schemas import SweepCase, SweepDetail, SweepListItem, SweepRequest
from web.backend.models.schemas import DateRange, SystemConfig
from web.backend.services.client_errors import public_error
from web.backend.services.history_store import HistoryStore
from web.backend.services.pool import SimPool
from web.backend.services.presets import hardware_by_id, scale_battery, tariff_by_id
from web.backend.services.sim_worker import CasePayload, execute_case
from web.backend.services.simulation_service import STRATEGY_MAP, SimulationService
from web.backend.services.span import inclusive_day_count

_MAX_CASES = 200
_MAX_CAPACITIES = 12
_MAX_CAPACITY_KWH = 200.0
_MAX_DAYS = 366

_HOURS_YEAR = 8760.0
_RESULT_POLL_S = 0.5

logger = logging.getLogger(__name__)


@dataclass
class SweepJob:
    sweep_id: str
    label: Optional[str]
    created_at: str
    status: str
    date_start: str
    date_end: str
    cases: list[SweepCase]
    request: dict
    cancel: object
    lock: threading.Lock = field(default_factory=threading.Lock)
    futures: list[Future] = field(default_factory=list)
    hours: dict[int, Optional[float]] = field(default_factory=dict)


class SweepBusy(Exception):
    def __init__(self) -> None:
        super().__init__("a sweep is already running")


class SweepService:
    def __init__(self, sim: SimulationService, pool: SimPool, history: HistoryStore) -> None:
        self.sim = sim
        self.pool = pool
        self.history = history
        self.jobs: dict[str, SweepJob] = {}
        self._lock = threading.Lock()

    def start(self, body: SweepRequest) -> tuple[str, int]:
        self._validate(body)
        frame = self.sim.data_service.get_frame(body.date_range.start, body.date_range.end, exclude_ev=body.exclude_ev)
        if len(frame) == 0 or not frame["valid"].any():
            raise ValueError(f"No data between {body.date_range.start} and {body.date_range.end}")
        cases = _case_list(body)
        now = datetime.now(ZoneInfo("Europe/Madrid")).isoformat()
        job = SweepJob(
            sweep_id=uuid.uuid4().hex[:12],
            label=body.label,
            created_at=now,
            status="running",
            date_start=body.date_range.start,
            date_end=body.date_range.end,
            cases=cases,
            request=body.model_dump(mode="json"),
            cancel=self.pool.manager.Event(),
        )
        with self._lock:
            if any(existing.status == "running" for existing in self.jobs.values()):
                raise SweepBusy()
            self.jobs[job.sweep_id] = job
        threading.Thread(target=self._run, args=(job, frame, body), daemon=True).start()
        return job.sweep_id, len(cases)

    def get(self, sweep_id: str) -> Optional[SweepDetail]:
        job = self.jobs.get(sweep_id)
        if job is not None:
            return _detail(job)
        row = self.history.get_sweep(sweep_id)
        if row is None:
            return None
        return SweepDetail(
            sweep_id=row["sweep_id"],
            label=row["label"],
            created_at=row["created_at"],
            status=row["status"],
            date_range=DateRange(start=row["date_start"], end=row["date_end"]),
            done=row["done"],
            total=row["total"],
            cases=[SweepCase.model_validate(case) for case in row["cases"]],
        )

    def list_sweeps(self, limit: int) -> list[SweepListItem]:
        seen: set[str] = set()
        items: list[SweepListItem] = []
        with self._lock:
            jobs = list(self.jobs.values())
        for job in sorted(jobs, key=lambda j: j.created_at, reverse=True):
            seen.add(job.sweep_id)
            items.append(_list_item(job))
        for row in self.history.list_sweeps(limit):
            if row["sweep_id"] in seen:
                continue
            items.append(SweepListItem(
                sweep_id=row["sweep_id"],
                label=row["label"],
                created_at=row["created_at"],
                status=row["status"],
                date_range=DateRange(start=row["date_start"], end=row["date_end"]),
                total=row["total"],
            ))
        items.sort(key=lambda item: item.created_at, reverse=True)
        return items[:limit]

    def cancel(self, sweep_id: str) -> Optional[str]:
        job = self.jobs.get(sweep_id)
        if job is None:
            row = self.history.get_sweep(sweep_id)
            return None if row is None else row["status"]
        job.cancel.set()
        for fut in job.futures:
            fut.cancel()
        return job.status

    def _run(self, job: SweepJob, frame: pd.DataFrame, body: SweepRequest) -> None:
        try:
            inputs = self.sim.prepare_inputs(
                frame, body.strategies, SystemConfig(exclude_ev=body.exclude_ev), load_pvpc=False,
            )
            prices = self._prices(frame, body)
            baseline_n = len(body.tariffs)
            groups = (
                list(enumerate(job.cases[:baseline_n])),
                list(enumerate(job.cases[baseline_n:], baseline_n)),
            )
            for group in groups:
                self._submit_wait(job, frame, body, prices, inputs, group)
                if job.cancel.is_set():
                    with job.lock:
                        self._cancel_pending(job)
                    break
            self._finish(job)
        except Exception:
            public_error("sweep failed")
            job.cancel.set()
            for fut in list(job.futures):
                fut.cancel()
            with job.lock:
                self._fail_open(job, "sweep failed")
                job.status = "failed"
            if self._persist_logged(job):
                self._drop(job)

    def _submit_wait(
        self, job, frame, body, prices, inputs, indexed: list[tuple[int, SweepCase]],
    ) -> None:
        futs: list[tuple[int, Future]] = []
        for pos, (idx, case) in enumerate(indexed):
            if job.cancel.is_set():
                with job.lock:
                    case.status = "cancelled"
                continue
            with job.lock:
                case.status = "running"
            try:
                fut = self._submit_one(job, frame, body, prices, inputs, idx, case)
            except BrokenProcessPool:
                logger.exception("worker process crashed")
                with job.lock:
                    case.status = "failed"
                    case.error = "worker process crashed"
                    for _later_idx, later in indexed[pos + 1:]:
                        if later.status in ("pending", "running"):
                            later.status = "failed"
                            later.error = "worker process crashed"
                break
            job.futures.append(fut)
            futs.append((idx, fut))
        for idx, fut in futs:
            self._consume(job, idx, fut)
        job.futures.clear()

    def _submit_one(self, job, frame, body, prices, inputs, idx: int, case: SweepCase) -> Future:
        payload = CasePayload(
            frame=frame,
            strategy_id=case.strategy_id,
            config=_case_config(body, case),
            pvpc_prices=prices if _tariff_uses_pvpc(case.tariff) else None,
            history=inputs.history,
            daily_forecasts=inputs.daily_forecasts,
            ghi=inputs.ghi,
            cancel=job.cancel,
            job_id=f"{job.sweep_id}:{idx}",
            metrics_only=True,
        )
        return self.pool.executor.submit(execute_case, payload)

    def _consume(self, job: SweepJob, idx: int, fut: Future) -> None:
        if fut.cancelled():
            with job.lock:
                job.cases[idx].status = "cancelled"
            return
        try:
            output = self._wait_result(job, fut)
        except CancelledError:
            with job.lock:
                job.cases[idx].status = "cancelled"
            return
        except BrokenProcessPool:
            logger.exception("worker process crashed")
            with job.lock:
                job.cases[idx].status = "failed"
                job.cases[idx].error = "worker process crashed"
            self._persist_logged(job)
            return
        except Exception:
            with job.lock:
                job.cases[idx].status = "failed"
                job.cases[idx].error = public_error("simulation failed")
            self._persist_logged(job)
            return
        with job.lock:
            job.hours[idx] = _apply(job.cases[idx], output)
            _savings(job.cases, job.hours)
            job.cases[idx].status = output.status
        self._persist_logged(job)

    def _wait_result(self, job: SweepJob, fut: Future):
        while True:
            if job.cancel.is_set():
                fut.cancel()
                raise CancelledError()
            try:
                return fut.result(timeout=0)
            except TimeoutError:
                if _executor_broken(self.pool):
                    raise BrokenProcessPool("worker process crashed")
            try:
                return fut.result(timeout=_RESULT_POLL_S)
            except TimeoutError:
                continue

    def _persist_logged(self, job: SweepJob) -> bool:
        try:
            self._persist(job)
        except (sqlite3.Error, OSError):
            logger.exception("sweep persist failed")
            return False
        return True

    def _prices(self, frame: pd.DataFrame, body: SweepRequest):
        if not any(_tariff_uses_pvpc(tariff_id) for tariff_id in body.tariffs):
            return None
        cfg = SystemConfig(tariff=tariff_by_id(body.tariffs[0]).tariff)
        window = self.sim._pvpc_window(frame, cfg)
        if window is None:
            return None
        return self.sim.pvpc_service.get_prices(*window, strict=False)

    def _finish(self, job: SweepJob) -> None:
        with job.lock:
            self._fail_open(job, "sweep failed")
            statuses = [case.status for case in job.cases]
            if job.cancel.is_set() or (statuses and all(status == "cancelled" for status in statuses)):
                job.status = "cancelled"
            elif any(status == "cancelled" for status in statuses) and not any(status == "completed" for status in statuses):
                job.status = "cancelled"
            elif statuses and all(status == "failed" for status in statuses):
                job.status = "failed"
            else:
                job.status = "completed"
        job.futures.clear()
        if self._persist_logged(job):
            self._drop(job)

    def _fail_open(self, job: SweepJob, message: str) -> None:
        for case in job.cases:
            if case.status in ("pending", "running"):
                case.status = "cancelled" if job.cancel.is_set() else "failed"
                if case.status == "failed":
                    case.error = message

    def _drop(self, job: SweepJob) -> None:
        job.futures.clear()
        with self._lock:
            self.jobs.pop(job.sweep_id, None)

    @staticmethod
    def _cancel_pending(job: SweepJob) -> None:
        for case in job.cases:
            if case.status in ("pending", "running"):
                case.status = "cancelled"

    def _persist(self, job: SweepJob) -> None:
        with job.lock:
            cases = [case.model_copy() for case in job.cases]
            record = {
                "sweep_id": job.sweep_id,
                "label": job.label,
                "created_at": job.created_at,
                "status": job.status,
                "date_start": job.date_start,
                "date_end": job.date_end,
                "done": sum(case.status not in ("pending", "running") for case in cases),
                "total": len(cases),
                "cases": [case.model_dump() for case in cases],
                "request": job.request,
            }
        self.history.upsert_sweep(record)

    @staticmethod
    def _validate(body: SweepRequest) -> None:
        unknown = sorted(set(body.strategies) - set(STRATEGY_MAP))
        if unknown:
            raise ValueError(f"Unknown strategies: {unknown}")
        try:
            hardware_by_id(body.hardware)
            for tariff_id in body.tariffs:
                tariff_by_id(tariff_id)
        except KeyError as exc:
            raise ValueError(f"Unknown preset: {exc.args[0]}") from exc
        if len(body.capacities_kwh) > _MAX_CAPACITIES:
            raise ValueError(f"at most {_MAX_CAPACITIES} capacities")
        if any(
            not math.isfinite(cap) or cap <= 0 or cap > _MAX_CAPACITY_KWH
            for cap in body.capacities_kwh
        ):
            raise ValueError(f"each capacity must be finite and in (0, {_MAX_CAPACITY_KWH:g}] kWh")
        if inclusive_day_count(body.date_range.start, body.date_range.end) > _MAX_DAYS:
            raise ValueError(f"date range exceeds {_MAX_DAYS} inclusive days")
        n_cases = len(body.tariffs) * (1 + len(body.capacities_kwh) * len(body.strategies))
        if n_cases > _MAX_CASES:
            raise ValueError(f"sweep would run {n_cases} cases; the limit is {_MAX_CASES}")


def _case_list(body: SweepRequest) -> list[SweepCase]:
    baselines = [
        SweepCase(tariff=tariff_id, capacity_kwh=0.0, strategy_id="no_battery", status="pending")
        for tariff_id in body.tariffs
    ]
    matrix = [
        SweepCase(tariff=tariff_id, capacity_kwh=cap, strategy_id=strategy_id, status="pending")
        for tariff_id in body.tariffs
        for cap in body.capacities_kwh
        for strategy_id in body.strategies
    ]
    return baselines + matrix


def _case_config(body: SweepRequest, case: SweepCase) -> SystemConfig:
    hardware = hardware_by_id(body.hardware)
    battery = hardware.battery if case.capacity_kwh == 0 else scale_battery(hardware.battery, case.capacity_kwh)
    return SystemConfig(
        battery=battery,
        grid=hardware.grid,
        tariff=tariff_by_id(case.tariff).tariff,
        exclude_ev=body.exclude_ev,
    )


def _tariff_uses_pvpc(tariff_id: str) -> bool:
    return tariff_by_id(tariff_id).tariff.use_pvpc


def _apply(case: SweepCase, output) -> Optional[float]:
    case.error = output.error
    metrics = output.metrics or {}
    if output.status != "completed":
        return None
    case.total_cost = metrics.get("total_cost")
    case.adjusted_cost = metrics.get("adjusted_cost")
    case.equivalent_full_cycles = metrics.get("equivalent_full_cycles")
    case.self_consumption_rate = metrics.get("self_consumption_rate")
    return metrics.get("valid_hours")


def _savings(cases: list[SweepCase], hours_by_index: dict[int, Optional[float]]) -> None:
    baselines = {
        case.tariff: case.adjusted_cost
        for case in cases
        if case.strategy_id == "no_battery" and case.capacity_kwh == 0 and case.adjusted_cost is not None
    }
    for idx, case in enumerate(cases):
        base = baselines.get(case.tariff)
        if case.capacity_kwh == 0 and case.strategy_id == "no_battery":
            case.savings_vs_baseline = 0.0 if case.adjusted_cost is not None else None
            case.annualised_savings = 0.0 if case.adjusted_cost is not None else None
            continue
        if base is None or case.adjusted_cost is None:
            continue
        savings = base - case.adjusted_cost
        case.savings_vs_baseline = savings
        hours = hours_by_index.get(idx) or 0
        case.annualised_savings = savings * _HOURS_YEAR / hours if hours else None


def _detail(job: SweepJob) -> SweepDetail:
    with job.lock:
        cases = [case.model_copy() for case in job.cases]
        status = job.status
    return SweepDetail(
        sweep_id=job.sweep_id,
        label=job.label,
        created_at=job.created_at,
        status=status,
        date_range=DateRange(start=job.date_start, end=job.date_end),
        done=sum(case.status not in ("pending", "running") for case in cases),
        total=len(cases),
        cases=cases,
    )


def _executor_broken(pool) -> bool:
    executor = getattr(pool, "executor", None)
    return bool(getattr(executor, "_broken", False))


def _list_item(job: SweepJob) -> SweepListItem:
    return SweepListItem(
        sweep_id=job.sweep_id,
        label=job.label,
        created_at=job.created_at,
        status=job.status,
        date_range=DateRange(start=job.date_start, end=job.date_end),
        total=len(job.cases),
    )
