"""Scenario matrix. One frame, process-pool cases, baseline per tariff."""

import threading
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from web.backend.models.feature_schemas import SweepCase, SweepDetail, SweepListItem, SweepRequest
from web.backend.models.schemas import DateRange, SystemConfig
from web.backend.services.history_store import HistoryStore
from web.backend.services.pool import SimPool
from web.backend.services.presets import hardware_by_id, scale_battery, tariff_by_id
from web.backend.services.sim_worker import CasePayload, execute_case
from web.backend.services.simulation_service import STRATEGY_MAP, SimulationService

_HOURS_YEAR = 8760.0


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


class SweepService:
    def __init__(self, sim: SimulationService, pool: SimPool, history: HistoryStore) -> None:
        self.sim = sim
        self.pool = pool
        self.history = history
        self.jobs: dict[str, SweepJob] = {}

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
        for job in sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True):
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
            prices = self._prices(frame, body)
            history = frame
            groups = (job.cases[:len(body.tariffs)], job.cases[len(body.tariffs):])
            for group in groups:
                self._submit_wait(job, frame, body, prices, history, group)
                if job.cancel.is_set():
                    self._cancel_pending(job)
                    break
            self._finish(job)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            job.status = "failed"
            job.cases[0].error = str(exc)
            self._persist(job)

    def _submit_wait(self, job, frame, body, prices, history, cases: list[SweepCase]) -> None:
        futs: list[tuple[int, Future]] = []
        for case in cases:
            idx = job.cases.index(case)
            if job.cancel.is_set():
                case.status = "cancelled"
                continue
            case.status = "running"
            payload = CasePayload(
                frame=frame,
                strategy_id=case.strategy_id,
                config=_case_config(body, case),
                pvpc_prices=prices if _tariff_uses_pvpc(case.tariff) else None,
                history=history,
                cancel=job.cancel,
                job_id=f"{job.sweep_id}:{idx}",
            )
            fut = self.pool.executor.submit(execute_case, payload)
            job.futures.append(fut)
            futs.append((idx, fut))
        for idx, fut in futs:
            if fut.cancelled():
                job.cases[idx].status = "cancelled"
                continue
            output = fut.result()
            job.hours[idx] = _apply(job.cases[idx], output)
            _savings(job.cases, job.hours)
            self._persist(job)

    def _prices(self, frame: pd.DataFrame, body: SweepRequest):
        if not any(_tariff_uses_pvpc(tariff_id) for tariff_id in body.tariffs):
            return None
        cfg = SystemConfig(tariff=tariff_by_id(body.tariffs[0]).tariff)
        window = self.sim._pvpc_window(frame, cfg)
        if window is None:
            return None
        return self.sim.pvpc_service.get_prices(*window, strict=False)

    def _finish(self, job: SweepJob) -> None:
        statuses = [case.status for case in job.cases]
        if job.cancel.is_set() or any(status == "cancelled" for status in statuses):
            job.status = "cancelled"
        elif any(status == "failed" for status in statuses):
            job.status = "failed"
        else:
            job.status = "completed"
        self._persist(job)

    @staticmethod
    def _cancel_pending(job: SweepJob) -> None:
        for case in job.cases:
            if case.status in ("pending", "running"):
                case.status = "cancelled"

    def _persist(self, job: SweepJob) -> None:
        done = sum(case.status not in ("pending", "running") for case in job.cases)
        self.history.upsert_sweep({
            "sweep_id": job.sweep_id,
            "label": job.label,
            "created_at": job.created_at,
            "status": job.status,
            "date_start": job.date_start,
            "date_end": job.date_end,
            "done": done,
            "total": len(job.cases),
            "cases": [case.model_dump() for case in job.cases],
            "request": job.request,
        })

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
        if any(cap <= 0 for cap in body.capacities_kwh):
            raise ValueError("capacities_kwh must be positive")


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
    case.status = output.status
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
    done = sum(case.status not in ("pending", "running") for case in job.cases)
    return SweepDetail(
        sweep_id=job.sweep_id,
        label=job.label,
        created_at=job.created_at,
        status=job.status,
        date_range=DateRange(start=job.date_start, end=job.date_end),
        done=done,
        total=len(job.cases),
        cases=job.cases,
    )


def _list_item(job: SweepJob) -> SweepListItem:
    return SweepListItem(
        sweep_id=job.sweep_id,
        label=job.label,
        created_at=job.created_at,
        status=job.status,
        date_range=DateRange(start=job.date_start, end=job.date_end),
        total=len(job.cases),
    )
