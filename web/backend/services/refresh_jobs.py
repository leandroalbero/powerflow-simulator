"""Background Influx refresh jobs. One at a time."""

import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.data.config import MissingCredentialError
from web.backend.models.feature_schemas import EntityRefreshSummary, RefreshJobResponse, RefreshSummaryBody
from web.backend.services.client_errors import public_error


@dataclass
class RefreshJob:
    job_id: str
    status: str = "running"
    progress_pct: float = 0.0
    message: str = ""
    summary: Optional[dict] = None
    error: Optional[str] = None
    _lines: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)


class RefreshRegistry:
    def __init__(self, reload: Callable[[], None]) -> None:
        self._reload = reload
        self._jobs: dict[str, RefreshJob] = {}
        self._lock = threading.Lock()

    def running(self) -> bool:
        with self._lock:
            return any(job.status == "running" for job in self._jobs.values())

    def get(self, job_id: str) -> Optional[RefreshJobResponse]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        summary = RefreshSummaryBody.model_validate(job.summary) if job.summary else None
        return RefreshJobResponse(
            job_id=job.job_id,
            status=job.status,  # type: ignore[arg-type]
            progress_pct=job.progress_pct,
            message=job.message,
            summary=summary,
            error=job.error,
        )

    def start(self, full: bool) -> str:
        self._require_credentials()
        with self._lock:
            if any(job.status == "running" for job in self._jobs.values()):
                raise RuntimeError("a refresh is already running")
            job = RefreshJob(job_id=uuid.uuid4().hex[:12])
            self._jobs[job.job_id] = job
        threading.Thread(target=self._run, args=(job, full), daemon=True).start()
        return job.job_id

    @staticmethod
    def _require_credentials() -> None:
        from src.data import config as influx_config

        try:
            influx_config.load_influx_config()
        except MissingCredentialError as exc:
            raise MissingCredentialError(exc.name) from exc

    def _run(self, job: RefreshJob, full: bool) -> None:
        from src.data import refresh as refresh_mod

        def progress(line: str) -> None:
            with job._lock:
                job._lines += 1
                job.message = line
                job.progress_pct = min(95.0, job._lines * 8.0)

        try:
            summary = refresh_mod.refresh(full=full, progress=progress, with_irradiance=True)
            self._reload()
            job.summary = _summary(summary)
            job.progress_pct = 100.0
            job.status = "completed"
        except MissingCredentialError as exc:
            self._fail(job, f"INFLUX_* not configured ({exc.name})")
        except Exception:
            self._fail(job, public_error("refresh failed"))

    @staticmethod
    def _fail(job: RefreshJob, message: str) -> None:
        job.status = "failed"
        job.error = message
        job.message = message


def _summary(raw: dict) -> dict:
    entities = {
        key: EntityRefreshSummary.model_validate(value).model_dump()
        for key, value in (raw.get("entities") or {}).items()
    }
    return {"updated_at": raw.get("updated_at") or "", "entities": entities}
