"""SQLite persistence for run and sweep summaries. Timeseries stay in memory."""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from web.backend.models.schemas import DateRange, SystemConfig
from web.backend.services.data_service import data_dir

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    label TEXT,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    date_start TEXT,
    date_end TEXT,
    config_json TEXT NOT NULL,
    strategies_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sweeps (
    sweep_id TEXT PRIMARY KEY,
    label TEXT,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    date_start TEXT NOT NULL,
    date_end TEXT NOT NULL,
    done INTEGER NOT NULL,
    total INTEGER NOT NULL,
    cases_json TEXT NOT NULL,
    request_json TEXT NOT NULL
);
"""


def runs_db_path() -> Path:
    return data_dir() / "runs.db"


def config_digest(config: SystemConfig | dict[str, Any]) -> str:
    payload = config.model_dump(mode="json") if isinstance(config, SystemConfig) else config
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


class HistoryStore:
    """WAL sqlite store. One connection, guarded by a lock."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or runs_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def upsert_run(
        self,
        run_id: str,
        label: Optional[str],
        created_at: str,
        status: str,
        date_start: Optional[str],
        date_end: Optional[str],
        config: dict[str, Any],
        strategies: list[dict[str, Any]],
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO runs (run_id, label, created_at, status, date_start, date_end,
                                  config_json, strategies_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    label=excluded.label,
                    status=excluded.status,
                    strategies_json=excluded.strategies_json,
                    config_json=excluded.config_json
                """,
                (
                    run_id, label, created_at, status, date_start, date_end,
                    json.dumps(config), json.dumps(strategies),
                ),
            )
            self._conn.commit()

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return _run_row(row) if row is not None else None

    def list_runs(self, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC, run_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_run_row(row) for row in rows]

    def delete_run(self, run_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            self._conn.commit()
        return cur.rowcount > 0

    def upsert_sweep(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sweeps (sweep_id, label, created_at, status, date_start, date_end,
                                    done, total, cases_json, request_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sweep_id) DO UPDATE SET
                    status=excluded.status,
                    done=excluded.done,
                    total=excluded.total,
                    cases_json=excluded.cases_json
                """,
                (
                    record["sweep_id"], record.get("label"), record["created_at"], record["status"],
                    record["date_start"], record["date_end"], record["done"], record["total"],
                    json.dumps(record["cases"]), json.dumps(record.get("request") or {}),
                ),
            )
            self._conn.commit()

    def get_sweep(self, sweep_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM sweeps WHERE sweep_id = ?", (sweep_id,)).fetchone()
        return _sweep_row(row) if row is not None else None

    def list_sweeps(self, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sweeps ORDER BY created_at DESC, sweep_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_sweep_row(row) for row in rows]


def _run_row(row: sqlite3.Row) -> dict[str, Any]:
    start, end = row["date_start"], row["date_end"]
    return {
        "run_id": row["run_id"],
        "label": row["label"],
        "created_at": row["created_at"],
        "status": row["status"],
        "date_range": DateRange(start=start, end=end) if start and end else None,
        "config": json.loads(row["config_json"]),
        "strategies": json.loads(row["strategies_json"]),
    }


def _sweep_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "sweep_id": row["sweep_id"],
        "label": row["label"],
        "created_at": row["created_at"],
        "status": row["status"],
        "date_start": row["date_start"],
        "date_end": row["date_end"],
        "done": row["done"],
        "total": row["total"],
        "cases": json.loads(row["cases_json"]),
        "request": json.loads(row["request_json"]),
    }
