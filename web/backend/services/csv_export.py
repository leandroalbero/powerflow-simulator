"""CSV attachments for runs and sweeps."""

import csv
import io
from typing import Iterator

import numpy as np
import pandas as pd

from web.backend.models.schemas import StrategyMetrics

_METRIC_FIELDS = [name for name in StrategyMetrics.model_fields if name != "monthly"]
_MONTHLY_FIELDS = (
    "month", "cost", "import_kwh", "export_kwh", "import_cost", "export_revenue",
    "solar_kwh", "load_kwh", "battery_out_kwh",
)
_TS_COLUMNS = (
    ("battery_level", "battery_level_kwh"),
    ("grid_import", "grid_import_kw"),
    ("grid_export", "grid_export_kw"),
    ("solar_power", "solar_kw"),
    ("house_consumption", "load_kw"),
    ("battery_charge", "battery_charge_kw"),
    ("battery_discharge", "battery_discharge_kw"),
    ("import_price", "import_price"),
    ("export_price", "export_price"),
    ("step_cost", "step_cost"),
    ("grid_charge", "grid_charge_kw"),
    ("curtailed", "curtailed_kw"),
)


def _rows(header: list[str], records: list[list[object]]) -> Iterator[str]:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    yield buf.getvalue()
    for record in records:
        buf.seek(0)
        buf.truncate()
        writer.writerow(record)
        yield buf.getvalue()


def summary_csv(strategies: list[dict]) -> Iterator[str]:
    header = ["strategy_id", "status", "error", *_METRIC_FIELDS]
    records = []
    for item in strategies:
        metrics = item.get("metrics") or {}
        records.append([
            item.get("strategy_id"), item.get("status"), item.get("error") or "",
            *[_cell(metrics.get(name)) for name in _METRIC_FIELDS],
        ])
    return _rows(header, records)


def monthly_csv(strategies: list[dict]) -> Iterator[str]:
    header = ["strategy", *_MONTHLY_FIELDS]
    records = []
    for item in strategies:
        metrics = item.get("metrics") or {}
        for month in metrics.get("monthly") or []:
            records.append([item.get("strategy_id"), *[_cell(month.get(name)) for name in _MONTHLY_FIELDS]])
    return _rows(header, records)


_CHUNK = 2048


def timeseries_csv(ts: dict[str, np.ndarray], capacity_kwh: float) -> Iterator[str]:
    ts_ms = np.asarray(ts["ts"], dtype=np.int64)
    header = ["ts", "soc_pct", *[name for name, _ in _TS_COLUMNS]]
    level = np.asarray(ts["battery_level_kwh"], dtype=float)
    soc = level / capacity_kwh * 100.0 if capacity_kwh else np.full(len(level), np.nan)
    columns = [soc, *[np.asarray(ts[key], dtype=float) for _, key in _TS_COLUMNS]]

    def _chunks() -> Iterator[str]:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(header)
        yield buf.getvalue()
        n = len(ts_ms)
        for start in range(0, n, _CHUNK):
            end = min(n, start + _CHUNK)
            stamps = pd.to_datetime(ts_ms[start:end], unit="ms", utc=True).tz_convert("Europe/Madrid")
            buf.seek(0)
            buf.truncate()
            for offset, stamp in enumerate(stamps):
                writer.writerow([stamp.isoformat(), *[_cell(col[start + offset]) for col in columns]])
            yield buf.getvalue()

    return _chunks()


def sweep_csv(cases: list[dict]) -> Iterator[str]:
    header = [
        "tariff", "capacity_kwh", "strategy_id", "status", "total_cost", "adjusted_cost",
        "savings_vs_baseline", "annualised_savings", "equivalent_full_cycles",
        "self_consumption_rate", "error",
    ]
    records = [[_cell(case.get(name)) for name in header] for case in cases]
    return _rows(header, records)


def _cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return value
