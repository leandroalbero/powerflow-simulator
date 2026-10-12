"""Map a raw SimResult timeseries onto the API payload, with envelope downsampling."""

from typing import Optional

import numpy as np
import pandas as pd

from src.data.frame import to_utc_bound
from web.backend.models.schemas import TimeseriesResponse
from web.backend.services.envelope import envelope_downsample


def present_timeseries(
    ts_data: dict[str, np.ndarray],
    capacity_kwh: float,
    start: Optional[str],
    end: Optional[str],
    max_points: int,
) -> TimeseriesResponse:
    ts_ms = np.asarray(ts_data["ts"], dtype=np.int64)
    level = np.asarray(ts_data["battery_level_kwh"], dtype=float)
    columns = {
        "battery_level": level,
        "grid_import": np.asarray(ts_data["grid_import_kw"], dtype=float),
        "grid_export": np.asarray(ts_data["grid_export_kw"], dtype=float),
        "solar_power": np.asarray(ts_data["solar_kw"], dtype=float),
        "house_consumption": np.asarray(ts_data["load_kw"], dtype=float),
        "battery_charge": np.asarray(ts_data["battery_charge_kw"], dtype=float),
        "battery_discharge": np.asarray(ts_data["battery_discharge_kw"], dtype=float),
        "soc_pct": level / capacity_kwh * 100.0 if capacity_kwh else np.full(len(level), np.nan),
        "import_price": np.asarray(ts_data["import_price"], dtype=float),
        "export_price": np.asarray(ts_data["export_price"], dtype=float),
    }
    cost = np.asarray(ts_data["step_cost"], dtype=float)
    mask = np.ones(len(ts_ms), dtype=bool)
    if start:
        mask &= ts_ms >= epoch_ms(start, is_end=False)
    if end:
        mask &= ts_ms < epoch_ms(end, is_end=True)
    ts_ms = ts_ms[mask]
    columns = {key: values[mask] for key, values in columns.items()}
    cost = cost[mask]
    env = envelope_downsample(ts_ms, columns, cost, max_points)
    stamps = pd.to_datetime(env.ts_ms, unit="ms", utc=True).tz_convert("Europe/Madrid")
    return TimeseriesResponse(
        timestamps=[stamp.isoformat() for stamp in stamps],
        point_count=len(env.ts_ms),
        ts=env.ts_ms.astype(int).tolist(),
        downsampled=env.downsampled,
        resolution_minutes=env.resolution_minutes,
        step_cost=_nan_to_none(env.step_cost),
        **{key: _nan_to_none(values) for key, values in env.columns.items()},
    )


def epoch_ms(value: str, is_end: bool) -> int:
    """Exclusive end, same convention as /simulate: a date-only end covers that whole local day."""
    text = str(value).strip()
    if text.isdigit():
        return int(text) + (1 if is_end else 0)
    bound = to_utc_bound(text, is_end=is_end)
    if bound is None:
        raise ValueError(f"empty bound {value!r}")
    return int(bound.value // 1_000_000)


def _nan_to_none(values: np.ndarray) -> list[Optional[float]]:
    return [None if v != v else float(v) for v in np.asarray(values, dtype=float).tolist()]
