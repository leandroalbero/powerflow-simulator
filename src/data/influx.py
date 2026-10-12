"""InfluxDB 1.8 /query client. Always 1-minute means, never raw points."""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from base64 import b64encode
from datetime import datetime, timezone

import pandas as pd

from src.data.config import InfluxConfig
from src.data.entities import CHUNK, Entity

_ATTEMPTS = 3
_BACKOFF_S = (0.5, 1.0, 2.0)
_TRANSIENT_HTTP = frozenset({429, 500, 502, 503, 504})
UTC = timezone.utc


class InfluxQueryError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class InfluxHttpError(Exception):
    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"influx http {status}")


def iter_chunks(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Half-open [start, end) windows of CHUNK. Adjacent, no overlap, no gap."""
    if start >= end:
        return []
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        nxt = min(cursor + CHUNK, end)
        chunks.append((cursor, nxt))
        cursor = nxt
    return chunks


def format_ts(ts: datetime) -> str:
    aware = ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)
    return aware.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_query(entity: Entity, start: datetime, end: datetime) -> str:
    return (
        f'SELECT mean("value"), count("value") FROM "{entity.measurement}" '
        f"WHERE \"entity_id\"='{entity.entity_id}' "
        f"AND time >= '{format_ts(start)}' AND time < '{format_ts(end)}' "
        f"GROUP BY time(1m) fill(none)"
    )


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.Series(dtype="datetime64[ns, UTC]"),
            "mean": pd.Series(dtype="float64"),
            "count": pd.Series(dtype="int32"),
        }
    )


def parse_grouped(body: dict[str, object]) -> pd.DataFrame:
    results = body.get("results")
    if not isinstance(results, list) or not results:
        return empty_frame()
    first = results[0]
    if not isinstance(first, dict):
        return empty_frame()
    error = first.get("error")
    if isinstance(error, str) and error:
        raise InfluxQueryError(error)
    series = first.get("series")
    if not isinstance(series, list) or not series:
        return empty_frame()
    block = series[0]
    if not isinstance(block, dict):
        return empty_frame()
    values = block.get("values")
    if not isinstance(values, list) or not values:
        return empty_frame()
    times: list[int] = []
    means: list[float] = []
    counts: list[int] = []
    for row in values:
        if not isinstance(row, list) or len(row) < 3:
            continue
        raw_time, raw_mean, raw_count = row[0], row[1], row[2]
        if raw_mean is None or raw_count is None or not isinstance(raw_time, int | float):
            continue
        times.append(int(raw_time))
        means.append(float(raw_mean))
        counts.append(int(raw_count))
    if not times:
        return empty_frame()
    ts = pd.to_datetime(times, unit="s", utc=True).astype("datetime64[ns, UTC]")
    return pd.DataFrame(
        {
            "ts": ts,
            "mean": pd.Series(means, dtype="float64"),
            "count": pd.Series(counts, dtype="int32"),
        }
    )


def _request(cfg: InfluxConfig, query: str) -> dict[str, object]:
    params = urllib.parse.urlencode({"db": cfg.db, "epoch": "s", "q": query})
    req = urllib.request.Request(f"{cfg.url}/query?{params}")
    token = b64encode(f"{cfg.user}:{cfg.password}".encode()).decode("ascii")
    req.add_header("Authorization", f"Basic {token}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise InfluxHttpError(exc.code) from None
    if not isinstance(payload, dict):
        raise InfluxQueryError("influx response was not an object")
    return payload


def query_with_retry(cfg: InfluxConfig, query: str) -> dict[str, object]:
    last_http: InfluxHttpError | None = None
    for attempt in range(_ATTEMPTS):
        try:
            return _request(cfg, query)
        except InfluxHttpError as exc:
            if exc.status not in _TRANSIENT_HTTP or attempt == _ATTEMPTS - 1:
                raise
            last_http = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == _ATTEMPTS - 1:
                raise
        time.sleep(_BACKOFF_S[attempt])
    if last_http is not None:
        raise last_http
    raise InfluxQueryError("influx query failed")


def fetch_chunk(cfg: InfluxConfig, entity: Entity, start: datetime, end: datetime) -> pd.DataFrame:
    return parse_grouped(query_with_retry(cfg, build_query(entity, start, end)))


def concat_frames(parts: list[pd.DataFrame]) -> pd.DataFrame:
    present = [part for part in parts if not part.empty]
    if not present:
        return empty_frame()
    combined = pd.concat(present, ignore_index=True)
    combined = combined.drop_duplicates(subset=["ts"], keep="last")
    return combined.sort_values("ts").reset_index(drop=True)


def fetch_range(cfg: InfluxConfig, entity: Entity, start: datetime, end: datetime) -> pd.DataFrame:
    parts = [fetch_chunk(cfg, entity, chunk_start, chunk_end) for chunk_start, chunk_end in iter_chunks(start, end)]
    return concat_frames(parts)


def drop_at_or_after(frame: pd.DataFrame, end: datetime) -> pd.DataFrame:
    if frame.empty:
        return frame
    end_ts = pd.Timestamp(end)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    return frame.loc[frame["ts"] < end_ts].reset_index(drop=True)
