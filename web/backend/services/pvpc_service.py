"""PVPC price service.

Fetches Spanish PVPC (regulated consumer) hourly prices from REE's public
API and caches them in a SQLite database keyed by UTC hour-truncated
ISO8601 timestamps. Range lookups only hit the network for hours missing
from the cache.

REE reference:
    https://apidatos.ree.es/es/datos/mercados/precios-mercados-tiempo-real
Prices are returned in €/MWh and converted to €/kWh for storage.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError

import pytz

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "pvpc_cache.db"

REE_ENDPOINT = (
    "https://apidatos.ree.es/es/datos/mercados/precios-mercados-tiempo-real"
)
MAX_WINDOW_DAYS = 28
_MADRID_TZ = pytz.timezone("Europe/Madrid")
_UTC = timezone.utc


class PvpcService:
    """Fetch and cache hourly PVPC prices from REE."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path: Path = db_path if db_path is not None else DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # Opening a fresh connection per call is the safest concurrency model
        # (simulation runs execute in a ThreadPoolExecutor). We still wrap
        # writes in a lock to avoid duplicate-fetch races across threads.
        self._write_lock = threading.Lock()
        self._init_schema()

    # -- public ------------------------------------------------------------------

    def get_prices(
        self, start: datetime, end: datetime, strict: bool = True
    ) -> Dict[datetime, float]:
        """Return {utc_hour: price_eur_kwh} for every hour in [start, end].

        ``start`` and ``end`` may be Madrid-local (tz-aware) or naive (assumed
        Madrid). Missing hours are fetched from REE in ≤30-day windows and
        inserted into the SQLite cache. Already-cached hours never trigger
        HTTP calls. With ``strict=False`` hours REE has not published are
        left out instead of raising.
        """
        start_utc = _normalise_to_utc_hour(start)
        end_utc = _normalise_to_utc_hour(end)
        if end_utc < start_utc:
            raise ValueError("end must be >= start")

        all_hours = _hour_range(start_utc, end_utc)
        cached = self._query_cached(start_utc, end_utc)
        missing = [h for h in all_hours if h not in cached]

        if missing:
            logger.info(
                "PVPC cache miss: %d hours missing between %s and %s",
                len(missing), start_utc.isoformat(), end_utc.isoformat(),
            )
            fetched = self._fetch_missing(missing, strict=strict)
            with self._write_lock:
                self._insert_prices(fetched)
            cached.update(fetched)
        else:
            logger.debug(
                "PVPC cache hit: %d hours between %s and %s",
                len(all_hours), start_utc.isoformat(), end_utc.isoformat(),
            )

        # Final result: only hours actually requested; raise if still missing
        # (non-strict callers rely on PvpcTariff's fallback for unpublished hours).
        result: Dict[datetime, float] = {}
        for h in all_hours:
            if h not in cached:
                if not strict:
                    continue
                raise RuntimeError(
                    f"PVPC fetch failed: hour {h.isoformat()} still missing "
                    "after REE fetch"
                )
            result[h] = cached[h]
        return result

    # -- cache plumbing ----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path), check_same_thread=False)

    def _init_schema(self) -> None:
        with self._write_lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS pvpc_prices (
                        ts_utc TEXT PRIMARY KEY,
                        price_eur_kwh REAL NOT NULL,
                        fetched_at TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def _query_cached(
        self, start_utc: datetime, end_utc: datetime
    ) -> Dict[datetime, float]:
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT ts_utc, price_eur_kwh FROM pvpc_prices "
                "WHERE ts_utc >= ? AND ts_utc <= ?",
                (start_utc.isoformat(), end_utc.isoformat()),
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        out: Dict[datetime, float] = {}
        for ts_str, price in rows:
            out[datetime.fromisoformat(ts_str)] = float(price)
        return out

    def _insert_prices(self, prices: Dict[datetime, float]) -> None:
        if not prices:
            return
        now_iso = datetime.now(_UTC).isoformat()
        rows = [
            (ts.isoformat(), float(price), now_iso)
            for ts, price in prices.items()
        ]
        conn = self._connect()
        try:
            conn.executemany(
                "INSERT OR REPLACE INTO pvpc_prices "
                "(ts_utc, price_eur_kwh, fetched_at) VALUES (?, ?, ?)",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    # -- fetching ----------------------------------------------------------------

    def _fetch_missing(
        self, missing_hours: List[datetime], strict: bool = True
    ) -> Dict[datetime, float]:
        """Group missing hours into contiguous ranges and fetch each.

        Ranges are split into ``MAX_WINDOW_DAYS`` windows (REE returns at
        most ~31 days per request). With ``strict=False`` a failed window
        that starts tomorrow (local) or later is logged and skipped: REE
        answers HTTP 502 for a window holding only unpublished hours (e.g.
        tomorrow before ~20:15). Earlier hours are published, so a failure
        there still raises rather than silently pricing them by fallback.
        """
        if not missing_hours:
            return {}

        missing_sorted = sorted(missing_hours)
        ranges: List[Tuple[datetime, datetime]] = []
        run_start = missing_sorted[0]
        run_end = missing_sorted[0]
        for h in missing_sorted[1:]:
            if h - run_end <= timedelta(hours=1):
                run_end = h
            else:
                ranges.append((run_start, run_end))
                run_start = h
                run_end = h
        ranges.append((run_start, run_end))

        out: Dict[datetime, float] = {}
        unpublished_from = _unpublished_from()
        for run_start, run_end in ranges:
            cursor = run_start
            while cursor <= run_end:
                window_end = min(cursor + timedelta(days=MAX_WINDOW_DAYS) - timedelta(hours=1), run_end)
                try:
                    out.update(self._fetch_window(cursor, window_end))
                except RuntimeError as exc:
                    if strict or cursor < unpublished_from:
                        raise
                    logger.warning(
                        "PVPC prices unavailable for %s..%s, using tariff fallback: %s",
                        cursor.isoformat(), window_end.isoformat(), exc,
                    )
                cursor = window_end + timedelta(hours=1)
        return out

    def _fetch_window(
        self, start_utc: datetime, end_utc: datetime
    ) -> Dict[datetime, float]:
        """Call REE for a single <=30-day window.

        REE expects Madrid-local naive timestamps in its query params. The
        returned datetimes include explicit UTC offsets, so we always trust
        the response's offset rather than the request-side assumption.
        """
        end_param = _ree_local(end_utc)
        request_start = start_utc
        # REE answers HTTP 400 when start_date == end_date (a one-hour window, or two UTC hours that
        # share a wall-clock label on the DST fall-back day). Earlier hours are always published, so
        # widen backwards and keep only the requested hours.
        while _ree_local(request_start) >= end_param:
            request_start -= timedelta(hours=1)
        params = (
            f"?start_date={_ree_local(request_start)}"
            f"&end_date={end_param}"
            f"&time_trunc=hour"
        )
        url = REE_ENDPOINT + params
        logger.info("PVPC fetching %s", url)

        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = resp.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"PVPC fetch failed: HTTP error calling REE: {exc}") from exc

        try:
            doc = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"PVPC fetch failed: invalid JSON from REE: {exc}") from exc

        prices = _parse_ree_response(doc)
        return {ts: p for ts, p in prices.items() if start_utc <= ts <= end_utc}


# ---------------------------------------------------------------------------
# Helpers


def _parse_ree_response(doc: dict) -> Dict[datetime, float]:
    """Extract the PVPC series from a REE response payload.

    Walks ``included`` for the entry with ``type == "PVPC"``, then emits
    ``{utc_hour: price_eur_kwh}`` from its ``attributes.values`` array.
    Prices come in €/MWh and are divided by 1000.
    """
    included = doc.get("included")
    if not isinstance(included, list):
        raise RuntimeError("PVPC fetch failed: malformed REE response (no 'included')")

    pvpc_entry = None
    for entry in included:
        if isinstance(entry, dict) and entry.get("type") == "PVPC":
            pvpc_entry = entry
            break
    if pvpc_entry is None:
        raise RuntimeError("PVPC fetch failed: no PVPC series in REE response")

    values = pvpc_entry.get("attributes", {}).get("values")
    if not isinstance(values, list):
        raise RuntimeError("PVPC fetch failed: PVPC entry missing attributes.values")

    out: Dict[datetime, float] = {}
    for item in values:
        if not isinstance(item, dict):
            continue
        dt_str = item.get("datetime")
        raw = item.get("value")
        if dt_str is None or raw is None:
            continue
        try:
            ts = datetime.fromisoformat(dt_str).astimezone(_UTC)
            ts = ts.replace(minute=0, second=0, microsecond=0)
        except ValueError as exc:
            raise RuntimeError(
                f"PVPC fetch failed: unparsable datetime {dt_str!r}: {exc}"
            ) from exc
        # €/MWh -> €/kWh
        out[ts] = float(raw) / 1000.0
    if not out:
        raise RuntimeError("PVPC fetch failed: PVPC values array was empty")
    return out


def _ree_local(ts_utc: datetime) -> str:
    return ts_utc.astimezone(_MADRID_TZ).strftime("%Y-%m-%dT%H:%M")


def _unpublished_from() -> datetime:
    """UTC start of tomorrow (local): the earliest hour REE may not have published yet."""
    tomorrow = datetime.now(_MADRID_TZ).date() + timedelta(days=1)
    return _MADRID_TZ.localize(datetime(tomorrow.year, tomorrow.month, tomorrow.day)).astimezone(_UTC)


def _normalise_to_utc_hour(dt: datetime) -> datetime:
    """Truncate to the hour and return UTC-tz aware.

    Naive inputs are assumed Madrid-local.
    """
    if dt.tzinfo is None:
        dt = _MADRID_TZ.localize(dt)
    return dt.astimezone(_UTC).replace(minute=0, second=0, microsecond=0)


def _hour_range(start_utc: datetime, end_utc: datetime) -> List[datetime]:
    """Inclusive hourly range [start_utc, end_utc]."""
    out: List[datetime] = []
    cursor = start_utc
    while cursor <= end_utc:
        out.append(cursor)
        cursor += timedelta(hours=1)
    return out
