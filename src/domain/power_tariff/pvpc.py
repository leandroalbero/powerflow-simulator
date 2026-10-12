"""PVPC tariff.

Drop-in replacement for ``PowerTariff`` backed by Spain's regulated PVPC
hourly import prices. Export is a flat rate, or optionally follows PVPC.

Prices are keyed by **UTC** hour. ``rate_at(ts)`` prices the UTC hour that
contains ``ts``, so both 02:00 hours of the October fall-back day get their own
price. Rate types come from the tertiles of the ts's Madrid-local day (23, 24 or
25 hourly prices): cheapest third VALLEY, middle PLAIN, most expensive PEAK.
Hours with the same price always share a RateType.

A missing hour falls back to the same local hour of the most recent earlier day
that has a price (prices for the day after tomorrow are never published), so a
lookup never raises.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from src.domain.power_tariff.model import (
    LOCAL_TZ,
    EnergyDirection,
    Rate,
    RateBlock,
    RateType,
    build_rate_blocks,
    local_date_hour,
)

_UTC = timezone.utc
_FALLBACK_MAX_DAYS = 400


def _to_utc_hour(dt: datetime) -> datetime:
    """Hour-truncated UTC datetime; naive datetimes are treated as UTC."""
    aware = dt.replace(tzinfo=_UTC) if dt.tzinfo is None else dt.astimezone(_UTC)
    return datetime(aware.year, aware.month, aware.day, aware.hour, tzinfo=_UTC)


def _local_midnight_utc(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=LOCAL_TZ).astimezone(_UTC)


def _local_day_utc_hours(d: date) -> List[datetime]:
    start = _local_midnight_utc(d)
    n = int((_local_midnight_utc(d + timedelta(days=1)) - start).total_seconds() // 3600)
    return [start + timedelta(hours=i) for i in range(n)]


@dataclass
class _Day:
    utc_hours: List[datetime]
    import_rates: List[Rate]
    export_rates: List[Rate]
    by_hour: Dict[datetime, int]
    blocks: List[RateBlock]


class PvpcTariff:
    """Hourly PVPC import tariff with flat (or PVPC-following) export."""

    rate_schedule: Dict[Tuple[int, int], Rate]

    def __init__(
        self,
        prices_eur_kwh: Dict[datetime, float],
        export_price: float,
        export_follows_pvpc: bool = False,
        export_pvpc_discount: float = 0.0,
    ) -> None:
        self._prices: Dict[datetime, float] = {
            _to_utc_hour(k): float(v) for k, v in prices_eur_kwh.items()
        }
        self._export_price = float(export_price)
        self._export_follows_pvpc = export_follows_pvpc
        self._export_pvpc_discount = max(0.0, float(export_pvpc_discount))
        self._days: Dict[date, _Day] = {}
        self._current_datetime: Optional[datetime] = None
        self.rate_schedule = {}
        if self._prices:
            first_day = local_date_hour(min(self._prices))[0]
            self._seed_day = first_day
            self.rate_schedule = self._schedule_for(first_day)
        else:
            self._seed_day = None

    # -- pure lookups ------------------------------------------------------------

    def rate_at(self, ts: datetime, direction: EnergyDirection) -> Rate:
        local_date = local_date_hour(ts)[0]
        day = self._day(local_date)
        i = day.by_hour[_to_utc_hour(ts)]
        return day.import_rates[i] if direction == EnergyDirection.IMPORT else day.export_rates[i]

    def day_rate_blocks(self, local_date: date) -> List[RateBlock]:
        return self._day(local_date).blocks

    # -- legacy stateful API -----------------------------------------------------

    def update_datetime(self, dt: datetime) -> None:
        self._current_datetime = dt
        self.rate_schedule = self._schedule_for(local_date_hour(dt)[0])

    def get_rate(self, hour: int, direction: EnergyDirection) -> Rate:
        if not 0 <= hour < 24:
            raise ValueError(f"Hour must be between 0 and 23, got {hour}")
        local_date = local_date_hour(self._current_datetime)[0] if self._current_datetime else self._seed_day
        if local_date is None:
            return Rate(price=0.0, energy_direction=direction)
        day = self._day(local_date)
        i = self._wall_hour_index(day, hour)
        return day.import_rates[i] if direction == EnergyDirection.IMPORT else day.export_rates[i]

    def get_import_rate(self, hour: int) -> float:
        return self.get_rate(hour, EnergyDirection.IMPORT).price

    def get_export_rate(self, hour: int) -> float:
        return self.get_rate(hour, EnergyDirection.EXPORT).price

    # -- internals ---------------------------------------------------------------

    def _price(self, utc_hour: datetime) -> float:
        price = self._prices.get(utc_hour)
        if price is not None:
            return price
        if not self._prices:
            return 0.0
        local = utc_hour.astimezone(LOCAL_TZ)
        for step in (-1, 1):
            for k in range(1, _FALLBACK_MAX_DAYS):
                d = local.date() + timedelta(days=step * k)
                cand = _to_utc_hour(datetime(d.year, d.month, d.day, local.hour, tzinfo=LOCAL_TZ))
                price = self._prices.get(cand)
                if price is not None:
                    return price
        return 0.0

    def _day(self, local_date: date) -> _Day:
        day = self._days.get(local_date)
        if day is not None:
            return day
        hours = _local_day_utc_hours(local_date)
        prices = [self._price(h) for h in hours]
        types = _tertile_rate_types(prices)
        imports = [Rate(price=p, energy_direction=EnergyDirection.IMPORT, _rate_type=t)
                   for p, t in zip(prices, types)]
        if self._export_follows_pvpc:
            exports = [Rate(price=max(0.0, p - self._export_pvpc_discount),
                            energy_direction=EnergyDirection.EXPORT, _rate_type=t)
                       for p, t in zip(prices, types)]
        else:
            flat = Rate(price=self._export_price, energy_direction=EnergyDirection.EXPORT,
                        _rate_type=RateType.PLAIN)
            exports = [flat] * len(hours)
        day = _Day(utc_hours=hours, import_rates=imports, export_rates=exports,
                   by_hour={h: i for i, h in enumerate(hours)}, blocks=[])
        day.blocks = build_rate_blocks([imports[self._wall_hour_index(day, h)] for h in range(24)])
        self._days[local_date] = day
        return day

    @staticmethod
    def _wall_hour_index(day: _Day, hour: int) -> int:
        """Index of the first UTC hour whose local wall hour is ``hour`` (or the next one on spring-forward)."""
        for target in range(hour, 24):
            for i, h in enumerate(day.utc_hours):
                if h.astimezone(LOCAL_TZ).hour == target:
                    return i
        return len(day.utc_hours) - 1

    def _schedule_for(self, local_date: date) -> Dict[Tuple[int, int], Rate]:
        return {(b["start"], b["end"]): b["rate"] for b in self._day(local_date).blocks}


def _type_for_rank(rank: int, n: int) -> RateType:
    if rank * 3 < n:
        return RateType.VALLEY
    if rank * 3 < 2 * n:
        return RateType.PLAIN
    return RateType.PEAK


def _tertile_rate_types(prices: List[float]) -> List[RateType]:
    """Classify a day's hourly prices into VALLEY/PLAIN/PEAK by rank tertile.

    A tie takes the type of its cheapest rank, so equal prices are never split
    across a tertile cut.
    """
    n = len(prices)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda h: prices[h])
    result: List[RateType] = [RateType.PLAIN] * n
    i = 0
    while i < n:
        price = prices[order[i]]
        j = i + 1
        while j < n and prices[order[j]] == price:
            j += 1
        rate_type = _type_for_rank(i, n)
        for k in range(i, j):
            result[order[k]] = rate_type
        i = j
    return result
