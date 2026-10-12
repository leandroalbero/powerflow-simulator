from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, TypedDict
from zoneinfo import ZoneInfo

from src.domain.power_tariff.holidays import is_offpeak_day

LOCAL_TZ = ZoneInfo("Europe/Madrid")


class RateType(Enum):
    PEAK = "peak"
    PLAIN = "plain"
    VALLEY = "valley"


class EnergyDirection(Enum):
    IMPORT = "import"
    EXPORT = "export"


@dataclass
class Rate:
    price: float
    energy_direction: EnergyDirection
    _rate_type: Optional[RateType] = None

    @property
    def rate_type(self) -> Optional[RateType]:
        return self._rate_type

    @rate_type.setter
    def rate_type(self, value: RateType) -> None:
        self._rate_type = value


class RateBlock(TypedDict):
    start: int
    end: int
    rate: Rate


def local_date_hour(ts: datetime) -> Tuple[date, int]:
    """Europe/Madrid calendar date and hour of ``ts``; naive values are local wall-clock."""
    if ts.tzinfo is not None:
        ts = ts.astimezone(LOCAL_TZ)
    return ts.date(), ts.hour


def build_rate_blocks(hourly: List[Rate]) -> List[RateBlock]:
    """Group consecutive hours with equal import price and rate type."""
    blocks: List[RateBlock] = []
    start = 0
    for h in range(1, len(hourly) + 1):
        if h == len(hourly) or (hourly[h].price, hourly[h].rate_type) != (hourly[start].price, hourly[start].rate_type):
            blocks.append(RateBlock(start=start, end=h, rate=hourly[start]))
            start = h
    return blocks


@dataclass
class PowerTariff:
    """Time-of-use tariff: an hourly schedule plus an all-day import rate on off-peak days.

    ``rate_at(ts, direction)`` is the pure lookup every caller should use. The
    ``update_datetime``/``get_rate`` pair is kept for backward compatibility.
    """

    rate_schedule: Dict[Tuple[int, int], Rate]
    weekend_rate: Optional[Rate] = None
    _current_datetime: Optional[datetime] = None
    _hourly: Dict[EnergyDirection, List[Rate]] = field(default_factory=dict, init=False, repr=False)
    _offpeak_import: Optional[Rate] = field(default=None, init=False, repr=False)
    _blocks_cache: Dict[date, List[RateBlock]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._set_rate_types()
        self._validate_schedule()
        self._hourly = {
            direction: [self._schedule_rate(h, direction) for h in range(24)]
            for direction in EnergyDirection
        }
        if self.weekend_rate is not None:
            # 2.0TD off-peak days are P3 all day: import only, always the valley period.
            self._offpeak_import = Rate(
                price=self.weekend_rate.price,
                energy_direction=EnergyDirection.IMPORT,
                _rate_type=RateType.VALLEY,
            )

    def _set_rate_types(self) -> None:
        for direction in EnergyDirection:
            direction_prices = [rate.price for rate in self.rate_schedule.values()
                                if rate.energy_direction == direction]
            if not direction_prices:
                continue

            valley_price = min(direction_prices)
            peak_price = max(direction_prices)

            for rate in self.rate_schedule.values():
                if rate.energy_direction == direction:
                    if rate.price == valley_price:
                        rate.rate_type = RateType.VALLEY
                    elif rate.price == peak_price:
                        rate.rate_type = RateType.PEAK
                    else:
                        rate.rate_type = RateType.PLAIN

    def _validate_schedule(self) -> None:
        import_periods: set[int] = set()
        export_periods: set[int] = set()

        for (start, end), rate in self.rate_schedule.items():
            if not (0 <= start < 24 and 0 <= end <= 24):
                raise ValueError(f"Invalid hours in schedule: {start}, {end}")
            if start >= end:
                raise ValueError(f"Start hour must be less than end hour in schedule: {start}, {end}")
            if rate.price < 0:
                raise ValueError(f"Negative rate in schedule: {rate.price}")

            current_hours = set(range(start, end))
            if rate.energy_direction == EnergyDirection.IMPORT:
                if current_hours & import_periods:
                    raise ValueError("Overlapping import periods in schedule")
                import_periods.update(current_hours)
            else:
                if current_hours & export_periods:
                    raise ValueError("Overlapping export periods in schedule")
                export_periods.update(current_hours)

        if import_periods != set(range(24)):
            missing_hours = set(range(24)) - import_periods
            raise ValueError(f"Incomplete import schedule. Missing hours: {missing_hours}")

        if export_periods != set(range(24)):
            missing_hours = set(range(24)) - export_periods
            raise ValueError(f"Incomplete export schedule. Missing hours: {missing_hours}")

    def _schedule_rate(self, hour: int, direction: EnergyDirection) -> Rate:
        for (start, end), rate in self.rate_schedule.items():
            if start <= hour < end and rate.energy_direction == direction:
                return rate
        return Rate(price=0.0, energy_direction=direction)

    def _rate_for(self, local_date: Optional[date], hour: int, direction: EnergyDirection) -> Rate:
        if (direction == EnergyDirection.IMPORT and self._offpeak_import is not None
                and local_date is not None and is_offpeak_day(local_date)):
            return self._offpeak_import
        return self._hourly[direction][hour]

    # -- pure lookups ------------------------------------------------------------

    def rate_at(self, ts: datetime, direction: EnergyDirection) -> Rate:
        """Rate in force at ``ts``; depends only on the local date and hour of ``ts``."""
        local_date, hour = local_date_hour(ts)
        return self._rate_for(local_date, hour, direction)

    def day_rate_blocks(self, local_date: date) -> List[RateBlock]:
        blocks = self._blocks_cache.get(local_date)
        if blocks is None:
            blocks = build_rate_blocks([self._rate_for(local_date, h, EnergyDirection.IMPORT) for h in range(24)])
            self._blocks_cache[local_date] = blocks
        return blocks

    # -- legacy stateful API -----------------------------------------------------

    def update_datetime(self, dt: datetime) -> None:
        self._current_datetime = dt

    def get_rate(self, hour: int, direction: EnergyDirection) -> Rate:
        if not 0 <= hour < 24:
            raise ValueError(f"Hour must be between 0 and 23, got {hour}")
        local_date = local_date_hour(self._current_datetime)[0] if self._current_datetime else None
        return self._rate_for(local_date, hour, direction)

    def get_import_rate(self, hour: int) -> float:
        return self.get_rate(hour, EnergyDirection.IMPORT).price

    def get_export_rate(self, hour: int) -> float:
        return self.get_rate(hour, EnergyDirection.EXPORT).price
