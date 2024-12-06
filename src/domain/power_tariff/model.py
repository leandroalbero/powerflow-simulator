from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Tuple


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
    def rate_type(self) -> RateType:
        return self._rate_type

    @rate_type.setter
    def rate_type(self, value: RateType) -> None:
        self._rate_type = value


@dataclass
class PowerTariff:
    rate_schedule: Dict[Tuple[int, int], Rate]
    weekend_rate: Optional[Rate] = None
    _current_datetime: Optional[datetime] = None

    def __post_init__(self) -> None:
        self._set_rate_types()
        self._validate_schedule()
        self._weekend_rate = self.weekend_rate or None

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

    def update_datetime(self, dt: datetime) -> None:
        self._current_datetime = dt

    def get_rate(self, hour: int, direction: EnergyDirection) -> Rate:
        if not 0 <= hour < 24:
            raise ValueError(f"Hour must be between 0 and 23, got {hour}")

        if self._current_datetime and self._current_datetime.weekday() >= 5 and self.weekend_rate:
            return self.weekend_rate

        for (start, end), rate in self.rate_schedule.items():
            if start <= hour < end and rate.energy_direction == direction:
                return rate

        return Rate(price=0.0, energy_direction=direction)

    def get_import_rate(self, hour: int) -> float:
        return self.get_rate(hour, EnergyDirection.IMPORT).price

    def get_export_rate(self, hour: int) -> float:
        return self.get_rate(hour, EnergyDirection.EXPORT).price
