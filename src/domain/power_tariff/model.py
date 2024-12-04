from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple


@dataclass
class PowerTariff:
    import_rate_schedule: Dict[Tuple[int, int], float]
    export_rate_schedule: Dict[Tuple[int, int], float]
    _current_datetime: Optional[datetime] = None

    def __post_init__(self) -> None:
        self._validate_schedule(self.import_rate_schedule, "import")
        self._validate_schedule(self.export_rate_schedule, "export")
        self._weekend_rate = min(self.import_rate_schedule.values())

    def _validate_schedule(self, schedule: Dict[Tuple[int, int], float], schedule_type: str) -> None:
        for (start, end), rate in schedule.items():
            if not (0 <= start < 24 and 0 <= end <= 24):
                raise ValueError(f"Invalid hours in {schedule_type} schedule: {start}, {end}")
            if start >= end:
                raise ValueError(f"Start hour must be less than end hour in {schedule_type} schedule: {start}, {end}")
            if rate < 0:
                raise ValueError(f"Negative rate in {schedule_type} schedule: {rate}")

        covered_hours: set[int] = set()
        for start, end in schedule.keys():
            current_hours = set(range(start, end))
            if current_hours & covered_hours:
                raise ValueError(f"Overlapping periods in {schedule_type} schedule")
            covered_hours.update(current_hours)

        if covered_hours != set(range(24)):
            missing_hours = set(range(24)) - covered_hours
            raise ValueError(f"Incomplete {schedule_type} schedule. Missing hours: {missing_hours}")

    def update_datetime(self, dt: datetime) -> None:
        self._current_datetime = dt

    def get_import_rate(self, hour: int) -> float:
        if not 0 <= hour < 24:
            raise ValueError(f"Hour must be between 0 and 23, got {hour}")

        if self._current_datetime and self._current_datetime.weekday() >= 5:
            return self._weekend_rate

        for (start, end), rate in self.import_rate_schedule.items():
            if start <= hour < end:
                return rate
        return 0.0

    def get_export_rate(self, hour: int) -> float:
        if not 0 <= hour < 24:
            raise ValueError(f"Hour must be between 0 and 23, got {hour}")
        for (start, end), rate in self.export_rate_schedule.items():
            if start <= hour < end:
                return rate
        return 0.0
