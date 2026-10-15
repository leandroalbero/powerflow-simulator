from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.domain.battery.models import Battery
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import EnergyDirection, PowerTariff, Rate

MAD = ZoneInfo("Europe/Madrid")


def make_td20() -> PowerTariff:
    schedule = {
        (0, 8): Rate(price=0.085, energy_direction=EnergyDirection.IMPORT),
        (8, 10): Rate(price=0.135, energy_direction=EnergyDirection.IMPORT),
        (10, 14): Rate(price=0.197, energy_direction=EnergyDirection.IMPORT),
        (14, 18): Rate(price=0.135, energy_direction=EnergyDirection.IMPORT),
        (18, 22): Rate(price=0.197, energy_direction=EnergyDirection.IMPORT),
        (22, 24): Rate(price=0.135, energy_direction=EnergyDirection.IMPORT),
        (0, 24): Rate(price=0.08, energy_direction=EnergyDirection.EXPORT),
    }
    return PowerTariff(rate_schedule=schedule,
                       weekend_rate=Rate(price=0.085, energy_direction=EnergyDirection.IMPORT))


def at(day: str, hour: int, minute: int = 0) -> datetime:
    y, m, d = (int(x) for x in day.split("-"))
    return datetime(y, m, d, hour, minute, tzinfo=MAD)


WEEKDAY = "2026-03-04"   # Wednesday
SATURDAY = "2026-03-07"


@pytest.fixture
def td20():
    return make_td20()


@pytest.fixture
def battery():
    return Battery(capacity=10.0, max_charge_rate=2.0, max_discharge_rate=2.0, efficiency=0.81, initial_soc=0.5)


@pytest.fixture
def grid():
    return Grid(max_import=5.5, max_export=5.5)
