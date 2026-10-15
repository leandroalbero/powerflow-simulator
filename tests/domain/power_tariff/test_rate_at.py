from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.domain.power_tariff.holidays import is_offpeak_day
from src.domain.power_tariff.model import EnergyDirection, PowerTariff, Rate, RateType

MAD = ZoneInfo("Europe/Madrid")


@pytest.fixture
def td20():
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


def test_good_friday_is_not_offpeak_but_national_day_is():
    assert not is_offpeak_day(date(2026, 4, 3))
    assert is_offpeak_day(date(2026, 10, 12))
    assert is_offpeak_day(date(2026, 1, 6))
    assert is_offpeak_day(date(2026, 3, 7))  # Saturday
    assert not is_offpeak_day(date(2026, 3, 9))


def test_weekend_export_equals_weekday_export(td20):
    sat = datetime(2026, 3, 7, 13, tzinfo=MAD)
    wed = datetime(2026, 3, 4, 13, tzinfo=MAD)
    assert td20.rate_at(sat, EnergyDirection.EXPORT).price == td20.rate_at(wed, EnergyDirection.EXPORT).price == 0.08
    td20.update_datetime(sat)
    assert td20.get_export_rate(13) == 0.08


@pytest.mark.parametrize("day", [date(2026, 3, 7), date(2026, 3, 8), date(2026, 10, 12), date(2026, 12, 8)])
def test_offpeak_day_import_is_valley_all_day(td20, day):
    for h in range(24):
        rate = td20.rate_at(datetime(day.year, day.month, day.day, h, tzinfo=MAD), EnergyDirection.IMPORT)
        assert rate.rate_type == RateType.VALLEY
        assert rate.price == 0.085
    blocks = td20.day_rate_blocks(day)
    assert len(blocks) == 1 and blocks[0]["rate"].rate_type == RateType.VALLEY


def test_good_friday_uses_weekday_schedule(td20):
    gf = datetime(2026, 4, 3, 19, tzinfo=MAD)
    assert td20.rate_at(gf, EnergyDirection.IMPORT).rate_type == RateType.PEAK


def test_rate_at_depends_only_on_its_argument(td20):
    ts = pd.Timestamp("2026-03-04 19:30", tz="Europe/Madrid")
    td20.update_datetime(datetime(2026, 3, 7, 3, tzinfo=MAD))
    assert td20.rate_at(ts, EnergyDirection.IMPORT).price == 0.197
    utc = ts.tz_convert("UTC")
    assert td20.rate_at(utc, EnergyDirection.IMPORT).price == 0.197
    assert td20.rate_at(ts.tz_localize(None), EnergyDirection.IMPORT).price == 0.197


def test_weekday_blocks(td20):
    blocks = td20.day_rate_blocks(date(2026, 3, 4))
    assert [(b["start"], b["end"]) for b in blocks] == [(0, 8), (8, 10), (10, 14), (14, 18), (18, 22), (22, 24)]
    assert [b["rate"].rate_type for b in blocks] == [RateType.VALLEY, RateType.PLAIN, RateType.PEAK,
                                                     RateType.PLAIN, RateType.PEAK, RateType.PLAIN]
    assert td20.day_rate_blocks(date(2026, 3, 4)) is blocks
