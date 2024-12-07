import pytest
from datetime import datetime

from src.domain.power_tariff.model import Rate, EnergyDirection, RateType, PowerTariff


@pytest.fixture
def basic_schedule():
    return {
        (0, 7): Rate(price=0.1, energy_direction=EnergyDirection.IMPORT),
        (7, 17): Rate(price=0.2, energy_direction=EnergyDirection.IMPORT),
        (17, 24): Rate(price=0.3, energy_direction=EnergyDirection.IMPORT),
        (0, 12): Rate(price=0.05, energy_direction=EnergyDirection.EXPORT),
        (12, 24): Rate(price=0.08, energy_direction=EnergyDirection.EXPORT),
    }


@pytest.fixture
def weekend_rate():
    return Rate(price=0.15, energy_direction=EnergyDirection.IMPORT)


def test_rate_type_assignment():
    schedule = {
        (0, 8): Rate(price=0.1, energy_direction=EnergyDirection.IMPORT),
        (8, 16): Rate(price=0.2, energy_direction=EnergyDirection.IMPORT),
        (16, 24): Rate(price=0.3, energy_direction=EnergyDirection.IMPORT),
        (0, 12): Rate(price=0.05, energy_direction=EnergyDirection.EXPORT),
        (12, 24): Rate(price=0.08, energy_direction=EnergyDirection.EXPORT),
    }
    tariff = PowerTariff(schedule)

    assert schedule[(0, 8)].rate_type == RateType.VALLEY
    assert schedule[(8, 16)].rate_type == RateType.PLAIN
    assert schedule[(16, 24)].rate_type == RateType.PEAK
    assert schedule[(0, 12)].rate_type == RateType.VALLEY
    assert schedule[(12, 24)].rate_type == RateType.PEAK


def test_invalid_hours():
    with pytest.raises(ValueError, match="Invalid hours in schedule"):
        PowerTariff({
            (0, 25): Rate(price=0.1, energy_direction=EnergyDirection.IMPORT),
            (0, 24): Rate(price=0.05, energy_direction=EnergyDirection.EXPORT),
        })


def test_overlapping_periods():
    with pytest.raises(ValueError, match="Overlapping import periods in schedule"):
        PowerTariff({
            (0, 8): Rate(price=0.1, energy_direction=EnergyDirection.IMPORT),
            (7, 16): Rate(price=0.2, energy_direction=EnergyDirection.IMPORT),
            (16, 24): Rate(price=0.3, energy_direction=EnergyDirection.IMPORT),
            (0, 24): Rate(price=0.05, energy_direction=EnergyDirection.EXPORT),
        })


def test_incomplete_schedule():
    with pytest.raises(ValueError, match="Incomplete import schedule"):
        PowerTariff({
            (0, 8): Rate(price=0.1, energy_direction=EnergyDirection.IMPORT),
            (9, 24): Rate(price=0.2, energy_direction=EnergyDirection.IMPORT),
            (0, 24): Rate(price=0.05, energy_direction=EnergyDirection.EXPORT),
        })


@pytest.mark.skip(reason="This test is not implemented yet.")
def test_negative_rate():
    with pytest.raises(ValueError, match="Negative rate in schedule"):
        PowerTariff({
            (0, 24): Rate(price=-0.1, energy_direction=EnergyDirection.IMPORT),
            (0, 24): Rate(price=0.05, energy_direction=EnergyDirection.EXPORT),
        })


def test_get_rate(basic_schedule):
    tariff = PowerTariff(basic_schedule)

    assert tariff.get_import_rate(3) == 0.1
    assert tariff.get_import_rate(12) == 0.2
    assert tariff.get_import_rate(20) == 0.3

    assert tariff.get_export_rate(6) == 0.05
    assert tariff.get_export_rate(18) == 0.08


def test_weekend_rate(basic_schedule, weekend_rate):
    tariff = PowerTariff(basic_schedule, weekend_rate=weekend_rate)

    # Test weekday
    tariff.update_datetime(datetime(2024, 1, 3))  # Wednesday
    assert tariff.get_import_rate(12) == 0.2

    # Test weekend
    tariff.update_datetime(datetime(2024, 1, 6))  # Saturday
    assert tariff.get_import_rate(12) == 0.15


@pytest.mark.skip(reason="This test is not implemented yet.")
def test_invalid_hour():
    tariff = PowerTariff(rate_schedule={
        (0, 23): Rate(price=0.1, energy_direction=EnergyDirection.IMPORT),
        (0, 23): Rate(price=0.05, energy_direction=EnergyDirection.EXPORT),
    })

    with pytest.raises(ValueError, match="Hour must be between 0 and 23"):
        tariff.get_import_rate(24)


def test_rate_type_property():
    rate = Rate(price=0.1, energy_direction=EnergyDirection.IMPORT)
    rate.rate_type = RateType.PEAK
    assert rate.rate_type == RateType.PEAK
