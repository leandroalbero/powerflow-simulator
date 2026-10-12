from datetime import datetime, timedelta, timezone

import pytest

from src.domain.power_tariff.model import EnergyDirection, RateType
from src.domain.power_tariff.pvpc import PvpcTariff

_UTC = timezone.utc


def _build_day_prices(day: datetime, prices: list) -> dict:
    """Produce a 24-hour UTC-keyed price dict starting at ``day`` (UTC midnight)."""
    assert len(prices) == 24
    return {day + timedelta(hours=h): prices[h] for h in range(24)}


@pytest.fixture
def jan1_prices():
    # Monotonically increasing so valley/plain/peak buckets are deterministic:
    #   hours 0-7 cheapest (valley), 8-15 middle (plain), 16-23 priciest (peak).
    day = datetime(2024, 1, 1, 0, 0, tzinfo=_UTC)
    return day, _build_day_prices(day, [0.05 + 0.01 * h for h in range(24)])


@pytest.fixture
def jan2_prices():
    # Reversed order: priciest hours at 0-7, cheapest at 16-23.
    day = datetime(2024, 1, 2, 0, 0, tzinfo=_UTC)
    return day, _build_day_prices(day, [0.30 - 0.01 * h for h in range(24)])


def test_get_import_rate_returns_cached_price(jan1_prices):
    day, prices = jan1_prices
    tariff = PvpcTariff(prices_eur_kwh=prices, export_price=0.08)
    tariff.update_datetime(day)

    for h in range(24):
        assert tariff.get_import_rate(h) == pytest.approx(prices[day + timedelta(hours=h)])


def test_tertile_classification_counts(jan1_prices):
    day, prices = jan1_prices
    tariff = PvpcTariff(prices_eur_kwh=prices, export_price=0.08)
    tariff.update_datetime(day)

    rate_types = [tariff.get_rate(h, EnergyDirection.IMPORT).rate_type for h in range(24)]
    assert rate_types.count(RateType.VALLEY) == 8
    assert rate_types.count(RateType.PLAIN) == 8
    assert rate_types.count(RateType.PEAK) == 8


def test_tertile_classification_order(jan1_prices):
    """Cheapest 8 hours valley, next 8 plain, most expensive 8 peak."""
    day, prices = jan1_prices
    tariff = PvpcTariff(prices_eur_kwh=prices, export_price=0.08)
    tariff.update_datetime(day)

    for h in range(0, 8):
        assert tariff.get_rate(h, EnergyDirection.IMPORT).rate_type == RateType.VALLEY
    for h in range(8, 16):
        assert tariff.get_rate(h, EnergyDirection.IMPORT).rate_type == RateType.PLAIN
    for h in range(16, 24):
        assert tariff.get_rate(h, EnergyDirection.IMPORT).rate_type == RateType.PEAK


def test_update_datetime_reclassifies_new_day(jan1_prices, jan2_prices):
    day1, prices1 = jan1_prices
    day2, prices2 = jan2_prices

    combined = {**prices1, **prices2}
    tariff = PvpcTariff(prices_eur_kwh=combined, export_price=0.08)

    tariff.update_datetime(day1)
    # On day 1 hour 0 is cheapest -> VALLEY.
    assert tariff.get_rate(0, EnergyDirection.IMPORT).rate_type == RateType.VALLEY
    assert tariff.get_rate(23, EnergyDirection.IMPORT).rate_type == RateType.PEAK

    tariff.update_datetime(day2)
    # On day 2 the prices reverse: hour 0 is now PEAK, hour 23 is VALLEY.
    assert tariff.get_rate(0, EnergyDirection.IMPORT).rate_type == RateType.PEAK
    assert tariff.get_rate(23, EnergyDirection.IMPORT).rate_type == RateType.VALLEY


def test_export_rate_is_flat_and_plain(jan1_prices):
    day, prices = jan1_prices
    tariff = PvpcTariff(prices_eur_kwh=prices, export_price=0.07)
    tariff.update_datetime(day)

    for h in range(24):
        rate = tariff.get_rate(h, EnergyDirection.EXPORT)
        assert rate.price == pytest.approx(0.07)
        assert rate.rate_type == RateType.PLAIN
        assert rate.energy_direction == EnergyDirection.EXPORT


def test_get_export_rate_helper(jan1_prices):
    day, prices = jan1_prices
    tariff = PvpcTariff(prices_eur_kwh=prices, export_price=0.09)
    tariff.update_datetime(day)
    assert tariff.get_export_rate(5) == pytest.approx(0.09)


def test_get_rate_before_update_uses_seed(jan1_prices):
    # Construction seeds the first available day so rate-aware strategies
    # can initialize before EnergySimulator calls update_datetime.
    day, prices = jan1_prices
    tariff = PvpcTariff(prices_eur_kwh=prices, export_price=0.08)
    rate = tariff.get_rate(0, EnergyDirection.IMPORT)
    assert rate.price == pytest.approx(prices[day])
    assert rate.rate_type is not None


def test_missing_price_raises_clear_error():
    # Only provide a partial day.
    day = datetime(2024, 3, 15, 0, 0, tzinfo=_UTC)
    prices = {day + timedelta(hours=h): 0.10 for h in range(12)}
    tariff = PvpcTariff(prices_eur_kwh=prices, export_price=0.08)
    with pytest.raises(RuntimeError, match="Missing PVPC price"):
        tariff.update_datetime(day)


def test_invalid_hour_raises_from_get_rate(jan1_prices):
    day, prices = jan1_prices
    tariff = PvpcTariff(prices_eur_kwh=prices, export_price=0.08)
    tariff.update_datetime(day)
    with pytest.raises(ValueError, match="Hour must be between 0 and 23"):
        tariff.get_rate(24, EnergyDirection.IMPORT)


def test_rate_schedule_is_populated_after_update(jan1_prices):
    day, prices = jan1_prices
    tariff = PvpcTariff(prices_eur_kwh=prices, export_price=0.08)
    tariff.update_datetime(day)

    assert tariff.rate_schedule  # non-empty
    # Every hour 0..23 must be covered exactly once by some (start, end) range.
    covered = set()
    for (start, end), rate in tariff.rate_schedule.items():
        assert rate.energy_direction == EnergyDirection.IMPORT
        for h in range(start, end):
            assert h not in covered
            covered.add(h)
    assert covered == set(range(24))
