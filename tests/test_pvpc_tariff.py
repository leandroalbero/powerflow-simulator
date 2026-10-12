from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from src.domain.power_tariff.model import EnergyDirection, RateType
from src.domain.power_tariff.pvpc import PvpcTariff, _tertile_rate_types

_UTC = timezone.utc
_MAD = ZoneInfo("Europe/Madrid")


def _local_day_prices(d: date, prices: list) -> dict:
    """UTC-keyed prices for each hour of Madrid-local day ``d``."""
    start = datetime(d.year, d.month, d.day, tzinfo=_MAD).astimezone(_UTC)
    return {start + timedelta(hours=h): p for h, p in enumerate(prices)}


def _local(d: date, hour: int, minute: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=_MAD)


JAN1 = date(2024, 1, 1)
JAN2 = date(2024, 1, 2)


@pytest.fixture
def jan1_prices():
    # Increasing through the local day: hours 0-7 valley, 8-15 plain, 16-23 peak.
    return JAN1, _local_day_prices(JAN1, [0.05 + 0.01 * h for h in range(24)])


@pytest.fixture
def jan2_prices():
    return JAN2, _local_day_prices(JAN2, [0.30 - 0.01 * h for h in range(24)])


def test_rate_at_returns_price_of_containing_utc_hour(jan1_prices):
    d, prices = jan1_prices
    tariff = PvpcTariff(prices_eur_kwh=prices, export_price=0.08)
    for h in range(24):
        ts = _local(d, h, 37)
        assert tariff.rate_at(ts, EnergyDirection.IMPORT).price == pytest.approx(0.05 + 0.01 * h)


def test_legacy_get_import_rate_uses_current_local_day(jan1_prices):
    d, prices = jan1_prices
    tariff = PvpcTariff(prices_eur_kwh=prices, export_price=0.08)
    tariff.update_datetime(_local(d, 0))
    for h in range(24):
        assert tariff.get_import_rate(h) == pytest.approx(0.05 + 0.01 * h)


def test_tertile_classification(jan1_prices):
    d, prices = jan1_prices
    tariff = PvpcTariff(prices_eur_kwh=prices, export_price=0.08)
    types = [tariff.rate_at(_local(d, h), EnergyDirection.IMPORT).rate_type for h in range(24)]
    assert types == [RateType.VALLEY] * 8 + [RateType.PLAIN] * 8 + [RateType.PEAK] * 8


def test_each_local_day_has_its_own_classification(jan1_prices, jan2_prices):
    tariff = PvpcTariff(prices_eur_kwh={**jan1_prices[1], **jan2_prices[1]}, export_price=0.08)
    assert tariff.rate_at(_local(JAN1, 0), EnergyDirection.IMPORT).rate_type == RateType.VALLEY
    assert tariff.rate_at(_local(JAN1, 23), EnergyDirection.IMPORT).rate_type == RateType.PEAK
    assert tariff.rate_at(_local(JAN2, 0), EnergyDirection.IMPORT).rate_type == RateType.PEAK
    assert tariff.rate_at(_local(JAN2, 23), EnergyDirection.IMPORT).rate_type == RateType.VALLEY
    assert tariff.day_rate_blocks(JAN1)[0]["rate"].rate_type == RateType.VALLEY
    assert tariff.day_rate_blocks(JAN2)[0]["rate"].rate_type == RateType.PEAK


def test_rate_at_is_independent_of_legacy_clock(jan1_prices, jan2_prices):
    tariff = PvpcTariff(prices_eur_kwh={**jan1_prices[1], **jan2_prices[1]}, export_price=0.08)
    before = tariff.rate_at(_local(JAN1, 20), EnergyDirection.IMPORT).price
    tariff.update_datetime(_local(JAN2, 20))
    assert tariff.rate_at(_local(JAN1, 20), EnergyDirection.IMPORT).price == before


def test_export_rate_is_flat_and_plain(jan1_prices):
    d, prices = jan1_prices
    tariff = PvpcTariff(prices_eur_kwh=prices, export_price=0.07)
    for h in range(24):
        rate = tariff.rate_at(_local(d, h), EnergyDirection.EXPORT)
        assert rate.price == pytest.approx(0.07)
        assert rate.rate_type == RateType.PLAIN
        assert rate.energy_direction == EnergyDirection.EXPORT
    tariff.update_datetime(_local(d, 5))
    assert tariff.get_export_rate(5) == pytest.approx(0.07)


def test_export_can_follow_pvpc_minus_discount(jan1_prices):
    d, prices = jan1_prices
    tariff = PvpcTariff(prices_eur_kwh=prices, export_price=0.07, export_follows_pvpc=True,
                        export_pvpc_discount=0.06)
    assert tariff.rate_at(_local(d, 0), EnergyDirection.EXPORT).price == pytest.approx(0.0)
    assert tariff.rate_at(_local(d, 20), EnergyDirection.EXPORT).price == pytest.approx(0.25 - 0.06)


def test_get_rate_before_update_uses_seed_day(jan1_prices):
    d, prices = jan1_prices
    tariff = PvpcTariff(prices_eur_kwh=prices, export_price=0.08)
    rate = tariff.get_rate(0, EnergyDirection.IMPORT)
    assert rate.price == pytest.approx(0.05)
    assert rate.rate_type is not None
    assert tariff.rate_schedule


def test_dst_fall_back_day_prices_25_distinct_utc_hours():
    d = date(2025, 10, 26)
    prices = _local_day_prices(d, [0.10 + 0.001 * i for i in range(25)])
    assert len(prices) == 25
    tariff = PvpcTariff(prices_eur_kwh=prices, export_price=0.05)
    first_two = datetime(2025, 10, 26, 0, 30, tzinfo=_UTC)   # 02:30 CEST
    second_two = datetime(2025, 10, 26, 1, 30, tzinfo=_UTC)  # 02:30 CET
    assert first_two.astimezone(_MAD).hour == second_two.astimezone(_MAD).hour == 2
    seen = set()
    for utc_hour in prices:
        seen.add(tariff.rate_at(utc_hour + timedelta(minutes=10), EnergyDirection.IMPORT).price)
    assert len(seen) == 25
    assert tariff.rate_at(first_two, EnergyDirection.IMPORT).price != tariff.rate_at(
        second_two, EnergyDirection.IMPORT).price
    types = [tariff.rate_at(h, EnergyDirection.IMPORT).rate_type for h in prices]
    assert types.count(RateType.VALLEY) == 9 and types.count(RateType.PEAK) == 8
    assert [b["start"] for b in tariff.day_rate_blocks(d)][0] == 0
    assert tariff.day_rate_blocks(d)[-1]["end"] == 24


def test_dst_spring_forward_day_has_23_prices():
    d = date(2026, 3, 29)
    prices = _local_day_prices(d, [0.10 + 0.01 * i for i in range(23)])
    tariff = PvpcTariff(prices_eur_kwh=prices, export_price=0.05)
    types = [tariff.rate_at(h, EnergyDirection.IMPORT).rate_type for h in prices]
    assert len(types) == 23
    assert tariff.day_rate_blocks(d)[-1]["end"] == 24


def test_missing_hour_falls_back_to_previous_day_same_local_hour(jan1_prices):
    d, prices = jan1_prices
    partial = _local_day_prices(JAN2, [0.5] * 12)  # JAN2 only has hours 0-11
    tariff = PvpcTariff(prices_eur_kwh={**prices, **partial}, export_price=0.08)
    assert tariff.rate_at(_local(JAN2, 5), EnergyDirection.IMPORT).price == pytest.approx(0.5)
    assert tariff.rate_at(_local(JAN2, 20), EnergyDirection.IMPORT).price == pytest.approx(0.05 + 0.01 * 20)
    tomorrow = date(2024, 1, 4)
    assert tariff.rate_at(_local(tomorrow, 18), EnergyDirection.IMPORT).price == pytest.approx(0.05 + 0.01 * 18)
    tariff.update_datetime(_local(tomorrow, 0))
    assert tariff.get_import_rate(18) == pytest.approx(0.05 + 0.01 * 18)


def test_never_raises_without_prices():
    tariff = PvpcTariff(prices_eur_kwh={}, export_price=0.08)
    assert tariff.rate_at(_local(JAN1, 3), EnergyDirection.IMPORT).price == 0.0


def test_invalid_hour_raises_from_get_rate(jan1_prices):
    d, prices = jan1_prices
    tariff = PvpcTariff(prices_eur_kwh=prices, export_price=0.08)
    with pytest.raises(ValueError, match="Hour must be between 0 and 23"):
        tariff.get_rate(24, EnergyDirection.IMPORT)


def test_day_blocks_cover_all_hours(jan1_prices):
    d, prices = jan1_prices
    tariff = PvpcTariff(prices_eur_kwh=prices, export_price=0.08)
    covered = []
    for b in tariff.day_rate_blocks(d):
        assert b["rate"].energy_direction == EnergyDirection.IMPORT
        covered.extend(range(b["start"], b["end"]))
    assert covered == list(range(24))


def test_tertiles_generalise_to_any_day_length():
    # Equal prices must share a type. Splitting a flat day by hour order
    # labelled it valley/plain/peak and made force_valleys grid-charge at hour 15.
    flat = _tertile_rate_types([1.0] * 24)
    assert len(set(flat)) == 1
    partial = _tertile_rate_types([0.05] * 8 + [0.20] * 16)
    assert len(set(partial[8:])) == 1
    assert len(_tertile_rate_types(list(range(23)))) == 23
    assert _tertile_rate_types([]) == []
