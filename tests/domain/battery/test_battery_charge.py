import math

from pytest import approx

from src.domain.battery.models import Battery


def test_charge_level():
    battery = Battery(capacity=100.0, max_charge_rate=20.0, max_discharge_rate=20.0)
    assert battery.charge_level == 0.5
    battery.current_charge = 75.0
    assert battery.charge_level == 0.75


def test_max_charge_power_below_and_at_taper_start():
    battery = Battery(100.0, 20.0, 20.0)
    battery.current_charge = 80.0
    assert battery.max_charge_power() == 20.0
    battery.current_charge = 90.0
    assert battery.max_charge_power() == 20.0


def test_max_charge_power_above_taper():
    battery = Battery(100.0, 20.0, 20.0)
    battery.current_charge = 95.0
    assert battery.max_charge_power() == approx(20.0 * (0.3 + 0.7 * 0.5))
    battery.current_charge = 100.0
    assert battery.max_charge_power() == approx(20.0 * 0.3)


def test_charge_limited_by_power_stores_with_one_way_efficiency():
    battery = Battery(100.0, 20.0, 20.0, efficiency=0.81)
    battery.current_charge = 0.0

    drawn = battery.charge(10.0, 1.0)

    assert drawn == 10.0
    assert battery.current_charge == approx(10.0 * 0.9)


def test_charge_limited_by_max_charge_rate():
    battery = Battery(100.0, 20.0, 20.0, efficiency=0.81)
    battery.current_charge = 0.0

    assert battery.charge(30.0, 1.0) == 20.0
    assert battery.current_charge == approx(20.0 * 0.9)


def test_charge_limited_by_capacity():
    battery = Battery(100.0, 20.0, 20.0, taper_factor=0.0)
    battery.current_charge = 95.0

    drawn = battery.charge(20.0, 1.0)

    assert drawn == approx(5.0 / math.sqrt(0.9))
    assert battery.current_charge == approx(100.0)


def test_taper_reduces_power_not_efficiency():
    eta_c = math.sqrt(0.9)
    for level in (50.0, 95.0, 99.0):
        battery = Battery(100.0, 20.0, 20.0, efficiency=0.9)
        battery.current_charge = level
        drawn = battery.charge(1.0, 0.01)
        assert battery.current_charge - level == approx(drawn * 0.01 * eta_c)
    tapered = Battery(100.0, 20.0, 20.0)
    tapered.current_charge = 99.0
    assert tapered.charge(20.0, 1 / 60) < 20.0


def test_round_trip_of_one_kwh_returns_round_trip_efficiency():
    battery = Battery(100.0, 50.0, 50.0, efficiency=0.9, initial_soc=0.2)
    start = battery.current_charge
    battery.charge(1.0, 1.0)
    stored = battery.current_charge - start
    delivered = battery.discharge(50.0, 1.0, floor_kwh=start)
    assert stored == approx(math.sqrt(0.9))
    assert delivered == approx(0.9)
    assert battery.current_charge == approx(start)
