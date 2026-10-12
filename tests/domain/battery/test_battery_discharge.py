import math

from pytest import approx

from src.domain.battery.models import Battery

ETA = math.sqrt(0.9)


def test_discharge_limited_by_power():
    battery = Battery(100.0, 20.0, 20.0)
    battery.current_charge = 100.0

    assert battery.discharge(10.0, 1.0) == 10.0
    assert battery.current_charge == approx(100.0 - 10.0 / ETA)


def test_discharge_limited_by_max_discharge_rate():
    battery = Battery(100.0, 20.0, 20.0)
    battery.current_charge = 100.0

    assert battery.discharge(30.0, 1.0) == 20.0
    assert battery.current_charge == approx(100.0 - 20.0 / ETA)


def test_discharge_limited_by_current_charge():
    battery = Battery(100.0, 20.0, 20.0)
    battery.current_charge = 10.0

    delivered = battery.discharge(20.0, 1.0)

    assert delivered == approx(10.0 * ETA)
    assert battery.current_charge == approx(0.0)


def test_discharge_never_goes_below_floor():
    battery = Battery(100.0, 20.0, 20.0)
    battery.current_charge = 12.0

    delivered = battery.discharge(20.0, 1.0, floor_kwh=10.0)

    assert delivered == approx(2.0 * ETA)
    assert battery.current_charge >= 10.0
    assert battery.current_charge == approx(10.0)


def test_discharge_below_floor_start_delivers_nothing():
    battery = Battery(100.0, 20.0, 20.0)
    battery.current_charge = 5.0
    assert battery.discharge(5.0, 1.0, floor_kwh=10.0) == 0.0
    assert battery.current_charge == 5.0
