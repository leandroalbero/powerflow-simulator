from src.domain.battery.models import Battery
from pytest import approx


def test_discharge_limited_by_power():
    # Arrange
    battery = Battery(100.0, 20.0, 20.0)
    battery.current_charge = 100.0
    power = 10.0
    duration = 1.0

    # Act
    actual_power = battery.discharge(power, duration)

    # Assert
    assert actual_power == power
    assert battery.current_charge == approx(100.0 - (power * duration / 0.95))


def test_discharge_limited_by_max_discharge_rate():
    # Arrange
    battery = Battery(100.0, 20.0, 20.0)
    battery.current_charge = 100.0

    # Act
    actual_power = battery.discharge(30.0, 1.0)

    # Assert
    assert actual_power == 20.0
    assert battery.current_charge == approx(100.0 - (20.0 / 0.95))


def test_discharge_limited_by_current_charge():
    # Arrange
    battery = Battery(100.0, 20.0, 20.0)
    battery.current_charge = 10.0

    # Act
    actual_power = battery.discharge(20.0, 1.0)

    # Assert
    assert actual_power < 20.0
    assert battery.current_charge >= 0.0
