from src.domain.battery.models import Battery
from pytest import approx


def test_charge_level():
    # Arrange
    battery = Battery(capacity=100.0, max_charge_rate=20.0, max_discharge_rate=20.0)
    initial_level = battery.charge_level
    battery.current_charge = 75.0

    # Act
    new_level = battery.charge_level

    # Assert
    assert initial_level == 0.5
    assert new_level == 0.75


def test_charge_efficiency_below_taper():
    # Arrange
    battery = Battery(100.0, 20.0, 20.0, efficiency=0.9)
    battery.current_charge = 80.0  # 80% < 90% taper start

    # Act
    efficiency = battery._get_charge_efficiency()

    # Assert
    assert efficiency == 0.9


def test_charge_efficiency_at_taper():
    # Arrange
    battery = Battery(100.0, 20.0, 20.0, efficiency=0.9)
    battery.current_charge = 90.0  # 90% = taper start

    # Act
    efficiency = battery._get_charge_efficiency()

    # Assert
    assert efficiency == 0.9


def test_charge_efficiency_above_taper():
    # Arrange
    battery = Battery(100.0, 20.0, 20.0, efficiency=0.9)
    battery.current_charge = 95.0  # 95% > 90% taper start
    expected = 0.9 * (0.3 + 0.7 * 0.5)  # Factor of 0.5 due to being halfway to full

    # Act
    efficiency = battery._get_charge_efficiency()

    # Assert
    assert efficiency == approx(expected)


def test_charge_efficiency_minimum():
    # Arrange
    battery = Battery(100.0, 20.0, 20.0, efficiency=0.9)
    battery.current_charge = 99.9  # Nearly full

    # Act
    efficiency = battery._get_charge_efficiency()

    # Assert
    assert efficiency >= 0.1


def test_max_charge_power_below_taper():
    # Arrange
    battery = Battery(100.0, 20.0, 20.0)
    battery.current_charge = 80.0  # 80% < 90% taper start

    # Act
    max_power = battery._get_max_charge_power()

    # Assert
    assert max_power == 20.0


def test_max_charge_power_at_taper():
    # Arrange
    battery = Battery(100.0, 20.0, 20.0)
    battery.current_charge = 90.0  # 90% = taper start

    # Act
    max_power = battery._get_max_charge_power()

    # Assert
    assert max_power == 20.0


def test_max_charge_power_above_taper():
    # Arrange
    battery = Battery(100.0, 20.0, 20.0)
    battery.current_charge = 95.0  # 95% > 90% taper start
    expected = 20.0 * (0.3 + 0.7 * 0.5)  # Factor of 0.5 due to being halfway to full

    # Act
    max_power = battery._get_max_charge_power()

    # Assert
    assert max_power == approx(expected)


def test_charge_limited_by_power():
    # Arrange
    battery = Battery(100.0, 20.0, 20.0)
    battery.current_charge = 0.0
    power = 10.0
    duration = 1.0

    # Act
    actual_power = battery.charge(power, duration)

    # Assert
    assert actual_power == power
    assert battery.current_charge == approx(power * duration * 0.95)


def test_charge_limited_by_max_charge_rate():
    # Arrange
    battery = Battery(100.0, 20.0, 20.0)
    battery.current_charge = 0.0

    # Act
    actual_power = battery.charge(30.0, 1.0)

    # Assert
    assert actual_power == 20.0
    assert battery.current_charge == approx(20.0 * 0.95)


def test_charge_limited_by_capacity():
    # Arrange
    battery = Battery(100.0, 20.0, 20.0)
    battery.current_charge = 95.0

    # Act
    actual_power = battery.charge(20.0, 1.0)

    # Assert
    assert actual_power < 20.0
    assert battery.current_charge <= 100.0