from src.domain.battery.models import Battery


def test_init():
    # Arrange
    capacity = 100.0
    max_charge_rate = 20.0
    max_discharge_rate = 20.0

    # Act
    battery = Battery(capacity, max_charge_rate, max_discharge_rate)

    # Assert
    assert battery.capacity == 100.0
    assert battery.current_charge == 50.0
    assert battery.max_charge_rate == 20.0
    assert battery.max_discharge_rate == 20.0
    assert battery.efficiency == 0.95
    assert battery._charge_taper_start == 0.9
    assert battery._taper_factor == 0.7