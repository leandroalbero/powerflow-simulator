from src.domain.battery.models import Battery


def test_init_defaults():
    battery = Battery(100.0, 20.0, 20.0)

    assert battery.capacity == 100.0
    assert battery.current_charge == 50.0
    assert battery.max_charge_rate == 20.0
    assert battery.max_discharge_rate == 20.0
    assert battery.efficiency == 0.90
    assert battery.taper_start == 0.9
    assert battery.taper_factor == 0.7


def test_init_from_config_values():
    battery = Battery(15.36, 4.5, 4.6, efficiency=0.81, initial_soc=0.1, taper_start=0.95, taper_factor=0.5)

    assert battery.current_charge == 15.36 * 0.1
    assert battery.charge_efficiency == 0.9
    assert battery.discharge_efficiency == 0.9
    assert battery.taper_start == 0.95
    assert battery.taper_factor == 0.5
