import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock

from src.domain.energy_simulator.models import EnergySimulator
from src.domain.battery.models import Battery
from src.domain.energy_load.model import EnergyLoad
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import PowerTariff
from src.domain.solar_generator.solar_generator import SolarGenerator
from src.domain.strategy.model import BaseEnergyStrategy, EnergyFlow


class MockStrategy(BaseEnergyStrategy):
    def calculate_energy_flows(self, solar_power: float, load_power: float, hour: int, duration: float) -> EnergyFlow:
        if solar_power < 0:
            raise ValueError("Solar generation cannot be negative")
        return EnergyFlow(
            direct_solar=min(solar_power, load_power),
            battery_charge=0.0,
            battery_discharge=0.0,
            grid_import=max(0, load_power - solar_power),
            grid_export=max(0, solar_power - load_power)
        )


class TestEnergySimulator:
    @pytest.fixture
    def mock_components(self):
        battery = Mock(spec=Battery)
        battery.capacity = 10.0
        battery.current_charge = 5.0

        load = Mock(spec=EnergyLoad)
        grid = Mock(spec=Grid)
        tariff = Mock(spec=PowerTariff)
        tariff.get_import_rate.return_value = 0.30
        tariff.get_export_rate.return_value = 0.10
        solar = Mock(spec=SolarGenerator)
        strategy = MockStrategy(battery, grid, tariff)

        return battery, load, grid, tariff, solar, strategy

    @pytest.fixture
    def simulator(self, mock_components):
        battery, load, grid, tariff, solar, strategy = mock_components
        return EnergySimulator(battery, load, grid, tariff, solar, strategy)

    def test_initialization(self, simulator, mock_components):
        battery, _, _, _, _, _ = mock_components
        assert simulator.battery == battery
        assert simulator.total_cost == 0.0
        assert simulator.total_solar_generated == 0.0
        assert len(simulator.timestamps) == 0

    def test_step_naive_datetime_raises_error(self, simulator):
        naive_datetime = datetime.now()
        with pytest.raises(ValueError, match="Timestamp must be timezone-aware"):
            simulator.step(naive_datetime)

    def test_step_duration_calculation(self, simulator, mock_components):
        _, load, _, _, solar, _ = mock_components
        current_time = datetime.now(timezone.utc)
        prev_time = current_time - timedelta(minutes=30)

        solar.get_generation_safe.return_value = 1000.0
        load.get_load_safe.return_value = 500.0

        simulator.step(current_time, prev_time)
        assert abs(simulator.total_solar_generated - 0.5) < 1e-10

    def test_battery_tracking(self, simulator, mock_components):
        battery, load, _, _, solar, _ = mock_components
        battery.current_charge = 7.5

        timestamp = datetime.now(timezone.utc)
        solar.get_generation_safe.return_value = 1000.0
        load.get_load_safe.return_value = 500.0

        simulator.step(timestamp)
        assert simulator.battery_levels[-1] == 7.5

    def test_negative_values_handling(self, simulator, mock_components):
        _, load, _, _, solar, _ = mock_components
        timestamp = datetime.now(timezone.utc)
        solar.get_generation_safe.return_value = -1000.0
        load.get_load_safe.return_value = 500.0

        with pytest.raises(ValueError, match="Solar generation cannot be negative"):
            simulator.step(timestamp)
