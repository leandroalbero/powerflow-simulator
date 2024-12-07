import pytest
from unittest.mock import Mock

from src.domain.strategy.model import SelfConsumeStrategy, ForceChargeValleyAndPrePeakStrategy, EnergyFlow, \
    ForceChargeAtNightStrategy, BaseEnergyStrategy
from src.domain.battery.models import Battery
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import PowerTariff, Rate, EnergyDirection, RateType


@pytest.fixture
def battery():
    battery = Mock(spec=Battery)
    battery.capacity = 10.0
    battery.current_charge = 5.0
    battery.max_discharge_rate = 2.0
    battery.max_charge_rate = 2.0
    battery.charge_level = 0.5
    return battery


@pytest.fixture
def grid():
    grid = Mock(spec=Grid)
    grid.import_power.return_value = 1.0
    grid.export_power.return_value = 1.0
    return grid


@pytest.fixture
def rates():
    return {
        0: Rate(price=0.1, energy_direction=EnergyDirection.IMPORT),
        8: Rate(price=0.2, energy_direction=EnergyDirection.IMPORT),
        16: Rate(price=0.3, energy_direction=EnergyDirection.IMPORT),
    }


@pytest.fixture
def tariff(rates):
    tariff = Mock(spec=PowerTariff)
    tariff.rate_schedule = rates
    tariff.get_rate.side_effect = lambda hour, direction: rates.get(hour - (hour % 8), rates[0])
    return tariff


class TestBaseEnergyStrategy:
    def test_calculate_initial_flows(self, battery, grid, tariff):
        class TestStrategy(BaseEnergyStrategy):
            def calculate_energy_flows(self, solar_power, load_power, hour, duration):
                return super()._calculate_initial_flows(solar_power, load_power)

        strategy = TestStrategy(battery, grid, tariff)
        flows = strategy._calculate_initial_flows(10.0, 5.0)

        assert isinstance(flows, EnergyFlow)
        assert flows.direct_solar == 5.0
        assert flows.remaining_solar == 5.0
        assert flows.remaining_load == 0.0

    def test_handle_remaining_solar(self, battery, grid, tariff):
        class TestStrategy(BaseEnergyStrategy):
            def calculate_energy_flows(self, solar_power, load_power, hour, duration):
                pass

        strategy = TestStrategy(battery, grid, tariff)
        flows = EnergyFlow(remaining_solar=3.0)
        battery.charge.return_value = 1.0
        grid.export_power.return_value = 1.0

        strategy._handle_remaining_solar(flows, 1.0)

        assert flows.battery_charge == 1.0
        assert flows.grid_export == 1.0
        assert flows.remaining_solar == 1.0

    def test_error_handling_discharge_battery(self, battery, grid, tariff):
        class TestStrategy(BaseEnergyStrategy):
            def calculate_energy_flows(self, solar_power, load_power, hour, duration):
                pass

        strategy = TestStrategy(battery, grid, tariff)
        battery.discharge.side_effect = ValueError("Battery error")

        with pytest.raises(ValueError, match="Battery error"):
            strategy._discharge_battery(5.0, 1.0)


class TestSelfConsumeStrategy:
    def test_calculate_energy_flows(self, battery, grid, tariff):
        battery.charge.return_value = 2.0
        grid.export_power.return_value = 1.0

        strategy = SelfConsumeStrategy(battery, grid, tariff)
        flows = strategy.calculate_energy_flows(10.0, 5.0, 0, 1.0)

        assert flows.direct_solar == 5.0
        assert flows.battery_charge == 2.0
        battery.charge.assert_called()
        grid.export_power.assert_called()

    def test_no_solar_high_load(self, battery, grid, tariff):
        battery.discharge.return_value = 2.0
        grid.import_power.return_value = 8.0

        strategy = SelfConsumeStrategy(battery, grid, tariff)
        flows = strategy.calculate_energy_flows(0.0, 10.0, 0, 1.0)

        assert flows.direct_solar == 0.0
        assert flows.battery_discharge == 2.0
        battery.discharge.assert_called()
        grid.import_power.assert_called()


class TestForceChargeAtNightStrategy:
    @pytest.mark.skip("Test not implemented")
    def test_valley_period_charging(self, battery, grid, tariff):
        battery.charge.return_value = 2.0
        grid.import_power.return_value = 1.0

        strategy = ForceChargeAtNightStrategy(battery, grid, tariff)
        battery.current_charge = 2.0
        flows = strategy.calculate_energy_flows(5.0, 2.0, 0, 1.0)

        assert flows.direct_solar == 2.0
        battery.charge.assert_called()
        grid.import_power.assert_called()

    def test_peak_period_discharging(self, battery, grid, tariff):
        battery.discharge.return_value = 4.0
        grid.import_power.return_value = 2.0

        strategy = ForceChargeAtNightStrategy(battery, grid, tariff)
        flows = strategy.calculate_energy_flows(2.0, 8.0, 16, 1.0)

        assert flows.direct_solar == 2.0
        assert flows.battery_discharge == 4.0
        battery.discharge.assert_called()

    def test_error_invalid_rate_schedule(self, battery, grid):
        invalid_tariff = Mock(spec=PowerTariff)
        invalid_tariff.rate_schedule = {0: Rate(price=0.1,
                                                energy_direction=EnergyDirection.IMPORT,
                                                )}

        with pytest.raises(ValueError, match="Tariff must have at least 3 different rates"):
            ForceChargeAtNightStrategy(battery, grid, invalid_tariff)


class TestForceChargeValleyAndPrePeakStrategy:
    def test_valley_charging(self, battery, grid, tariff):
        battery.charge.return_value = 1.0
        grid.import_power.return_value = 1.0

        strategy = ForceChargeValleyAndPrePeakStrategy(battery, grid, tariff)
        battery.current_charge = 2.0
        flows = strategy.calculate_energy_flows(3.0, 1.0, 0, 1.0)

        assert flows.direct_solar == 1.0
        assert flows.battery_charge == 1.0
        battery.charge.assert_called()

    def test_peak_discharge(self, battery, grid, tariff):
        battery.discharge.return_value = 2.0
        grid.import_power.return_value = 2.0

        strategy = ForceChargeValleyAndPrePeakStrategy(battery, grid, tariff)
        flows = strategy.calculate_energy_flows(1.0, 5.0, 16, 1.0)

        assert flows.direct_solar == 1.0
        assert flows.battery_discharge == 2.0
        battery.discharge.assert_called()

    def test_after_valley_behavior(self, battery, grid, tariff):
        strategy = ForceChargeValleyAndPrePeakStrategy(battery, grid, tariff)
        flows = strategy.calculate_energy_flows(2.0, 3.0, 8, 1.0)

        assert flows.direct_solar == 2.0
        assert strategy._is_after_valley_block(8)


class TestBoundaryConditions:
    @pytest.mark.parametrize("hour", [-1, 24, 25])
    def test_invalid_hours(self, battery, grid, tariff, hour):
        battery.charge.return_value = 1.0
        grid.import_power.return_value = 1.0

        strategy = ForceChargeValleyAndPrePeakStrategy(battery, grid, tariff)
        flows = strategy.calculate_energy_flows(1.0, 1.0, hour, 1.0)

        assert isinstance(flows, EnergyFlow)
        assert flows.battery_charge == 1.0

    def test_zero_duration(self, battery, grid, tariff):
        strategy = SelfConsumeStrategy(battery, grid, tariff)
        with pytest.raises(ZeroDivisionError):
            strategy.calculate_energy_flows(1.0, 1.0, 0, 0.0)

    def test_negative_power_values(self, battery, grid, tariff):
        strategy = SelfConsumeStrategy(battery, grid, tariff)
        flows = strategy.calculate_energy_flows(-1.0, -1.0, 0, 1.0)

        assert flows.direct_solar == 0.0
        assert flows.remaining_solar == 0.0
        assert flows.remaining_load == 0.0
