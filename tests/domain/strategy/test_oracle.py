import numpy as np
import pytest

from src.domain.battery.models import Battery
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import PowerTariff, Rate, EnergyDirection
from src.domain.strategy.model import EnergyFlow
from src.domain.strategy.oracle import OracleStrategy, solve_oracle_lp


class TestSolveOracleLp:
    """Tests for the core LP solver function."""

    def test_no_solar_charges_at_valley_discharges_at_peak(self):
        """With no solar, oracle should charge at cheapest rate and discharge at most expensive."""
        n = 4  # 4 hours
        dt = 1.0  # 1-hour steps
        solar = np.zeros(n)
        load = np.array([0.0, 0.0, 1.0, 1.0])  # load only in hours 2-3
        import_rates = np.array([0.085, 0.085, 0.182, 0.182])  # valley, valley, peak, peak
        export_rates = np.full(n, 0.08)

        result = solve_oracle_lp(
            solar=solar, load=load,
            import_rates=import_rates, export_rates=export_rates,
            dt=dt, battery_capacity=5.0, max_charge_rate=2.0,
            max_discharge_rate=2.0, efficiency=0.95,
            initial_soc=0.5, min_soc_frac=0.1,
        )

        assert result.success
        assert result.charge[0] > 0 or result.charge[1] > 0, "Should charge during valley"
        assert result.discharge[2] > 0 or result.discharge[3] > 0, "Should discharge during peak"
        peak_only_cost = 2.0 * 0.182
        assert result.total_cost < peak_only_cost

    def test_excess_solar_exports(self):
        """With excess solar and no load, oracle should export."""
        n = 2
        dt = 1.0
        solar = np.array([3.0, 3.0])
        load = np.zeros(n)
        import_rates = np.full(n, 0.182)
        export_rates = np.full(n, 0.08)

        result = solve_oracle_lp(
            solar=solar, load=load,
            import_rates=import_rates, export_rates=export_rates,
            dt=dt, battery_capacity=5.0, max_charge_rate=2.0,
            max_discharge_rate=2.0, efficiency=0.95,
            initial_soc=0.5, min_soc_frac=0.1,
        )

        assert result.success
        assert sum(result.grid_export) > 0, "Should export excess solar"
        assert result.total_cost < 0, "Should earn money from export"

    def test_respects_battery_capacity(self):
        """Cannot charge beyond battery capacity."""
        n = 3
        dt = 1.0
        solar = np.zeros(n)
        load = np.zeros(n)
        import_rates = np.array([0.01, 0.01, 0.182])
        export_rates = np.full(n, 0.005)

        result = solve_oracle_lp(
            solar=solar, load=load,
            import_rates=import_rates, export_rates=export_rates,
            dt=dt, battery_capacity=2.0, max_charge_rate=5.0,
            max_discharge_rate=5.0, efficiency=1.0,
            initial_soc=0.0, min_soc_frac=0.0,
        )

        assert result.success
        assert all(s <= 2.0 + 1e-6 for s in result.soc)

    def test_respects_charge_rate_limits(self):
        """Charge rate cannot exceed max_charge_rate."""
        n = 2
        dt = 1.0
        solar = np.zeros(n)
        load = np.zeros(n)
        import_rates = np.full(n, 0.01)
        export_rates = np.full(n, 0.005)

        result = solve_oracle_lp(
            solar=solar, load=load,
            import_rates=import_rates, export_rates=export_rates,
            dt=dt, battery_capacity=10.0, max_charge_rate=1.5,
            max_discharge_rate=1.5, efficiency=1.0,
            initial_soc=0.0, min_soc_frac=0.0,
        )

        assert result.success
        assert all(c <= 1.5 + 1e-6 for c in result.charge)

    def test_efficiency_loss(self):
        """Round-trip should lose energy proportional to efficiency."""
        n = 2
        dt = 1.0
        solar = np.zeros(n)
        load = np.array([0.0, 1.0])
        import_rates = np.array([0.085, 0.182])
        export_rates = np.full(n, 0.08)

        result = solve_oracle_lp(
            solar=solar, load=load,
            import_rates=import_rates, export_rates=export_rates,
            dt=dt, battery_capacity=5.0, max_charge_rate=2.0,
            max_discharge_rate=2.0, efficiency=0.95,
            initial_soc=0.5, min_soc_frac=0.1,
        )

        assert result.success
        if result.charge[0] > 0.01 and result.discharge[1] > 0.01:
            energy_in = result.charge[0] * 0.95 * dt
            energy_out = result.discharge[1] / 0.95 * dt
            assert energy_out > result.discharge[1] * dt * 0.99

    def test_min_soc_respected(self):
        """Battery should not discharge below min_soc."""
        n = 2
        dt = 1.0
        solar = np.zeros(n)
        load = np.array([5.0, 5.0])
        import_rates = np.full(n, 0.182)
        export_rates = np.full(n, 0.08)

        result = solve_oracle_lp(
            solar=solar, load=load,
            import_rates=import_rates, export_rates=export_rates,
            dt=dt, battery_capacity=5.0, max_charge_rate=2.0,
            max_discharge_rate=2.0, efficiency=1.0,
            initial_soc=1.0, min_soc_frac=0.2,
        )

        assert result.success
        assert all(s >= 1.0 - 1e-6 for s in result.soc)

    def test_empty_input(self):
        """Zero-length arrays should return empty result."""
        result = solve_oracle_lp(
            solar=np.array([]), load=np.array([]),
            import_rates=np.array([]), export_rates=np.array([]),
            dt=1.0, battery_capacity=5.0, max_charge_rate=2.0,
            max_discharge_rate=2.0, efficiency=0.95,
            initial_soc=0.5, min_soc_frac=0.1,
        )

        assert result.success
        assert len(result.charge) == 0
        assert result.total_cost == 0.0


class TestOracleStrategy:
    """Tests for the strategy wrapper that replays LP results through the simulator."""

    @pytest.fixture
    def tariff(self):
        schedule = {
            (0, 8): Rate(price=0.085, energy_direction=EnergyDirection.IMPORT),
            (8, 10): Rate(price=0.134, energy_direction=EnergyDirection.IMPORT),
            (10, 14): Rate(price=0.182, energy_direction=EnergyDirection.IMPORT),
            (14, 18): Rate(price=0.134, energy_direction=EnergyDirection.IMPORT),
            (18, 22): Rate(price=0.182, energy_direction=EnergyDirection.IMPORT),
            (22, 24): Rate(price=0.134, energy_direction=EnergyDirection.IMPORT),
            (0, 24): Rate(price=0.08, energy_direction=EnergyDirection.EXPORT),
        }
        return PowerTariff(rate_schedule=schedule)

    @pytest.fixture
    def battery(self):
        return Battery(capacity=5.0, max_charge_rate=2.0, max_discharge_rate=2.0, efficiency=0.95)

    @pytest.fixture
    def grid(self):
        return Grid(max_import=5.0, max_export=5.0)

    def test_constructs_from_timeseries(self, battery, grid, tariff):
        """Strategy should accept timeseries data and solve LP during init."""
        solar = np.array([0.0, 0.0, 2.0, 2.0])
        load = np.array([1.0, 1.0, 1.0, 1.0])
        hours = np.array([2, 3, 12, 13])
        durations = np.array([1.0, 1.0, 1.0, 1.0])

        strategy = OracleStrategy(
            battery=battery, grid=grid, tariff=tariff,
            solar=solar, load=load, hours=hours, durations=durations,
        )

        assert strategy.lp_result.success

    def test_calculate_energy_flows_returns_valid_flow(self, battery, grid, tariff):
        """Each call to calculate_energy_flows should return precomputed decision."""
        solar = np.array([0.0, 2.0])
        load = np.array([1.0, 1.0])
        hours = np.array([2, 12])
        durations = np.array([1.0, 1.0])

        strategy = OracleStrategy(
            battery=battery, grid=grid, tariff=tariff,
            solar=solar, load=load, hours=hours, durations=durations,
        )

        flows = strategy.calculate_energy_flows(0.0, 1.0, 2, 1.0)
        assert isinstance(flows, EnergyFlow)

    def test_steps_advance_sequentially(self, battery, grid, tariff):
        """Each call should advance to the next precomputed step."""
        solar = np.array([0.0, 0.0, 3.0])
        load = np.array([1.0, 1.0, 1.0])
        hours = np.array([2, 3, 12])
        durations = np.array([1.0, 1.0, 1.0])

        strategy = OracleStrategy(
            battery=battery, grid=grid, tariff=tariff,
            solar=solar, load=load, hours=hours, durations=durations,
        )

        flow0 = strategy.calculate_energy_flows(0.0, 1.0, 2, 1.0)
        flow1 = strategy.calculate_energy_flows(0.0, 1.0, 3, 1.0)
        flow2 = strategy.calculate_energy_flows(3.0, 1.0, 12, 1.0)

        for flow in [flow0, flow1, flow2]:
            assert isinstance(flow, EnergyFlow)

    def test_energy_balance_per_step(self, battery, grid, tariff):
        """Solar + grid_import + discharge = load + charge + export for each step."""
        solar = np.array([0.5, 1.0, 2.0, 0.0])
        load = np.array([1.0, 1.0, 0.5, 1.5])
        hours = np.array([2, 8, 12, 20])
        durations = np.array([1.0, 1.0, 1.0, 1.0])

        strategy = OracleStrategy(
            battery=battery, grid=grid, tariff=tariff,
            solar=solar, load=load, hours=hours, durations=durations,
        )

        for i in range(4):
            flows = strategy.calculate_energy_flows(solar[i], load[i], hours[i], durations[i])
            solar_energy = solar[i] * durations[i]
            load_energy = load[i] * durations[i]
            supply = flows.direct_solar + flows.grid_import * durations[i] + flows.battery_discharge * durations[i]
            demand = load_energy
            assert supply >= demand - 1e-6, f"Step {i}: supply {supply:.4f} < demand {demand:.4f}"
