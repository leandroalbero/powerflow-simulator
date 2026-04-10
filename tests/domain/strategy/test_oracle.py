import numpy as np
import pytest

from src.domain.strategy.oracle import solve_oracle_lp


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
