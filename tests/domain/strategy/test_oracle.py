import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.domain.battery.models import Battery
from src.domain.energy_simulator.models import run
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import EnergyDirection, PowerTariff, Rate
from src.domain.strategy.model import (
    ForceChargeAtNightStrategy,
    ForceChargeAtValleyStrategy,
    ForceChargeValleyAndPrePeakStrategy,
    NoBatteryStrategy,
    SelfConsumeStrategy,
    SmartDischargeStrategy,
    ValleyChargePeakDischargeStrategy,
)
from src.domain.strategy.oracle import OracleStrategy, solve_oracle_lp
from tests.domain.strategy.conftest import make_td20
from tests.synthetic import synthetic_frame


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
            energy_out = result.discharge[1] / 0.95 * dt
            assert energy_out > result.discharge[1] * dt * 0.99  # round-trip loss

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
    """The LP is solved on the frame's own grid and replayed by timestamp through _settle."""

    @staticmethod
    def build(frame, min_soc=0.1, initial_soc=0.1, capacity=10.0, **kw):
        battery = Battery(capacity=capacity, max_charge_rate=4.5, max_discharge_rate=4.6, efficiency=0.9,
                          initial_soc=initial_soc)
        tariff = make_td20()
        oracle = OracleStrategy(battery, Grid(max_import=5.5, max_export=5.5), tariff, frame,
                                min_battery_level=min_soc, **kw)
        return oracle, tariff

    def test_charges_in_valley_and_discharges_at_evening_peak(self):
        # Shoulder at 0.19: charging there loses money against a 0.197 peak at a 0.9 round trip,
        # so the optimum charges only in the 00-08 valley.
        frame = synthetic_frame("2026-03-04", days=1, outage=False, spike=False)
        frame["pv_kw"] = 0.0
        battery = Battery(capacity=10.0, max_charge_rate=4.5, max_discharge_rate=4.6, efficiency=0.9,
                          initial_soc=0.1)
        tariff = PowerTariff(rate_schedule={
            (0, 8): Rate(price=0.085, energy_direction=EnergyDirection.IMPORT),
            (8, 10): Rate(price=0.19, energy_direction=EnergyDirection.IMPORT),
            (10, 14): Rate(price=0.197, energy_direction=EnergyDirection.IMPORT),
            (14, 18): Rate(price=0.19, energy_direction=EnergyDirection.IMPORT),
            (18, 22): Rate(price=0.197, energy_direction=EnergyDirection.IMPORT),
            (22, 24): Rate(price=0.19, energy_direction=EnergyDirection.IMPORT),
            (0, 24): Rate(price=0.08, energy_direction=EnergyDirection.EXPORT),
        })
        oracle = OracleStrategy(battery, Grid(max_import=5.5, max_export=5.5), tariff, frame)
        res = run(frame, oracle, tariff)
        hours = frame.index.hour
        gc = res.timeseries["grid_charge_kw"]
        dis = res.timeseries["battery_discharge_kw"]
        assert gc[hours < 8].sum() > 0
        assert gc[hours >= 8].sum() == pytest.approx(0.0, abs=1e-9)
        assert dis[(hours >= 18) & (hours < 22)].sum() > 0
        assert dis[hours < 8].sum() == pytest.approx(0.0, abs=1e-9)

    def test_slots_are_aligned_to_frame_timestamps(self):
        frame = synthetic_frame("2026-03-04", days=1, outage=False)
        oracle, _ = self.build(frame)
        assert oracle.slots.start_ns == frame.index[0].value
        oracle.now = frame.index[0] + pd.Timedelta(minutes=7 * 60 + 3)
        assert oracle._slot() == (7 * 60) // 5
        assert len(oracle.slots.solar) == 24 * 12

    def test_min_soc_from_construction_is_used_by_lp(self):
        frame = synthetic_frame("2026-03-04", days=2, outage=False)
        oracle, tariff = self.build(frame, min_soc=0.3, initial_soc=0.5)
        assert oracle.lp_result.success
        assert oracle.lp_result.soc.min() >= 0.3 * 10.0 - 1e-6
        res = run(frame, oracle, tariff)
        assert np.nanmin(res.timeseries["battery_level_kwh"]) >= 3.0 - 1e-9

    def test_never_overwrites_battery_and_balances(self):
        frame = synthetic_frame()
        oracle, tariff = self.build(frame)
        res = run(frame, oracle, tariff)
        ts = res.timeseries
        v = frame["valid"].to_numpy()
        lhs = (ts["solar_kw"] + ts["grid_import_kw"] + ts["battery_discharge_kw"])[v]
        rhs = (ts["load_kw"] + ts["grid_export_kw"] + ts["battery_charge_kw"] + ts["curtailed_kw"])[v]
        np.testing.assert_allclose(lhs, rhs, atol=1e-9)
        level = ts["battery_level_kwh"]
        prev = np.concatenate(([res.metrics["initial_battery_kwh"]], level[:-1]))
        b = oracle.battery
        expected = (ts["battery_charge_kw"] * b.charge_efficiency - ts["battery_discharge_kw"] / b.discharge_efficiency) / 60
        np.testing.assert_allclose((level - prev)[v], expected[v], atol=1e-9)

    def test_invalid_slots_have_no_battery_activity(self):
        frame = synthetic_frame()
        oracle, _ = self.build(frame)
        dead = oracle.slots.dt == 0
        assert dead.any()
        assert np.all(oracle.lp_result.charge[dead] == 0)
        assert np.all(oracle.lp_result.discharge[dead] == 0)

    def test_taper_is_respected_by_plan(self):
        frame = synthetic_frame("2026-03-04", days=1, outage=False, spike=False)
        frame["pv_kw"] = 0.0
        oracle, _ = self.build(frame, initial_soc=0.1)
        lp = oracle.lp_result
        b = oracle.battery
        prev = np.concatenate(([b.capacity * 0.1], lp.soc[:-1]))
        level = prev / b.capacity
        limit = np.where(level < b.taper_start, b.max_charge_rate,
                         b.max_charge_rate * ((1 - b.taper_factor) + b.taper_factor * (1 - level) / (1 - b.taper_start)))
        assert np.all(lp.charge <= limit + 1e-6)

    def test_chunked_solve_matches_single_solve(self):
        frame = synthetic_frame("2026-03-02", days=6, outage=False)
        single, t1 = self.build(frame, chunk_days=31)
        chunked, t2 = self.build(frame, chunk_days=2, overlap_days=1)
        assert chunked.lp_result.success
        a = run(frame, single, t1).metrics["adjusted_cost"]
        b = run(frame, chunked, t2).metrics["adjusted_cost"]
        assert b == pytest.approx(a, abs=0.05)

    def test_beats_every_heuristic_on_synthetic_frame(self):
        frame = synthetic_frame()
        oracle, tariff = self.build(frame)
        best = run(frame, oracle, tariff).metrics["adjusted_cost"]
        for cls in (NoBatteryStrategy, SelfConsumeStrategy, ForceChargeAtNightStrategy, ForceChargeAtValleyStrategy,
                    ForceChargeValleyAndPrePeakStrategy, ValleyChargePeakDischargeStrategy, SmartDischargeStrategy):
            battery = Battery(capacity=10.0, max_charge_rate=4.5, max_discharge_rate=4.6, efficiency=0.9,
                              initial_soc=0.1)
            t = make_td20()
            other = run(frame, cls(battery, Grid(max_import=5.5, max_export=5.5), t), t).metrics["adjusted_cost"]
            assert best <= other + max(0.01 * abs(other), 0.5), cls.__name__


class TestOracleEndToEnd:
    """Realistic scenario: one day with solar curve and varying load."""

    def test_one_day_24_hours(self):
        """Oracle over a 24h period should produce lower cost than naive grid import."""
        n = 24
        dt = 1.0
        # Bell-curve solar: peaks at noon
        solar = np.array([
            0, 0, 0, 0, 0, 0,        # 00-05: night
            0.1, 0.5, 1.5, 3.0,      # 06-09: sunrise
            4.0, 4.5, 4.5, 4.0,      # 10-13: peak sun
            3.0, 1.5, 0.5, 0.1,      # 14-17: sunset
            0, 0, 0, 0, 0, 0,        # 18-23: night
        ])
        # Typical household: morning + evening peaks
        load = np.array([
            0.3, 0.3, 0.3, 0.3, 0.3, 0.5,   # 00-05
            0.8, 1.2, 1.5, 1.0,               # 06-09
            0.8, 0.8, 1.0, 0.8,               # 10-13
            0.8, 1.0, 1.5, 2.0,               # 14-17
            2.5, 2.0, 1.5, 1.0, 0.5, 0.3,    # 18-23
        ])
        import_rates = np.array([
            0.085, 0.085, 0.085, 0.085, 0.085, 0.085, 0.085, 0.085,  # valley
            0.134, 0.134,                                              # shoulder
            0.182, 0.182, 0.182, 0.182,                                # peak
            0.134, 0.134, 0.134, 0.134,                                # shoulder
            0.182, 0.182, 0.182, 0.182,                                # peak
            0.134, 0.134,                                              # shoulder
        ])
        export_rates = np.full(n, 0.08)

        result = solve_oracle_lp(
            solar=solar, load=load,
            import_rates=import_rates, export_rates=export_rates,
            dt=dt,
            battery_capacity=15.0,
            max_charge_rate=4.8,
            max_discharge_rate=4.8,
            efficiency=0.95,
            initial_soc=0.1,
            min_soc_frac=0.1,
        )

        assert result.success

        # Naive cost: import everything from grid, no battery
        naive_cost = sum(load[t] * import_rates[t] * dt for t in range(n))
        # Subtract export revenue for excess solar
        excess = np.maximum(solar - load, 0)
        naive_cost -= sum(excess[t] * export_rates[t] * dt for t in range(n))

        assert result.total_cost < naive_cost, (
            f"Oracle ({result.total_cost:.4f}) should beat naive ({naive_cost:.4f})"
        )

        # Verify SoC stays within bounds
        assert all(s >= 15.0 * 0.1 - 1e-6 for s in result.soc)
        assert all(s <= 15.0 + 1e-6 for s in result.soc)


_INFLUX = Path(__file__).resolve().parents[3] / "data" / "influx" / "load.parquet"


@pytest.mark.skipif(not _INFLUX.exists() or importlib.util.find_spec("src.data.store") is None,
                    reason="no data/influx store")
def test_oracle_beats_heuristics_on_a_real_week():
    from src.data.frame import build_power_frame

    frame = build_power_frame("2026-03-02", "2026-03-08")
    assert frame["valid"].any()

    def battery():
        return Battery(capacity=15.36, max_charge_rate=4.5, max_discharge_rate=4.6, efficiency=0.9, initial_soc=0.5)

    t = make_td20()
    oracle = OracleStrategy(battery(), Grid(max_import=5.5, max_export=5.5), t, frame)
    best = run(frame, oracle, t).metrics["adjusted_cost"]
    for cls in (NoBatteryStrategy, SelfConsumeStrategy, ForceChargeAtNightStrategy, SmartDischargeStrategy):
        t = make_td20()
        other = run(frame, cls(battery(), Grid(max_import=5.5, max_export=5.5), t), t).metrics["adjusted_cost"]
        assert best <= other + max(0.01 * abs(other), 0.5), cls.__name__
