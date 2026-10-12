from datetime import date, timedelta
from unittest.mock import Mock

import numpy as np
import pytest

from src.domain.power_tariff.model import EnergyDirection, PowerTariff, Rate
from src.domain.strategy.model import (
    EnergyFlow,
    ForceChargeAtNightStrategy,
    ForceChargeAtValleyStrategy,
    ForceChargeValleyAndPrePeakStrategy,
    ForecastChargeStrategy,
    NoBatteryStrategy,
    SelfConsumeStrategy,
    SmartDischargeStrategy,
    ValleyChargePeakDischargeStrategy,
)
from tests.domain.strategy.conftest import SATURDAY, WEEKDAY, at

DT = 1 / 60
HEURISTICS = [SelfConsumeStrategy, ForceChargeAtNightStrategy, ForceChargeAtValleyStrategy,
              ForceChargeValleyAndPrePeakStrategy, ValleyChargePeakDischargeStrategy, SmartDischargeStrategy,
              NoBatteryStrategy]


def step(strategy, ts, solar, load, dt=DT) -> EnergyFlow:
    strategy.now = ts
    return strategy.calculate_energy_flows(solar, load, ts.hour, dt)


def assert_balanced(f: EnergyFlow, solar, load, dt, level_before, battery):
    lhs = solar * dt + f.grid_import + f.battery_discharge
    rhs = (load * dt - f.unserved) + f.grid_export + f.battery_charge + f.curtailed
    assert lhs == pytest.approx(rhs, abs=1e-9)
    delta = battery.current_charge - level_before
    expected = f.battery_charge * battery.charge_efficiency - f.battery_discharge / battery.discharge_efficiency
    assert delta == pytest.approx(expected, abs=1e-9)


class TestSettle:
    def test_discharge_limited_to_load_deficit(self, battery, grid, td20):
        s = SelfConsumeStrategy(battery, grid, td20)
        f = s._settle(0.0, 0.5, -2.0, 1.0, allow_grid_charge=False)
        assert f.battery_discharge == pytest.approx(0.5)
        assert f.grid_export == 0.0

    def test_export_discharge_only_when_allowed(self, battery, grid, td20):
        s = SelfConsumeStrategy(battery, grid, td20)
        f = s._settle(0.0, 0.5, -2.0, 1.0, allow_grid_charge=False, allow_export_discharge=True)
        assert f.battery_discharge == pytest.approx(2.0)
        assert f.grid_export == pytest.approx(1.5)

    def test_grid_part_of_charge_capped_by_grid_charge_power(self, battery, grid, td20):
        s = SelfConsumeStrategy(battery, grid, td20, max_charge_power=0.5)
        f = s._settle(1.5, 1.0, 2.0, 1.0, allow_grid_charge=True)
        assert f.battery_charge_from_grid == pytest.approx(0.5)
        assert f.battery_charge == pytest.approx(1.0)

    def test_min_soc_floor_never_violated(self, battery, grid, td20):
        s = SelfConsumeStrategy(battery, grid, td20, min_battery_level=0.3)
        for _ in range(500):
            s._settle(0.0, 3.0, -3.0, DT, allow_grid_charge=False)
            assert battery.current_charge >= 0.3 * battery.capacity - 1e-12
        assert battery.current_charge == pytest.approx(3.0)

    def test_floor_accounts_for_discharge_efficiency(self, battery, grid, td20):
        battery.current_charge = 1.5
        s = SelfConsumeStrategy(battery, grid, td20, min_battery_level=0.1)
        f = s._settle(0.0, 5.0, -2.0, 1.0, allow_grid_charge=False)
        assert f.battery_discharge == pytest.approx(0.5 * 0.9)
        assert battery.current_charge == pytest.approx(1.0)

    def test_solar_charges_first_then_grid(self, battery, grid, td20):
        s = SelfConsumeStrategy(battery, grid, td20)
        f = s._settle(1.5, 1.0, 2.0, 1.0, allow_grid_charge=True)
        assert f.battery_charge == pytest.approx(2.0)
        assert f.battery_charge_from_grid == pytest.approx(1.5)
        assert f.grid_import == pytest.approx(1.5)
        assert f.grid_export == 0.0

    def test_no_grid_charge_unless_allowed(self, battery, grid, td20):
        s = SelfConsumeStrategy(battery, grid, td20)
        f = s._settle(1.5, 1.0, 2.0, 1.0, allow_grid_charge=False)
        assert f.battery_charge == pytest.approx(0.5)
        assert f.grid_import == 0.0

    def test_grid_charge_reduced_to_respect_import_cap(self, battery, grid, td20):
        s = SelfConsumeStrategy(battery, grid, td20)
        f = s._settle(0.0, 4.5, 2.0, 1.0, allow_grid_charge=True)
        assert f.battery_charge_from_grid == pytest.approx(1.0)
        assert f.grid_import == pytest.approx(5.5)
        assert f.over_limit == 0.0

    def test_load_above_cap_is_served_and_reported(self, battery, grid, td20):
        s = SelfConsumeStrategy(battery, grid, td20)
        f = s._settle(0.0, 7.0, 2.0, 1.0, allow_grid_charge=True)
        assert f.battery_charge == 0.0
        assert f.grid_import == pytest.approx(7.0)
        assert f.unserved == 0.0
        assert f.over_limit == pytest.approx(1.5)

    def test_export_capped_and_excess_curtailed(self, battery, grid, td20):
        battery.current_charge = battery.capacity
        s = SelfConsumeStrategy(battery, grid, td20)
        f = s._settle(8.0, 0.5, 7.5, 1.0, allow_grid_charge=False)
        assert f.grid_export == pytest.approx(5.5)
        assert f.curtailed == pytest.approx(2.0)

    def test_balance_invariant_random(self, battery, grid, td20):
        rng = np.random.default_rng(0)
        s = SelfConsumeStrategy(battery, grid, td20)
        for _ in range(2000):
            solar, load = rng.uniform(0, 8), rng.uniform(0, 9)
            cmd = rng.uniform(-5, 5)
            before = battery.current_charge
            f = s._settle(solar, load, cmd, DT, bool(rng.integers(2)), bool(rng.integers(2)))
            assert_balanced(f, solar, load, DT, before, battery)
            assert f.grid_import - f.over_limit <= grid.max_import * DT + 1e-12
            assert f.grid_export <= grid.max_export * DT + 1e-12
            assert battery.current_charge >= 0.1 * battery.capacity - 1e-12

    def test_zero_duration_raises(self, battery, grid, td20):
        with pytest.raises(ZeroDivisionError):
            SelfConsumeStrategy(battery, grid, td20).calculate_energy_flows(1.0, 1.0, 0, 0.0)

    def test_negative_powers_are_clamped(self, battery, grid, td20):
        f = step(SelfConsumeStrategy(battery, grid, td20), at(WEEKDAY, 3), -1.0, -1.0)
        assert f == EnergyFlow()


class TestStrategies:
    def test_no_battery_never_moves(self, battery, grid, td20):
        s = NoBatteryStrategy(battery, grid, td20)
        before = battery.current_charge
        f1 = step(s, at(WEEKDAY, 12), 4.0, 1.0)
        f2 = step(s, at(WEEKDAY, 20), 0.0, 3.0)
        assert battery.current_charge == before
        assert f1.grid_export == pytest.approx(3.0 * DT)
        assert f2.grid_import == pytest.approx(3.0 * DT)

    def test_charge_night_grid_charges_in_weekday_valley_only(self, battery, grid, td20):
        s = ForceChargeAtNightStrategy(battery, grid, td20)
        f = step(s, at(WEEKDAY, 3), 0.0, 0.5)
        assert f.battery_charge_from_grid == pytest.approx(2.0 * DT)
        assert f.battery_discharge == 0.0
        f = step(s, at(WEEKDAY, 19), 0.0, 1.0)
        assert f.battery_discharge == pytest.approx(1.0 * DT)

    def test_single_rate_day_behaves_like_self_consume(self, battery, grid, td20):
        for cls in HEURISTICS[1:-1]:
            s = cls(battery, grid, td20)
            for hour in (3, 9, 12, 15, 19, 23):
                f = step(s, at(SATURDAY, hour), 0.0, 1.0)
                assert f.battery_charge_from_grid == 0.0, (cls.__name__, hour)
                assert f.battery_discharge == pytest.approx(1.0 * DT), (cls.__name__, hour)
            f = step(s, at(SATURDAY, 13), 3.0, 1.0)
            assert f.battery_charge == pytest.approx(2.0 * DT), cls.__name__

    def test_sunday_night_does_not_grid_charge_but_monday_does(self, battery, grid, td20):
        s = ForceChargeAtNightStrategy(battery, grid, td20)
        assert step(s, at("2026-03-08", 23), 0.0, 0.5).battery_charge_from_grid == 0.0
        assert step(s, at("2026-03-09", 1), 0.0, 0.5).battery_charge_from_grid > 0.0

    def test_valley_charge_stops_at_target(self, battery, grid, td20):
        s = ForceChargeAtNightStrategy(battery, grid, td20, valley_charge_target=0.6)
        for m in range(240):
            step(s, at(WEEKDAY, 2 + m // 60, m % 60), 0.0, 0.2)
        assert battery.current_charge == pytest.approx(6.0, abs=1e-6)

    def test_grid_charge_power_defaults_to_battery_rate_and_is_clipped(self, battery, grid, td20):
        assert ForceChargeAtNightStrategy(battery, grid, td20).grid_charge_power == 2.0
        assert ForceChargeAtNightStrategy(battery, grid, td20, max_charge_power=4.5).grid_charge_power == 2.0
        assert ForceChargeAtNightStrategy(battery, grid, td20, max_charge_power=1.0).grid_charge_power == 1.0

    def test_force_valleys_windows(self, battery, grid, td20):
        s = ForceChargeAtValleyStrategy(battery, grid, td20)
        assert step(s, at(WEEKDAY, 9, 30), 0.0, 0.5).battery_charge_from_grid > 0
        assert step(s, at(WEEKDAY, 8, 30), 0.0, 0.5).battery_charge_from_grid == 0.0
        held = step(s, at(WEEKDAY, 15), 0.0, 1.0)
        assert held.battery_discharge == 0.0 and held.battery_charge_from_grid == 0.0
        assert step(s, at(WEEKDAY, 17, 10), 0.0, 0.5).battery_charge_from_grid > 0
        assert step(s, at(WEEKDAY, 12), 0.0, 1.0).battery_discharge == pytest.approx(DT)

    def test_force_valleys_pre_peak_windows(self, battery, grid, td20):
        s = ForceChargeValleyAndPrePeakStrategy(battery, grid, td20)
        assert step(s, at(WEEKDAY, 3), 0.0, 0.5).battery_charge_from_grid > 0
        after_valley = step(s, at(WEEKDAY, 8, 30), 0.0, 1.0)
        assert after_valley.battery_charge_from_grid == 0.0 and after_valley.battery_discharge == 0.0
        assert step(s, at(WEEKDAY, 15), 0.0, 0.5).battery_charge_from_grid > 0
        assert step(s, at(WEEKDAY, 19), 0.0, 1.0).battery_discharge == pytest.approx(DT)
        assert step(s, at(WEEKDAY, 23), 0.0, 1.0).battery_discharge == 0.0

    @pytest.mark.parametrize("day", ["2026-10-05", "2026-10-13"], ids=["monday", "after_holiday"])
    def test_force_valleys_pre_peak_charges_in_valley_after_offpeak_day(self, battery, grid, td20, day):
        s = ForceChargeValleyAndPrePeakStrategy(battery, grid, td20)
        assert step(s, at(day, 3), 0.0, 0.5).battery_charge_from_grid > 0
        after_valley = step(s, at(day, 8, 30), 0.0, 1.0)
        assert after_valley.battery_charge_from_grid == 0.0 and after_valley.battery_discharge == 0.0

    def test_valley_charge_peak_discharge(self, battery, grid, td20):
        s = ValleyChargePeakDischargeStrategy(battery, grid, td20)
        assert step(s, at(WEEKDAY, 3), 0.0, 0.5).battery_charge_from_grid > 0
        assert step(s, at(WEEKDAY, 9), 0.0, 1.0).battery_discharge == 0.0
        assert step(s, at(WEEKDAY, 19), 0.0, 1.0).battery_discharge == pytest.approx(DT)

    def test_smart_discharge_holds_between_peaks(self, battery, grid, td20):
        s = SmartDischargeStrategy(battery, grid, td20)
        assert step(s, at(WEEKDAY, 15), 0.0, 1.0).battery_discharge == 0.0
        assert step(s, at(WEEKDAY, 9), 0.0, 1.0).battery_discharge == pytest.approx(DT)
        assert step(s, at(WEEKDAY, 23), 0.0, 1.0).battery_discharge == pytest.approx(DT)

    def test_forecast_charge_target_follows_forecast(self, battery, grid, td20):
        battery.current_charge = 1.0
        sunny = {date(2026, 3, 4): 5000.0}
        s = ForecastChargeStrategy(battery, grid, td20, daily_forecasts=sunny)
        s.now = at(WEEKDAY, 3)
        expected = max(0.1, 1.0 - 5000.0 * 4.592 / 1000 / battery.capacity)
        assert s._get_charge_target() == pytest.approx(expected)
        s.now = at("2026-03-05", 3)
        assert s._get_charge_target() == 1.0

    def test_rate_aware_requires_three_rates(self, battery, grid):
        tariff = Mock(spec=PowerTariff)
        tariff.rate_schedule = {0: Rate(price=0.1, energy_direction=EnergyDirection.IMPORT)}
        with pytest.raises(ValueError, match="Tariff must have at least 3 different rates"):
            ForceChargeAtNightStrategy(battery, grid, tariff)

    def test_blocks_follow_the_current_pvpc_day(self, battery, grid):
        from datetime import datetime, timedelta, timezone
        from zoneinfo import ZoneInfo

        from src.domain.power_tariff.pvpc import PvpcTariff

        mad = ZoneInfo("Europe/Madrid")
        prices = {}
        for d, f in ((date(2026, 3, 4), lambda h: 0.05 + 0.01 * h), (date(2026, 3, 5), lambda h: 0.30 - 0.01 * h)):
            start = datetime(d.year, d.month, d.day, tzinfo=mad).astimezone(timezone.utc)
            prices.update({start + timedelta(hours=h): f(h) for h in range(24)})
        s = ForceChargeAtNightStrategy(battery, grid, PvpcTariff(prices, export_price=0.05))
        assert step(s, at("2026-03-04", 2), 0.0, 0.5).battery_charge_from_grid > 0
        peak_now = step(s, at("2026-03-05", 2), 0.0, 1.0)
        assert peak_now.battery_charge_from_grid == 0.0 and peak_now.battery_discharge > 0
        assert step(s, at("2026-03-05", 18), 0.0, 0.5).battery_charge_from_grid > 0

    @pytest.mark.parametrize("cls", HEURISTICS + [ForecastChargeStrategy])
    def test_energy_balance_every_step(self, cls, battery, grid, td20):
        kwargs = {"daily_forecasts": {}} if cls is ForecastChargeStrategy else {}
        s = cls(battery, grid, td20, **kwargs)
        rng = np.random.default_rng(1)
        for i in range(3 * 1440):
            ts = at("2026-03-06", 0) + timedelta(minutes=i)
            solar = max(0.0, 5 * np.sin((ts.hour - 7) / 12 * np.pi)) * rng.uniform(0.3, 1)
            load = rng.uniform(0.2, 3.0) + (5.0 if i % 997 == 0 else 0.0)
            before = battery.current_charge
            f = step(s, ts, solar, load)
            assert_balanced(f, solar, load, DT, before, battery)
            assert battery.current_charge >= s.min_battery_level * battery.capacity - 1e-9
            assert f.unserved == 0.0
