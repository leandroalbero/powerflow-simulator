from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from src.domain.battery.models import Battery
from src.domain.energy_simulator.models import run
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import EnergyDirection
from src.domain.strategy.irradiance import HourlyGhi, irradiance_dir, load_irradiance
from src.domain.strategy.mpc import (
    MPC_VARIANTS,
    LoadForecaster,
    MpcStrategy,
    PerfectSolarForecaster,
    SolarForecaster,
    build_mpc,
    learn_monthly_ghi_factors,
)
from src.domain.strategy.oracle import OracleStrategy
from tests.domain.strategy.conftest import MAD, make_td20
from tests.synthetic import synthetic_frame


def _ghi_frame(start="2026-02-01", days=60, missing=()):
    """Hourly GHI on UTC hour starts: a clean bell around 12 UTC."""
    idx = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")
    vals = np.clip(np.sin((idx.hour - 6) / 12 * np.pi), 0, None) * 800.0
    df = pd.DataFrame({"ghi": vals, "ghi_forecast": vals}, index=idx)
    for d in missing:
        df = df[df.index.normalize() != pd.Timestamp(d, tz="UTC")]
    return df


class TestLoadForecaster:
    @pytest.fixture
    def load_df(self):
        """28 days of hourly load: 1 kW at night, 2 kW during the day (legacy W 'state' frame)."""
        idx = pd.date_range("2024-06-01", periods=28 * 24, freq="h", tz="Europe/Madrid")
        values = [1000.0 if ts.hour < 8 or ts.hour >= 22 else 2000.0 for ts in idx]
        return pd.DataFrame({"state": values}, index=idx)

    def test_forecast_reflects_pattern(self, load_df):
        f = LoadForecaster(load_df).forecast_24h(pd.Timestamp("2024-06-29 00:00", tz="Europe/Madrid"), 96, 15)
        assert len(f) == 96
        assert np.mean(f[:32]) == pytest.approx(1.0, abs=0.1)
        assert np.mean(f[40:60]) == pytest.approx(2.0, abs=0.1)

    def test_no_history_gives_zeros(self):
        empty = pd.DataFrame({"state": []}, index=pd.DatetimeIndex([], tz="Europe/Madrid"))
        f = LoadForecaster(empty).forecast_24h(pd.Timestamp("2024-06-29 10:00", tz="Europe/Madrid"), 96, 15)
        assert np.all(f == 0.0)

    def test_uses_only_data_before_the_plan_day(self):
        idx = pd.date_range("2026-03-01", "2026-04-10", freq="1min", tz="Europe/Madrid", inclusive="left")
        load = pd.Series(1.0, index=idx)
        load[idx >= pd.Timestamp("2026-04-01", tz="Europe/Madrid")] = 100.0
        fc = LoadForecaster(load)
        f = fc.forecast_24h(pd.Timestamp("2026-04-01 13:00", tz="Europe/Madrid"), 96, 15)
        assert np.allclose(f, 1.0)
        later = fc.forecast_24h(pd.Timestamp("2026-04-05 13:00", tz="Europe/Madrid"), 96, 15)
        assert later.max() > 1.0

    def test_profile_computed_once_per_day(self):
        idx = pd.date_range("2026-03-01", "2026-03-20", freq="1min", tz="Europe/Madrid", inclusive="left")
        fc = LoadForecaster(pd.Series(1.0, index=idx))
        fc.forecast_24h(pd.Timestamp("2026-03-15 01:00", tz="Europe/Madrid"))
        fc.forecast_24h(pd.Timestamp("2026-03-15 23:00", tz="Europe/Madrid"))
        assert list(fc._cache) == [date(2026, 3, 15)]

    def test_offpeak_days_have_their_own_profile(self):
        idx = pd.date_range("2026-03-02", "2026-03-30", freq="1min", tz="Europe/Madrid", inclusive="left")
        load = pd.Series(np.where(idx.weekday >= 5, 3.0, 1.0), index=idx)
        fc = LoadForecaster(load)
        sat = fc.forecast_24h(pd.Timestamp("2026-04-04 12:00", tz="Europe/Madrid"), 4, 15)
        mon = fc.forecast_24h(pd.Timestamp("2026-03-30 12:00", tz="Europe/Madrid"), 4, 15)
        assert np.allclose(sat, 3.0) and np.allclose(mon, 1.0)


class TestSolarForecasting:
    def test_irradiance_dir_is_repo_relative_and_overridable(self, monkeypatch, tmp_path):
        monkeypatch.delenv("POWERFLOW_DATA_DIR", raising=False)
        assert irradiance_dir().parent.parent == __import__("pathlib").Path(__file__).resolve().parents[3]
        monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
        assert irradiance_dir() == tmp_path / "irradiance"

    def test_missing_day_falls_back_to_climatology(self):
        ghi = HourlyGhi(_ghi_frame(missing=["2026-03-10"]))
        f = SolarForecaster(ghi).forecast_24h(pd.Timestamp("2026-03-10 00:00", tz="Europe/Madrid"), 96, 15)
        assert f.max() > 1.0
        assert f[:16].max() == 0.0

    def test_uses_utc_hour_of_each_step(self):
        ghi = HourlyGhi(_ghi_frame())
        f = SolarForecaster(ghi, ghi_to_pv_factor=1000.0).forecast_24h(
            pd.Timestamp("2026-03-10 13:00", tz="Europe/Madrid"), 4, 15)
        expected = np.sin((12 - 6) / 12 * np.pi) * 800.0
        assert f == pytest.approx([expected] * 4)

    def test_factors_learned_only_before_run_start(self):
        ghi = HourlyGhi(_ghi_frame(start="2026-01-01", days=90))
        idx = pd.date_range("2026-01-01", "2026-03-31", freq="1min", tz="Europe/Madrid", inclusive="left")
        pred = ghi.at(idx.tz_convert("UTC").asi8) * 4.592 / 1000
        pv = pd.Series(pred * 0.5, index=idx)
        pv[idx >= pd.Timestamp("2026-03-01", tz="Europe/Madrid")] *= 4.0
        factors = learn_monthly_ghi_factors(pv, ghi, before=pd.Timestamp("2026-03-01", tz="Europe/Madrid"))
        assert factors[1] == pytest.approx(0.5, rel=0.02)
        assert factors[2] == pytest.approx(0.5, rel=0.02)
        assert factors[3] == 1.0

    def test_perfect_forecaster_averages_actual_minutes(self):
        idx = pd.date_range("2026-03-10", periods=120, freq="1min", tz="Europe/Madrid")
        pv = pd.Series(np.arange(120, dtype=float), index=idx)
        f = PerfectSolarForecaster(pv).forecast_24h(idx[10], 3, 15)
        assert f == pytest.approx([np.mean(range(10, 25)), np.mean(range(25, 40)), np.mean(range(40, 55))])

    def test_real_irradiance_files_load_if_present(self):
        irr = load_irradiance()
        if irr.empty:
            pytest.skip("no irradiance files")
        assert str(irr.index.tz) == "UTC"
        assert irr.index.is_unique and irr.index.is_monotonic_increasing
        summer = irr[irr.index.month == 7]["ghi"]
        peak_hour = summer.groupby(summer.index.hour).mean().idxmax()
        assert peak_hour in (11, 12)  # solar noon at the site is ~12:04 UTC


_HISTORY = synthetic_frame("2026-02-01", days=40, outage=False, spike=False)


def _mpc(frame, **kw):
    battery = Battery(capacity=10.0, max_charge_rate=4.5, max_discharge_rate=4.6, efficiency=0.9, initial_soc=0.2)
    tariff = make_td20()
    strategy = MpcStrategy(battery, Grid(max_import=5.5, max_export=5.5), tariff,
                           load_forecaster=LoadForecaster(_HISTORY["load_kw"]),
                           solar_forecaster=PerfectSolarForecaster(frame["pv_kw"]), **kw)
    return strategy, tariff


class TestMpcStrategy:
    def test_resolves_every_resolve_interval(self):
        frame = synthetic_frame(days=1, outage=False)
        s, _ = _mpc(frame)
        t0 = pd.Timestamp("2026-03-04 10:00", tz="Europe/Madrid")
        for i in range(15):
            s.now = t0 + pd.Timedelta(minutes=i)
            s.calculate_energy_flows(3.0, 1.5, 10, 1 / 60)
        assert s._solve_count == 1
        s.now = t0 + pd.Timedelta(minutes=15)
        s.calculate_energy_flows(3.0, 1.5, 10, 1 / 60)
        assert s._solve_count == 2

    def test_planning_does_not_touch_tariff_or_current_step_price(self):
        frame = synthetic_frame("2026-03-06", days=2, outage=False)  # Friday -> Saturday
        s, tariff = _mpc(frame)
        friday_23 = datetime(2026, 3, 6, 23, 30, tzinfo=MAD)
        before = tariff.rate_at(friday_23, EnergyDirection.IMPORT).price
        tariff.update_datetime(friday_23)
        legacy_before = tariff.get_import_rate(23)
        s.now = friday_23
        s.calculate_energy_flows(0.0, 1.0, 23, 1 / 60)
        assert s._solve_count == 1
        assert tariff.rate_at(friday_23, EnergyDirection.IMPORT).price == before == 0.135
        assert tariff._current_datetime == friday_23
        assert tariff.get_import_rate(23) == legacy_before
        res = run(frame, s, tariff)
        idx = frame.index
        expected = np.array([tariff.rate_at(t, EnergyDirection.IMPORT).price for t in idx[::37]])
        assert np.array_equal(res.timeseries["import_price"][::37], expected)

    def test_grid_charges_in_valley_ahead_of_peak(self):
        frame = synthetic_frame(days=1, outage=False)
        frame["pv_kw"] = 0.0
        s, tariff = _mpc(frame)
        res = run(frame, s, tariff)
        hours = frame.index.hour
        assert res.timeseries["grid_charge_kw"][hours < 8].sum() > 0
        assert res.timeseries["battery_discharge_kw"][(hours >= 18) & (hours < 22)].sum() > 0

    def test_grid_charge_power_limits_plan_and_replay(self):
        frame = synthetic_frame(days=1, outage=False)
        frame["pv_kw"] = 0.0
        s, tariff = _mpc(frame, max_charge_power=2.0)
        res = run(frame, s, tariff)
        gc = res.timeseries["grid_charge_kw"]
        assert gc.max() <= 2.0 + 1e-9
        assert gc.max() > 1.9
        assert np.all(s._cached_plan["charge"] <= 2.0 + 1e-6)

    def test_mpc_cannot_beat_oracle(self):
        frame = synthetic_frame()
        s, tariff = _mpc(frame)
        mpc_cost = run(frame, s, tariff).metrics["adjusted_cost"]
        battery = Battery(capacity=10.0, max_charge_rate=4.5, max_discharge_rate=4.6, efficiency=0.9, initial_soc=0.2)
        t2 = make_td20()
        oracle = OracleStrategy(battery, Grid(max_import=5.5, max_export=5.5), t2, frame)
        best = run(frame, oracle, t2).metrics["adjusted_cost"]
        assert best <= mpc_cost + max(0.01 * abs(mpc_cost), 0.5)

    @pytest.mark.parametrize("sid", list(MPC_VARIANTS))
    def test_factory_builds_every_variant(self, sid):
        frame = synthetic_frame(days=1, outage=False)
        battery = Battery(capacity=10.0, max_charge_rate=4.5, max_discharge_rate=4.6)
        ghi = HourlyGhi(_ghi_frame())
        s = build_mpc(sid, battery, Grid(5.5, 5.5), make_td20(), frame, ghi=ghi, min_battery_level=0.2)
        assert s.min_battery_level == 0.2
        assert s._step_minutes == MPC_VARIANTS[sid].step_minutes
        s.now = frame.index[600]
        f = s.calculate_energy_flows(2.0, 1.0, 10, 1 / 60)
        assert f.unserved == 0.0
