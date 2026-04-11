import numpy as np
import pandas as pd
import pytest
import pytz

from src.domain.strategy.mpc import LoadForecaster

LOCAL_TZ = pytz.timezone("Europe/Madrid")


class TestLoadForecaster:
    @pytest.fixture
    def load_df(self):
        """28 days of hourly load data with a simple pattern: 1kW at night, 2kW during day."""
        idx = pd.date_range("2024-06-01", periods=28 * 24, freq="h", tz=LOCAL_TZ)
        values = [1000.0 if ts.hour < 8 or ts.hour >= 22 else 2000.0 for ts in idx]
        return pd.DataFrame({"state": values}, index=idx)

    def test_forecast_length(self, load_df):
        forecaster = LoadForecaster(load_df)
        ts = pd.Timestamp("2024-06-29 10:00", tz=LOCAL_TZ)
        forecast = forecaster.forecast_24h(ts, steps=96, step_minutes=15)
        assert len(forecast) == 96

    def test_forecast_reflects_pattern(self, load_df):
        forecaster = LoadForecaster(load_df)
        ts = pd.Timestamp("2024-06-29 00:00", tz=LOCAL_TZ)
        forecast = forecaster.forecast_24h(ts, steps=96, step_minutes=15)
        assert np.mean(forecast[:32]) == pytest.approx(1.0, abs=0.1)
        assert np.mean(forecast[40:60]) == pytest.approx(2.0, abs=0.1)

    def test_forecast_with_no_history(self):
        empty_df = pd.DataFrame({"state": []}, index=pd.DatetimeIndex([], tz=LOCAL_TZ))
        forecaster = LoadForecaster(empty_df)
        ts = pd.Timestamp("2024-06-29 10:00", tz=LOCAL_TZ)
        forecast = forecaster.forecast_24h(ts, steps=96, step_minutes=15)
        assert len(forecast) == 96
        assert np.all(forecast == 0.0)


from datetime import date
from src.domain.strategy.mpc import SolarForecaster, GHI_TO_PV_FACTOR


class TestSolarForecaster:
    @pytest.fixture
    def hourly_forecasts(self):
        forecasts = {}
        for d in [date(2024, 6, 29), date(2024, 6, 30)]:
            hourly = {}
            for h in range(24):
                if 6 <= h <= 18:
                    hourly[h] = max(0, 800 - abs(h - 12) * 120)
                else:
                    hourly[h] = 0.0
            forecasts[d] = hourly
        return forecasts

    def test_forecast_length(self, hourly_forecasts):
        forecaster = SolarForecaster(hourly_forecasts)
        ts = pd.Timestamp("2024-06-29 00:00", tz=LOCAL_TZ)
        forecast = forecaster.forecast_24h(ts, steps=96, step_minutes=15)
        assert len(forecast) == 96

    def test_forecast_night_is_zero(self, hourly_forecasts):
        forecaster = SolarForecaster(hourly_forecasts)
        ts = pd.Timestamp("2024-06-29 00:00", tz=LOCAL_TZ)
        forecast = forecaster.forecast_24h(ts, steps=96, step_minutes=15)
        assert np.all(forecast[:24] == 0.0)

    def test_forecast_peak_positive(self, hourly_forecasts):
        forecaster = SolarForecaster(hourly_forecasts)
        ts = pd.Timestamp("2024-06-29 00:00", tz=LOCAL_TZ)
        forecast = forecaster.forecast_24h(ts, steps=96, step_minutes=15)
        assert forecast[48] > 0

    def test_forecast_missing_date_returns_zeros(self, hourly_forecasts):
        forecaster = SolarForecaster(hourly_forecasts)
        ts = pd.Timestamp("2024-07-15 00:00", tz=LOCAL_TZ)
        forecast = forecaster.forecast_24h(ts, steps=96, step_minutes=15)
        assert np.all(forecast == 0.0)


from pathlib import Path
from src.domain.strategy.mpc import load_hourly_ghi_forecasts


class TestLoadHourlyGhiForecasts:
    def test_loads_from_irradiance_dir(self):
        forecasts = load_hourly_ghi_forecasts()
        assert len(forecasts) > 0
        some_date = next(iter(forecasts))
        assert isinstance(forecasts[some_date], dict)
        assert all(0 <= h <= 23 for h in forecasts[some_date])

    def test_summer_noon_has_positive_ghi(self):
        forecasts = load_hourly_ghi_forecasts()
        d = date(2024, 6, 15)
        if d in forecasts:
            assert forecasts[d].get(12, 0.0) > 0


from datetime import timedelta
from src.domain.battery.models import Battery
from src.domain.grid.model import Grid
from src.domain.power_tariff.model import PowerTariff, Rate, EnergyDirection
from src.domain.strategy.model import EnergyFlow
from src.domain.strategy.mpc import MpcStrategy, SolarForecaster


class TestMpcStrategy:
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
        return Battery(capacity=15.0, max_charge_rate=4.8, max_discharge_rate=4.8, efficiency=0.95)

    @pytest.fixture
    def grid(self):
        return Grid(max_import=5.0, max_export=5.0)

    @pytest.fixture
    def load_forecaster(self):
        idx = pd.date_range("2024-06-01", periods=28 * 24, freq="h", tz=LOCAL_TZ)
        df = pd.DataFrame({"state": [1500.0] * len(idx)}, index=idx)
        return LoadForecaster(df)

    @pytest.fixture
    def solar_forecaster(self):
        from datetime import date as date_cls
        forecasts = {}
        for day_offset in range(60):
            d = date_cls(2024, 6, 1) + timedelta(days=day_offset)
            hourly = {}
            for h in range(24):
                if 8 <= h <= 16:
                    ghi = max(0, 600 - abs(h - 12) * 100)
                else:
                    ghi = 0.0
                hourly[h] = ghi
            forecasts[d] = hourly
        return SolarForecaster(forecasts)

    def test_returns_valid_flow(self, battery, grid, tariff, load_forecaster, solar_forecaster):
        strategy = MpcStrategy(
            battery=battery, grid=grid, tariff=tariff,
            load_forecaster=load_forecaster,
            solar_forecaster=solar_forecaster,
        )
        ts = pd.Timestamp("2024-06-15 10:00", tz=LOCAL_TZ)
        strategy.set_timestamp(ts)
        flows = strategy.calculate_energy_flows(3.0, 1.5, 10, 1.0 / 60.0)
        assert isinstance(flows, EnergyFlow)

    def test_resolves_at_15min_intervals(self, battery, grid, tariff, load_forecaster, solar_forecaster):
        strategy = MpcStrategy(
            battery=battery, grid=grid, tariff=tariff,
            load_forecaster=load_forecaster,
            solar_forecaster=solar_forecaster,
        )
        ts1 = pd.Timestamp("2024-06-15 10:00", tz=LOCAL_TZ)
        strategy.set_timestamp(ts1)
        strategy.calculate_energy_flows(3.0, 1.5, 10, 1.0 / 60.0)
        solves_after_first = strategy._solve_count

        for i in range(1, 14):
            ts = ts1 + timedelta(minutes=i)
            strategy.set_timestamp(ts)
            strategy.calculate_energy_flows(3.0, 1.5, 10, 1.0 / 60.0)
        assert strategy._solve_count == solves_after_first

        ts2 = ts1 + timedelta(minutes=15)
        strategy.set_timestamp(ts2)
        strategy.calculate_energy_flows(3.0, 1.5, 10, 1.0 / 60.0)
        assert strategy._solve_count == solves_after_first + 1

    def test_charges_during_valley(self, battery, grid, tariff, load_forecaster, solar_forecaster):
        strategy = MpcStrategy(
            battery=battery, grid=grid, tariff=tariff,
            load_forecaster=load_forecaster,
            solar_forecaster=solar_forecaster,
        )
        ts = pd.Timestamp("2024-06-15 03:00", tz=LOCAL_TZ)
        strategy.set_timestamp(ts)
        flows = strategy.calculate_energy_flows(0.0, 1.5, 3, 1.0 / 60.0)
        assert flows.battery_charge > 0 or flows.grid_import > 0
