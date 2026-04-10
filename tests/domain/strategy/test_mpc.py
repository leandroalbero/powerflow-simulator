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
