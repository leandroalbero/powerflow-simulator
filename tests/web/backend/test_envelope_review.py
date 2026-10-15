"""Envelope must keep per-column spikes, break at partial NaN gaps, and cap points."""

import numpy as np
import pytest

from web.backend.services.envelope import envelope_downsample


def test_keeps_spike_when_another_column_has_a_larger_level():
    n = 10_000
    ts = np.arange(n, dtype=np.int64) * 60_000
    grid = np.full(n, 0.3)
    grid[5050] = 6.0
    columns = {
        "battery_level": np.full(n, 10.0),
        "solar_power": np.full(n, 3.0),
        "grid_import": grid,
    }
    env = envelope_downsample(ts, columns, np.ones(n), 100)
    assert 6.0 in env.columns["grid_import"]
    assert len(env.ts_ms) <= 100
    assert np.nansum(env.step_cost) == pytest.approx(n)


def test_partial_nan_in_a_bucket_emits_a_null_for_that_column():
    n = 10_000
    ts = np.arange(n, dtype=np.int64) * 60_000
    grid = np.linspace(0.2, 0.8, n)
    grid[5000:5030] = np.nan
    solar = np.full(n, 2.0)
    env = envelope_downsample(ts, {"grid_import": grid, "solar_power": solar}, np.zeros(n), 100)
    assert np.isnan(env.columns["grid_import"]).any()
    assert len(env.ts_ms) <= 100
    # The other column stays finite at the gap row so only the broken series gaps.
    gap_rows = np.isnan(env.columns["grid_import"])
    assert np.isfinite(env.columns["solar_power"][gap_rows]).all()


def test_week_of_minutes_keeps_half_hour_resolution_at_the_default_cap():
    n = 7 * 1440
    ts = np.arange(n, dtype=np.int64) * 60_000
    minute = np.arange(n)
    hod = (minute % 1440) / 60.0
    solar = np.clip(np.sin((hod - 7) / 12 * np.pi), 0, None) * 5
    load = 0.4 + 0.25 * np.sin(minute / 40.0)
    net = solar - load
    charge = np.clip(net, 0, 4.5)
    discharge = np.clip(-net, 0, 4.6)
    level = np.clip(7 + np.cumsum((charge - discharge) / 60.0), 0, 15.36)
    columns = {
        "battery_level": level,
        "grid_import": np.clip(-net - discharge, 0, None),
        "grid_export": np.clip(net - charge, 0, None),
        "solar_power": solar,
        "house_consumption": load,
        "battery_charge": charge,
        "battery_discharge": discharge,
        "soc_pct": level / 15.36 * 100.0,
        "import_price": np.repeat(np.linspace(0.05, 0.25, n // 60 + 1), 60)[:n],
        "export_price": np.repeat(np.linspace(0.02, 0.08, n // 60 + 1), 60)[:n],
    }
    env = envelope_downsample(ts, columns, np.ones(n), 2000)
    assert env.resolution_minutes <= 30
    assert len(env.ts_ms) <= 2000
    assert np.all(np.diff(env.ts_ms) > 0)
    assert np.nansum(env.step_cost) == pytest.approx(n)


def test_soc_and_prices_follow_power_rows():
    n = 4000
    ts = np.arange(n, dtype=np.int64) * 60_000
    power = {
        "solar_power": np.sin(np.linspace(0, 12, n)),
        "house_consumption": np.cos(np.linspace(0, 8, n)),
    }
    price = np.full(n, 0.1)
    price[1234] = 40.0
    followed = {**power, "soc_pct": np.linspace(0, 100, n), "import_price": price}
    alone = envelope_downsample(ts, power, np.ones(n), 80)
    with_followers = envelope_downsample(ts, followed, np.ones(n), 80)
    assert list(with_followers.ts_ms) == list(alone.ts_ms)
    assert 40.0 not in with_followers.columns["import_price"]


def test_tiny_budget_keeps_priority_extremes_and_bucket_cost():
    n = 3000
    ts = np.arange(n, dtype=np.int64) * 60_000
    solar = np.zeros(n)
    solar[100] = 12.0
    solar[2000] = -3.0
    columns = {
        "solar_power": solar,
        "house_consumption": np.full(n, 0.4),
        "grid_import": np.full(n, 0.2),
        "grid_export": np.full(n, 0.1),
        "battery_charge": np.full(n, 0.0),
        "battery_discharge": np.full(n, 0.0),
        "battery_level": np.full(n, 5.0),
        "soc_pct": np.full(n, 40.0),
        "import_price": np.full(n, 0.1),
    }
    env = envelope_downsample(ts, columns, np.ones(n), 10)
    assert len(env.ts_ms) <= 10
    assert 12.0 in env.columns["solar_power"]
    assert -3.0 in env.columns["solar_power"]
    assert np.nansum(env.step_cost) == pytest.approx(n)
    assert int(np.sum(np.nan_to_num(env.step_cost) > 0)) >= 2

