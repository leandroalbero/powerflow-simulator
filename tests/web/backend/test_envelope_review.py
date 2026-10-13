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
