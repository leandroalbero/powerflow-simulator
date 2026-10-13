"""Timeseries CSV must not materialise every row before the first byte."""

import tracemalloc

import numpy as np

from web.backend.services.csv_export import timeseries_csv


def test_timeseries_csv_streams_without_building_every_row():
    n = 20_000
    ts = {
        "ts": np.arange(n, dtype=np.int64) * 60_000,
        "battery_level_kwh": np.full(n, 5.0),
        "grid_import_kw": np.full(n, 0.2),
        "grid_export_kw": np.zeros(n),
        "solar_kw": np.full(n, 1.0),
        "load_kw": np.full(n, 0.4),
        "battery_charge_kw": np.zeros(n),
        "battery_discharge_kw": np.zeros(n),
        "import_price": np.full(n, 0.1),
        "export_price": np.full(n, 0.05),
        "step_cost": np.full(n, 0.001),
        "grid_charge_kw": np.zeros(n),
        "curtailed_kw": np.zeros(n),
    }
    tracemalloc.start()
    chunks = timeseries_csv(ts, 10.0)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # A boxed row list for 20k x 14 cells is well above this. Numpy copies are not.
    assert peak < 6_000_000
    first = next(chunks)
    assert first.startswith("ts,")
    body = "".join(chunks)
    assert body.count("\n") == n
    assert "5.0" in body
