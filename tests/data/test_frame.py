import importlib.util

import numpy as np
import pandas as pd
import pytest

from src.data.frame import build_power_frame, csv_to_entity_frame, frame_coverage, memory_reader
from tests.data.conftest import parquet_reader, write_entity


def _minutes(start_utc: str, n: int) -> list:
    return list(pd.date_range(start_utc, periods=n, freq="1min", tz="UTC"))


@pytest.fixture
def two_days(data_dir):
    # 2026-03-02 and 2026-03-03 local (UTC+1): load every minute, except a 3 h outage
    # and a single 30-min silence; PV only 08:00-17:00 local.
    load_t = _minutes("2026-03-01T23:00:00Z", 2 * 1440)
    load_v = [1000.0] * len(load_t)
    keep = [i for i, t in enumerate(load_t)
            if not (pd.Timestamp("2026-03-02T09:00Z") <= t < pd.Timestamp("2026-03-02T12:00Z"))
            and not (pd.Timestamp("2026-03-03T02:01Z") <= t < pd.Timestamp("2026-03-03T02:31Z"))]
    write_entity(data_dir, "load", [load_t[i] for i in keep], [load_v[i] for i in keep])
    pv_t = [t for t in load_t if 7 <= t.hour < 16]
    write_entity(data_dir, "pv", pv_t, [2000.0] * len(pv_t))
    ev_t = _minutes("2026-03-02T20:00:00Z", 60)
    write_entity(data_dir, "ev", ev_t, [700.0] * len(ev_t))
    return data_dir


def test_inclusive_end_whole_local_days(two_days):
    f = build_power_frame("2026-03-02", "2026-03-03", reader=parquet_reader(two_days))
    assert len(f) == 2 * 1440
    assert str(f.index.tz) == "Europe/Madrid"
    assert f.index[0] == pd.Timestamp("2026-03-02 00:00", tz="Europe/Madrid")
    assert f.index[-1] == pd.Timestamp("2026-03-03 23:59", tz="Europe/Madrid")
    one = build_power_frame("2026-03-02", "2026-03-02", reader=parquet_reader(two_days))
    assert len(one) == 1440


def test_ffill_up_to_15_minutes_then_invalid(two_days):
    f = build_power_frame("2026-03-02", "2026-03-03", reader=parquet_reader(two_days))
    utc = f.index.tz_convert("UTC")
    silence = (utc >= pd.Timestamp("2026-03-03T02:01Z")) & (utc < pd.Timestamp("2026-03-03T02:31Z"))
    filled = f.loc[silence, "valid"].to_numpy()
    assert filled[:15].all()
    assert not filled[15:].any()
    assert np.isnan(f.loc[silence, "load_kw"].to_numpy()[15:]).all()
    assert f.loc[silence, "load_kw"].to_numpy()[:15] == pytest.approx(1.0)


def test_pv_night_is_zero_and_outage_invalid(two_days):
    f = build_power_frame("2026-03-02", "2026-03-03", reader=parquet_reader(two_days))
    assert f["pv_kw"].notna().all()
    night = f.between_time("20:00", "05:00")
    assert (night["pv_kw"] == 0).all()
    noon = f.loc["2026-03-02 09:00":"2026-03-02 09:59"]
    assert (noon["pv_kw"] == 2.0).all()
    utc = f.index.tz_convert("UTC")
    outage = (utc >= pd.Timestamp("2026-03-02T09:16Z")) & (utc < pd.Timestamp("2026-03-02T12:00Z"))
    assert not f.loc[outage, "valid"].any()
    cov = frame_coverage(f)
    assert len(cov["gaps"]) == 1
    assert cov["gaps"][0]["hours"] == pytest.approx((180 - 15) / 60, abs=0.01)
    assert 0.9 < cov["valid_fraction"] < 1.0


def test_exclude_ev_and_measured_columns(two_days):
    r = parquet_reader(two_days)
    base = build_power_frame("2026-03-02", "2026-03-02", reader=r)
    noev = build_power_frame("2026-03-02", "2026-03-02", exclude_ev=True, reader=r)
    window = slice("2026-03-02 21:00", "2026-03-02 21:59")
    assert (base.loc[window, "ev_kw"] == 0.7).all()
    assert noev.loc[window, "load_kw"].to_numpy() == pytest.approx(0.3)
    assert (noev["load_kw"].dropna() >= 0).all()
    m = build_power_frame("2026-03-02", "2026-03-02", include_measured=True, reader=r)
    assert {"grid_kw", "battery_kw", "soc_pct"} <= set(m.columns)


def test_soc_ffill_unlimited_within_valid(data_dir):
    t = _minutes("2026-03-01T23:00:00Z", 1440)
    write_entity(data_dir, "load", t, [500.0] * len(t))
    write_entity(data_dir, "soc", [t[0], t[600]], [40.0, 55.0])
    f = build_power_frame("2026-03-02", "2026-03-02", include_measured=True, reader=parquet_reader(data_dir))
    assert f["soc_pct"].iloc[599] == 40.0
    assert f["soc_pct"].iloc[-1] == 55.0


def test_dst_days_have_23_and_25_hours(data_dir):
    t = _minutes("2026-03-28T23:00:00Z", 3 * 1440)
    write_entity(data_dir, "load", t, [500.0] * len(t))
    r = parquet_reader(data_dir)
    assert len(build_power_frame("2026-03-29", "2026-03-29", reader=r)) == 23 * 60
    t2 = _minutes("2025-10-25T22:00:00Z", 3 * 1440)
    write_entity(data_dir, "load", t2, [500.0] * len(t2))
    f = build_power_frame("2025-10-26", "2025-10-26", reader=r)
    assert len(f) == 25 * 60
    assert f.index.is_unique


def test_clamped_to_available_data(two_days):
    f = build_power_frame("2026-02-01", "2026-04-30", reader=parquet_reader(two_days))
    assert f.index[0] == pd.Timestamp("2026-03-02 00:00", tz="Europe/Madrid")
    assert f.index[-1] == pd.Timestamp("2026-03-03 23:59", tz="Europe/Madrid")


def test_no_load_data_gives_empty_frame(data_dir):
    f = build_power_frame("2026-03-02", "2026-03-02", reader=parquet_reader(data_dir))
    assert len(f) == 0
    assert {"pv_kw", "load_kw", "ev_kw", "valid"} <= set(f.columns)


def test_csv_upload_conversion_matches_store_format():
    csv = (
        b'"last_changed","state"\n'
        b'"2026-03-02T00:00:10.000+01:00","100"\n'
        b'"2026-03-02T00:00:40.000+01:00","300"\n'
        b'"2026-03-02T00:00:40.000+01:00","300"\n'
        b'"2026-03-01T23:01:00Z","500"\n'
        b'"2026-03-02T00:02:00.000+01:00",\n'
        b'"2026-03-02T00:03:00.000+01:00","unavailable"\n'
    )
    e = csv_to_entity_frame(csv)
    assert str(e.index.tz) == "UTC"
    assert e.index.name == "ts"
    assert list(e["mean"]) == [200.0, 500.0]
    assert list(e["count"]) == [2, 1]
    assert e["count"].dtype == np.int32
    f = build_power_frame("2026-03-02", "2026-03-02", reader=memory_reader({"load": e, "pv": e}))
    assert len(f) == 2
    assert list(f["load_kw"]) == pytest.approx([0.2, 0.5])


def test_four_year_frame_is_fast():
    import time

    idx = pd.date_range("2022-01-01", "2026-01-01", freq="1min", tz="UTC", inclusive="left", name="ts")
    e = pd.DataFrame({"mean": np.full(len(idx), 800.0), "count": np.ones(len(idx), dtype=np.int32)}, index=idx)
    t0 = time.perf_counter()
    f = build_power_frame(reader=memory_reader({"load": e, "pv": e, "ev": e}))
    assert len(f) == len(idx)
    assert time.perf_counter() - t0 < 5.0


@pytest.mark.skipif(importlib.util.find_spec("src.data.store") is None, reason="data-layer store not present")
def test_default_reader_uses_store(two_days):
    f = build_power_frame("2026-03-02", "2026-03-03")
    assert len(f) == 2 * 1440


def test_fixture_files_match_store_on_disk_format(two_days):
    import pyarrow.parquet as pq

    table = pq.read_table(two_days / "influx" / "load.parquet")
    assert table.schema.names == ["ts", "mean", "count"]
    assert str(table.schema.field("ts").type) == "timestamp[ns, tz=UTC]"


def _fake_store(monkeypatch, read_entity):
    import sys
    import types

    mod = types.ModuleType("src.data.store")
    mod.read_entity = read_entity
    monkeypatch.setitem(sys.modules, "src.data.store", mod)


def test_store_reader_treats_unknown_entity_as_missing(monkeypatch):
    from src.data.frame import store_reader

    def read_entity(key, start=None, end=None):
        raise KeyError(key)

    _fake_store(monkeypatch, read_entity)
    assert store_reader("ev", None, None) is None


def test_store_reader_propagates_malformed_file_errors(monkeypatch):
    from src.data.frame import store_reader

    def read_entity(key, start=None, end=None):
        raise KeyError("ts")

    _fake_store(monkeypatch, read_entity)
    with pytest.raises(KeyError, match="ts"):
        store_reader("load", None, None)


@pytest.fixture
def edge_outage(data_dir):
    # Load 2026-03-01 .. 2026-03-02 19:08 local, then nothing until 2026-03-05 12:00 local, then a day more.
    before = _minutes("2026-02-28T23:00:00Z", 1440 + 19 * 60 + 8)
    after = _minutes("2026-03-05T11:00:00Z", 1440)
    write_entity(data_dir, "load", before + after, [500.0] * (len(before) + len(after)))
    return data_dir


def test_outage_at_window_end_stays_as_invalid_rows(edge_outage):
    f = build_power_frame("2026-03-02", "2026-03-02", reader=parquet_reader(edge_outage))
    assert len(f) == 1440
    assert f.index[-1] == pd.Timestamp("2026-03-02 23:59", tz="Europe/Madrid")
    assert f["valid"].sum() == 19 * 60 + 8 + 15
    assert frame_coverage(f)["excluded_hours"] == pytest.approx((1440 - (19 * 60 + 23)) / 60)


def test_outage_at_window_start_stays_as_invalid_rows(edge_outage):
    f = build_power_frame("2026-03-03", "2026-03-05", reader=parquet_reader(edge_outage))
    assert len(f) == 3 * 1440
    assert f.index[0] == pd.Timestamp("2026-03-03 00:00", tz="Europe/Madrid")
    assert f["valid"].sum() == 12 * 60
    assert not f["valid"].iloc[: 2 * 1440 + 12 * 60].any()


def test_window_entirely_inside_outage_is_all_invalid(edge_outage):
    f = build_power_frame("2026-03-03", "2026-03-04", reader=parquet_reader(edge_outage))
    assert len(f) == 2 * 1440
    assert not f["valid"].any()


def test_clamp_uses_global_data_span(edge_outage):
    f = build_power_frame("2026-02-20", "2026-03-20", reader=parquet_reader(edge_outage))
    assert f.index[0] == pd.Timestamp("2026-03-01 00:00", tz="Europe/Madrid")
    assert f.index[-1] == pd.Timestamp("2026-03-06 11:59", tz="Europe/Madrid")
