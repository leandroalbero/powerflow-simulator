from pathlib import Path

import pandas as pd
import pytest


def write_entity(data_dir: Path, key: str, minutes_utc: list, values: list) -> None:
    """Write a parquet file in the store's format: index ts (UTC minute), mean float64, count int32."""
    idx = pd.DatetimeIndex(pd.to_datetime(minutes_utc, utc=True), name="ts")
    df = pd.DataFrame({"mean": pd.Series(values, dtype="float64").to_numpy(),
                       "count": pd.Series([1] * len(values), dtype="int32").to_numpy()}, index=idx)
    out = data_dir / "influx"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / f"{key}.parquet")


def parquet_reader(data_dir: Path):
    """Stand-in for src.data.store.read_entity with the same contract."""

    def _read(key, start=None, end=None):
        path = data_dir / "influx" / f"{key}.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index < end]
        return df

    return _read


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("POWERFLOW_DATA_DIR", str(tmp_path))
    return tmp_path
