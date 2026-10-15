from pathlib import Path

import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

STORE_SCHEMA = pa.schema([
    pa.field("ts", pa.timestamp("ns", tz="UTC")),
    pa.field("mean", pa.float64()),
    pa.field("count", pa.int32()),
])


def write_entity(data_dir: Path, key: str, minutes_utc: list, values: list) -> None:
    """Write a parquet file in the store's on-disk format: COLUMNS ts (UTC minute), mean float64, count int32."""
    df = pd.DataFrame({
        "ts": pd.to_datetime(minutes_utc, utc=True).astype("datetime64[ns, UTC]"),
        "mean": pd.Series(values, dtype="float64").to_numpy(),
        "count": pd.Series([1] * len(values), dtype="int32").to_numpy(),
    })
    out = data_dir / "influx"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, schema=STORE_SCHEMA, preserve_index=False), out / f"{key}.parquet")


def parquet_reader(data_dir: Path):
    """Stand-in for src.data.store.read_entity with the same contract."""

    def _read(key, start=None, end=None):
        path = data_dir / "influx" / f"{key}.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path, columns=["ts", "mean", "count"]).set_index("ts")
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
