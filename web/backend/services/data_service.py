import io
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import pytz

from web.backend.models.schemas import DataInfo, DateRange

LOCAL_TZ = pytz.timezone("Europe/Madrid")

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


class DataService:
    """Loads and manages CSV data for solar generation and house consumption."""

    def __init__(self) -> None:
        self.solar_data: Optional[pd.DataFrame] = None
        self.load_data: Optional[pd.DataFrame] = None

    # --- public API ---

    def load_default_data(self) -> None:
        """Load CSV files from the default data/ directory."""
        solar_path = DATA_DIR / "pv_power_highres.csv"
        load_path = DATA_DIR / "house_consumption_highres.csv"

        if solar_path.exists():
            self.solar_data = self._read_csv(solar_path)
        if load_path.exists():
            self.load_data = self._read_csv(load_path)

    def upload_solar(self, content: bytes) -> None:
        self.solar_data = self._read_csv_bytes(content)

    def upload_load(self, content: bytes) -> None:
        self.load_data = self._read_csv_bytes(content)

    def get_date_range(self) -> Optional[DateRange]:
        """Return the overlapping date range across both datasets."""
        if self.solar_data is None or self.load_data is None:
            return None

        start = max(self.solar_data.index.min(), self.load_data.index.min())
        end = min(self.solar_data.index.max(), self.load_data.index.max())

        return DateRange(
            start=start.isoformat(),
            end=end.isoformat(),
        )

    def get_data_info(self) -> DataInfo:
        return DataInfo(
            solar_file_loaded=self.solar_data is not None,
            load_file_loaded=self.load_data is not None,
            date_range=self.get_date_range(),
            solar_point_count=len(self.solar_data) if self.solar_data is not None else 0,
            load_point_count=len(self.load_data) if self.load_data is not None else 0,
        )

    def get_filtered_data(
        self, start: Optional[str] = None, end: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (solar_df, load_df) filtered to the given date range and sorted."""
        if self.solar_data is None or self.load_data is None:
            raise ValueError("Data not loaded. Load or upload CSV files first.")

        solar = self.solar_data.copy()
        load = self.load_data.copy()

        if start is not None:
            ts_start = pd.Timestamp(start)
            if ts_start.tzinfo is None:
                ts_start = ts_start.tz_localize(LOCAL_TZ)
            solar = solar[solar.index >= ts_start]
            load = load[load.index >= ts_start]

        if end is not None:
            ts_end = pd.Timestamp(end)
            if ts_end.tzinfo is None:
                ts_end = ts_end.tz_localize(LOCAL_TZ)
            solar = solar[solar.index <= ts_end]
            load = load[load.index <= ts_end]

        return solar.sort_index(), load.sort_index()

    # --- internal helpers ---

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        df = pd.read_csv(str(path), index_col="last_changed")
        return DataService._prepare_dataframe(df)

    @staticmethod
    def _read_csv_bytes(content: bytes) -> pd.DataFrame:
        df = pd.read_csv(io.BytesIO(content), index_col="last_changed")
        return DataService._prepare_dataframe(df)

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        if "state" not in df.columns:
            raise ValueError("CSV must contain a 'state' column")

        df.index = pd.to_datetime(df.index, utc=True).tz_convert(LOCAL_TZ)
        df["state"] = pd.to_numeric(df["state"], errors="coerce")
        df = df.dropna(subset=["state"])
        df = df.sort_index()
        return df
