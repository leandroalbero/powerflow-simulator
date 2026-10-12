from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.data.frame import build_power_frame, csv_to_entity_frame, memory_reader
from web.backend.models.schemas import DataInfo, DateRange

LOCAL_TZ = "Europe/Madrid"

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


class DataService:
    """Holds per-minute entity data and builds the fixed simulation frame from it."""

    def __init__(self) -> None:
        self.entities: Dict[str, pd.DataFrame] = {}

    # --- loading ---

    def load_default_data(self) -> None:
        """Load the high-resolution HA CSV exports from the data/ directory."""
        for key, name in (("pv", "pv_power_highres.csv"), ("load", "house_consumption_highres.csv")):
            path = DATA_DIR / name
            if path.exists():
                self.entities[key] = csv_to_entity_frame(path)

    def upload_solar(self, content: bytes) -> None:
        self.entities["pv"] = self._parse_upload(content)

    def upload_load(self, content: bytes) -> None:
        self.entities["load"] = self._parse_upload(content)

    @staticmethod
    def _parse_upload(content: bytes) -> pd.DataFrame:
        try:
            return csv_to_entity_frame(content)
        except ValueError as exc:
            raise ValueError(f"CSV must contain 'last_changed' and 'state' columns: {exc}") from exc

    # --- queries ---

    def get_frame(self, start: Optional[str] = None, end: Optional[str] = None,
                  exclude_ev: bool = False) -> pd.DataFrame:
        if "load" not in self.entities:
            raise ValueError("Data not loaded. Load or upload CSV files first.")
        return build_power_frame(start, end, exclude_ev=exclude_ev, reader=memory_reader(self.entities))

    def get_date_range(self) -> Optional[DateRange]:
        pv, load = self.entities.get("pv"), self.entities.get("load")
        if pv is None or load is None or pv.empty or load.empty:
            return None
        start = max(pv.index.min(), load.index.min()).tz_convert(LOCAL_TZ)
        end = min(pv.index.max(), load.index.max()).tz_convert(LOCAL_TZ)
        return DateRange(start=start.isoformat(), end=end.isoformat())

    def get_data_info(self) -> DataInfo:
        pv, load = self.entities.get("pv"), self.entities.get("load")
        return DataInfo(
            solar_file_loaded=pv is not None,
            load_file_loaded=load is not None,
            date_range=self.get_date_range(),
            solar_point_count=len(pv) if pv is not None else 0,
            load_point_count=len(load) if load is not None else 0,
        )
