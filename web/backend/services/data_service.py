"""Per-minute power data for simulations.

Data comes from the parquet store (``src.data.store``) through
``build_power_frame``. Uploaded ``last_changed,state`` CSVs are converted to the
same per-minute entity format and override the store for that entity until the
next reload. Without a store, the bundled high-resolution CSVs are used.
"""

import importlib.util
import json
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.data.frame import (
    EntityReader,
    build_power_frame,
    csv_to_entity_frame,
    frame_coverage,
    memory_reader,
    store_reader,
)
from web.backend.models.schemas import DataGap, DataInfo, DateRange

logger = logging.getLogger(__name__)

LOCAL_TZ = "Europe/Madrid"
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
STORE_ENTITIES = ("pv", "load", "battery", "grid", "soc", "ev")
LEGACY_CSVS = {"pv": "pv_power_highres.csv", "load": "house_consumption_highres.csv"}
FRAME_CACHE_SIZE = 4

FrameKey = Tuple[Optional[str], Optional[str], bool]


def data_dir() -> Path:
    base = os.environ.get("POWERFLOW_DATA_DIR")
    return Path(base) if base else DATA_DIR


def store_available() -> bool:
    return importlib.util.find_spec("src.data.store") is not None and (data_dir() / "influx").is_dir()


class DataService:
    def __init__(self) -> None:
        self.uploads: Dict[str, pd.DataFrame] = {}
        self.csv_entities: Dict[str, pd.DataFrame] = {}
        self.use_store = False
        self._frames: "OrderedDict[FrameKey, pd.DataFrame]" = OrderedDict()
        self._lock = threading.Lock()
        self._info: Optional[DataInfo] = None

    # --- loading ---

    def load_default_data(self) -> None:
        """(Re)load: use the parquet store when present, else the bundled CSVs; drops uploads."""
        with self._lock:
            self.uploads = {}
            self.csv_entities = {}
            self.use_store = store_available()
            if not self.use_store:
                for key, name in LEGACY_CSVS.items():
                    path = data_dir() / name
                    if path.exists():
                        self.csv_entities[key] = csv_to_entity_frame(path)
            self._invalidate()

    def upload_solar(self, content: bytes) -> None:
        self._upload("pv", content)

    def upload_load(self, content: bytes) -> None:
        self._upload("load", content)

    def _upload(self, key: str, content: bytes) -> None:
        try:
            entity = csv_to_entity_frame(content)
        except ValueError as exc:
            raise ValueError(f"CSV must contain 'last_changed' and 'state' columns: {exc}") from exc
        if entity.empty:
            raise ValueError("CSV contains no numeric samples")
        with self._lock:
            self.uploads[key] = entity
            self._invalidate()

    def _invalidate(self) -> None:
        self._frames.clear()
        self._info = None

    def _reader(self) -> EntityReader:
        if self.use_store:
            return memory_reader(self.uploads, fallback=store_reader)
        return memory_reader({**self.csv_entities, **self.uploads})

    def has_data(self) -> bool:
        return self.use_store or "load" in self.uploads or "load" in self.csv_entities

    # --- frames ---

    def get_frame(self, start: Optional[str] = None, end: Optional[str] = None,
                  exclude_ev: bool = False) -> pd.DataFrame:
        if not self.has_data():
            raise ValueError("Data not loaded. Load or upload CSV files first.")
        key: FrameKey = (start, end, exclude_ev)
        with self._lock:
            frame = self._frames.get(key)
            if frame is not None:
                self._frames.move_to_end(key)
                return frame
            reader = self._reader()
        frame = build_power_frame(start, end, exclude_ev=exclude_ev, reader=reader)
        with self._lock:
            self._frames[key] = frame
            while len(self._frames) > FRAME_CACHE_SIZE:
                self._frames.popitem(last=False)
        return frame

    # --- info ---

    def _entities_available(self) -> List[str]:
        present = set(self.uploads)
        if self.use_store:
            present |= {k for k in STORE_ENTITIES if (data_dir() / "influx" / f"{k}.parquet").exists()}
        else:
            present |= set(self.csv_entities)
        return [k for k in STORE_ENTITIES if k in present]

    @staticmethod
    def _manifest_updated_at() -> Optional[str]:
        path = data_dir() / "influx" / "manifest.json"
        try:
            with open(path) as f:
                value = json.load(f).get("updated_at")
            return str(value) if value is not None else None
        except (OSError, ValueError, AttributeError):
            return None

    def get_date_range(self) -> Optional[DateRange]:
        info = self.get_data_info()
        return info.date_range

    def get_data_info(self) -> DataInfo:
        if self._info is not None:
            return self._info
        if not self.has_data():
            return DataInfo()
        frame = self.get_frame()
        coverage = frame_coverage(frame)
        start = frame.index[0].isoformat() if len(frame) else None
        end = frame.index[-1].isoformat() if len(frame) else None
        entities = self._entities_available()
        pv = self.uploads.get("pv", self.csv_entities.get("pv"))
        load = self.uploads.get("load", self.csv_entities.get("load"))
        info = DataInfo(
            solar_file_loaded="pv" in entities,
            load_file_loaded="load" in entities,
            date_range=DateRange(start=start, end=end) if start and end else None,
            solar_point_count=len(pv) if pv is not None else int(frame["pv_kw"].gt(0).sum()),
            load_point_count=len(load) if load is not None else int(frame["valid"].sum()),
            data_from=start,
            data_until=end,
            coverage=round(coverage["valid_fraction"], 6),
            gaps=[DataGap(**g) for g in coverage["gaps"]],
            manifest_updated_at=self._manifest_updated_at() if self.use_store else None,
            entities=entities,
            source="store" if self.use_store else "csv",
            uploads=sorted(self.uploads),
        )
        self._info = info
        return info
