"""Influx entity registry. Measurement is the Influx unit name."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final
from zoneinfo import ZoneInfo

UTC: Final = ZoneInfo("UTC")
DATA_START: Final = datetime(2022, 8, 25, tzinfo=UTC)
CHUNK: Final = timedelta(days=7)
OVERLAP: Final = timedelta(hours=3)
MAX_WORKERS: Final = 3


@dataclass(frozen=True, slots=True)
class Entity:
    key: str
    measurement: str
    entity_id: str
    unit: str
    semantics: str


# ev lives in measurement kW, not W. W is empty for both wall-connector ids.
# Values are kilowatts (observed max ~7.4). Chosen over tesla_wall_connector_total_power
# because wall_connector_power has the larger recent sample count and is not nonsense.
ENTITIES: Final[dict[str, Entity]] = {
    "pv": Entity("pv", "W", "pv_power", "W", "PV DC power, >=0"),
    "load": Entity("load", "W", "house_consumption", "W", "whole-house load incl. EV charger"),
    "battery": Entity("battery", "W", "battery_power", "W", ">0 discharging, <0 charging"),
    "grid": Entity("grid", "W", "on_grid_export_power", "W", ">0 exporting, <0 importing"),
    "soc": Entity("soc", "%", "battery_state_of_charge", "%", "0-100"),
    "ev": Entity("ev", "kW", "wall_connector_power", "kW", "Tesla wall connector power, part of load"),
}


def resolve_entities(keys: list[str] | None) -> tuple[Entity, ...]:
    if keys is None:
        return tuple(ENTITIES[key] for key in ENTITIES)
    missing = [key for key in keys if key not in ENTITIES]
    if missing:
        raise KeyError(",".join(missing))
    return tuple(ENTITIES[key] for key in keys)
