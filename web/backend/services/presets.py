"""Hardware and tariff presets named by the API contract."""

from typing import Final

from web.backend.models.feature_schemas import HardwarePreset, TariffPreset
from web.backend.models.schemas import BatteryConfig, GridConfig, SystemConfig, TariffConfig

REFERENCE_CAPACITY_KWH: Final[float] = 15.36
SWITCH_DATE: Final[str] = "2026-04-07"


def hardware_presets() -> list[HardwarePreset]:
    grid = GridConfig()
    return [
        HardwarePreset(
            id="current",
            name="Current GoodWe",
            description="15.36 kWh, 4.5/4.6 kW, in service from 2026-04-07",
            battery=BatteryConfig(capacity=15.36, max_charge_rate=4.5, max_discharge_rate=4.6),
            grid=grid,
            valid_from=SWITCH_DATE,
        ),
        HardwarePreset(
            id="pre_2026_04",
            name="Pre-April 2026",
            description="15.36 kWh, 2.2/2.1 kW, through 2026-04-06",
            battery=BatteryConfig(capacity=15.36, max_charge_rate=2.2, max_discharge_rate=2.1),
            grid=grid,
            valid_until="2026-04-06",
        ),
        HardwarePreset(
            id="double_battery",
            name="Double battery",
            description="30.72 kWh at the current inverter rates",
            battery=BatteryConfig(capacity=30.72, max_charge_rate=4.5, max_discharge_rate=4.6),
            grid=grid,
        ),
    ]


def tariff_presets() -> list[TariffPreset]:
    flat = TariffConfig()
    pvpc = flat.model_copy(update={"use_pvpc": True})
    indexed = flat.model_copy(update={
        "use_pvpc": True,
        "export_follows_pvpc": True,
        "export_pvpc_discount": 0.02,
    })
    return [
        TariffPreset(id="2_0td", name="2.0TD", description="Default time-of-use tariff", tariff=flat),
        TariffPreset(id="pvpc", name="PVPC", description="PVPC import, flat export", tariff=pvpc),
        TariffPreset(
            id="pvpc_indexed_export",
            name="PVPC indexed export",
            description="PVPC import and export with a 0.02 EUR/kWh discount",
            tariff=indexed,
        ),
    ]


def hardware_by_id(preset_id: str) -> HardwarePreset:
    for preset in hardware_presets():
        if preset.id == preset_id:
            return preset
    raise KeyError(preset_id)


def tariff_by_id(preset_id: str) -> TariffPreset:
    for preset in tariff_presets():
        if preset.id == preset_id:
            return preset
    raise KeyError(preset_id)


def scale_battery(battery: BatteryConfig, capacity_kwh: float) -> BatteryConfig:
    """Override capacity and scale charge/discharge by capacity / 15.36 kWh."""
    scale = capacity_kwh / REFERENCE_CAPACITY_KWH
    return battery.model_copy(update={
        "capacity": capacity_kwh,
        "max_charge_rate": battery.max_charge_rate * scale,
        "max_discharge_rate": battery.max_discharge_rate * scale,
    })


def config_for_hardware(preset_id: str, capacity_kwh: float | None = None) -> SystemConfig:
    preset = hardware_by_id(preset_id)
    battery = preset.battery if capacity_kwh is None else scale_battery(preset.battery, capacity_kwh)
    return SystemConfig(battery=battery, grid=preset.grid)
