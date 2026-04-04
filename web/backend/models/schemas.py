from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# --- Strategy ---

class StrategyInfo(BaseModel):
    id: str
    name: str
    description: str


# --- Battery / Grid / Tariff config ---

class BatteryConfig(BaseModel):
    capacity: float = Field(5.4, gt=0, description="Battery capacity in kWh")
    max_charge_rate: float = Field(2.1, gt=0, description="Max charge rate in kW")
    max_discharge_rate: float = Field(2.1, gt=0, description="Max discharge rate in kW")
    efficiency: float = Field(0.95, gt=0, le=1.0, description="Round-trip efficiency")
    initial_soc: float = Field(0.1, ge=0, le=1.0, description="Initial state of charge (0-1)")
    taper_start: float = Field(0.9, gt=0, le=1.0, description="SoC threshold where charge tapering begins")
    taper_factor: float = Field(0.7, ge=0, le=1.0, description="Tapering intensity (0=no taper, 1=full taper)")


class GridConfig(BaseModel):
    max_import: float = Field(5.0, gt=0, description="Max grid import in kW")
    max_export: float = Field(5.0, gt=0, description="Max grid export in kW")


class TariffRate(BaseModel):
    start_hour: int = Field(ge=0, lt=24)
    end_hour: int = Field(gt=0, le=24)
    price: float = Field(ge=0)
    direction: str = Field(description="'import' or 'export'")


class TariffConfig(BaseModel):
    rates: List[TariffRate] = Field(default_factory=lambda: [
        TariffRate(start_hour=0, end_hour=8, price=0.085, direction="import"),
        TariffRate(start_hour=8, end_hour=10, price=0.134, direction="import"),
        TariffRate(start_hour=10, end_hour=14, price=0.182, direction="import"),
        TariffRate(start_hour=14, end_hour=18, price=0.134, direction="import"),
        TariffRate(start_hour=18, end_hour=22, price=0.182, direction="import"),
        TariffRate(start_hour=22, end_hour=24, price=0.134, direction="import"),
        TariffRate(start_hour=0, end_hour=24, price=0.08, direction="export"),
    ])
    weekend_rate: Optional[TariffRate] = Field(None, description="Flat override rate for weekends")


class StrategyConfig(BaseModel):
    min_battery_level: float = Field(0.1, ge=0, le=1.0, description="Minimum SoC before discharge stops")
    max_charge_power: float = Field(2.05, gt=0, description="Max grid charge power in kW")
    valley_charge_target: float = Field(1.0, gt=0, le=1.0, description="Target SoC during valley charging")


class SystemConfig(BaseModel):
    battery: BatteryConfig = Field(default_factory=BatteryConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    tariff: TariffConfig = Field(default_factory=TariffConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)


class DateRange(BaseModel):
    start: str = Field(description="ISO datetime string")
    end: str = Field(description="ISO datetime string")


class ConfigResponse(BaseModel):
    battery: BatteryConfig
    grid: GridConfig
    tariff: TariffConfig
    strategy: StrategyConfig
    data_date_range: Optional[DateRange] = None


# --- Simulation ---

class SimulationRequest(BaseModel):
    strategies: List[str]
    date_range: Optional[DateRange] = None


class SimulationStartResponse(BaseModel):
    run_id: str
    status: str = "running"


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StrategyMetrics(BaseModel):
    total_cost: float
    total_solar_generated: float
    total_solar_consumed: float
    total_solar_exported: float
    total_grid_imported: float
    total_battery_in: float
    total_battery_out: float
    total_house_consumption: float
    battery_level: float
    battery_capacity: float
    self_consumption_rate: float
    solar_fraction: float


class StrategyResult(BaseModel):
    strategy_id: str
    status: str
    metrics: Optional[StrategyMetrics] = None
    error: Optional[str] = None


class RunSummary(BaseModel):
    run_id: str
    status: str
    strategies: List[StrategyResult]


class TimeseriesResponse(BaseModel):
    timestamps: List[str]
    battery_level: List[float]
    grid_import: List[float]
    grid_export: List[float]
    solar_power: List[float]
    house_consumption: List[float]
    point_count: int


# --- Data info ---

class DataInfo(BaseModel):
    solar_file_loaded: bool = False
    load_file_loaded: bool = False
    date_range: Optional[DateRange] = None
    solar_point_count: int = 0
    load_point_count: int = 0
