from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# --- Strategy ---

class StrategyInfo(BaseModel):
    id: str
    name: str
    description: str
    group: Literal["core", "advanced", "experimental"] = "advanced"
    available: bool = True
    unavailable_reason: Optional[str] = None


# --- Battery / Grid / Tariff config ---

class BatteryConfig(BaseModel):
    capacity: float = Field(15.36, gt=0, description="Battery capacity in kWh")
    max_charge_rate: float = Field(4.5, gt=0, description="Max charge rate in kW")
    max_discharge_rate: float = Field(4.6, gt=0, description="Max discharge rate in kW")
    efficiency: float = Field(0.90, gt=0, le=1.0, description="Round-trip efficiency")
    initial_soc: float = Field(0.5, ge=0, le=1.0, description="Initial state of charge (0-1)")
    taper_start: float = Field(0.9, gt=0, le=1.0, description="SoC threshold where charge tapering begins")
    taper_factor: float = Field(0.7, ge=0, le=1.0, description="Tapering intensity (0=no taper, 1=full taper)")
    cycle_cost_eur_kwh: float = Field(
        0.0,
        ge=0,
        description="Battery wear cost per kWh cycled (EUR/kWh). "
        "LFP example: €1500 / (4000 cycles × 15 kWh) ≈ 0.025. "
        "Used by MPC/Oracle LP as a cycling penalty to prevent spurious "
        "arbitrage. Not included in reported total_cost.",
    )


class GridConfig(BaseModel):
    max_import: float = Field(5.5, gt=0, description="Max grid import in kW (caps grid charging; load is always served)")
    max_export: float = Field(5.5, gt=0, description="Max grid export in kW")


class TariffRate(BaseModel):
    start_hour: int = Field(ge=0, lt=24)
    end_hour: int = Field(gt=0, le=24)
    price: float = Field(ge=0)
    direction: str = Field(description="'import' or 'export'")


class TariffConfig(BaseModel):
    rates: List[TariffRate] = Field(default_factory=lambda: [
        TariffRate(start_hour=0, end_hour=8, price=0.085, direction="import"),
        TariffRate(start_hour=8, end_hour=10, price=0.135, direction="import"),
        TariffRate(start_hour=10, end_hour=14, price=0.197, direction="import"),
        TariffRate(start_hour=14, end_hour=18, price=0.135, direction="import"),
        TariffRate(start_hour=18, end_hour=22, price=0.197, direction="import"),
        TariffRate(start_hour=22, end_hour=24, price=0.135, direction="import"),
        TariffRate(start_hour=0, end_hour=24, price=0.08, direction="export"),
    ])
    weekend_rate: Optional[TariffRate] = Field(
        default_factory=lambda: TariffRate(start_hour=0, end_hour=24, price=0.085, direction="import"),
        description="Flat override rate for weekends (Spain 2.0TD: valley all day)",
    )
    use_pvpc: bool = Field(
        False,
        description="Replace import rate schedule with hourly PVPC prices from REE. "
        "Export uses the flat export rate from `rates`.",
    )
    export_follows_pvpc: bool = Field(
        False,
        description="When PVPC is on, also use the hourly PVPC price for export "
        "compensation instead of the flat rate. Approximates contracts that "
        "pay spot-indexed export.",
    )
    export_pvpc_discount: float = Field(
        0.0,
        ge=0,
        description="When export_follows_pvpc is on, subtract this amount (EUR/kWh) "
        "from the hourly PVPC price to model retailer margin/taxes. Clamped to 0. "
        "Octopus-style: try ~0.06 for PVPC minus margin, ~0.02 for near-pass-through.",
    )
    mpc_horizon_hours: int = Field(
        24,
        ge=1,
        le=48,
        description="LP horizon (hours) for MPC strategies. 48h works when PVPC "
        "next-day prices are published (after ~20:15) and models look further ahead. "
        "Simulation always has full foresight, so 48h is always valid here.",
    )


class StrategyConfig(BaseModel):
    min_battery_level: float = Field(0.1, ge=0, le=1.0, description="Minimum SoC before discharge stops")
    max_charge_power: float = Field(4.5, gt=0, description="Max grid charge power in kW (clipped to the battery's max charge rate)")
    valley_charge_target: float = Field(1.0, gt=0, le=1.0, description="Target SoC during valley charging")


class SystemConfig(BaseModel):
    battery: BatteryConfig = Field(default_factory=BatteryConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    tariff: TariffConfig = Field(default_factory=TariffConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    exclude_ev: bool = Field(False, description="Subtract the EV charger's power from the house load")


class DateRange(BaseModel):
    start: str = Field(description="ISO datetime string")
    end: str = Field(description="ISO datetime string")


class ConfigResponse(BaseModel):
    battery: BatteryConfig
    grid: GridConfig
    tariff: TariffConfig
    strategy: StrategyConfig
    data_date_range: Optional[DateRange] = None
    exclude_ev: bool = False


# --- Simulation ---

class SimulationRequest(BaseModel):
    strategies: List[str]
    date_range: Optional[DateRange] = None
    config: Optional[SystemConfig] = Field(
        None, description="Configuration for this run; defaults to the server's current config")
    label: Optional[str] = None


class SimulationStartResponse(BaseModel):
    run_id: str
    status: str = "running"


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class MonthlyMetrics(BaseModel):
    month: str
    cost: float
    import_kwh: float
    export_kwh: float
    import_cost: float
    export_revenue: float
    solar_kwh: float
    load_kwh: float
    battery_out_kwh: float


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
    import_cost: Optional[float] = None
    export_revenue: Optional[float] = None
    adjusted_cost: Optional[float] = None
    terminal_value_eur: Optional[float] = None
    stored_energy_delta_kwh: Optional[float] = None
    initial_battery_kwh: Optional[float] = None
    final_battery_kwh: Optional[float] = None
    unserved_kwh: Optional[float] = None
    curtailed_kwh: Optional[float] = None
    over_limit_kwh: Optional[float] = None
    grid_charge_kwh: Optional[float] = None
    valid_hours: Optional[float] = None
    excluded_hours: Optional[float] = None
    equivalent_full_cycles: Optional[float] = None
    ev_kwh: Optional[float] = None
    monthly: Optional[List[MonthlyMetrics]] = None


class StrategyResult(BaseModel):
    strategy_id: str
    status: str
    progress: int = 0
    metrics: Optional[StrategyMetrics] = None
    error: Optional[str] = None


class RunSummary(BaseModel):
    run_id: str
    status: str
    strategies: List[StrategyResult]
    label: Optional[str] = None
    created_at: Optional[str] = None
    date_range: Optional[DateRange] = None
    config: Optional[SystemConfig] = None


class TimeseriesResponse(BaseModel):
    timestamps: List[str]
    battery_level: List[Optional[float]]
    grid_import: List[Optional[float]]
    grid_export: List[Optional[float]]
    solar_power: List[Optional[float]]
    house_consumption: List[Optional[float]]
    point_count: int
    ts: List[int] = Field(default_factory=list)
    battery_charge: List[Optional[float]] = Field(default_factory=list)
    battery_discharge: List[Optional[float]] = Field(default_factory=list)
    soc_pct: List[Optional[float]] = Field(default_factory=list)
    import_price: List[Optional[float]] = Field(default_factory=list)
    export_price: List[Optional[float]] = Field(default_factory=list)
    step_cost: List[Optional[float]] = Field(default_factory=list)
    downsampled: bool = False
    resolution_minutes: float = 1.0


# --- Data info ---

class DataGap(BaseModel):
    start: str
    end: str
    hours: float


class DataInfo(BaseModel):
    solar_file_loaded: bool = False
    load_file_loaded: bool = False
    date_range: Optional[DateRange] = None
    solar_point_count: int = 0
    load_point_count: int = 0
    data_from: Optional[str] = None
    data_until: Optional[str] = None
    coverage: Optional[float] = Field(None, description="Fraction of minutes with known house load")
    gaps: List[DataGap] = Field(default_factory=list, description="Load outages of an hour or more")
    manifest_updated_at: Optional[str] = None
    entities: List[str] = Field(default_factory=list)
    source: Optional[str] = Field(None, description="'store' (parquet) or 'csv'")
    uploads: List[str] = Field(default_factory=list, description="Entities overridden by an uploaded CSV")
    has_ev: bool = False
    has_measured: bool = False
