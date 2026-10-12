"""Additive API models for the feature-phase contract. Existing models stay in schemas.py."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from web.backend.models.schemas import BatteryConfig, DateRange, GridConfig, TariffConfig


class RunStrategyBrief(BaseModel):
    strategy_id: str
    status: str
    total_cost: Optional[float] = None
    adjusted_cost: Optional[float] = None


class RunListItem(BaseModel):
    run_id: str
    label: Optional[str] = None
    created_at: str
    status: str
    date_range: Optional[DateRange] = None
    strategies: List[RunStrategyBrief]
    config_digest: str


class HardwarePreset(BaseModel):
    id: str
    name: str
    description: str
    battery: BatteryConfig
    grid: GridConfig
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None


class TariffPreset(BaseModel):
    id: str
    name: str
    description: str
    tariff: TariffConfig


class PresetsResponse(BaseModel):
    hardware: List[HardwarePreset]
    tariffs: List[TariffPreset]


class RefreshRequest(BaseModel):
    full: bool = False


class RefreshStartResponse(BaseModel):
    job_id: str


class EntityRefreshSummary(BaseModel):
    rows_added: int = 0
    rows: int = 0
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None
    gaps_over_1h: int = 0
    fetched: int = 0


class RefreshSummaryBody(BaseModel):
    updated_at: str = ""
    entities: Dict[str, EntityRefreshSummary] = Field(default_factory=dict)


class RefreshJobResponse(BaseModel):
    job_id: str
    status: Literal["running", "completed", "failed"]
    progress_pct: float
    message: str = ""
    summary: Optional[RefreshSummaryBody] = None
    error: Optional[str] = None


class SweepRequest(BaseModel):
    date_range: DateRange
    strategies: List[str] = Field(min_length=1)
    tariffs: List[str] = Field(min_length=1)
    capacities_kwh: List[float] = Field(min_length=1)
    hardware: str = "current"
    exclude_ev: bool = False
    label: Optional[str] = None


class SweepStartResponse(BaseModel):
    sweep_id: str
    total_cases: int


class SweepCase(BaseModel):
    tariff: str
    capacity_kwh: float
    strategy_id: str
    status: str
    total_cost: Optional[float] = None
    adjusted_cost: Optional[float] = None
    savings_vs_baseline: Optional[float] = None
    annualised_savings: Optional[float] = None
    equivalent_full_cycles: Optional[float] = None
    self_consumption_rate: Optional[float] = None
    error: Optional[str] = None


class SweepDetail(BaseModel):
    sweep_id: str
    label: Optional[str] = None
    created_at: str
    status: str
    date_range: DateRange
    done: int
    total: int
    cases: List[SweepCase]


class SweepListItem(BaseModel):
    sweep_id: str
    label: Optional[str] = None
    created_at: str
    status: str
    date_range: DateRange
    total: int


class HardwareSegment(BaseModel):
    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)

    preset_id: str
    from_: str = Field(alias="from")
    to: str


class DailyBacktest(BaseModel):
    date: str
    measured_import_kwh: float
    sim_import_kwh: float
    measured_export_kwh: float
    sim_export_kwh: float
    measured_cost: float
    sim_cost: float


class SocSeries(BaseModel):
    ts: List[int]
    measured_pct: List[Optional[float]]
    sim_pct: List[Optional[float]]


class BacktestMetrics(BaseModel):
    import_error_pct: float
    export_error_pct: float
    cost_error_pct: float
    daily_import_mae_kwh: float
    soc_rmse_pct: float
    measured_cost: float
    sim_cost: float


class BacktestRequest(BaseModel):
    date_range: DateRange
    strategy_id: str = "self_consume"
    hardware: str = "auto"


class BacktestResponse(BaseModel):
    strategy_id: str
    hardware_used: List[HardwareSegment]
    daily: List[DailyBacktest]
    soc: SocSeries
    metrics: BacktestMetrics
