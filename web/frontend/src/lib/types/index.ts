// Shapes mirror web/backend/models/schemas.py plus the additive fields of API contract v2.
// Fields from the contract that the current backend may not send yet are optional.

export type StrategyGroup = 'core' | 'advanced' | 'experimental';

export interface StrategyInfo {
  id: string;
  name: string;
  description: string;
  group: StrategyGroup;
  available?: boolean;
  unavailable_reason?: string | null;
}

// ---- Config ----

export interface BatteryConfig {
  capacity: number;
  max_charge_rate: number;
  max_discharge_rate: number;
  efficiency: number;
  initial_soc: number;
  taper_start: number;
  taper_factor: number;
  cycle_cost_eur_kwh: number;
}

export interface GridConfig {
  max_import: number;
  max_export: number;
}

export type RateDirection = 'import' | 'export';

export interface TariffRate {
  start_hour: number;
  end_hour: number;
  price: number;
  direction: RateDirection;
}

export interface TariffConfig {
  rates: TariffRate[];
  weekend_rate: TariffRate | null;
  use_pvpc: boolean;
  export_follows_pvpc: boolean;
  export_pvpc_discount: number;
  mpc_horizon_hours: number;
}

export interface StrategyConfig {
  min_battery_level: number;
  max_charge_power: number;
  valley_charge_target: number;
}

export interface SystemConfig {
  battery: BatteryConfig;
  grid: GridConfig;
  tariff: TariffConfig;
  strategy: StrategyConfig;
  exclude_ev: boolean;
}

export interface DateRange {
  start: string;
  end: string;
}

export interface ConfigResponse {
  battery: BatteryConfig;
  grid: GridConfig;
  tariff: TariffConfig;
  strategy: StrategyConfig;
  exclude_ev?: boolean;
  data_date_range: DateRange | null;
}

// ---- Presets (new) ----

export interface HardwarePreset {
  id: string;
  name: string;
  description: string;
  battery: BatteryConfig;
  grid: GridConfig;
  valid_from: string | null;
  valid_until: string | null;
}

export interface TariffPreset {
  id: string;
  name: string;
  description: string;
  tariff: TariffConfig;
}

export interface PresetsResponse {
  hardware: HardwarePreset[];
  tariffs: TariffPreset[];
}

// ---- Simulation ----

export interface SimulationRequest {
  strategies: string[];
  date_range: DateRange;
  config?: SystemConfig;
  label?: string;
}

export interface SimulationStartResponse {
  run_id: string;
  status: string;
}

export interface MonthlyMetrics {
  month: string;
  cost: number;
  import_kwh: number;
  export_kwh: number;
  import_cost: number;
  export_revenue: number;
  solar_kwh: number;
  load_kwh: number;
  battery_out_kwh: number;
}

export interface StrategyMetrics {
  total_cost: number;
  total_solar_generated: number;
  total_solar_consumed: number;
  total_solar_exported: number;
  total_grid_imported: number;
  total_battery_in: number;
  total_battery_out: number;
  total_house_consumption: number;
  battery_level: number;
  battery_capacity: number;
  /** Percent, 0–100. */
  self_consumption_rate: number;
  /** Share of load not imported from the grid, percent 0–100. */
  solar_fraction: number;
  import_cost?: number | null;
  export_revenue?: number | null;
  adjusted_cost?: number | null;
  terminal_value_eur?: number | null;
  stored_energy_delta_kwh?: number | null;
  initial_battery_kwh?: number | null;
  final_battery_kwh?: number | null;
  unserved_kwh?: number | null;
  curtailed_kwh?: number | null;
  over_limit_kwh?: number | null;
  grid_charge_kwh?: number | null;
  valid_hours?: number | null;
  excluded_hours?: number | null;
  equivalent_full_cycles?: number | null;
  ev_kwh?: number | null;
  monthly?: MonthlyMetrics[] | null;
}

export type StrategyStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface StrategyResult {
  strategy_id: string;
  status: StrategyStatus;
  progress?: number;
  metrics?: StrategyMetrics | null;
  error?: string | null;
}

export type RunStatusValue = 'running' | 'completed' | 'failed' | 'cancelled';

export interface RunSummary {
  run_id: string;
  status: RunStatusValue;
  strategies: StrategyResult[];
  label?: string | null;
  created_at?: string | null;
  date_range?: DateRange | null;
  config?: SystemConfig | null;
}

// ---- Timeseries ----

/** Raw response; v2 fields are optional until the backend ships them. */
export interface TimeseriesResponse {
  timestamps: string[];
  battery_level: (number | null)[];
  grid_import: (number | null)[];
  grid_export: (number | null)[];
  solar_power: (number | null)[];
  house_consumption: (number | null)[];
  point_count: number;
  ts?: number[];
  battery_charge?: (number | null)[];
  battery_discharge?: (number | null)[];
  soc_pct?: (number | null)[];
  import_price?: (number | null)[];
  export_price?: (number | null)[];
  step_cost?: (number | null)[];
  downsampled?: boolean;
  resolution_minutes?: number;
}

/** Normalised timeseries used by the charts. Absent series stay undefined. */
export interface Timeseries {
  /** Epoch seconds (uPlot's native x unit). */
  x: number[];
  solar: (number | null)[];
  load: (number | null)[];
  gridImport: (number | null)[];
  gridExport: (number | null)[];
  batteryKwh: (number | null)[];
  socPct?: (number | null)[];
  batteryCharge?: (number | null)[];
  batteryDischarge?: (number | null)[];
  importPrice?: (number | null)[];
  exportPrice?: (number | null)[];
  stepCost?: (number | null)[];
}

// ---- Data ----

export interface DataGap {
  start: string;
  end: string;
  hours: number;
}

export interface DataInfo {
  solar_file_loaded?: boolean;
  load_file_loaded?: boolean;
  date_range: DateRange | null;
  solar_point_count: number;
  load_point_count: number;
  data_from: string | null;
  data_until: string | null;
  /** Fraction 0–1. */
  coverage: number | null;
  gaps: DataGap[];
  manifest_updated_at: string | null;
  entities: string[];
  source: 'store' | 'csv' | null;
  uploads: string[];
  has_ev?: boolean;
  has_measured?: boolean;
}

export interface RefreshStartResponse {
  job_id: string;
}

export interface RefreshJob {
  job_id: string;
  status: 'running' | 'completed' | 'failed';
  progress_pct: number;
  message: string;
  summary?: Record<string, unknown> | null;
  error?: string | null;
}

// ---- Runs history (new) ----

export interface RunListItem {
  run_id: string;
  label: string | null;
  created_at: string;
  status: RunStatusValue;
  date_range: DateRange;
  strategies: { strategy_id: string; status: StrategyStatus; total_cost: number | null; adjusted_cost: number | null }[];
  config_digest: string;
}

export type ExportKind = 'summary' | 'monthly' | 'timeseries';

// ---- WebSocket ----

export interface WsProgress {
  strategy: string;
  percent: number;
}

export interface WsStrategyDone {
  strategy: string;
  status: StrategyStatus;
  metrics?: StrategyMetrics | null;
  error?: string | null;
}

export interface WsRunComplete {
  status: 'run_complete';
}

export interface WsError {
  error: string;
}

export type WsMessage = WsProgress | WsStrategyDone | WsRunComplete | WsError;

// ---- Sweep (new) ----

export interface SweepRequest {
  date_range: DateRange;
  strategies: string[];
  tariffs: string[];
  capacities_kwh: number[];
  hardware?: string;
  exclude_ev?: boolean;
  label?: string;
}

export interface SweepCase {
  tariff: string;
  capacity_kwh: number;
  strategy_id: string;
  status: StrategyStatus;
  total_cost: number | null;
  adjusted_cost: number | null;
  savings_vs_baseline: number | null;
  annualised_savings: number | null;
  equivalent_full_cycles: number | null;
  self_consumption_rate: number | null;
  error?: string | null;
}

export interface SweepResult {
  sweep_id: string;
  label: string | null;
  created_at: string;
  status: RunStatusValue;
  date_range: DateRange;
  done: number;
  total: number;
  cases: SweepCase[];
}

export interface SweepListItem {
  sweep_id: string;
  label: string | null;
  created_at: string;
  status: RunStatusValue;
  date_range: DateRange;
  total: number;
}

export type WsSweepMessage =
  | { type: 'case_complete'; case: SweepCase }
  | { type: 'progress'; done: number; total: number }
  | { type: 'sweep_complete'; status: RunStatusValue };

// ---- Backtest (new) ----

export interface BacktestRequest {
  date_range: DateRange;
  strategy_id?: string;
  hardware?: string;
}

export interface BacktestDaily {
  date: string;
  measured_import_kwh: number;
  sim_import_kwh: number;
  measured_export_kwh: number;
  sim_export_kwh: number;
  measured_cost: number;
  sim_cost: number;
}

export interface BacktestResult {
  strategy_id: string;
  hardware_used: { preset_id: string; from: string; to: string }[];
  daily: BacktestDaily[];
  soc: { ts: number[]; measured_pct: (number | null)[]; sim_pct: number[] };
  metrics: {
    import_error_pct: number;
    export_error_pct: number;
    cost_error_pct: number;
    daily_import_mae_kwh: number;
    soc_rmse_pct: number;
    measured_cost: number;
    sim_cost: number;
  };
}
