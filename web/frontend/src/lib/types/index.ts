// ---- Strategy ----

export interface StrategyInfo {
  id: string;
  name: string;
  description: string;
}

// ---- System config ----

export interface BatteryConfig {
  capacity: number;
  max_charge_rate: number;
  max_discharge_rate: number;
  efficiency: number;
}

export interface GridConfig {
  max_import: number;
  max_export: number;
}

export interface TariffRate {
  start_hour: number;
  end_hour: number;
  price: number;
  direction: 'import' | 'export';
}

export interface TariffConfig {
  rates: TariffRate[];
}

export interface SystemConfig {
  battery: BatteryConfig;
  grid: GridConfig;
  tariff: TariffConfig;
}

export interface DateRange {
  start: string;
  end: string;
}

/** GET /api/config returns SystemConfig fields plus an optional data_date_range. */
export interface ConfigResponse {
  battery: BatteryConfig;
  grid: GridConfig;
  tariff: TariffConfig;
  data_date_range: DateRange | null;
}

// ---- Simulation ----

export interface SimulationRequest {
  strategies: string[];
  date_range?: DateRange;
}

export interface SimulationStartResponse {
  run_id: string;
  status: string;
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
  self_consumption_rate: number;
  solar_fraction: number;
}

export interface StrategyResult {
  strategy_id: string;
  status: string;
  metrics?: StrategyMetrics;
  error?: string;
}

export interface RunSummary {
  run_id: string;
  status: string;
  strategies: StrategyResult[];
}

// ---- Timeseries ----

export interface TimeseriesResponse {
  timestamps: string[];
  battery_level: number[];
  grid_import: number[];
  grid_export: number[];
  solar_power: number[];
  house_consumption: number[];
  point_count: number;
}

// ---- Data ----

export interface DataInfo {
  solar_file_loaded: boolean;
  load_file_loaded: boolean;
  date_range: DateRange | null;
  solar_point_count: number;
  load_point_count: number;
}

// ---- WebSocket messages ----

export interface WsProgress {
  strategy: string;
  percent: number;
}

export interface WsStrategyDone {
  strategy: string;
  status: string;
  metrics?: StrategyMetrics;
  error?: string;
}

export interface WsRunComplete {
  status: 'run_complete';
}

export type WsMessage = WsProgress | WsStrategyDone | WsRunComplete;

// ---- UI state helpers ----

export type LogLevel = 'info' | 'warn' | 'error';

export interface LogEntry {
  timestamp: number;
  message: string;
  level: LogLevel;
}

export interface StrategyRunState {
  progress: number;
  status: string;
  metrics?: StrategyMetrics;
  error?: string;
}

export interface CurrentRun {
  runId: string;
  status: string;
  results: Map<string, StrategyRunState>;
}
