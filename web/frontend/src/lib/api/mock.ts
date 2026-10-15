// In-memory stand-ins for the v2 endpoints, active only with ?mock=1 or VITE_MOCK=1.
import type {
  BacktestRequest,
  BacktestResult,
  BatteryConfig,
  ExportKind,
  GridConfig,
  PresetsResponse,
  RefreshJob,
  RefreshStartResponse,
  RunListItem,
  SweepCase,
  SweepListItem,
  SweepRequest,
  SweepResult,
  TariffConfig,
  TariffRate,
} from '../types';

const delay = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

const RATES: TariffRate[] = [
  { start_hour: 0, end_hour: 8, price: 0.085, direction: 'import' },
  { start_hour: 8, end_hour: 10, price: 0.135, direction: 'import' },
  { start_hour: 10, end_hour: 14, price: 0.197, direction: 'import' },
  { start_hour: 14, end_hour: 18, price: 0.135, direction: 'import' },
  { start_hour: 18, end_hour: 22, price: 0.197, direction: 'import' },
  { start_hour: 22, end_hour: 24, price: 0.135, direction: 'import' },
  { start_hour: 0, end_hour: 24, price: 0.08, direction: 'export' },
];

const tariff = (over: Partial<TariffConfig>): TariffConfig => ({
  rates: RATES.map((r) => ({ ...r })),
  weekend_rate: { start_hour: 0, end_hour: 24, price: 0.085, direction: 'import' },
  use_pvpc: false,
  export_follows_pvpc: false,
  export_pvpc_discount: 0,
  mpc_horizon_hours: 24,
  ...over,
});

const battery = (over: Partial<BatteryConfig>): BatteryConfig => ({
  capacity: 15.36,
  max_charge_rate: 4.5,
  max_discharge_rate: 4.6,
  efficiency: 0.9,
  initial_soc: 0.5,
  taper_start: 0.9,
  taper_factor: 0.7,
  cycle_cost_eur_kwh: 0,
  ...over,
});

const grid: GridConfig = { max_import: 5.5, max_export: 5.5 };

const PRESETS: PresetsResponse = {
  hardware: [
    {
      id: 'current',
      name: 'Current (4.5 / 4.6 kW)',
      description: 'GoodWe 15.36 kWh, charge 4.5 kW, discharge 4.6 kW since 7 Apr 2026.',
      battery: battery({}),
      grid: { ...grid },
      valid_from: '2026-04-07',
      valid_until: null,
    },
    {
      id: 'pre_2026_04',
      name: 'Before Apr 2026 (2.2 / 2.1 kW)',
      description: 'Same battery with the earlier inverter limits.',
      battery: battery({ max_charge_rate: 2.2, max_discharge_rate: 2.1 }),
      grid: { ...grid },
      valid_from: null,
      valid_until: '2026-04-06',
    },
    {
      id: 'double_battery',
      name: 'Double battery (30.72 kWh)',
      description: 'Two battery stacks, current inverter limits.',
      battery: battery({ capacity: 30.72 }),
      grid: { ...grid },
      valid_from: null,
      valid_until: null,
    },
  ],
  tariffs: [
    { id: '2_0td', name: '2.0TD time-of-use', description: 'Valley, flat and peak periods; weekends all valley.', tariff: tariff({}) },
    { id: 'pvpc', name: 'PVPC hourly', description: 'Hourly regulated prices, flat export rate.', tariff: tariff({ use_pvpc: true }) },
    {
      id: 'pvpc_indexed_export',
      name: 'PVPC, indexed export',
      description: 'Hourly prices for import and export, export minus €0.02/kWh.',
      tariff: tariff({ use_pvpc: true, export_follows_pvpc: true, export_pvpc_discount: 0.02 }),
    },
  ],
};

const runs: RunListItem[] = [];
const refreshJobs = new Map<string, { started: number; full: boolean }>();
const sweeps = new Map<string, { req: SweepRequest; started: number; cancelled: boolean; created: string }>();

function csvBlob(rows: (string | number)[][]): Blob {
  return new Blob([rows.map((r) => r.join(',')).join('\n') + '\n'], { type: 'text/csv' });
}

function sweepCases(req: SweepRequest, done: number): SweepCase[] {
  const out: SweepCase[] = [];
  let i = 0;
  for (const t of req.tariffs) {
    const baseline = 900 + t.length * 20;
    out.push({
      tariff: t, capacity_kwh: 0, strategy_id: 'no_battery', status: 'completed',
      total_cost: baseline, adjusted_cost: baseline, savings_vs_baseline: 0, annualised_savings: 0,
      equivalent_full_cycles: 0, self_consumption_rate: 52,
    });
    for (const c of req.capacities_kwh) {
      for (const s of req.strategies) {
        if (i++ >= done) return out;
        const save = Math.min(c, 30) * (8 + (s.length % 5)) * (t.includes('pvpc') ? 1.15 : 1);
        out.push({
          tariff: t, capacity_kwh: c, strategy_id: s, status: 'completed',
          total_cost: baseline - save, adjusted_cost: baseline - save, savings_vs_baseline: save,
          annualised_savings: save, equivalent_full_cycles: 250 - c * 2, self_consumption_rate: 70 + c / 2,
        });
      }
    }
  }
  return out;
}

export const mock = {
  async getPresets(): Promise<PresetsResponse> {
    await delay(120);
    return structuredClone(PRESETS);
  },

  async cancelRun(_runId: string): Promise<{ status: string }> {
    await delay(150);
    return { status: 'cancelling' };
  },

  async listRuns(limit: number): Promise<RunListItem[]> {
    await delay(150);
    return runs.slice(0, limit);
  },

  async deleteRun(runId: string): Promise<void> {
    await delay(100);
    const i = runs.findIndex((r) => r.run_id === runId);
    if (i >= 0) runs.splice(i, 1);
  },

  async exportCsv(runId: string, kind: ExportKind, strategy?: string): Promise<{ blob: Blob; filename: string }> {
    await delay(200);
    const rows: (string | number)[][] =
      kind === 'summary'
        ? [['strategy_id', 'adjusted_cost', 'total_cost'], ['mock', 0, 0]]
        : kind === 'monthly'
          ? [['strategy_id', 'month', 'cost'], ['mock', '2026-10', 0]]
          : [['ts', 'solar_kw', 'load_kw'], ['2026-10-12T00:00:00+02:00', 0, 0.4]];
    return { blob: csvBlob(rows), filename: `run-${runId}-${kind}${strategy ? `-${strategy}` : ''}.csv` };
  },

  async startDataRefresh(full: boolean): Promise<RefreshStartResponse> {
    await delay(150);
    const job_id = `mock-${Date.now().toString(36)}`;
    refreshJobs.set(job_id, { started: Date.now(), full });
    return { job_id };
  },

  async getRefreshJob(jobId: string): Promise<RefreshJob> {
    await delay(80);
    const job = refreshJobs.get(jobId);
    if (!job) throw new Error(`Refresh job ${jobId} not found`);
    const pct = Math.min(100, Math.round(((Date.now() - job.started) / 6000) * 100));
    return {
      job_id: jobId,
      status: pct >= 100 ? 'completed' : 'running',
      progress_pct: pct,
      message: pct < 40 ? 'Fetching InfluxDB series' : pct < 80 ? 'Writing parquet store' : pct < 100 ? 'Fetching irradiance' : 'Up to date',
      summary: pct >= 100 ? { new_rows: 1440 } : null,
    };
  },

  async startSweep(req: SweepRequest): Promise<{ sweep_id: string; total_cases: number }> {
    await delay(150);
    const sweep_id = `sw-${Date.now().toString(36)}`;
    sweeps.set(sweep_id, { req, started: Date.now(), cancelled: false, created: new Date().toISOString() });
    return { sweep_id, total_cases: req.tariffs.length * (req.capacities_kwh.length * req.strategies.length + 1) };
  },

  async getSweep(sweepId: string): Promise<SweepResult> {
    await delay(100);
    const s = sweeps.get(sweepId);
    if (!s) throw new Error(`Sweep ${sweepId} not found`);
    const total = s.req.tariffs.length * s.req.capacities_kwh.length * s.req.strategies.length;
    const done = Math.min(total, Math.floor((Date.now() - s.started) / 400));
    const status = s.cancelled ? 'cancelled' : done >= total ? 'completed' : 'running';
    return {
      sweep_id: sweepId, label: s.req.label ?? null, created_at: s.created, status, date_range: s.req.date_range,
      done: done + s.req.tariffs.length, total: total + s.req.tariffs.length, cases: sweepCases(s.req, done),
    };
  },

  async cancelSweep(sweepId: string): Promise<{ status: string }> {
    const s = sweeps.get(sweepId);
    if (s) s.cancelled = true;
    return { status: 'cancelled' };
  },

  async listSweeps(limit: number): Promise<SweepListItem[]> {
    await delay(100);
    return [...sweeps.entries()].slice(0, limit).map(([id, s]) => ({
      sweep_id: id, label: s.req.label ?? null, created_at: s.created, status: s.cancelled ? 'cancelled' : 'completed',
      date_range: s.req.date_range, total: s.req.tariffs.length * (s.req.capacities_kwh.length * s.req.strategies.length + 1),
    }));
  },

  async runBacktest(req: BacktestRequest): Promise<BacktestResult> {
    await delay(600);
    const start = new Date(`${req.date_range.start}T00:00:00`);
    const end = new Date(`${req.date_range.end}T00:00:00`);
    const daily: BacktestResult['daily'] = [];
    for (let d = new Date(start), i = 0; d <= end; d.setDate(d.getDate() + 1), i++) {
      const mi = 8 + Math.sin(i / 3) * 3;
      const me = 6 + Math.cos(i / 4) * 2;
      const iso = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
      daily.push({
        date: iso, measured_import_kwh: mi, sim_import_kwh: mi * 0.95, measured_export_kwh: me, sim_export_kwh: me * 1.04,
        measured_cost: mi * 0.14 - me * 0.08, sim_cost: mi * 0.95 * 0.14 - me * 1.04 * 0.08,
      });
    }
    const ts: number[] = [];
    const m: (number | null)[] = [];
    const s: number[] = [];
    for (let t = start.getTime(), i = 0; t <= end.getTime() + 86_400_000; t += 900_000, i++) {
      ts.push(t);
      const v = 50 + 40 * Math.sin(i / 20);
      m.push(i % 300 < 6 ? null : v);
      s.push(v + 3 * Math.sin(i / 7));
    }
    return {
      strategy_id: req.strategy_id ?? 'self_consume',
      hardware_used: [{ preset_id: 'current', from: req.date_range.start, to: req.date_range.end }],
      daily,
      soc: { ts, measured_pct: m, sim_pct: s },
      metrics: {
        import_error_pct: -5, export_error_pct: 4, cost_error_pct: -6.5, daily_import_mae_kwh: 0.6, soc_rmse_pct: 4.1,
        measured_cost: daily.reduce((a, b) => a + b.measured_cost, 0), sim_cost: daily.reduce((a, b) => a + b.sim_cost, 0),
      },
    };
  },
};
