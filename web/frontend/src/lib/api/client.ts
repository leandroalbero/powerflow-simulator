import type {
  BacktestRequest,
  BacktestResult,
  ConfigResponse,
  DataInfo,
  ExportKind,
  PresetsResponse,
  RefreshJob,
  RefreshStartResponse,
  RunListItem,
  RunSummary,
  SimulationRequest,
  SimulationStartResponse,
  StrategyInfo,
  SweepListItem,
  SweepRequest,
  SweepResult,
  Timeseries,
} from '../types';
import { getBlob, getJson, probeStatus, sendJson } from './http';
import { adaptTimeseries } from './timeseries';
import type { TimeseriesResponse } from '../types';
import { mock } from './mock';

export { ApiError, errorMessage, isUnavailable } from './http';

function readMockFlag(): boolean {
  try {
    if (import.meta.env.VITE_MOCK === '1') return true;
    if (new URLSearchParams(location.search).get('mock') === '1') return true;
    const q = location.hash.indexOf('?');
    return q >= 0 && new URLSearchParams(location.hash.slice(q + 1)).get('mock') === '1';
  } catch {
    return false;
  }
}

/** Mocks replace only the endpoints that the v2 contract adds; existing ones are always real. */
export const MOCK_MODE = readMockFlag();

const enc = encodeURIComponent;

// ---- Existing endpoints (always real) ----

export const getStrategies = () => getJson<StrategyInfo[]>('/api/strategies');
export const getConfig = () => getJson<ConfigResponse>('/api/config');
export const getDataInfo = () => getJson<DataInfo>('/api/data/info');
export const reloadData = () => sendJson<DataInfo>('POST', '/api/data/reload');

export const startSimulation = (req: SimulationRequest) =>
  sendJson<SimulationStartResponse>('POST', '/api/simulate', req);

export const getRunSummary = (runId: string, signal?: AbortSignal) =>
  getJson<RunSummary>(`/api/runs/${enc(runId)}/summary`, signal);

export async function getTimeseries(
  runId: string,
  strategy: string,
  opts: { start?: number; end?: number; maxPoints?: number; signal?: AbortSignal } = {},
): Promise<Timeseries> {
  const p = new URLSearchParams({ strategy, max_points: String(opts.maxPoints ?? 2000) });
  if (opts.start != null) p.set('start', new Date(opts.start).toISOString());
  if (opts.end != null) p.set('end', new Date(opts.end).toISOString());
  const r = await getJson<TimeseriesResponse>(`/api/runs/${enc(runId)}/timeseries?${p}`, opts.signal);
  return adaptTimeseries(r);
}

// ---- New endpoints (mocked only in mock mode) ----

export const getPresets = (): Promise<PresetsResponse> =>
  MOCK_MODE ? mock.getPresets() : getJson<PresetsResponse>('/api/presets');

export const cancelRun = (runId: string): Promise<{ status: string }> =>
  MOCK_MODE ? mock.cancelRun(runId) : sendJson('POST', `/api/runs/${enc(runId)}/cancel`);

export const listRuns = (limit = 50): Promise<RunListItem[]> =>
  MOCK_MODE ? mock.listRuns(limit) : getJson<RunListItem[]>(`/api/runs?limit=${limit}`);

export const deleteRun = (runId: string): Promise<void> =>
  MOCK_MODE ? mock.deleteRun(runId) : sendJson<void>('DELETE', `/api/runs/${enc(runId)}`);

export function exportCsv(
  runId: string,
  kind: ExportKind,
  strategy?: string,
): Promise<{ blob: Blob; filename: string | null }> {
  if (MOCK_MODE) return mock.exportCsv(runId, kind, strategy);
  const p = new URLSearchParams({ kind });
  if (strategy) p.set('strategy', strategy);
  return getBlob(`/api/runs/${enc(runId)}/export.csv?${p}`);
}

export const startDataRefresh = (full = false): Promise<RefreshStartResponse> =>
  MOCK_MODE ? mock.startDataRefresh(full) : sendJson('POST', '/api/data/refresh', { full });

export const getRefreshJob = (jobId: string): Promise<RefreshJob> =>
  MOCK_MODE ? mock.getRefreshJob(jobId) : getJson<RefreshJob>(`/api/data/refresh/${enc(jobId)}`);

export const startSweep = (req: SweepRequest): Promise<{ sweep_id: string; total_cases: number }> =>
  MOCK_MODE ? mock.startSweep(req) : sendJson('POST', '/api/sweep', req);

export const getSweep = (sweepId: string): Promise<SweepResult> =>
  MOCK_MODE ? mock.getSweep(sweepId) : getJson<SweepResult>(`/api/sweep/${enc(sweepId)}`);

export const cancelSweep = (sweepId: string): Promise<{ status: string }> =>
  MOCK_MODE ? mock.cancelSweep(sweepId) : sendJson('POST', `/api/sweep/${enc(sweepId)}/cancel`);

export const listSweeps = (limit = 20): Promise<SweepListItem[]> =>
  MOCK_MODE ? mock.listSweeps(limit) : getJson<SweepListItem[]>(`/api/sweeps?limit=${limit}`);

export function sweepExportUrl(sweepId: string): string {
  return `/api/sweep/${enc(sweepId)}/export.csv`;
}

export const runBacktest = (req: BacktestRequest): Promise<BacktestResult> =>
  MOCK_MODE ? mock.runBacktest(req) : sendJson('POST', '/api/backtest', req);

// ---- Capability probes ----

export type Capability = 'presets' | 'cancel' | 'runs' | 'export' | 'refresh' | 'sweep' | 'backtest';

const PROBE_ID = '__probe__';

/**
 * Detects which v2 routes exist without side effects. Probes always use the route's real method:
 * when the built frontend is mounted at "/", a method mismatch falls through to the static
 * files mount and looks like a missing route. A missing route answers 404 "Not Found" (or 405
 * from the static mount for POST); a known route answers with its own 404 message (unknown id)
 * or a 422 (empty body), which both mean the route exists.
 */
const PROBES: Record<Capability, { path: string; method?: 'POST'; body?: unknown }> = {
  presets: { path: '/api/presets' },
  cancel: { path: `/api/runs/${PROBE_ID}/cancel`, method: 'POST' },
  runs: { path: '/api/runs?limit=1' },
  export: { path: `/api/runs/${PROBE_ID}/export.csv?kind=summary` },
  // Never POST /api/data/refresh as a probe: that would start a real refresh.
  refresh: { path: `/api/data/refresh/${PROBE_ID}` },
  sweep: { path: '/api/sweeps?limit=1' },
  backtest: { path: '/api/backtest', method: 'POST', body: {} },
};

/** The route each capability needs, as it appears in FastAPI's OpenAPI document. */
const ROUTES: Record<Capability, string> = {
  presets: 'GET /api/presets',
  cancel: 'POST /api/runs/{}/cancel',
  runs: 'GET /api/runs',
  export: 'GET /api/runs/{}/export.csv',
  refresh: 'GET /api/data/refresh/{}',
  sweep: 'GET /api/sweeps',
  backtest: 'POST /api/backtest',
};

/**
 * Routes the backend declares, from its OpenAPI document; null when that isn't served.
 * One successful request instead of a probe per capability, so missing routes don't fill the
 * console with 404s on every check.
 */
export async function fetchRouteList(): Promise<Set<string> | null> {
  if (MOCK_MODE) return null;
  try {
    const res = await fetch('/openapi.json', { headers: { Accept: 'application/json' } });
    if (!res.ok || !(res.headers.get('Content-Type') ?? '').includes('json')) return null;
    const doc: unknown = await res.json();
    if (!doc || typeof doc !== 'object' || !('paths' in doc) || typeof doc.paths !== 'object' || !doc.paths) return null;
    const out = new Set<string>();
    for (const [path, ops] of Object.entries(doc.paths as Record<string, unknown>)) {
      if (!ops || typeof ops !== 'object') continue;
      const norm = path.replace(/\{[^}]*\}/g, '{}');
      for (const m of Object.keys(ops)) out.add(`${m.toUpperCase()} ${norm}`);
    }
    return out.size ? out : null;
  } catch {
    return null;
  }
}

export function routeDeclared(routes: Set<string>, cap: Capability): boolean {
  return routes.has(ROUTES[cap]);
}

export async function probeCapability(cap: Capability): Promise<boolean> {
  if (MOCK_MODE) return true;
  const probe = PROBES[cap];
  const { status, detail } = await probeStatus(probe.path, probe.method, probe.body);
  if (status === 0 || status === 405 || status === 501) return false;
  if (status === 404) return detail !== 'Not Found';
  return true;
}
