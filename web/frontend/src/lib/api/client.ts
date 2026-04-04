import type {
  ConfigResponse,
  DataInfo,
  RunSummary,
  SimulationRequest,
  SimulationStartResponse,
  StrategyInfo,
  SystemConfig,
  TimeseriesResponse,
} from '../types/index';

/**
 * Base URL for API requests.
 * Empty string works with both the Vite dev proxy (/api/* -> localhost:8000)
 * and production mode (FastAPI serves frontend static files).
 */
const BASE = '';

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.text();
    throw new ApiError(res.status, `${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

function json(body: unknown): RequestInit {
  return {
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  };
}

// ---- Strategies ----

export async function getStrategies(): Promise<StrategyInfo[]> {
  return request<StrategyInfo[]>('/api/strategies');
}

// ---- Config ----

export async function getConfig(): Promise<ConfigResponse> {
  return request<ConfigResponse>('/api/config');
}

export async function updateConfig(config: SystemConfig): Promise<ConfigResponse> {
  return request<ConfigResponse>('/api/config', {
    method: 'PUT',
    ...json(config),
  });
}

// ---- Simulation ----

export async function startSimulation(req: SimulationRequest): Promise<SimulationStartResponse> {
  return request<SimulationStartResponse>('/api/simulate', {
    method: 'POST',
    ...json(req),
  });
}

export async function getRunSummary(runId: string): Promise<RunSummary> {
  return request<RunSummary>(`/api/runs/${encodeURIComponent(runId)}/summary`);
}

export async function getTimeseries(
  runId: string,
  strategy: string,
  opts?: { start?: string; end?: string; maxPoints?: number },
): Promise<TimeseriesResponse> {
  const params = new URLSearchParams();
  params.set('strategy', strategy);
  if (opts?.start) params.set('start', opts.start);
  if (opts?.end) params.set('end', opts.end);
  if (opts?.maxPoints) params.set('max_points', String(opts.maxPoints));
  return request<TimeseriesResponse>(
    `/api/runs/${encodeURIComponent(runId)}/timeseries?${params}`,
  );
}

// ---- Data ----

export async function getDataInfo(): Promise<DataInfo> {
  return request<DataInfo>('/api/data/info');
}

export async function uploadData(files: {
  solarFile?: File;
  loadFile?: File;
}): Promise<DataInfo> {
  const form = new FormData();
  if (files.solarFile) form.append('solar_file', files.solarFile);
  if (files.loadFile) form.append('load_file', files.loadFile);
  return request<DataInfo>('/api/data/upload', {
    method: 'POST',
    body: form,
  });
}

// ---- Health ----

export async function healthCheck(): Promise<{ status: string }> {
  return request<{ status: string }>('/api/health');
}
