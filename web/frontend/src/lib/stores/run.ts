import { derived, get, writable } from 'svelte/store';
import { ApiError, cancelRun, errorMessage, getRunSummary, getTimeseries, isUnavailable, startSimulation } from '../api/client';
import { RunSocket } from '../api/ws';
import type {
  DateRange,
  RunStatusValue,
  RunSummary,
  StrategyMetrics,
  StrategyResult,
  StrategyStatus,
  SystemConfig,
} from '../types';
import { defaults, setCapability, strategyById, strategyName } from './app';
import { draft } from './draft';
import { logActivity, toast } from './toasts';
import { fmtRange } from '../ui/format';
import { hashParams, hrefFor, isRouteHash, replaceHashParam } from './route';

export const BASELINE_ID = 'no_battery';
/** Below three days an annual figure is mostly noise. */
export const MIN_HOURS_FOR_ANNUAL = 72;

export interface RunState {
  runId: string | null;
  status: RunStatusValue | 'starting';
  range: DateRange;
  /** Includes the baseline. */
  strategyIds: string[];
  config: SystemConfig;
  results: Record<string, StrategyResult>;
  startedAt: number;
  finishedAt: number | null;
  error: string | null;
  cancelling: boolean;
  /** Loaded from history (History tab or a #/simulate?run= link) rather than started in this tab. */
  reopened?: boolean;
  createdAt?: string | null;
  label?: string | null;
  /** The server still has the results but has dropped the per-minute data (it keeps only recent runs). */
  timeseriesExpired?: boolean;
  /** False when the server didn't return the run's settings and `config` is a stand-in. */
  configFromRun?: boolean;
  /** Shown because of a History or link action, so Simulate's period and strategy controls follow it once. */
  adoptControls?: boolean;
}

export const run = writable<RunState | null>(null);

let socket: RunSocket | null = null;
let poll: ReturnType<typeof setInterval> | null = null;
let generation = 0;

function stopWatching(): void {
  socket?.close();
  socket = null;
  if (poll) clearInterval(poll);
  poll = null;
}

function patchStrategy(id: string, patch: Partial<StrategyResult>): void {
  run.update((r) => {
    if (!r) return r;
    const prev = r.results[id] ?? { strategy_id: id, status: 'pending' as StrategyStatus };
    // A late progress tick must never downgrade a finished strategy.
    if (isFinal(prev.status) && patch.status && !isFinal(patch.status)) return r;
    return { ...r, results: { ...r.results, [id]: { ...prev, ...patch } } };
  });
}

const isFinal = (s: StrategyStatus) => s === 'completed' || s === 'failed' || s === 'cancelled';

function applySummary(s: RunSummary): void {
  run.update((r) => {
    if (!r || r.runId !== s.run_id) return r;
    const results = { ...r.results };
    for (const sr of s.strategies) {
      const prev = results[sr.strategy_id];
      const progress = sr.progress ?? (isFinal(sr.status) ? 100 : prev?.progress ?? 0);
      if (prev && isFinal(prev.status) && !isFinal(sr.status)) continue;
      results[sr.strategy_id] = { ...prev, ...sr, progress };
    }
    return { ...r, results };
  });
  if (s.status !== 'running') finish(s.status);
}

function finish(status: RunStatusValue): void {
  const r = get(run);
  if (!r || r.finishedAt) return;
  stopWatching();
  run.set({ ...r, status, finishedAt: Date.now(), cancelling: false });
  const failed = Object.values(r.results).filter((x) => x.status === 'failed');
  const secs = ((Date.now() - r.startedAt) / 1000).toFixed(1);
  if (status === 'cancelled') toast('info', 'Simulation cancelled');
  else if (failed.length) {
    toast('warning', `Simulation finished with ${failed.length} failed`, failed.map((f) => strategyName(f.strategy_id)).join(', '));
  } else logActivity(`Simulation finished in ${secs} s`, 'success');
}

async function pollOnce(runId: string, gen: number): Promise<void> {
  try {
    const s = await getRunSummary(runId);
    if (gen === generation) applySummary(s);
  } catch (err) {
    if (gen !== generation) return;
    // A 404 means the server forgot the run (restart); keep polling through blips otherwise.
    if (err instanceof ApiError && err.status === 404) {
      stopWatching();
      run.update((r) => (r ? { ...r, status: 'failed', error: errorMessage(err), finishedAt: Date.now() } : r));
      toast('error', 'Simulation lost', errorMessage(err));
    }
  }
}

export async function startRun(opts: {
  range: DateRange;
  strategies: string[];
  config: SystemConfig;
  label?: string;
  adoptControls?: boolean;
}): Promise<void> {
  stopWatching();
  remember(null);
  setRunParam(null);
  loadingId = null;
  openingRun.set(null);
  const gen = ++generation;
  const ids = [BASELINE_ID, ...opts.strategies.filter((s) => s !== BASELINE_ID)];
  const results: Record<string, StrategyResult> = {};
  for (const id of ids) results[id] = { strategy_id: id, status: 'pending', progress: 0 };
  run.set({
    runId: null,
    status: 'starting',
    range: opts.range,
    strategyIds: ids,
    config: opts.config,
    results,
    startedAt: Date.now(),
    finishedAt: null,
    error: null,
    cancelling: false,
    label: opts.label?.trim() || null,
    adoptControls: opts.adoptControls,
  });
  logActivity(`Simulating ${ids.length} strategies, ${fmtRange(opts.range.start, opts.range.end)}`);

  let runId: string;
  try {
    const res = await startSimulation({ strategies: ids, date_range: opts.range, config: opts.config, label: opts.label });
    runId = res.run_id;
  } catch (err) {
    if (gen !== generation) return;
    run.update((r) => (r ? { ...r, status: 'failed', error: errorMessage(err), finishedAt: Date.now() } : r));
    toast('error', 'Could not start the simulation', errorMessage(err));
    return;
  }
  if (gen !== generation) return;
  run.update((r) => (r ? { ...r, runId, status: 'running' } : r));
  if (isSimulateHash()) setRunParam(runId);
  remember({ runId, range: opts.range, strategyIds: ids, config: opts.config, startedAt: get(run)?.startedAt ?? Date.now() });
  watch(runId, gen);
}

function watch(runId: string, gen: number): void {
  socket = new RunSocket(runId, {
    onProgress: (m) => gen === generation && patchStrategy(m.strategy, { status: 'running', progress: m.percent }),
    onStrategyDone: (m) =>
      gen === generation &&
      patchStrategy(m.strategy, { status: m.status, metrics: m.metrics ?? null, error: m.error ?? null, progress: 100 }),
    // Fetch the summary once more so the final state comes from one consistent source.
    onRunComplete: () => gen === generation && void pollOnce(runId, gen),
  });
  poll = setInterval(() => void pollOnce(runId, gen), 2000);
}

// ---- Reattach after a reload ----

/*
 * The summary endpoint doesn't return the run's date range or config yet, so what the results
 * view needs besides the summary is kept per tab in sessionStorage.
 */
const LAST_RUN_KEY = 'pf.lastRun.v1';

interface LastRun {
  runId: string;
  range: DateRange;
  strategyIds: string[];
  config: SystemConfig;
  startedAt: number;
  reopened?: boolean;
}

function remember(v: LastRun | null): void {
  try {
    if (v) sessionStorage.setItem(LAST_RUN_KEY, JSON.stringify(v));
    else sessionStorage.removeItem(LAST_RUN_KEY);
  } catch {
    // Storage unavailable: a reload just starts empty.
  }
}

function recall(): LastRun | null {
  try {
    const raw = sessionStorage.getItem(LAST_RUN_KEY);
    const v = raw ? (JSON.parse(raw) as Partial<LastRun>) : null;
    if (
      v &&
      typeof v.runId === 'string' &&
      v.range &&
      typeof v.range.start === 'string' &&
      typeof v.range.end === 'string' &&
      Array.isArray(v.strategyIds) &&
      v.config &&
      typeof v.startedAt === 'number'
    ) {
      return v as LastRun;
    }
  } catch {
    // Unreadable entry: treat as absent.
  }
  return null;
}

/** Show this tab's last run again after a reload, and keep following it if it is still running. */
export async function resumeLastRun(): Promise<void> {
  listenForRunParam();
  const linked = isSimulateHash() ? hashParams().get(RUN_PARAM) : null;
  // A link names the run explicitly, so it wins over whatever this tab showed last.
  if (linked) {
    // A reload of a run started in this tab shouldn't present it as reopened from history.
    const own = recall();
    await loadRun(linked, { reopened: !(own?.runId === linked && !own.reopened) });
    return;
  }
  const last = recall();
  if (!last || get(run)) return;
  const gen = ++generation;
  let s: RunSummary;
  let expired = false;
  try {
    s = await getRunSummary(last.runId);
    if (s.status !== 'running') expired = await timeseriesExpired(last.runId, s);
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) remember(null);
    return;
  }
  if (gen !== generation || get(run)) return;
  const results: Record<string, StrategyResult> = {};
  for (const id of last.strategyIds) results[id] = { strategy_id: id, status: 'pending', progress: 0 };
  for (const sr of s.strategies) {
    results[sr.strategy_id] = { ...sr, progress: sr.progress ?? (isFinal(sr.status) ? 100 : 0) };
  }
  const running = s.status === 'running';
  run.set({
    runId: last.runId,
    status: s.status,
    range: s.date_range ?? last.range,
    strategyIds: last.strategyIds,
    config: s.config ?? last.config,
    results,
    startedAt: last.startedAt,
    finishedAt: running ? null : last.startedAt,
    error: null,
    cancelling: false,
    reopened: last.reopened ?? false,
    createdAt: s.created_at ?? null,
    label: s.label ?? null,
    timeseriesExpired: expired,
    configFromRun: s.config != null,
  });
  setRunParam(last.runId);
  if (running) {
    logActivity('Reattached to the running simulation');
    watch(last.runId, gen);
  }
}

export async function cancelCurrentRun(): Promise<void> {
  const r = get(run);
  if (!r?.runId || r.finishedAt || r.cancelling) return;
  run.set({ ...r, cancelling: true });
  try {
    await cancelRun(r.runId);
    logActivity('Cancel requested');
  } catch (err) {
    run.update((x) => (x ? { ...x, cancelling: false } : x));
    if (isUnavailable(err)) {
      setCapability('cancel', 'unavailable');
      toast('info', 'Cancel needs a backend update');
    } else toast('error', 'Could not cancel', errorMessage(err));
  }
}

// ---- Derived results ----

export interface Ranked {
  id: string;
  status: StrategyStatus;
  error: string | null;
  metrics: StrategyMetrics | null;
  /** adjusted_cost, falling back to total_cost. */
  cost: number | null;
  rawCost: number | null;
  savings: number | null;
  savingsPerYear: number | null;
  cyclesPerDay: number | null;
  validHours: number | null;
  isBaseline: boolean;
  progress: number;
}

export const effectiveCost = (m: StrategyMetrics | null | undefined): number | null =>
  m ? (m.adjusted_cost ?? m.total_cost ?? null) : null;

export function annualise(v: number | null, validHours: number | null): number | null {
  if (v == null || !validHours || validHours < MIN_HOURS_FOR_ANNUAL) return null;
  return (v * 8760) / validHours;
}

export function rankResults(r: RunState): Ranked[] {
  const base = r.results[BASELINE_ID];
  const baseCost = base?.status === 'completed' ? effectiveCost(base.metrics) : null;
  const rows: Ranked[] = r.strategyIds.map((id) => {
    const res = r.results[id];
    const m = res?.metrics ?? null;
    const cost = res?.status === 'completed' ? effectiveCost(m) : null;
    const vh = m?.valid_hours ?? null;
    const savings = cost != null && baseCost != null && id !== BASELINE_ID ? baseCost - cost : null;
    const cycles = m?.equivalent_full_cycles ?? null;
    return {
      id,
      status: res?.status ?? 'pending',
      error: res?.error ?? null,
      metrics: m,
      cost,
      rawCost: m?.total_cost ?? null,
      savings,
      savingsPerYear: annualise(savings, vh),
      cyclesPerDay: cycles != null && vh ? cycles / (vh / 24) : null,
      validHours: vh,
      isBaseline: id === BASELINE_ID,
      progress: res?.progress ?? 0,
    };
  });
  return rows.sort((a, b) => {
    if (a.cost == null && b.cost == null) return 0;
    if (a.cost == null) return 1;
    if (b.cost == null) return -1;
    return a.cost - b.cost;
  });
}

export const ranked = derived(run, ($r) => ($r ? rankResults($r) : []));

/** Savings under half a cent display as €0.00, so they count as a tie, not a win or a loss. */
export const TIE_EUR = 0.005;

export interface Verdict {
  /** Cheapest battery strategy, described by the KPI cards even when it loses. */
  best: Ranked | null;
  /** 'win': best saves at least a cent; 'tie': within half a cent; 'loss': costs more. */
  outcome: 'win' | 'tie' | 'loss' | null;
  /** Row that gets the "Best" badge: the winning strategy, or the baseline on a tie or loss. */
  winnerId: string | null;
}

export function verdict(rows: Ranked[]): Verdict {
  const best = rows.find((r) => !r.isBaseline && r.cost != null) ?? null;
  if (!best || best.savings == null) return { best, outcome: null, winnerId: null };
  const outcome = best.savings >= TIE_EUR ? 'win' : best.savings > -TIE_EUR ? 'tie' : 'loss';
  return { best, outcome, winnerId: outcome === 'win' ? best.id : BASELINE_ID };
}


export const runBusy = derived(run, ($r) => Boolean($r && !$r.finishedAt));

export const runProgress = derived(run, ($r) => {
  if (!$r) return 0;
  const vals = $r.strategyIds.map((id) => {
    const s = $r.results[id];
    return !s ? 0 : isFinal(s.status) ? 100 : s.progress ?? 0;
  });
  return vals.reduce((a, b) => a + b, 0) / Math.max(1, vals.length);
});

// ---- Reopen a past run (History tab, #/simulate?run=<id> links) ----

/** Hash query parameter naming the run shown on Simulate: `#/simulate?run=<id>`. */
export const RUN_PARAM = 'run';

/** Id of the run being fetched for reopening, or null. */
export const openingRun = writable<string | null>(null);
let loadingId: string | null = null;

const isSimulateHash = () => isRouteHash('simulate');

/** Records the run on screen in the URL, but only while the Simulate tab is the one showing. */
function setRunParam(id: string | null): void {
  if (isSimulateHash()) replaceHashParam(RUN_PARAM, id);
}

function onHashChange(): void {
  if (!isSimulateHash()) return;
  const id = hashParams().get(RUN_PARAM);
  const shown = get(run)?.runId ?? null;
  // A bare #/simulate (tab link) shows whatever is on screen, so name it for sharing and reloads.
  if (!id) {
    if (shown && loadingId == null) setRunParam(shown);
    return;
  }
  if (id === loadingId || shown === id) return;
  void loadRun(id);
}

let listening = false;
function listenForRunParam(): void {
  if (listening) return;
  listening = true;
  window.addEventListener('hashchange', onHashChange);
}
// Registered at import, ahead of the route store's listener: Svelte re-renders between listeners,
// so Simulate must already see the run being opened when it mounts, or it fetches the old run's charts.
if (typeof window !== 'undefined') listenForRunParam();

if (import.meta.hot) {
  import.meta.hot.dispose(() => window.removeEventListener('hashchange', onHashChange));
}

/** One cheap request tells whether the server still has this run's per-minute data. */
async function timeseriesExpired(runId: string, s: RunSummary): Promise<boolean> {
  const done = s.strategies.filter((x) => x.status === 'completed');
  const probe = done.find((x) => x.strategy_id !== BASELINE_ID) ?? done[0];
  if (!probe) return false;
  try {
    await getTimeseries(runId, probe.strategy_id, { maxPoints: 10 });
    return false;
  } catch (err) {
    return err instanceof ApiError && err.status === 410;
  }
}

/** Show a stored run on Simulate (without navigating). Resolves to false when it can't be opened. */
export async function loadRun(runId: string, opts: { reopened?: boolean } = {}): Promise<boolean> {
  const reopened = opts.reopened ?? true;
  const prev = get(run);
  if (prev?.runId === runId) return true;
  stopWatching();
  const gen = ++generation;
  loadingId = runId;
  openingRun.set(runId);
  try {
    const s = await getRunSummary(runId);
    const running = s.status === 'running';
    const expired = running ? false : await timeseriesExpired(runId, s);
    if (gen !== generation) return false;
    const config = s.config ?? get(draft) ?? get(defaults);
    if (!s.date_range || !config) throw new Error('The server did not return this run’s date range or settings.');
    const ids = [BASELINE_ID, ...s.strategies.map((x) => x.strategy_id).filter((id) => id !== BASELINE_ID)];
    const results: Record<string, StrategyResult> = {};
    for (const sr of s.strategies) {
      results[sr.strategy_id] = { ...sr, progress: sr.progress ?? (isFinal(sr.status) ? 100 : 0) };
    }
    const created = s.created_at ? Date.parse(s.created_at) : NaN;
    const startedAt = Number.isFinite(created) ? created : Date.now();
    run.set({
      runId,
      status: s.status,
      range: s.date_range,
      strategyIds: ids,
      config,
      results,
      startedAt,
      finishedAt: running ? null : startedAt,
      error: null,
      cancelling: false,
      reopened,
      createdAt: s.created_at ?? null,
      label: s.label ?? null,
      timeseriesExpired: expired,
      configFromRun: s.config != null,
      adoptControls: reopened,
    });
    remember({ runId, range: s.date_range, strategyIds: ids, config, startedAt, reopened });
    if (running) watch(runId, gen);
    return true;
  } catch (err) {
    if (gen !== generation) return false;
    // The link keeps naming the run that stays on screen, so it can still be copied and shared.
    if (hashParams().get(RUN_PARAM) === runId) setRunParam(get(run)?.runId ?? null);
    // Keep following the run that stays on screen.
    if (prev?.runId && !prev.finishedAt) watch(prev.runId, gen);
    toast('error', `Couldn't open run ${runId}`, errorMessage(err));
    return false;
  } finally {
    if (gen === generation) {
      loadingId = null;
      openingRun.set(null);
    }
  }
}

/**
 * Run a stored run again with its own date range, strategies and settings, and show it on
 * Simulate. The baseline is added back by startRun; strategies that are no longer available
 * are dropped.
 */
export async function rerun(runId: string): Promise<void> {
  let s: RunSummary;
  try {
    s = await getRunSummary(runId);
  } catch (err) {
    toast('error', 'Could not re-run', errorMessage(err));
    return;
  }
  const range = s.date_range;
  if (!range) {
    toast('error', 'Could not re-run', 'The server did not return this run’s date range.');
    return;
  }
  const known = get(strategyById);
  const ids = s.strategies.map((x) => x.strategy_id).filter((id) => id !== BASELINE_ID);
  const usable = ids.filter((id) => known.get(id)?.available !== false);
  if (usable.length < ids.length) {
    toast('warning', 'Some strategies were left out', `${ids.filter((id) => !usable.includes(id)).map(strategyName).join(', ')} not available any more`);
  }
  let config = s.config ?? null;
  if (!config) {
    config = get(draft);
    if (!config) {
      toast('error', 'Could not re-run', 'Settings have not loaded yet.');
      return;
    }
    toast('warning', 'Original settings unavailable', 'Re-running with the current settings instead.');
  }
  location.hash = hrefFor('simulate');
  await startRun({ range, strategies: usable, config: structuredClone(config), label: s.label ?? undefined, adoptControls: true });
}

/** After a run was deleted on the server: stop showing it and stop restoring it on reload. */
export function forgetRun(runId: string): void {
  const r = get(run);
  if (r?.runId === runId) {
    if (r.finishedAt) {
      stopWatching();
      ++generation;
      run.set(null);
    }
    remember(null);
  }
  if (hashParams().get(RUN_PARAM) === runId) setRunParam(null);
}
