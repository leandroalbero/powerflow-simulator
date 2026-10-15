import { derived, get, writable } from 'svelte/store';
import { cancelSweep, errorMessage, getSweep, startSweep } from '../../api/client';
import { dataBounds } from '../../stores/app';
import { hashParams, isRouteHash, replaceHashParam } from '../../stores/route';
import { loadJson, saveJson } from '../../stores/storage';
import { logActivity, toast } from '../../stores/toasts';
import type { DateRange, RunStatusValue, SweepCase, SweepRequest, SweepResult } from '../../types';
import { PERIOD_OPTIONS, presetFor, rangeFor, type PeriodPreset } from '../Simulate/state';
import { caseKey, isBaseline } from './derive';
import { SweepSocket } from './sweepSocket';

// ---- Form ----

export const DEFAULT_TARIFFS = ['2_0td', 'pvpc'];
export const DEFAULT_CAPACITIES = [5, 10, 15.36, 20, 30.72];
export const DEFAULT_STRATEGIES = ['self_consume', 'charge_night', 'mpc', 'oracle'];
export const DEFAULT_HARDWARE = 'current';
export const MAX_CAPACITY = 200;
/** Above this many cases a sweep over a long period takes a long time and heats the machine. */
export const WARN_CASES = 60;
/** Backend limits (API contract): beyond these it answers 422. */
export const MAX_CASES = 200;
export const MAX_CAPACITIES = 12;
export const MAX_DAYS = 366;

interface Persisted {
  period: PeriodPreset;
  custom: DateRange | null;
  tariffs: string[];
  capacities: number[];
  strategies: string[];
  hardware: string;
  excludeEv: boolean;
  label: string;
}

const KEY = 'pf.compare.v1';
const SWEEP_KEY = 'pf.compare.sweep';
const stored = loadJson<Partial<Persisted>>(KEY);

const strings = (v: unknown): v is string[] => Array.isArray(v) && v.every((x) => typeof x === 'string');
const numbers = (v: unknown): v is number[] =>
  Array.isArray(v) && v.every((x) => typeof x === 'number' && Number.isFinite(x) && x > 0 && x <= MAX_CAPACITY);

export const period = writable<PeriodPreset>(
  PERIOD_OPTIONS.some((o) => o.value === stored?.period) ? (stored?.period as PeriodPreset) : 'last12m',
);
export const customRange = writable<DateRange | null>(
  stored?.custom && typeof stored.custom.start === 'string' && typeof stored.custom.end === 'string' ? stored.custom : null,
);
export const tariffs = writable<string[]>(strings(stored?.tariffs) ? stored.tariffs : DEFAULT_TARIFFS);
export const capacities = writable<number[]>(numbers(stored?.capacities) && stored.capacities.length ? stored.capacities : DEFAULT_CAPACITIES);
export const strategies = writable<string[]>(strings(stored?.strategies) ? stored.strategies : DEFAULT_STRATEGIES);
export const hardware = writable<string>(typeof stored?.hardware === 'string' ? stored.hardware : DEFAULT_HARDWARE);
export const excludeEv = writable<boolean>(typeof stored?.excludeEv === 'boolean' ? stored.excludeEv : false);
export const label = writable<string>(typeof stored?.label === 'string' ? stored.label : '');

const persist = derived(
  [period, customRange, tariffs, capacities, strategies, hardware, excludeEv, label],
  ([p, c, t, cap, s, h, ev, l]): Persisted => ({ period: p, custom: c, tariffs: t, capacities: cap, strategies: s, hardware: h, excludeEv: ev, label: l }),
);
persist.subscribe((v) => saveJson(KEY, v));

export const activeRange = derived([period, customRange, dataBounds], ([p, c, b]) => (b ? rangeFor(p, b, c) : null));

export function toggleIn(store: typeof tariffs | typeof strategies, id: string): void {
  store.update((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));
}

export function addCapacity(v: number): void {
  const r = Math.round(v * 100) / 100;
  capacities.update((c) => (c.includes(r) ? c : [...c, r].sort((a, b) => a - b)));
}

export function removeCapacity(v: number): void {
  capacities.update((c) => c.filter((x) => x !== v));
}

/** The empty state's one-click comparison. */
export function applyQuickPreset(): void {
  period.set('last12m');
  tariffs.set([...DEFAULT_TARIFFS]);
  capacities.set([...DEFAULT_CAPACITIES]);
}

// ---- Sweep ----

export type SweepPhase = 'starting' | RunStatusValue;

export interface SweepState {
  id: string | null;
  status: SweepPhase;
  label: string | null;
  createdAt: string | null;
  range: DateRange;
  done: number;
  total: number;
  cases: SweepCase[];
  error: string | null;
  /** Client clock when this session started or attached to it, for the ETA. */
  attachedAt: number;
  /** Cases already finished when this session attached, so the ETA uses only the observed rate. */
  doneAtAttach: number;
  cancelling: boolean;
}

export const sweep = writable<SweepState | null>(null);
export const sweepBusy = derived(sweep, ($s) => $s?.status === 'starting' || $s?.status === 'running');

let socket: SweepSocket | null = null;
let pollTimer: ReturnType<typeof setInterval> | null = null;
let refetchTimer: ReturnType<typeof setTimeout> | null = null;
let tracked: string | null = null;

function stopTracking(): void {
  socket?.close();
  socket = null;
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
  if (refetchTimer) clearTimeout(refetchTimer);
  refetchTimer = null;
  tracked = null;
}

function fromResult(r: SweepResult, prev: SweepState | null): SweepState {
  const same = prev?.id === r.sweep_id;
  return {
    id: r.sweep_id,
    status: r.status,
    label: r.label,
    createdAt: r.created_at,
    range: r.date_range,
    done: r.done,
    total: r.total,
    cases: r.cases,
    error: r.status === 'failed' ? r.cases.find((c) => c.error)?.error ?? null : null,
    attachedAt: same ? prev.attachedAt : Date.now(),
    doneAtAttach: same ? prev.doneAtAttach : r.done,
    cancelling: same && r.status === 'running' ? prev.cancelling : false,
  };
}

async function refetch(id: string): Promise<SweepResult | null> {
  try {
    const r = await getSweep(id);
    if (tracked !== id && get(sweep)?.id !== id) return r;
    sweep.update((s) => (s && s.id === id ? fromResult(r, s) : s));
    if (r.status !== 'running' && tracked === id) finish(r.status);
    return r;
  } catch (err) {
    if (get(sweep)?.id === id) logActivity(`Could not refresh the comparison: ${errorMessage(err)}`, 'warning');
    return null;
  }
}

function scheduleRefetch(id: string, ms = 600): void {
  if (refetchTimer) return;
  refetchTimer = setTimeout(() => {
    refetchTimer = null;
    void refetch(id);
  }, ms);
}

function finish(status: RunStatusValue): void {
  const s = get(sweep);
  stopTracking();
  if (!s) return;
  if (status === 'completed') logActivity(`Comparison finished: ${s.total} cases`, 'success');
  else if (status === 'cancelled') logActivity('Comparison cancelled');
  else if (status === 'failed') toast('error', 'Comparison failed', s.error ?? undefined);
}

function mergeCase(c: SweepCase): void {
  sweep.update((s) => {
    if (!s) return s;
    const k = caseKey(c.tariff, c.capacity_kwh, c.strategy_id);
    const i = s.cases.findIndex((x) => caseKey(x.tariff, x.capacity_kwh, x.strategy_id) === k);
    const cases = [...s.cases];
    if (i >= 0) cases[i] = c;
    else cases.push(c);
    return { ...s, cases };
  });
}

function track(id: string): void {
  stopTracking();
  tracked = id;
  socket = new SweepSocket(id, {
    onCase: (c) => {
      mergeCase(c);
      // The backend marks a case completed a moment before it fills in the savings, so a
      // case can arrive without them; it is sent only once, so fetch the full state again.
      if (c.status === 'completed' && !isBaseline(c) && c.savings_vs_baseline == null) scheduleRefetch(id);
    },
    onProgress: (done, total) =>
      sweep.update((s) => (s && s.id === id && (s.done !== done || s.total !== total) ? { ...s, done, total } : s)),
    onComplete: () => void refetch(id),
    onError: (msg) => logActivity(`Comparison progress: ${msg}`, 'warning'),
  });
  // Fallback while the socket is quiet or gone.
  pollTimer = setInterval(() => {
    if (socket && Date.now() - socket.lastMessageAt < 3000) return;
    void refetch(id);
  }, 2000);
}

export async function runSweep(req: SweepRequest): Promise<void> {
  if (get(sweepBusy)) return;
  stopTracking();
  // A sweep still being opened must not replace the one starting now.
  ++loadGen;
  openingSweep.set(null);
  setSweepParam(null);
  sweep.set({
    id: null,
    status: 'starting',
    label: req.label ?? null,
    createdAt: null,
    range: req.date_range,
    done: 0,
    total: req.tariffs.length * (req.capacities_kwh.length * req.strategies.length + 1),
    cases: [],
    error: null,
    attachedAt: Date.now(),
    doneAtAttach: 0,
    cancelling: false,
  });
  try {
    const { sweep_id } = await startSweep(req);
    saveJson(SWEEP_KEY, sweep_id);
    setSweepParam(sweep_id);
    sweep.update((s) => (s ? { ...s, id: sweep_id, status: 'running' } : s));
    logActivity(`Comparison started: ${req.tariffs.length * req.capacities_kwh.length * req.strategies.length} runs`);
    track(sweep_id);
    await refetch(sweep_id);
  } catch (err) {
    const msg = errorMessage(err);
    sweep.update((s) => (s ? { ...s, status: 'failed', error: msg } : s));
    toast('error', "The comparison didn't start", msg);
  }
}

export async function cancelCurrentSweep(): Promise<void> {
  const s = get(sweep);
  if (!s?.id || s.status !== 'running' || s.cancelling) return;
  sweep.update((x) => (x ? { ...x, cancelling: true } : x));
  try {
    await cancelSweep(s.id);
    scheduleRefetch(s.id, 300);
  } catch (err) {
    sweep.update((x) => (x ? { ...x, cancelling: false } : x));
    toast('error', 'Could not cancel', errorMessage(err));
  }
}

// ---- Which sweep is shown (#/compare?sweep=<id>) ----

export const SWEEP_PARAM = 'sweep';

/** Id of the sweep being fetched for display, or null. */
export const openingSweep = writable<string | null>(null);
let loadGen = 0;

/** Records the sweep on screen in the URL, but only while the Compare tab is the one showing. */
function setSweepParam(id: string | null): void {
  if (isRouteHash('compare')) replaceHashParam(SWEEP_PARAM, id);
}

/** Point the form at a sweep the user opened, so "Run comparison" means what's on screen. */
function applyToForm(r: SweepResult): void {
  const ts: string[] = [];
  const caps = new Set<number>();
  const ss: string[] = [];
  for (const c of r.cases) {
    if (!ts.includes(c.tariff)) ts.push(c.tariff);
    if (isBaseline(c)) continue;
    caps.add(Math.round(c.capacity_kwh * 100) / 100);
    if (!ss.includes(c.strategy_id)) ss.push(c.strategy_id);
  }
  if (ts.length) tariffs.set(ts);
  if (caps.size) capacities.set([...caps].sort((a, b) => a - b));
  if (ss.length) strategies.set(ss);
  label.set('');
  const p = presetFor(r.date_range, get(dataBounds));
  if (p === 'custom') customRange.set({ start: r.date_range.start, end: r.date_range.end });
  period.set(p);
}

/**
 * Show a stored sweep; keeps following it if it is still running. `quiet` is the silent restore
 * on page load: no error toast. The form is pointed at the sweep unless `keepForm` or `quiet`.
 */
export async function loadSweep(id: string, quiet = false, keepForm = false): Promise<boolean> {
  const prev = get(sweep);
  if (prev?.id === id) {
    setSweepParam(id);
    return true;
  }
  const gen = ++loadGen;
  openingSweep.set(id);
  try {
    const r = await getSweep(id);
    if (gen !== loadGen) return false;
    stopTracking();
    sweep.set(fromResult(r, null));
    saveJson(SWEEP_KEY, id);
    setSweepParam(id);
    if (!quiet && !keepForm) applyToForm(r);
    if (r.status === 'running') track(id);
    return true;
  } catch (err) {
    if (gen !== loadGen) return false;
    if (quiet) saveJson(SWEEP_KEY, null);
    else toast('error', `Couldn't open comparison ${id}`, errorMessage(err));
    // Keep naming what stays on screen rather than the sweep that failed to load.
    setSweepParam(get(sweep)?.id ?? null);
    return false;
  } finally {
    if (gen === loadGen) openingSweep.set(null);
  }
}

/**
 * Bring the page in line with the URL: a ?sweep= link wins; a bare #/compare shows what is already
 * on screen, or the last sweep this browser showed, and names it in the URL.
 */
export function syncSweepFromHash(): void {
  if (!isRouteHash('compare')) return;
  const id = hashParams().get(SWEEP_PARAM);
  const cur = get(sweep);
  if (id) {
    // A reload names the sweep this browser showed last; the form may hold edits since, so keep it.
    if (cur?.id !== id && get(openingSweep) !== id) void loadSweep(id, false, loadJson<string>(SWEEP_KEY) === id);
    return;
  }
  if (get(openingSweep)) return;
  if (cur) {
    if (cur.id) setSweepParam(cur.id);
    return;
  }
  const last = loadJson<string>(SWEEP_KEY);
  if (typeof last === 'string') void loadSweep(last, true);
}
