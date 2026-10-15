import { derived, get, writable } from 'svelte/store';
import { errorMessage, runBacktest } from '../../api/client';
import { dataBounds, lastFullDay } from '../../stores/app';
import { loadJson, saveJson } from '../../stores/storage';
import { logActivity } from '../../stores/toasts';
import type { BacktestResult, DateRange } from '../../types';
import { addDays, dayCount, isoDate, parseIsoDate } from '../../ui/format';

export type BtPeriod = 'last7' | 'last30' | 'upgrade' | 'last12m' | 'custom';

/** The day the battery inverter limits went from 2.2/2.1 kW to 4.5/4.6 kW. */
export const UPGRADE_DATE = '2026-04-07';
/**
 * The backend refuses spans over 366 days between the two midnights; capping the inclusive day
 * count at 366 stays inside that even across a daylight-saving change.
 */
export const MAX_DAYS = 366;

export const PERIOD_OPTIONS: { value: BtPeriod; label: string; short: string }[] = [
  { value: 'last7', label: 'Last 7 days', short: '7 d' },
  { value: 'last30', label: 'Last 30 days', short: '30 d' },
  { value: 'upgrade', label: 'Since battery upgrade', short: 'Upgrade' },
  { value: 'last12m', label: 'Last 12 months', short: '12 m' },
  { value: 'custom', label: 'Custom', short: 'Custom' },
];

interface Persisted {
  period: BtPeriod;
  custom: DateRange | null;
  strategy: string;
  hardware: string;
}

const KEY = 'pf.backtest.v1';
const stored = loadJson<Partial<Persisted>>(KEY);

export const period = writable<BtPeriod>(
  PERIOD_OPTIONS.some((o) => o.value === stored?.period) ? (stored?.period as BtPeriod) : 'last7',
);
export const customRange = writable<DateRange | null>(
  stored?.custom && typeof stored.custom.start === 'string' && typeof stored.custom.end === 'string' ? stored.custom : null,
);
export const strategyId = writable<string>(typeof stored?.strategy === 'string' ? stored.strategy : 'self_consume');
export const hardware = writable<string>(typeof stored?.hardware === 'string' ? stored.hardware : 'auto');

derived([period, customRange, strategyId, hardware], ([p, c, s, h]) => ({ period: p, custom: c, strategy: s, hardware: h })).subscribe(
  (v) => saveJson(KEY, v),
);

/**
 * Inclusive range for a preset, ending on the last complete day with data. Custom ranges are
 * clamped to the data but not to a year.
 */
export function rangeFor(p: BtPeriod, bounds: { min: string; max: string; until?: string | null }, custom: DateRange | null): DateRange {
  const lastFull = lastFullDay(bounds);
  const end = parseIsoDate(lastFull);
  const clampStart = (s: string) => (s < bounds.min ? bounds.min : s > lastFull ? lastFull : s);
  switch (p) {
    case 'last7':
      return { start: clampStart(isoDate(addDays(end, -6))), end: lastFull };
    case 'last30':
      return { start: clampStart(isoDate(addDays(end, -29))), end: lastFull };
    case 'upgrade':
      return { start: clampStart(UPGRADE_DATE), end: lastFull };
    case 'last12m':
      return { start: clampStart(isoDate(new Date(end.getFullYear() - 1, end.getMonth(), end.getDate() + 1))), end: lastFull };
    case 'custom': {
      const s = custom?.start ?? isoDate(addDays(end, -6));
      const cs = s < bounds.min ? bounds.min : s > bounds.max ? bounds.max : s;
      const e = custom?.end ?? lastFull;
      const ce = e > bounds.max ? bounds.max : e < cs ? cs : e;
      return { start: cs, end: ce };
    }
  }
}

export const activeRange = derived([period, customRange, dataBounds], ([p, c, b]) => (b ? rangeFor(p, b, c) : null));

/** The last year of a too-long range, keeping its end date. */
export function shortenToYear(r: DateRange): DateRange {
  return { start: isoDate(addDays(parseIsoDate(r.end), -(MAX_DAYS - 1))), end: r.end };
}

export function tooLong(r: DateRange | null): boolean {
  return r != null && dayCount(r.start, r.end) > MAX_DAYS;
}

// ---- Results ----

export type EntryStatus = 'pending' | 'running' | 'done' | 'failed' | 'skipped';

export interface Entry {
  status: EntryStatus;
  result?: BacktestResult;
  error?: string;
  /** HTTP status of a failure, 0 when the server couldn't be reached. */
  httpStatus?: number;
  ms?: number;
  startedAt?: number;
}

export interface Session {
  key: string;
  range: DateRange;
  hardware: string;
  /** Results per strategy, all for the same range and hardware. */
  entries: Record<string, Entry>;
  /** Strategies in the comparison, in run order; empty until "compare all" is used. */
  compareIds: string[];
}

/*
 * Kept per tab in sessionStorage so a reload doesn't throw away a long comparison. A run that was
 * still in flight can't be resumed (the request died with the page), so it comes back as skipped.
 */
const SESSION_KEY = 'pf.backtest.session.v1';

function restoreSession(): { s: Session; shown: string | null } | null {
  try {
    const raw = sessionStorage.getItem(SESSION_KEY);
    const v = raw ? (JSON.parse(raw) as { s?: Partial<Session>; shown?: unknown }) : null;
    const x = v?.s;
    if (
      !x ||
      typeof x.key !== 'string' ||
      typeof x.hardware !== 'string' ||
      !x.range ||
      typeof x.range.start !== 'string' ||
      typeof x.range.end !== 'string' ||
      !x.entries ||
      typeof x.entries !== 'object' ||
      !Array.isArray(x.compareIds)
    ) {
      return null;
    }
    const entries: Record<string, Entry> = {};
    for (const [id, e] of Object.entries(x.entries)) {
      if (e.status === 'done' && e.result) entries[id] = e;
      else if (e.status === 'failed') entries[id] = e;
      else entries[id] = { status: 'skipped' };
    }
    if (!Object.values(entries).some((e) => e.status === 'done' || e.status === 'failed')) return null;
    const want = typeof v?.shown === 'string' ? v.shown : null;
    const shown = want && entries[want]?.status === 'done' ? want : Object.keys(entries).find((id) => entries[id].status === 'done') ?? want;
    return { s: { ...(x as Session), entries }, shown };
  } catch {
    return null;
  }
}

const restored = restoreSession();
export const session = writable<Session | null>(restored?.s ?? null);
/** Strategy whose result fills the KPIs and charts. */
export const shownId = writable<string | null>(restored?.shown ?? null);

derived([session, shownId], ([s, shown]) => ({ s, shown })).subscribe((v) => {
  try {
    if (v.s) sessionStorage.setItem(SESSION_KEY, JSON.stringify(v));
    else sessionStorage.removeItem(SESSION_KEY);
  } catch {
    // Storage full or unavailable: a reload just starts empty.
  }
});
export const busy = writable(false);
/** Strategy being computed right now (single run or comparison). */
export const runningId = writable<string | null>(null);
export const comparing = writable(false);

let stopRequested = false;

const keyOf = (range: DateRange, hw: string) => `${range.start}|${range.end}|${hw}`;

function ensureSession(range: DateRange, hw: string): Session {
  const key = keyOf(range, hw);
  const cur = get(session);
  if (cur && cur.key === key) return cur;
  const s: Session = { key, range: { ...range }, hardware: hw, entries: {}, compareIds: [] };
  session.set(s);
  return s;
}

function patch(id: string, e: Entry): void {
  session.update((s) => (s ? { ...s, entries: { ...s.entries, [id]: e } } : s));
}

async function runOne(range: DateRange, hw: string, id: string): Promise<boolean> {
  const startedAt = performance.now();
  runningId.set(id);
  patch(id, { status: 'running', startedAt: Date.now() });
  try {
    const result = await runBacktest({ date_range: { start: range.start, end: range.end }, strategy_id: id, hardware: hw });
    patch(id, { status: 'done', result, ms: performance.now() - startedAt });
    return true;
  } catch (err) {
    const httpStatus = err && typeof err === 'object' && 'status' in err && typeof err.status === 'number' ? err.status : undefined;
    patch(id, { status: 'failed', error: errorMessage(err), httpStatus, ms: performance.now() - startedAt });
    logActivity(`Backtest ${id}: ${errorMessage(err)}`, 'error');
    return false;
  } finally {
    runningId.set(null);
  }
}

export async function runSingle(range: DateRange, hw: string, id: string): Promise<void> {
  if (get(busy)) return;
  busy.set(true);
  ensureSession(range, hw);
  shownId.set(id);
  try {
    await runOne(range, hw, id);
  } finally {
    busy.set(false);
  }
}

/**
 * One request at a time: each backtest is a synchronous, CPU-heavy call on the server, and
 * parallel ones would only compete for the same cores. Results already computed for this
 * range and hardware are reused.
 */
export async function compareAll(range: DateRange, hw: string, ids: string[], preferShown: string): Promise<void> {
  if (get(busy) || !ids.length) return;
  busy.set(true);
  comparing.set(true);
  stopRequested = false;
  const s = ensureSession(range, hw);
  // The strategy picked in the form goes first, so the KPIs and charts fill in for it straight away.
  const order = ids.includes(preferShown) ? [preferShown, ...ids.filter((x) => x !== preferShown)] : [...ids];
  session.update((x) => (x ? { ...x, compareIds: order } : x));
  for (const id of order) if (s.entries[id]?.status !== 'done') patch(id, { status: 'pending' });
  shownId.set(order[0]);
  try {
    for (const id of order) {
      if (stopRequested) break;
      if (get(session)?.entries[id]?.status === 'done') continue;
      await runOne(range, hw, id);
      const entries = get(session)?.entries ?? {};
      const sh = get(shownId);
      if (entries[id]?.status === 'done' && (!sh || entries[sh]?.status === 'failed')) shownId.set(id);
    }
    if (stopRequested) {
      session.update((x) => {
        if (!x) return x;
        const entries = { ...x.entries };
        for (const id of order) if (entries[id]?.status === 'pending') entries[id] = { status: 'skipped' };
        return { ...x, entries };
      });
    }
  } finally {
    stopRequested = false;
    comparing.set(false);
    busy.set(false);
  }
}

/** Stops a comparison after the request in flight; the server can't abort a backtest midway. */
export function stopCompare(): void {
  stopRequested = true;
}
