import { derived, get, writable } from 'svelte/store';
import { dataBounds, lastFullDay } from '../../stores/app';
import { BASELINE_ID, run } from '../../stores/run';
import type { DateRange } from '../../types';
import { loadJson, saveJson } from '../../stores/storage';
import { addDays, isoDate, parseIsoDate } from '../../ui/format';

export type PeriodPreset = 'last7' | 'last30' | 'last12m' | 'ytd' | 'custom';

export const PERIOD_OPTIONS: { value: PeriodPreset; label: string; short: string }[] = [
  { value: 'last7', label: 'Last 7 days', short: '7 d' },
  { value: 'last30', label: 'Last 30 days', short: '30 d' },
  { value: 'last12m', label: 'Last 12 months', short: '12 m' },
  { value: 'ytd', label: 'Year to date', short: 'YTD' },
  { value: 'custom', label: 'Custom', short: 'Custom' },
];

export const DEFAULT_STRATEGIES = ['self_consume', 'charge_night', 'mpc', 'oracle'];

interface Persisted {
  period: PeriodPreset;
  custom: { start: string; end: string } | null;
  strategies: string[];
}

const KEY = 'pf.simulate.v1';
const stored = loadJson<Partial<Persisted>>(KEY);

export const period = writable<PeriodPreset>(
  PERIOD_OPTIONS.some((o) => o.value === stored?.period) ? (stored?.period as PeriodPreset) : 'last7',
);
export const customRange = writable<{ start: string; end: string } | null>(
  stored?.custom && typeof stored.custom.start === 'string' && typeof stored.custom.end === 'string' ? stored.custom : null,
);
export const selected = writable<string[]>(
  Array.isArray(stored?.strategies) && stored.strategies.every((s) => typeof s === 'string') ? stored.strategies : DEFAULT_STRATEGIES,
);

const persist = derived([period, customRange, selected], ([p, c, s]) => ({ period: p, custom: c, strategies: s }));
persist.subscribe((v) => saveJson(KEY, v));

export function toggleStrategy(id: string): void {
  selected.update((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));
}

/**
 * Inclusive date range for a preset, clamped to the dataset. Presets end on the last complete day,
 * the same as on Backtest, so "Last 30 days" is one window on every tab; a custom range can still
 * include a partial newest day.
 */
export function rangeFor(
  p: PeriodPreset,
  bounds: { min: string; max: string; until?: string | null },
  custom: { start: string; end: string } | null,
) {
  const lastFull = lastFullDay(bounds);
  const end = parseIsoDate(lastFull);
  let start: Date;
  switch (p) {
    case 'last7':
      start = addDays(end, -6);
      break;
    case 'last30':
      start = addDays(end, -29);
      break;
    case 'last12m': {
      start = new Date(end.getFullYear() - 1, end.getMonth(), end.getDate() + 1);
      break;
    }
    case 'ytd':
      start = new Date(end.getFullYear(), 0, 1);
      break;
    case 'custom': {
      const s = custom?.start ?? isoDate(addDays(end, -6));
      const e = custom?.end ?? lastFull;
      const cs = s < bounds.min ? bounds.min : s > bounds.max ? bounds.max : s;
      const ce = e > bounds.max ? bounds.max : e < cs ? cs : e;
      return { start: cs, end: ce };
    }
  }
  const s = isoDate(start);
  return { start: s < bounds.min ? bounds.min : s, end: lastFull };
}

export const activeRange = derived([period, customRange, dataBounds], ([p, c, b]) => (b ? rangeFor(p, b, c) : null));

/** The preset that gives this range today, or 'custom'. */
export function presetFor(r: DateRange, bounds: { min: string; max: string; until?: string | null } | null): PeriodPreset {
  if (!bounds) return 'custom';
  for (const o of PERIOD_OPTIONS) {
    if (o.value === 'custom') continue;
    const x = rangeFor(o.value, bounds, null);
    if (x.start === r.start && x.end === r.end) return o.value;
  }
  return 'custom';
}

/*
 * A run reopened or re-run from History (or opened from a link) takes over the controls once, so
 * "Run simulation" means the run on screen rather than whatever the controls held before. Later
 * edits are left alone. A re-run has no id until the server answers, so the key is its start time.
 */
let syncedRun: string | null = null;
run.subscribe((r) => {
  if (!r?.adoptControls) return;
  const key = `${r.startedAt}|${r.range.start}|${r.range.end}|${r.strategyIds.join()}`;
  if (key === syncedRun) return;
  syncedRun = key;
  const ids = r.strategyIds.filter((id) => id !== BASELINE_ID);
  if (ids.length) selected.set(ids);
  const p = presetFor(r.range, get(dataBounds));
  if (p === 'custom') customRange.set({ start: r.range.start, end: r.range.end });
  period.set(p);
});

/** Optional label for the next run; cleared once that run starts. Not persisted. */
export const runLabel = writable('');
export const LABEL_MAX = 80;

/** Strategy shown in the power chart. Follows the best result until the user picks one. */
export const focusStrategy = writable<string | null>(null);
export const focusPinned = writable(false);

export function pickFocus(id: string): void {
  focusPinned.set(true);
  focusStrategy.set(id);
}

export function resetFocus(): void {
  focusPinned.set(false);
  focusStrategy.set(null);
}
