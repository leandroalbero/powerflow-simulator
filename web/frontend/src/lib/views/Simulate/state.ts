import { derived, writable } from 'svelte/store';
import { dataBounds } from '../../stores/app';
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

/** Inclusive date range for a preset, ending at the last day with data and clamped to the dataset. */
export function rangeFor(p: PeriodPreset, bounds: { min: string; max: string }, custom: { start: string; end: string } | null) {
  const end = parseIsoDate(bounds.max);
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
      const e = custom?.end ?? bounds.max;
      const cs = s < bounds.min ? bounds.min : s > bounds.max ? bounds.max : s;
      const ce = e > bounds.max ? bounds.max : e < cs ? cs : e;
      return { start: cs, end: ce };
    }
  }
  const s = isoDate(start);
  return { start: s < bounds.min ? bounds.min : s, end: bounds.max };
}

export const activeRange = derived([period, customRange, dataBounds], ([p, c, b]) => (b ? rangeFor(p, b, c) : null));

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
