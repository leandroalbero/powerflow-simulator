import { BASELINE_ID, MIN_HOURS_FOR_ANNUAL } from '../../stores/run';
import type { DateRange, SweepCase } from '../../types';
import { dayCount } from '../../ui/format';

const HOURS_YEAR = 8760;

export const caseKey = (tariff: string, capacity: number, strategy: string) => `${tariff}|${capRound(capacity)}|${strategy}`;

/** Capacities come back as floats (15.36); a fixed precision keeps them usable as map keys. */
export const capRound = (v: number) => Math.round(v * 1000) / 1000;

export const isBaseline = (c: SweepCase) => c.strategy_id === BASELINE_ID && c.capacity_kwh === 0;

/**
 * Hours with data in the sweep. A case carries no valid_hours, but the backend annualises with
 * them (savings × 8760 / hours), so they can be read back from a case with non-zero savings.
 * Every case runs on the same frame, so one is enough; the largest one is used because the backend
 * rounds both figures to 6 decimals. Until such a case exists the hours are unknown (null): the
 * calendar span would overstate them wherever data is missing.
 */
export function sweepHours(cases: SweepCase[]): number | null {
  let best: SweepCase | null = null;
  for (const c of cases) {
    const s = c.savings_vs_baseline;
    const a = c.annualised_savings;
    if (s == null || a == null || Math.abs(s) < 1e-3 || Math.abs(a) < 1e-3) continue;
    if (!best || Math.abs(a) > Math.abs(best.annualised_savings as number)) best = c;
  }
  return best ? ((best.savings_vs_baseline as number) * HOURS_YEAR) / (best.annualised_savings as number) : null;
}

export interface Cell {
  tariff: string;
  capacity: number;
  strategy: string;
  c: SweepCase;
}

export interface SweepModel {
  tariffs: string[];
  capacities: number[];
  strategies: string[];
  baselines: Map<string, SweepCase>;
  byKey: Map<string, SweepCase>;
  /** Hours with data, null until a finished battery case reveals them. */
  hours: number | null;
  /** Per-year figures are shown only with enough data to annualise. */
  perYear: boolean;
  /** The period would be long enough for per-year figures, but the hours aren't known yet. */
  hoursPending: boolean;
  completed: SweepCase[];
}

export function buildModel(cases: SweepCase[], range: DateRange | null): SweepModel {
  const tariffs: string[] = [];
  const caps = new Set<number>();
  const strategies: string[] = [];
  const baselines = new Map<string, SweepCase>();
  const byKey = new Map<string, SweepCase>();
  for (const c of cases) {
    if (!tariffs.includes(c.tariff)) tariffs.push(c.tariff);
    byKey.set(caseKey(c.tariff, c.capacity_kwh, c.strategy_id), c);
    if (isBaseline(c)) {
      baselines.set(c.tariff, c);
      continue;
    }
    caps.add(capRound(c.capacity_kwh));
    if (!strategies.includes(c.strategy_id)) strategies.push(c.strategy_id);
  }
  const hours = sweepHours(cases);
  const calendarHours = range ? dayCount(range.start, range.end) * 24 : 0;
  return {
    tariffs,
    capacities: [...caps].sort((a, b) => a - b),
    strategies,
    baselines,
    byKey,
    hours,
    perYear: hours != null && hours >= MIN_HOURS_FOR_ANNUAL,
    hoursPending: hours == null && calendarHours >= MIN_HOURS_FOR_ANNUAL,
    completed: cases.filter((c) => c.status === 'completed' && c.adjusted_cost != null),
  };
}

export function getCase(m: SweepModel, tariff: string, capacity: number, strategy: string): SweepCase | undefined {
  return m.byKey.get(caseKey(tariff, capacity, strategy));
}

/** Savings shown for a case: per year when the sweep is long enough, else over the period. */
export function shownSavings(m: SweepModel, c: SweepCase | undefined): number | null {
  if (!c || c.status !== 'completed') return null;
  return m.perYear ? c.annualised_savings : c.savings_vs_baseline;
}

/** Net cost scaled the same way as the savings. */
export function shownCost(m: SweepModel, c: SweepCase | undefined): number | null {
  if (!c || c.adjusted_cost == null) return null;
  if (!m.perYear || !m.hours) return c.adjusted_cost;
  return (c.adjusted_cost * HOURS_YEAR) / m.hours;
}

export function rawCost(m: SweepModel, c: SweepCase): number | null {
  if (c.total_cost == null) return null;
  if (!m.perYear || !m.hours) return c.total_cost;
  return (c.total_cost * HOURS_YEAR) / m.hours;
}

/** Best strategy for one tariff and capacity, by savings (same as lowest net cost within a tariff). */
export function bestInCell(m: SweepModel, tariff: string, capacity: number): SweepCase | undefined {
  let best: SweepCase | undefined;
  for (const s of m.strategies) {
    const c = getCase(m, tariff, capacity, s);
    if (!c || c.status !== 'completed' || c.savings_vs_baseline == null) continue;
    if (!best || c.savings_vs_baseline > (best.savings_vs_baseline ?? -Infinity)) best = c;
  }
  return best;
}

/** The combination that costs least overall, across tariffs. This is what the user would pay. */
export function cheapest(m: SweepModel, strategy: string | null = null): SweepCase | null {
  let best: SweepCase | null = null;
  for (const c of m.completed) {
    if (isBaseline(c) || (strategy && c.strategy_id !== strategy)) continue;
    if (!best || (c.adjusted_cost as number) < (best.adjusted_cost as number)) best = c;
  }
  return best;
}

/** The combination with the largest savings against its own tariff's no-battery baseline. */
export function mostSaving(m: SweepModel): SweepCase | null {
  let best: SweepCase | null = null;
  for (const c of m.completed) {
    if (isBaseline(c) || c.savings_vs_baseline == null) continue;
    if (!best || c.savings_vs_baseline > (best.savings_vs_baseline as number)) best = c;
  }
  return best;
}

export function cheapestBaseline(m: SweepModel): SweepCase | null {
  let best: SweepCase | null = null;
  for (const c of m.baselines.values()) {
    if (c.adjusted_cost == null) continue;
    if (!best || c.adjusted_cost < (best.adjusted_cost as number)) best = c;
  }
  return best;
}

/** Shared colour domain for every heatmap view, so switching strategy doesn't rescale the colours. */
export function savingsDomain(m: SweepModel): { min: number; max: number } {
  let min = 0;
  let max = 0;
  for (const c of m.completed) {
    if (isBaseline(c)) continue;
    const v = shownSavings(m, c);
    if (v == null) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return { min, max };
}

// ---- Run-time estimate ----

/** Seconds per simulated day for one case, measured on this machine; a rough guide only. */
const SECONDS_PER_DAY: Record<string, number> = {
  no_battery: 0.1,
  self_consume: 0.12,
  charge_night: 0.12,
  forecast_charge: 0.15,
  force_valleys: 0.12,
  force_valleys_pre_peak: 0.12,
  smart_discharge: 0.15,
  valley_charge_peak_discharge: 0.12,
  mpc: 0.65,
  mpc_perfect: 0.65,
  mpc_calibrated: 0.7,
  oracle: 0.65,
  mpc_5min: 2,
  mpc_5min_perfect: 2,
};
const WORKERS = 4;

export function estimateSeconds(days: number, tariffs: number, capacities: number, strategies: string[]): number {
  const perCombo = strategies.reduce((s, id) => s + (SECONDS_PER_DAY[id] ?? 0.6), 0);
  const work = days * (tariffs * capacities * perCombo + tariffs * SECONDS_PER_DAY.no_battery);
  return 3 + work / WORKERS;
}

export function fmtDuration(s: number): string {
  if (s < 45) return 'under a minute';
  const m = Math.round(s / 60);
  if (m < 60) return `about ${m} min`;
  const h = Math.floor(m / 60);
  const r = Math.round((m % 60) / 5) * 5;
  return r ? `about ${h} h ${r} min` : `about ${h} h`;
}
