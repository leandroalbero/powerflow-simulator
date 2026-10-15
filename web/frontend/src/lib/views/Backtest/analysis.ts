import type { BacktestDaily, BacktestResult } from '../../types';
import { addDays, eur, fmtDay, isoDate, num, parseIsoDate } from '../../ui/format';

/** Below this the measured cost is too close to zero (or negative) for a relative error to mean anything. */
export const MIN_COST_FOR_PCT = 1;
/** Export error % is shown only when measured export is at least this share of import. */
export const MIN_EXPORT_SHARE = 0.05;
/** Strategies that never buy grid energy for the battery. */
const NO_GRID_CHARGE = new Set(['no_battery', 'self_consume', 'smart_discharge']);

export interface Totals {
  measuredImport: number;
  simImport: number;
  measuredExport: number;
  simExport: number;
  days: number;
  /** Days on which neither the meters nor the simulation saw any grid flow: the data has a gap there. */
  emptyDays: number;
}

export const isEmptyDay = (d: BacktestDaily) =>
  d.measured_import_kwh === 0 && d.measured_export_kwh === 0 && d.sim_import_kwh === 0 && d.sim_export_kwh === 0;

export function totals(r: BacktestResult): Totals {
  const t: Totals = { measuredImport: 0, simImport: 0, measuredExport: 0, simExport: 0, days: r.daily.length, emptyDays: 0 };
  for (const d of r.daily) {
    t.measuredImport += d.measured_import_kwh;
    t.simImport += d.sim_import_kwh;
    t.measuredExport += d.measured_export_kwh;
    t.simExport += d.sim_export_kwh;
    if (isEmptyDay(d)) t.emptyDays++;
  }
  return t;
}

export const exportIsTiny = (t: Totals) => t.measuredExport < MIN_EXPORT_SHARE * t.measuredImport;
export const costPctMeaningful = (r: BacktestResult) => r.metrics.measured_cost >= MIN_COST_FOR_PCT;

/** How much less (positive) or more (negative) the real system paid than the simulated strategy. */
export function realVsSim(r: BacktestResult): { eur: number; pctOfSim: number | null } {
  const { measured_cost: m, sim_cost: s } = r.metrics;
  const d = s - m;
  return { eur: d, pctOfSim: s >= MIN_COST_FOR_PCT ? (d / s) * 100 : null };
}

// ---- SoC ----

/** Max gap in measured SoC, in ms, that still counts as a sensor blip rather than missing data. */
const SOC_GAP_MS = 2 * 3600_000;

/**
 * x in epoch seconds for uPlot. Where the measured series has a long hole the data itself is
 * missing, but the simulator holds its last level across it; that flat line would read as data,
 * so it is blanked too.
 */
export function socSeries(r: BacktestResult): { x: number[]; measured: (number | null)[]; sim: (number | null)[] } {
  const { ts, measured_pct: m, sim_pct: s } = r.soc;
  const sim: (number | null)[] = s.map((v) => (Number.isFinite(v) ? v : null));
  let i = 0;
  while (i < m.length) {
    if (m[i] != null) {
      i++;
      continue;
    }
    let j = i;
    while (j < m.length && m[j] == null) j++;
    const from = ts[Math.max(0, i - 1)];
    const to = ts[Math.min(ts.length - 1, j)];
    if (to - from > SOC_GAP_MS) for (let k = i; k < j; k++) sim[k] = null;
    i = j;
  }
  return { x: ts.map((t) => t / 1000), measured: m, sim };
}

/**
 * Nights on which the battery charge rose by at least `minRise` points between midnight and 07:00.
 * With no sun in that window, a rise means the battery was charged from the grid.
 */
export function nightCharging(ts: number[], pct: (number | null)[], minRise = 8): { nights: number; rising: number } {
  const byNight = new Map<string, number[]>();
  for (let i = 0; i < ts.length; i++) {
    const v = pct[i];
    if (v == null || !Number.isFinite(v)) continue;
    const d = new Date(ts[i]);
    if (d.getHours() >= 7) continue;
    const k = isoDate(d);
    let arr = byNight.get(k);
    if (!arr) byNight.set(k, (arr = []));
    arr.push(v);
  }
  let nights = 0;
  let rising = 0;
  for (const vals of byNight.values()) {
    if (vals.length < 3) continue;
    nights++;
    let lo = Infinity;
    let rise = 0;
    for (const v of vals) {
      lo = Math.min(lo, v);
      rise = Math.max(rise, v - lo);
    }
    if (rise >= minRise) rising++;
  }
  return { nights, rising };
}

// ---- Daily chart rows ----

export interface DayRow {
  key: string;
  label: string;
  tipLabel: string;
  a: number | null;
  b: number | null;
}

/** Above this many days, daily pairs get too thin to read and the chart switches to weeks. */
export const WEEKLY_ABOVE = 62;

export function dailyRows(daily: BacktestDaily[], pick: (d: BacktestDaily) => [number, number]): { rows: DayRow[]; weekly: boolean } {
  const multiYear = daily.length > 0 && daily[0].date.slice(0, 4) !== daily[daily.length - 1].date.slice(0, 4);
  if (daily.length <= WEEKLY_ABOVE) {
    return {
      weekly: false,
      rows: daily.map((d) => {
        const [a, b] = pick(d);
        const empty = isEmptyDay(d);
        const day = parseIsoDate(d.date);
        return {
          key: d.date,
          label: fmtDay(day, false),
          tipLabel: `${['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][day.getDay()]} ${fmtDay(day, multiYear || undefined)}`,
          a: empty ? null : a,
          b: empty ? null : b,
        };
      }),
    };
  }
  // Weeks start on Monday; a partial first or last week is still summed, and the tooltip says so.
  const weeks = new Map<string, { a: number; b: number; days: number; from: string; to: string }>();
  for (const d of daily) {
    if (isEmptyDay(d)) continue;
    const day = parseIsoDate(d.date);
    const monday = isoDate(addDays(day, -((day.getDay() + 6) % 7)));
    const [a, b] = pick(d);
    const w = weeks.get(monday) ?? { a: 0, b: 0, days: 0, from: d.date, to: d.date };
    w.a += a;
    w.b += b;
    w.days++;
    w.to = d.date;
    weeks.set(monday, w);
  }
  return {
    weekly: true,
    rows: [...weeks.entries()].map(([monday, w]) => ({
      key: monday,
      label: fmtDay(parseIsoDate(monday), false),
      tipLabel: `${fmtDay(parseIsoDate(w.from), multiYear || undefined)} – ${fmtDay(parseIsoDate(w.to), multiYear || undefined)}${w.days < 7 ? ` · ${w.days} ${w.days === 1 ? 'day' : 'days'} with data` : ''}`,
      a: w.a,
      b: w.b,
    })),
  };
}

export function cumulativeRows(daily: BacktestDaily[]): DayRow[] {
  let a = 0;
  let b = 0;
  const multiYear = daily.length > 0 && daily[0].date.slice(0, 4) !== daily[daily.length - 1].date.slice(0, 4);
  return daily.map((d) => {
    a += d.measured_cost;
    b += d.sim_cost;
    const day = parseIsoDate(d.date);
    return { key: d.date, label: fmtDay(day, false), tipLabel: `Up to ${fmtDay(day, multiYear || undefined)}`, a, b };
  });
}

// ---- Plain-language reading ----

export interface Reading {
  headline: string;
  details: string[];
}

/**
 * A hedged, number-driven reading of one backtest. Every clause is derived from the result;
 * none of it claims to know how the real controller works.
 */
export function readResult(
  r: BacktestResult,
  name: (id: string) => string,
  opts: { days: number; noBattery?: BacktestResult | null },
): Reading {
  const s = name(r.strategy_id);
  const { measured_cost: meas, sim_cost: sim } = r.metrics;
  const diff = realVsSim(r);
  const tol = Math.max(0.05 * Math.abs(sim), 0.1);
  const pctTxt = diff.pctOfSim != null ? ` (${num(Math.abs(diff.pctOfSim), 1)}%)` : '';
  const isRef = r.strategy_id === 'no_battery';

  let headline: string;
  const same = Math.abs(diff.eur) < tol;
  if (same) headline = `Your real system cost about the same as ${isRef ? 'no battery at all' : s} would have: ${eur(meas)} measured, ${eur(sim)} simulated.`;
  else {
    const other = isRef ? 'the house would have without a battery' : `${s} would have`;
    // With a net income on either side, "paid less" reads backwards; "better off" works for any sign.
    const earning = meas < 0 || sim < 0;
    const amount = `${eur(Math.abs(diff.eur))}${pctTxt}`;
    headline = earning
      ? `Your real system came out ${amount} ${diff.eur > 0 ? 'better' : 'worse'} off than ${other}.`
      : `Your real system paid ${amount} ${diff.eur > 0 ? 'less' : 'more'} than ${other}.`;
  }

  const details: string[] = [];
  const measNights = nightCharging(r.soc.ts, r.soc.measured_pct);
  const simNights = nightCharging(r.soc.ts, r.soc.sim_pct);
  const realChargesAtNight = measNights.rising >= Math.max(1, Math.ceil(measNights.nights * 0.25)) && measNights.rising > simNights.rising;

  if (!isRef && !same && diff.eur > 0) {
    if (realChargesAtNight && NO_GRID_CHARGE.has(r.strategy_id)) {
      details.push(
        `The measured battery charge rose overnight on ${measNights.rising} of ${measNights.nights} nights, so your controller probably charges from the grid at night. Try ${name('charge_night')} or ${name('mpc')}.`,
      );
    } else {
      details.push(`Your controller may time its charging better than ${s}, or the simulator may overestimate some losses.`);
    }
  } else if (!isRef && !same && diff.eur < 0) {
    details.push(`${s} may simply be the better strategy for this period, or the simulator may leave out losses and standby use that the real system has.`);
  }

  if (!isRef) {
    const rmse = r.metrics.soc_rmse_pct;
    if (rmse <= 10) details.push(`The battery charge level tracks ${s} closely (typically ${num(rmse, 0)} points apart).`);
    else if (rmse <= 20) details.push(`The battery charge level differs from ${s} by about ${num(rmse, 0)} points on average.`);
    else
      details.push(
        `The battery charge level is about ${num(rmse, 0)} points away from ${s} on average, so the real controller likely behaves differently${same ? ' and the matching cost may be partly coincidence' : ''}.`,
      );
  }

  const nb = opts.noBattery;
  if (nb && !isRef) {
    const saved = nb.metrics.sim_cost - meas;
    if (Math.abs(saved) >= 0.05)
      details.push(
        saved > 0
          ? `Against a simulated house without a battery (${eur(nb.metrics.sim_cost)}), your real battery saved roughly ${eur(saved)}.`
          : `A simulated house without a battery would have paid ${eur(nb.metrics.sim_cost)}, ${eur(-saved)} less than the real system.`,
      );
  }

  if (opts.days < 7) details.push(`This covers just ${opts.days} ${opts.days === 1 ? 'day' : 'days'}, so treat it as a rough indication.`);
  return { headline, details };
}
