import { fmtDay } from './format';

/*
 * The backend thins long timeseries by keeping, per bucket, the rows where the smallest and the
 * largest value across ALL series fall. For any one series those rows are arbitrary samples, so
 * drawn as a line it zig-zags between unrelated values. At long ranges the charts therefore
 * regroup the samples into calendar-aligned buckets (hour, day, week) and draw each series as
 * its min–max range per bucket, which is what the samples can honestly tell.
 */

const HOUR = 3600;
const DAY = 86400;
/**
 * Bucket length in seconds for a visible span, or null when the raw samples are fine. Up to about
 * ten days the samples are dense enough to draw as lines. Beyond that, sub-day buckets turn the
 * daily cycle into saw-teeth, so a span gets daily ranges, and past four months weekly ones.
 * Weeks rather than two-day buckets, which would alias with the weekday/weekend tariff rhythm.
 */
export function pickBucket(spanSec: number): number | null {
  if (spanSec <= 10.5 * DAY) return null;
  return spanSec <= 120 * DAY ? DAY : 7 * DAY;
}

/** Local-time offset in seconds at a given instant (DST-aware). */
const offsetAt = (t: number) => -new Date(t * 1000).getTimezoneOffset() * 60;

/** Index of the local, calendar-aligned bucket containing t. Weeks start on Monday. */
function bucketKey(t: number, b: number): number {
  const local = t + offsetAt(t);
  // 1970-01-05 was a Monday.
  return Math.floor((b >= 7 * DAY ? local - 4 * DAY : local) / b);
}

function bucketStart(key: number, b: number): number {
  const local = key * b + (b >= 7 * DAY ? 4 * DAY : 0);
  // Close enough across a DST change: the offset is taken at the local start itself.
  return local - offsetAt(local);
}

export interface Bucketed {
  /** Bucket centre, epoch seconds (where the point is drawn). */
  x: number[];
  /** Bucket start, epoch seconds. */
  start: number[];
  lo: (number | null)[][];
  hi: (number | null)[][];
  mean: (number | null)[][];
}

type Col = (number | null | undefined)[];

/**
 * Regroup samples (sorted by x, epoch seconds) into buckets of `b` seconds. Missing buckets and
 * buckets without any value become nulls, so data gaps stay gaps instead of being bridged.
 */
export function bucketize(x: number[], cols: Col[], b: number): Bucketed {
  const out: Bucketed = { x: [], start: [], lo: cols.map(() => []), hi: cols.map(() => []), mean: cols.map(() => []) };
  const n = cols.length;
  const lo = new Array<number>(n);
  const hi = new Array<number>(n);
  const sum = new Array<number>(n);
  const cnt = new Array<number>(n);
  let cur: number | null = null;

  const reset = () => {
    lo.fill(Infinity);
    hi.fill(-Infinity);
    sum.fill(0);
    cnt.fill(0);
  };
  const push = (key: number, empty = false) => {
    const s = bucketStart(key, b);
    out.start.push(s);
    out.x.push(s + b / 2);
    for (let c = 0; c < n; c++) {
      const has = !empty && cnt[c] > 0;
      out.lo[c].push(has ? lo[c] : null);
      out.hi[c].push(has ? hi[c] : null);
      out.mean[c].push(has ? sum[c] / cnt[c] : null);
    }
  };

  reset();
  for (let i = 0; i < x.length; i++) {
    const k = bucketKey(x[i], b);
    if (cur !== null && k !== cur) {
      push(cur);
      // One empty bucket is enough to break the line across a run of missing buckets.
      if (k > cur + 1) push(cur + 1, true);
      reset();
    }
    cur = k;
    for (let c = 0; c < n; c++) {
      const v = cols[c][i];
      if (v == null || !Number.isFinite(v)) continue;
      if (v < lo[c]) lo[c] = v;
      if (v > hi[c]) hi[c] = v;
      sum[c] += v;
      cnt[c]++;
    }
  }
  if (cur !== null) push(cur);
  return out;
}

const pad = (n: number) => String(n).padStart(2, '0');
const hm = (d: Date) => `${pad(d.getHours())}:${pad(d.getMinutes())}`;

/** Cursor label for the bucket whose centre is `centre`: "12 Oct", "12 Oct 10:00–12:00", "Week of 6 Oct". */
export function bucketLabel(centre: number, b: number): string {
  const s = new Date((centre - b / 2) * 1000);
  if (b >= 7 * DAY) return `Week of ${fmtDay(s)}`;
  if (b >= 2 * DAY) {
    const e = new Date((centre + b / 2 - 1) * 1000);
    return `${fmtDay(s)} – ${fmtDay(e)}`;
  }
  if (b >= DAY) return fmtDay(s);
  const e = new Date((centre + b / 2) * 1000);
  return `${fmtDay(s)} ${hm(s)}–${hm(e)}`;
}

/** "daily", "2-hourly", "weekly" for subtitles. */
export function bucketWord(b: number): string {
  if (b >= 7 * DAY) return 'weekly';
  if (b === DAY) return 'daily';
  if (b > DAY) return `${b / DAY}-day`;
  if (b === HOUR) return 'hourly';
  return `${b / HOUR}-hour`;
}
