const LOCALE = 'en-GB';

const eurFmt = new Map<number, Intl.NumberFormat>();
function eurFormatter(decimals: number): Intl.NumberFormat {
  let f = eurFmt.get(decimals);
  if (!f) {
    f = new Intl.NumberFormat(LOCALE, {
      style: 'currency',
      currency: 'EUR',
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
    eurFmt.set(decimals, f);
  }
  return f;
}

export const DASH = '—';

const isNum = (v: number | null | undefined): v is number => typeof v === 'number' && Number.isFinite(v);

/** €1,234.56; uses a true minus sign for negatives. */
export function eur(v: number | null | undefined, decimals = 2): string {
  if (!isNum(v)) return DASH;
  const s = eurFormatter(decimals).format(Math.abs(v));
  return v < 0 && Math.abs(v) >= 0.5 * 10 ** -decimals ? `−${s}` : s;
}

/** Signed money for deltas: +€4.47 / −€1.20. */
export function eurSigned(v: number | null | undefined, decimals = 2): string {
  if (!isNum(v)) return DASH;
  const rounded = Number(v.toFixed(decimals));
  if (rounded === 0) return eurFormatter(decimals).format(0);
  return `${rounded > 0 ? '+' : '−'}${eurFormatter(decimals).format(Math.abs(v))}`;
}

/** Whole euros for large annual figures, cents below €100. */
export function eurAuto(v: number | null | undefined): string {
  if (!isNum(v)) return DASH;
  return eur(v, Math.abs(v) >= 100 ? 0 : 2);
}

export function num(v: number | null | undefined, decimals = 1): string {
  if (!isNum(v)) return DASH;
  const s = Math.abs(v).toLocaleString(LOCALE, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  return v < 0 && Number(Math.abs(v).toFixed(decimals)) !== 0 ? `−${s}` : s;
}

/** Up to `maxDecimals`, without trailing zeros: 7.5, 15.36, 20. */
export function numTrim(v: number | null | undefined, maxDecimals = 2): string {
  if (!isNum(v)) return DASH;
  const s = Math.abs(v).toLocaleString(LOCALE, { minimumFractionDigits: 0, maximumFractionDigits: maxDecimals });
  return v < 0 && Number(Math.abs(v).toFixed(maxDecimals)) !== 0 ? `−${s}` : s;
}

export function kwh(v: number | null | undefined, decimals = 1): string {
  return isNum(v) ? `${num(v, decimals)} kWh` : DASH;
}

/** Value already in percent (0–100). */
export function pct(v: number | null | undefined, decimals = 0): string {
  return isNum(v) ? `${num(v, decimals)}%` : DASH;
}

// ---- Dates (local time, en-GB, 24 h) ----

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const pad = (n: number) => String(n).padStart(2, '0');

/** 'YYYY-MM-DD' from local date parts (never toISOString, which shifts near midnight). */
export function isoDate(d: Date): string {
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

/** Parse 'YYYY-MM-DD' as a local date at midnight. */
export function parseIsoDate(s: string): Date {
  const [y, m, d] = s.slice(0, 10).split('-').map(Number);
  return new Date(y, m - 1, d);
}

export function addDays(d: Date, n: number): Date {
  const r = new Date(d);
  r.setDate(r.getDate() + n);
  return r;
}

/** Inclusive day count between two 'YYYY-MM-DD' strings. */
export function dayCount(start: string, end: string): number {
  return Math.round((parseIsoDate(end).getTime() - parseIsoDate(start).getTime()) / 86_400_000) + 1;
}

/** 12 Oct 2026 */
export function fmtDate(d: Date | string | null | undefined): string {
  if (d == null) return DASH;
  const x = typeof d === 'string' ? (d.length === 10 ? parseIsoDate(d) : new Date(d)) : d;
  if (Number.isNaN(x.getTime())) return DASH;
  return `${x.getDate()} ${MONTHS[x.getMonth()]} ${x.getFullYear()}`;
}

/** 12 Oct (year added when not the current year) */
export function fmtDay(d: Date | string, withYear?: boolean): string {
  const x = typeof d === 'string' ? (d.length === 10 ? parseIsoDate(d) : new Date(d)) : d;
  const showYear = withYear ?? x.getFullYear() !== new Date().getFullYear();
  return `${x.getDate()} ${MONTHS[x.getMonth()]}${showYear ? ` ${x.getFullYear()}` : ''}`;
}

/** 12 Oct 21:16 */
export function fmtDateTime(d: Date | string | null | undefined, withYear?: boolean): string {
  if (d == null) return DASH;
  const x = typeof d === 'string' ? new Date(d) : d;
  if (Number.isNaN(x.getTime())) return DASH;
  return `${fmtDay(x, withYear)} ${pad(x.getHours())}:${pad(x.getMinutes())}`;
}

export function fmtTime(d: Date): string {
  return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** 'YYYY-MM' → 'Oct 2026' (or 'Oct' when short). */
export function fmtMonth(ym: string, short = false): string {
  const [y, m] = ym.split('-').map(Number);
  return short ? MONTHS[m - 1] : `${MONTHS[m - 1]} ${y}`;
}

/** "12–18 Oct 2026", "28 Sep – 4 Oct 2026", "1 Dec 2025 – 3 Jan 2026" */
export function fmtRange(start: string, end: string): string {
  const a = parseIsoDate(start);
  const b = parseIsoDate(end);
  if (start === end) return fmtDate(a);
  if (a.getFullYear() !== b.getFullYear()) return `${fmtDate(a)} – ${fmtDate(b)}`;
  if (a.getMonth() !== b.getMonth()) return `${a.getDate()} ${MONTHS[a.getMonth()]} – ${fmtDate(b)}`;
  return `${a.getDate()}–${fmtDate(b)}`;
}

/** "3 h ago", "2 days ago" */
export function fmtAgo(d: Date | string, now = Date.now()): string {
  const t = typeof d === 'string' ? Date.parse(d) : d.getTime();
  const mins = Math.max(0, Math.round((now - t) / 60_000));
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins} min ago`;
  const h = Math.round(mins / 60);
  if (h < 48) return `${h} h ago`;
  return `${Math.round(h / 24)} days ago`;
}

export function fmtHours(h: number): string {
  if (h < 48) return `${num(h, h < 10 ? 1 : 0)} h`;
  return `${num(h / 24, 1)} days`;
}
