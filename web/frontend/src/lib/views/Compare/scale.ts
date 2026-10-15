import type { ThemeMode } from '../../ui/colors';

// One green ramp for savings (light → dark on a light surface, dark → bright on a dark one),
// and a short red ramp for combinations that cost more than having no battery.
const POS: Record<ThemeMode, string[]> = {
  light: ['#eef7f2', '#cdeadb', '#9dd4b7', '#5db58b', '#2a8d62', '#0c6443'],
  dark: ['#1a2621', '#1d4232', '#1f6244', '#27855b', '#3aaa77', '#6fd5a2'],
};
const NEG: Record<ThemeMode, string[]> = {
  light: ['#fbecea', '#f4c4be', '#e48d83'],
  dark: ['#2e1f1f', '#5a2c29', '#8a3a33'],
};
const ZERO: Record<ThemeMode, string> = { light: '#f1f3f5', dark: '#1f242c' };

function hexToRgb(h: string): [number, number, number] {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function mix(stops: string[], t: number): string {
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = hexToRgb(stops[i]);
  const b = hexToRgb(stops[i + 1]);
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * f));
  return `#${c.map((v) => v.toString(16).padStart(2, '0')).join('')}`;
}

function luminance(hex: string): number {
  const [r, g, b] = hexToRgb(hex).map((v) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

const contrast = (a: number, b: number) => (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
const INK_DARK = '#10161d';
const INK_LIGHT = '#f4f7fa';
const MIN_CONTRAST = 4.6;

function toward(hex: string, target: number, f: number): string {
  const c = hexToRgb(hex).map((v) => Math.round(v + (target - v) * f));
  return `#${c.map((v) => v.toString(16).padStart(2, '0')).join('')}`;
}

/**
 * Between the two inks there is a band of mid luminance where neither reaches 4.5:1. A fill in
 * that band is lightened or darkened, whichever is the smaller change, just far enough for one
 * ink to pass; the ramp stays monotonic because fills only move to the nearer edge of the band.
 */
function readable(bg: string): { bg: string; fg: string } {
  const lDark = luminance(INK_DARK);
  const lLight = luminance(INK_LIGHT);
  if (contrast(luminance(bg), lDark) >= MIN_CONTRAST) return { bg, fg: INK_DARK };
  if (contrast(luminance(bg), lLight) >= MIN_CONTRAST) return { bg, fg: INK_LIGHT };
  // Smallest blend toward white (or black) that passes, so every fill in the band lands on the same edge.
  const push = (target: number, ink: number): string => {
    let lo = 0;
    let hi = 1;
    for (let i = 0; i < 16; i++) {
      const mid = (lo + hi) / 2;
      if (contrast(luminance(toward(bg, target, mid)), ink) >= MIN_CONTRAST) hi = mid;
      else lo = mid;
    }
    return toward(bg, target, hi);
  };
  const lighter = push(255, lDark);
  const darker = push(0, lLight);
  const l = luminance(bg);
  return Math.abs(luminance(lighter) - l) <= Math.abs(l - luminance(darker))
    ? { bg: lighter, fg: INK_DARK }
    : { bg: darker, fg: INK_LIGHT };
}

/** Fill and a text colour that reaches 4.5:1 on it. */
export function cellColors(v: number | null, domain: { min: number; max: number }, mode: ThemeMode): { bg: string; fg: string } {
  let bg: string;
  if (v == null || Math.abs(v) < 1e-9) bg = ZERO[mode];
  else if (v > 0) bg = mix(POS[mode], domain.max > 0 ? 0.12 + 0.88 * (v / domain.max) : 0);
  else bg = mix(NEG[mode], domain.min < 0 ? 0.25 + 0.75 * (v / domain.min) : 0);
  return readable(bg);
}

/** Gradient for the legend bar, from no savings to the largest. */
export function legendGradient(mode: ThemeMode): string {
  const pos = POS[mode].map((c, i, a) => `${c} ${Math.round(12 + (i / (a.length - 1)) * 88)}%`);
  return `linear-gradient(90deg, ${ZERO[mode]} 0%, ${pos.join(', ')})`;
}

export function negativeSwatch(mode: ThemeMode): string {
  return NEG[mode][1];
}
