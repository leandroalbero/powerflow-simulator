// Fixed colour per strategy id, independent of selection or rank. Hues are the validated
// colour-blind-safe categorical set; with more strategies than safe hues, advanced ones reuse
// a hue and are told apart by a hollow dot and a dashed line. Red is kept off strategies: it
// reads as a loss, and next to Force Charge at Night's orange it fails the CVD and
// normal-vision separation checks.

export type ThemeMode = 'light' | 'dark';

const HUES = {
  blue: ['#2a78d6', '#3987e5'],
  orange: ['#eb6834', '#d95926'],
  aqua: ['#1baf7a', '#199e70'],
  yellow: ['#eda100', '#c98500'],
  magenta: ['#e87ba4', '#d55181'],
  green: ['#008300', '#008300'],
  violet: ['#6250d6', '#9085e9'],
} as const;

type Hue = keyof typeof HUES;

interface StrategyStyle {
  hue: Hue | 'neutral';
  hollow: boolean;
}

const STYLE: Record<string, StrategyStyle> = {
  no_battery: { hue: 'neutral', hollow: false },
  self_consume: { hue: 'blue', hollow: false },
  charge_night: { hue: 'orange', hollow: false },
  oracle: { hue: 'aqua', hollow: false },
  mpc_perfect: { hue: 'yellow', hollow: false },
  forecast_charge: { hue: 'magenta', hollow: false },
  force_valleys: { hue: 'green', hollow: false },
  mpc: { hue: 'violet', hollow: false },
  smart_discharge: { hue: 'aqua', hollow: true },
  force_valleys_pre_peak: { hue: 'green', hollow: true },
  valley_charge_peak_discharge: { hue: 'orange', hollow: true },
  mpc_calibrated: { hue: 'violet', hollow: true },
  mpc_5min: { hue: 'blue', hollow: true },
  mpc_5min_perfect: { hue: 'yellow', hollow: true },
  dqn_agent: { hue: 'magenta', hollow: true },
};

const NEUTRAL = ['#7d8794', '#8e98a6'] as const;
const FALLBACK_ORDER: Hue[] = ['blue', 'orange', 'aqua', 'yellow', 'magenta', 'green', 'violet'];

function styleFor(id: string): StrategyStyle {
  const s = STYLE[id];
  if (s) return s;
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0;
  return { hue: FALLBACK_ORDER[h % FALLBACK_ORDER.length], hollow: true };
}

export function strategyColor(id: string, mode: ThemeMode): string {
  const s = styleFor(id);
  const pair = s.hue === 'neutral' ? NEUTRAL : HUES[s.hue];
  return pair[mode === 'dark' ? 1 : 0];
}

export function strategyHollow(id: string): boolean {
  return styleFor(id).hollow;
}

/** Line dash for charts: baseline dotted, hollow (advanced) strategies dashed. */
export function strategyDash(id: string): number[] | undefined {
  if (id === 'no_battery') return [2, 4];
  return styleFor(id).hollow ? [7, 4] : undefined;
}

/**
 * Power-flow series colours (power chart). Kept off the strategy hues they sit next to in the
 * strategy switcher: import takes the red no strategy uses, export a cyan, and the battery one
 * plum for both directions (charge is drawn dashed). Validated as a set with the dataviz
 * palette checker (all pairs, both modes); load stays neutral ink.
 */
export const FLOW = {
  solar: { light: '#d99400', dark: '#c98500' },
  load: { light: '#3b4552', dark: '#c9d0da' },
  gridImport: { light: '#e34948', dark: '#d84a5a' },
  gridExport: { light: '#0096b4', dark: '#1a9cb8' },
  batteryCharge: { light: '#7a3fa0', dark: '#9d5bd8' },
  batteryDischarge: { light: '#7a3fa0', dark: '#9d5bd8' },
  price: { light: '#66707e', dark: '#8e98a6' },
} as const;

export type FlowKey = keyof typeof FLOW;

export function flowColor(k: FlowKey, mode: ThemeMode): string {
  return FLOW[k][mode];
}

/** Hex → rgba with alpha. */
export function alpha(hex: string, a: number): string {
  const h = hex.replace('#', '');
  const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}
