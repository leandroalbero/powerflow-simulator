import { derived, get, writable } from 'svelte/store';
import type { HardwarePreset, SystemConfig, TariffConfig, TariffPreset, TariffRate } from '../types';
import { defaults, presets } from './app';
import { loadJson, saveJson } from './storage';

const KEY = 'pf.configDraft.v1';

/** The settings sent with every run; null until defaults have loaded. */
export const draft = writable<SystemConfig | null>(null);

function isObj(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

/**
 * Overlay a stored draft onto fresh defaults, keeping only keys that exist in the defaults and
 * values of the same primitive type, so a stale or partial stored draft can't break the form.
 */
function mergeKnown<T>(base: T, over: unknown): T {
  if (Array.isArray(base)) return (Array.isArray(over) ? over : base) as T;
  if (isObj(base)) {
    if (!isObj(over)) return base;
    const out: Record<string, unknown> = { ...base };
    for (const k of Object.keys(base)) {
      if (k in over) out[k] = mergeKnown((base as Record<string, unknown>)[k], over[k]);
    }
    return out as T;
  }
  if (base === null) return (over === null || isObj(over) ? over : base) as T;
  return (typeof over === typeof base ? over : base) as T;
}

const clone = <T>(v: T): T => structuredClone(v);

let initialised = false;
defaults.subscribe((d) => {
  if (!d || initialised) return;
  initialised = true;
  const stored = loadJson<unknown>(KEY);
  draft.set(stored ? mergeKnown(clone(d), stored) : clone(d));
});

draft.subscribe((d) => {
  if (!d) return;
  const def = get(defaults);
  saveJson(KEY, def && sameJson(d, def) ? null : d);
});

function sameJson(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

export const draftModified = derived([draft, defaults], ([$d, $def]) => Boolean($d && $def && !sameJson($d, $def)));

export function resetDraft(): void {
  const d = get(defaults);
  if (d) draft.set(clone(d));
}

// ---- Presets ----

function tariffKey(t: TariffConfig): string {
  const { mpc_horizon_hours: _h, ...rest } = t;
  return JSON.stringify(rest);
}

export const hardwarePresetId = derived([draft, presets], ([$d, $p]) => {
  if (!$d || !$p) return null;
  const m = $p.hardware.find((h) => sameJson(h.battery, $d.battery) && sameJson(h.grid, $d.grid));
  return m?.id ?? null;
});

export const tariffPresetId = derived([draft, presets], ([$d, $p]) => {
  if (!$d || !$p) return null;
  const k = tariffKey($d.tariff);
  return $p.tariffs.find((t) => tariffKey(t.tariff) === k)?.id ?? null;
});

export function applyHardwarePreset(p: HardwarePreset): void {
  draft.update((d) => (d ? { ...d, battery: clone(p.battery), grid: clone(p.grid) } : d));
}

/** The MPC horizon is a strategy parameter in the UI, so a tariff preset keeps the user's value. */
export function applyTariffPreset(p: TariffPreset): void {
  draft.update((d) =>
    d ? { ...d, tariff: { ...clone(p.tariff), mpc_horizon_hours: d.tariff.mpc_horizon_hours } } : d,
  );
}

// ---- Validation ----

export type DraftErrors = Record<string, string>;

interface RangeOpts {
  minExclusive?: boolean;
  integer?: boolean;
  /** Display scale and unit of the field, so limits read as the user sees them (0.9 → 90%). */
  scale?: number;
  unit?: string;
}

function range(errors: DraftErrors, path: string, v: number, min: number, max: number, label: string, opts: RangeOpts = {}) {
  const scale = opts.scale ?? 1;
  const show = (x: number) => {
    const n = Math.round(x * scale * 1e6) / 1e6;
    if (!opts.unit) return String(n);
    return opts.unit === '%' ? `${n}%` : `${n} ${opts.unit}`;
  };
  if (!Number.isFinite(v)) errors[path] = `${label} is required`;
  else if (opts.integer && !Number.isInteger(v)) errors[path] = `${label} must be a whole number`;
  else if (opts.minExclusive ? v <= min : v < min) errors[path] = `${label} must be ${opts.minExclusive ? 'more than' : 'at least'} ${show(min)}`;
  else if (v > max) errors[path] = `${label} must be at most ${show(max)}`;
}

const PCT = { scale: 100, unit: '%' };
const fmtHours = (hs: number[]) => {
  const spans: string[] = [];
  for (let i = 0; i < hs.length; ) {
    let j = i;
    while (j + 1 < hs.length && hs[j + 1] === hs[j] + 1) j++;
    spans.push(`${hs[i]}–${hs[j] + 1} h`);
    i = j + 1;
  }
  return spans.join(', ');
};

/** The backend rejects a fixed-rate schedule unless each direction covers 0–24 h exactly once. */
function checkCoverage(rates: TariffRate[], dir: 'import' | 'export'): string | null {
  const count = new Array<number>(24).fill(0);
  for (const r of rates) {
    if (!Number.isInteger(r.start_hour) || !Number.isInteger(r.end_hour) || r.end_hour <= r.start_hour) return null;
    for (let h = Math.max(0, r.start_hour); h < Math.min(24, r.end_hour); h++) count[h]++;
  }
  const overlap = count.flatMap((c, h) => (c > 1 ? [h] : []));
  if (overlap.length) return `${dir === 'import' ? 'Import periods' : 'Export rates'} overlap at ${fmtHours(overlap)}`;
  const missing = count.flatMap((c, h) => (c === 0 ? [h] : []));
  if (missing.length) return `No ${dir} price for ${fmtHours(missing)}`;
  return null;
}

/** Mirrors the backend's pydantic bounds and tariff checks so a bad value is caught before the run fails. */
export function validateConfig(c: SystemConfig): DraftErrors {
  const e: DraftErrors = {};
  const b = c.battery;
  range(e, 'battery.capacity', b.capacity, 0, 200, 'Capacity', { minExclusive: true, unit: 'kWh' });
  range(e, 'battery.max_charge_rate', b.max_charge_rate, 0, 50, 'Max charge power', { minExclusive: true, unit: 'kW' });
  range(e, 'battery.max_discharge_rate', b.max_discharge_rate, 0, 50, 'Max discharge power', { minExclusive: true, unit: 'kW' });
  range(e, 'battery.efficiency', b.efficiency, 0, 1, 'Round-trip efficiency', { minExclusive: true, ...PCT });
  range(e, 'battery.initial_soc', b.initial_soc, 0, 1, 'Initial charge', PCT);
  range(e, 'battery.taper_start', b.taper_start, 0, 1, 'Taper starts at', { minExclusive: true, ...PCT });
  range(e, 'battery.taper_factor', b.taper_factor, 0, 1, 'Taper strength', PCT);
  range(e, 'battery.cycle_cost_eur_kwh', b.cycle_cost_eur_kwh, 0, 1, 'Wear cost', { unit: '€/kWh' });
  range(e, 'grid.max_import', c.grid.max_import, 0, 50, 'Max import', { minExclusive: true, unit: 'kW' });
  range(e, 'grid.max_export', c.grid.max_export, 0, 50, 'Max export', { minExclusive: true, unit: 'kW' });
  range(e, 'strategy.min_battery_level', c.strategy.min_battery_level, 0, 1, 'Minimum charge', PCT);
  range(e, 'strategy.valley_charge_target', c.strategy.valley_charge_target, 0, 1, 'Valley charge target', { minExclusive: true, ...PCT });
  range(e, 'strategy.max_charge_power', c.strategy.max_charge_power, 0, 50, 'Max grid charge power', { minExclusive: true, unit: 'kW' });
  range(e, 'tariff.mpc_horizon_hours', c.tariff.mpc_horizon_hours, 1, 48, 'MPC horizon', { integer: true, unit: 'h' });
  range(e, 'tariff.export_pvpc_discount', c.tariff.export_pvpc_discount, 0, 1, 'Export discount', { unit: '€/kWh' });
  c.tariff.rates.forEach((r, i) => {
    range(e, `tariff.rates.${i}.start_hour`, r.start_hour, 0, 23, 'Start', { integer: true, unit: 'h' });
    range(e, `tariff.rates.${i}.end_hour`, r.end_hour, 1, 24, 'End', { integer: true, unit: 'h' });
    range(e, `tariff.rates.${i}.price`, r.price, 0, 5, 'Price', { unit: '€/kWh' });
    if (!e[`tariff.rates.${i}.end_hour`] && !e[`tariff.rates.${i}.start_hour`] && r.end_hour <= r.start_hour) {
      e[`tariff.rates.${i}.end_hour`] = 'End must be after start';
    }
  });
  if (c.tariff.weekend_rate) range(e, 'tariff.weekend_rate.price', c.tariff.weekend_rate.price, 0, 5, 'Weekend price', { unit: '€/kWh' });
  const imports = c.tariff.rates.filter((r) => r.direction === 'import');
  const exports = c.tariff.rates.filter((r) => r.direction === 'export');
  if (!c.tariff.use_pvpc) {
    if (!imports.length) e['tariff.rates'] = 'Add at least one import period';
    else {
      const msg = checkCoverage(imports, 'import');
      if (msg) e['tariff.rates'] = msg;
    }
  }
  if (!exports.length) e['tariff.rates.export'] = 'Add an export rate (use price 0 for none)';
  else if (!c.tariff.use_pvpc) {
    const msg = checkCoverage(exports, 'export');
    if (msg) e['tariff.rates.export'] = msg;
  }
  return e;
}

export const draftErrors = derived(draft, ($d) => ($d ? validateConfig($d) : {}));
export const draftValid = derived(draftErrors, ($e) => Object.keys($e).length === 0);
