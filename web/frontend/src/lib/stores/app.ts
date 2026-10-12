import { derived, get, writable } from 'svelte/store';
import {
  errorMessage,
  fetchRouteList,
  getConfig,
  getDataInfo,
  getPresets,
  getStrategies,
  isUnavailable,
  probeCapability,
  routeDeclared,
  type Capability,
} from '../api/client';
import type { DataInfo, PresetsResponse, StrategyInfo, SystemConfig } from '../types';
import { toast } from './toasts';

export type LoadState = 'loading' | 'ready' | 'error';

export const strategies = writable<StrategyInfo[]>([]);
export const dataInfo = writable<DataInfo | null>(null);
/** Server defaults from GET /api/config, normalised to a full SystemConfig. */
export const defaults = writable<SystemConfig | null>(null);
export const presets = writable<PresetsResponse | null>(null);
export const bootState = writable<LoadState>('loading');
export const bootError = writable<string | null>(null);

export type CapabilityState = 'unknown' | 'available' | 'unavailable';
export const capabilities = writable<Record<Capability, CapabilityState>>({
  presets: 'unknown',
  cancel: 'unknown',
  runs: 'unknown',
  export: 'unknown',
  refresh: 'unknown',
  sweep: 'unknown',
  backtest: 'unknown',
});

export function setCapability(cap: Capability, state: CapabilityState): void {
  capabilities.update((c) => (c[cap] === state ? c : { ...c, [cap]: state }));
}

export const strategyById = derived(strategies, ($s) => new Map($s.map((x) => [x.id, x])));

export function strategyName(id: string): string {
  return get(strategyById).get(id)?.name ?? id.replace(/_/g, ' ');
}

/** Inclusive local date bounds of the dataset, 'YYYY-MM-DD'. */
export const dataBounds = derived(dataInfo, ($d) => {
  const from = $d?.data_from ?? $d?.date_range?.start ?? null;
  const until = $d?.data_until ?? $d?.date_range?.end ?? null;
  if (!from || !until) return null;
  return { min: localDay(from), max: localDay(until), until };
});

function localDay(iso: string): string {
  const d = new Date(iso);
  const p = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

/** EV-related setting only makes sense when the data has an EV channel. */
export const hasEv = derived(dataInfo, ($d) => Boolean($d?.has_ev || $d?.entities?.some((e) => /(^|_)ev($|_)/i.test(e))));

export async function refreshDataInfo(): Promise<void> {
  try {
    dataInfo.set(await getDataInfo());
  } catch (err) {
    toast('error', 'Could not load data info', errorMessage(err));
  }
}

async function loadPresets(): Promise<void> {
  try {
    presets.set(await getPresets());
    setCapability('presets', 'available');
  } catch (err) {
    if (isUnavailable(err)) setCapability('presets', 'unavailable');
    else {
      setCapability('presets', 'unavailable');
      toast('warning', 'Presets unavailable', errorMessage(err));
    }
  }
}

const PROBED: Capability[] = ['cancel', 'runs', 'export', 'refresh', 'sweep', 'backtest'];

/** Re-check which v2 endpoints the backend has (it is being extended while the UI runs). */
export async function probeCapabilities(): Promise<void> {
  const routes = await fetchRouteList();
  if (routes) {
    for (const c of PROBED) setCapability(c, routeDeclared(routes, c) ? 'available' : 'unavailable');
    if (!routeDeclared(routes, 'presets')) setCapability('presets', 'unavailable');
    else if (get(capabilities).presets !== 'available') await loadPresets();
    return;
  }
  await Promise.all(
    PROBED.map(async (c) => setCapability(c, (await probeCapability(c)) ? 'available' : 'unavailable')),
  );
  if (get(capabilities).presets !== 'available') await loadPresets();
}

export async function bootstrap(): Promise<void> {
  bootState.set('loading');
  try {
    const [strats, cfg, info] = await Promise.all([getStrategies(), getConfig(), getDataInfo()]);
    strategies.set(strats);
    dataInfo.set(info);
    defaults.set({
      battery: cfg.battery,
      grid: cfg.grid,
      tariff: cfg.tariff,
      strategy: cfg.strategy,
      exclude_ev: cfg.exclude_ev ?? false,
    });
    bootState.set('ready');
    bootError.set(null);
  } catch (err) {
    bootError.set(errorMessage(err));
    bootState.set('error');
  }
  void probeCapabilities();
}
