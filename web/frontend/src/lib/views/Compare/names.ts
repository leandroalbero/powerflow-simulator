import { derived } from 'svelte/store';
import { presets, strategyById } from '../../stores/app';
import { numTrim } from '../../ui/format';

const FALLBACK: Record<string, string> = { '2_0td': '2.0TD', pvpc: 'PVPC', pvpc_indexed_export: 'PVPC indexed export' };

export const tariffNames = derived(presets, ($p) => {
  const m = new Map<string, string>(Object.entries(FALLBACK));
  for (const t of $p?.tariffs ?? []) m.set(t.id, t.name);
  return (id: string) => m.get(id) ?? id;
});

export function capLabel(v: number): string {
  return `${numTrim(v)} kWh`;
}

/** Reactive, unlike strategyName(): a sweep restored on page load can render before the strategy list arrives. */
export const stratNames = derived(strategyById, ($m) => (id: string) => $m.get(id)?.name ?? id.replace(/_/g, ' '));
