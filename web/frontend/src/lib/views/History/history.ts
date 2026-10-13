import { exportCsv } from '../../api/client';
import { BASELINE_ID, TIE_EUR } from '../../stores/run';
import { hrefWith, type Route } from '../../stores/route';
import type { ExportKind, RunListItem, RunStatusValue } from '../../types';

export interface RunDigest {
  /** Non-baseline strategies, in the order they were run. */
  strategyIds: string[];
  bestId: string | null;
  bestCost: number | null;
  /** Baseline net cost minus the best strategy's, for the run's whole period. */
  savings: number | null;
  outcome: 'win' | 'tie' | 'loss' | null;
  failed: number;
}

type ListStrategy = RunListItem['strategies'][number];

const cost = (s: ListStrategy): number | null =>
  s.status === 'completed' ? (s.adjusted_cost ?? s.total_cost ?? null) : null;

/** Same rules as the Simulate KPIs: rank by adjusted cost, half a cent counts as a tie. */
export function digestRun(r: RunListItem): RunDigest {
  const base = r.strategies.find((s) => s.strategy_id === BASELINE_ID);
  const baseCost = base ? cost(base) : null;
  const others = r.strategies.filter((s) => s.strategy_id !== BASELINE_ID);
  let best: ListStrategy | null = null;
  for (const s of others) {
    const c = cost(s);
    if (c == null) continue;
    if (!best || c < (cost(best) as number)) best = s;
  }
  const bestCost = best ? cost(best) : null;
  const savings = bestCost != null && baseCost != null ? baseCost - bestCost : null;
  const outcome = savings == null ? null : savings >= TIE_EUR ? 'win' : savings > -TIE_EUR ? 'tie' : 'loss';
  return {
    strategyIds: others.map((s) => s.strategy_id),
    bestId: best?.strategy_id ?? null,
    bestCost,
    savings,
    outcome,
    failed: r.strategies.filter((s) => s.status === 'failed').length,
  };
}

/** "Self Consume vs 3 more", or "Self Consume vs no battery" for a single strategy. */
export function runTitle(r: RunListItem, ids: string[], name: (id: string) => string): string {
  if (r.label?.trim()) return r.label.trim();
  if (!ids.length) return 'No battery only';
  return ids.length === 1 ? `${name(ids[0])} vs no battery` : `${name(ids[0])} vs ${ids.length - 1} more`;
}

export const STATUS_LABEL: Record<RunStatusValue, string> = {
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

/**
 * Stable hue per settings fingerprint, so runs with identical settings are easy to spot by eye.
 * Purely a recognition aid: the digest text is what identifies the settings.
 */
export function digestHue(digest: string): number {
  const n = parseInt(digest.slice(0, 6), 16);
  // Golden-angle steps spread nearby codes far apart on the colour wheel.
  return Number.isFinite(n) ? Math.round((n * 137.508) % 360) : 0;
}

/** Link to a tab showing one run or sweep, keeping app-wide params such as mock=1. */
export const hashHref = (route: Route, key: string, value: string): string => hrefWith(route, key, value);

export async function downloadRunCsv(runId: string, kind: ExportKind): Promise<void> {
  const { blob, filename } = await exportCsv(runId, kind);
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename ?? `powerflow-${runId}-${kind}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
