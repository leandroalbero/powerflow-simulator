import { writable } from 'svelte/store';
import type { CurrentRun, DateRange, LogEntry, StrategyInfo } from '../types/index';

/** All available strategies (loaded from GET /api/strategies). */
export const strategies = writable<StrategyInfo[]>([]);

/** Strategy IDs selected for the next simulation run. */
export const selectedStrategies = writable<Set<string>>(new Set());

/** Toggle a strategy in/out of the selection (returns a new Set for reactivity). */
export function toggleStrategy(id: string): void {
  selectedStrategies.update((s) => {
    const next = new Set(s);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    return next;
  });
}

/** Date range for the next simulation run (null = use full data range). */
export const selectedDateRange = writable<DateRange | null>(null);

/** Active or most-recent simulation run state. */
export const currentRun = writable<CurrentRun | null>(null);

/** Application log buffer shown in the log panel. */
export const logs = writable<LogEntry[]>([]);

/** Append a log entry (capped at 500 entries). */
export function addLog(message: string, level: LogEntry['level'] = 'info'): void {
  logs.update((entries) => [
    ...entries.slice(-499),
    { timestamp: Date.now(), message, level },
  ]);
}
