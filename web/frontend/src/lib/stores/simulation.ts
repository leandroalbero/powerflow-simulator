import { writable } from 'svelte/store';
import type { CurrentRun, LogEntry, StrategyInfo } from '../types/index';

/** All available strategies (loaded from GET /api/strategies). */
export const strategies = writable<StrategyInfo[]>([]);

/** Strategy IDs selected for the next simulation run. */
export const selectedStrategies = writable<Set<string>>(new Set());

/** Active or most-recent simulation run state. */
export const currentRun = writable<CurrentRun | null>(null);

/** Application log buffer shown in the log panel. */
export const logs = writable<LogEntry[]>([]);

/** Append a log entry. */
export function addLog(message: string, level: LogEntry['level'] = 'info'): void {
  logs.update((entries) => [
    ...entries,
    { timestamp: Date.now(), message, level },
  ]);
}
