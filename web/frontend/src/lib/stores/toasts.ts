import { writable } from 'svelte/store';

export type ToastKind = 'info' | 'success' | 'error' | 'warning';

export interface Toast {
  id: number;
  kind: ToastKind;
  title: string;
  message?: string;
}

export interface ActivityEntry {
  id: number;
  at: number;
  kind: ToastKind;
  text: string;
}

export const toasts = writable<Toast[]>([]);
export const activity = writable<ActivityEntry[]>([]);

let seq = 0;

export function dismissToast(id: number): void {
  toasts.update((t) => t.filter((x) => x.id !== id));
}

/** Record an event in the activity log without interrupting the user. */
export function logActivity(text: string, kind: ToastKind = 'info'): void {
  activity.update((a) => [{ id: ++seq, at: Date.now(), kind, text }, ...a].slice(0, 200));
}

/** Show a toast (and log it). Errors stay until dismissed. */
export function toast(kind: ToastKind, title: string, message?: string, timeoutMs?: number): number {
  const id = ++seq;
  toasts.update((t) => [...t.slice(-3), { id, kind, title, message }]);
  logActivity(message ? `${title}: ${message}` : title, kind);
  const ms = timeoutMs ?? (kind === 'error' ? 0 : 4500);
  if (ms > 0) setTimeout(() => dismissToast(id), ms);
  return id;
}
