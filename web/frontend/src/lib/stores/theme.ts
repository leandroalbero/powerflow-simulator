import { derived, readable, writable } from 'svelte/store';
import type { ThemeMode } from '../ui/colors';
import { loadJson, saveJson } from './storage';

export type ThemePref = 'system' | 'light' | 'dark';

const KEY = 'pf.theme';

function initialPref(): ThemePref {
  const v = loadJson<string>(KEY);
  return v === 'light' || v === 'dark' ? v : 'system';
}

export const themePref = writable<ThemePref>(initialPref());

const systemDark = readable<boolean>(false, (set) => {
  if (typeof matchMedia !== 'function') return;
  const mq = matchMedia('(prefers-color-scheme: dark)');
  set(mq.matches);
  const on = (e: MediaQueryListEvent) => set(e.matches);
  mq.addEventListener('change', on);
  return () => mq.removeEventListener('change', on);
});

/** The theme actually shown, for canvas code that can't read CSS variables. */
export const theme = derived<[typeof themePref, typeof systemDark], ThemeMode>(
  [themePref, systemDark],
  ([$p, $d]) => ($p === 'system' ? ($d ? 'dark' : 'light') : $p),
);

themePref.subscribe((p) => {
  saveJson(KEY, p === 'system' ? null : p);
  if (typeof document === 'undefined') return;
  if (p === 'system') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = p;
});

export function cycleTheme(): void {
  themePref.update((p) => (p === 'system' ? 'light' : p === 'light' ? 'dark' : 'system'));
}
