import { readable } from 'svelte/store';

export const ROUTES = ['simulate', 'compare', 'backtest', 'history'] as const;
export type Route = (typeof ROUTES)[number];

/** Hash params that name what one tab shows (`#/simulate?run=…`, `#/compare?sweep=…`); they never follow you to another tab. */
const VIEW_PARAMS = ['run', 'sweep'];

export function hashParams(): URLSearchParams {
  const q = location.hash.indexOf('?');
  return new URLSearchParams(q >= 0 ? location.hash.slice(q + 1) : '');
}

export function hashPath(): string {
  const q = location.hash.indexOf('?');
  return q >= 0 ? location.hash.slice(0, q) : location.hash;
}

export const isRouteHash = (r: Route): boolean => new RegExp(`^#/?${r}$`).test(hashPath());

function parse(): Route {
  const h = location.hash.replace(/^#\/?/, '').split(/[/?]/)[0];
  if ((ROUTES as readonly string[]).includes(h)) return h as Route;
  // Unknown or empty hash: show Simulate and make the URL say so, without adding a history entry.
  const q = location.hash.indexOf('?');
  history.replaceState(history.state, '', `${location.pathname}${location.search}#/simulate${q >= 0 ? location.hash.slice(q) : ''}`);
  return 'simulate';
}

export const route = readable<Route>(parse(), (set) => {
  const on = () => set(parse());
  window.addEventListener('hashchange', on);
  return () => window.removeEventListener('hashchange', on);
});

/**
 * Sets or clears one hash param in place, keeping the others. It adds no history entry and fires
 * no hashchange, so it only records what the page already shows.
 */
export function replaceHashParam(key: string, value: string | null): void {
  const p = hashParams();
  if (p.get(key) === value) return;
  if (value) p.set(key, value);
  else p.delete(key);
  const qs = p.toString();
  history.replaceState(history.state, '', `${location.pathname}${location.search}${hashPath() || '#/simulate'}${qs ? `?${qs}` : ''}`);
}

/** Tab link: keeps app-wide params such as mock=1, drops the ones naming what a tab shows. */
export function hrefFor(r: Route): string {
  const p = hashParams();
  for (const k of VIEW_PARAMS) p.delete(k);
  const qs = p.toString();
  return `#/${r}${qs ? `?${qs}` : ''}`;
}

/** Link to a tab showing one run or sweep, keeping app-wide params. */
export function hrefWith(r: Route, key: string, value: string): string {
  const p = hashParams();
  for (const k of VIEW_PARAMS) p.delete(k);
  p.set(key, value);
  return `#/${r}?${p}`;
}
