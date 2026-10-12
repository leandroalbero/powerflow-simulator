import { readable } from 'svelte/store';

export const ROUTES = ['simulate', 'compare', 'backtest', 'history'] as const;
export type Route = (typeof ROUTES)[number];

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

/** Keeps any '?…' suffix in the hash (e.g. mock=1) when switching tabs. */
export function hrefFor(r: Route): string {
  const q = location.hash.indexOf('?');
  return `#/${r}${q >= 0 ? location.hash.slice(q) : ''}`;
}
