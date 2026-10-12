<script lang="ts">
  import { route, ROUTES, hrefFor } from '../stores/route';
  import { MOCK_MODE } from '../api/client';
  import Tabs from '../ui/Tabs.svelte';
  import Badge from '../ui/Badge.svelte';
  import DataBadge from './DataBadge.svelte';
  import ThemeToggle from './ThemeToggle.svelte';

  const LABELS = { simulate: 'Simulate', compare: 'Compare', backtest: 'Backtest', history: 'History' } as const;
  $: items = ROUTES.map((r) => ({ id: r, label: LABELS[r], href: hrefFor(r) }));
</script>

<header class="bar">
  <div class="inner">
    <a class="brand" href={hrefFor('simulate')} aria-label="Powerflow, home">
      <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
        <rect x="3" y="6" width="15" height="12" rx="2.5" fill="none" stroke="currentColor" stroke-width="1.8" />
        <path d="M21 10.5v3" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" />
        <path d="M11.5 8.5 8.5 12.5h3l-1 3.5 3.5-4.5h-3l1-3Z" fill="currentColor" />
      </svg>
      <span>Powerflow</span>
      {#if MOCK_MODE}<Badge tone="warning" title="New endpoints are simulated in the browser">Mock</Badge>{/if}
    </a>
    <div class="tabs"><Tabs {items} current={$route} label="Main" /></div>
    <div class="right">
      <DataBadge />
      <ThemeToggle />
    </div>
  </div>
</header>

<style>
  .bar {
    position: sticky;
    top: 0;
    z-index: 20;
    background: color-mix(in srgb, var(--surface) 88%, transparent);
    backdrop-filter: saturate(1.4) blur(10px);
    -webkit-backdrop-filter: saturate(1.4) blur(10px);
    border-bottom: 1px solid var(--border);
  }
  .inner {
    max-width: var(--content-max);
    margin: 0 auto;
    padding: 0 var(--gutter);
    height: var(--topbar-h);
    display: flex;
    align-items: stretch;
    gap: var(--s-5);
  }
  .brand {
    display: inline-flex;
    align-items: center;
    gap: var(--s-2);
    color: var(--text);
    text-decoration: none;
    font-weight: 700;
    font-size: var(--fs-lg);
    letter-spacing: -0.02em;
    flex: none;
  }
  .brand svg {
    color: var(--accent);
  }
  .tabs {
    flex: 1;
    min-width: 0;
    display: flex;
  }
  .right {
    display: flex;
    align-items: center;
    gap: var(--s-2);
    flex: none;
  }
  @media (max-width: 720px) {
    .inner {
      flex-wrap: wrap;
      height: auto;
      row-gap: 0;
      gap: 0 var(--s-3);
    }
    .brand {
      height: 52px;
      flex: 1;
    }
    .tabs {
      order: 3;
      flex: 1 0 100%;
      height: 44px;
      margin: 0 calc(var(--s-3) * -1);
    }
  }
</style>
