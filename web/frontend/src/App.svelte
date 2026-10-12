<script lang="ts">
  import { onMount } from 'svelte';
  import './styles/tokens.css';
  import './styles/base.css';
  import TopBar from './lib/shell/TopBar.svelte';
  import SettingsDrawer from './lib/shell/SettingsDrawer.svelte';
  import ActivityFooter from './lib/shell/ActivityFooter.svelte';
  import ToastStack from './lib/ui/ToastStack.svelte';
  import SimulateView from './lib/views/Simulate/SimulateView.svelte';
  import CompareView from './lib/views/Compare/CompareView.svelte';
  import BacktestView from './lib/views/Backtest/BacktestView.svelte';
  import HistoryView from './lib/views/History/HistoryView.svelte';
  import { route } from './lib/stores/route';
  import { bootError, bootState, bootstrap, probeCapabilities } from './lib/stores/app';
  import { resumeLastRun } from './lib/stores/run';
  import Button from './lib/ui/Button.svelte';

  const TITLES = { simulate: 'Simulate', compare: 'Compare', backtest: 'Backtest', history: 'History' } as const;
  $: if (typeof document !== 'undefined') document.title = `${TITLES[$route]} · Powerflow`;

  // Hash navigation keeps the scroll position, which would land a short tab under the sticky bar.
  let shownRoute = $route;
  $: if ($route !== shownRoute) {
    shownRoute = $route;
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' });
  }

  onMount(() => {
    void bootstrap();
    void resumeLastRun();
    // The backend gains endpoints while the UI runs; re-check when the tab comes back.
    const onVis = () => document.visibilityState === 'visible' && void probeCapabilities();
    document.addEventListener('visibilitychange', onVis);
    return () => document.removeEventListener('visibilitychange', onVis);
  });
</script>

<TopBar />

<main class="main">
  {#if $bootState === 'error'}
    <div class="boot-error" role="alert">
      <h1>Can't reach the simulator backend</h1>
      <p>{$bootError}</p>
      <p class="muted">Check that the API server is running, then try again.</p>
      <Button variant="primary" icon="refresh" on:click={bootstrap}>Try again</Button>
    </div>
  {:else if $route === 'simulate'}
    <SimulateView />
  {:else if $route === 'compare'}
    <CompareView />
  {:else if $route === 'backtest'}
    <BacktestView />
  {:else}
    <HistoryView />
  {/if}
</main>

<ActivityFooter />
<SettingsDrawer />
<ToastStack />

<style>
  .main {
    max-width: var(--content-max);
    margin: 0 auto;
    padding: var(--s-5) var(--gutter) var(--s-6);
    min-height: calc(100vh - var(--topbar-h) - 60px);
  }
  .boot-error {
    max-width: 56ch;
    padding: var(--s-7) 0;
    display: grid;
    gap: var(--s-3);
    justify-items: start;
  }
  .boot-error h1 {
    font-size: var(--fs-xl);
  }
  @media (max-width: 720px) {
    .main {
      padding-top: var(--s-4);
    }
  }
</style>
