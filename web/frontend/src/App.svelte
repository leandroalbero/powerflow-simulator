<script lang="ts">
  import { onMount } from 'svelte';
  import './styles/fonts.css';
  import './styles/theme.css';

  import Toolbar from './lib/components/Toolbar.svelte';
  import ConfigPanel from './lib/components/ConfigPanel.svelte';
  import Chart from './lib/components/Chart.svelte';
  import ResultsPanel from './lib/components/ResultsPanel.svelte';
  import LogPanel from './lib/components/LogPanel.svelte';
  import StatusBar from './lib/components/StatusBar.svelte';

  import { strategies, toggleStrategy } from './lib/stores/simulation';
  import { systemConfig, dataInfo } from './lib/stores/config';
  import { addLog } from './lib/stores/simulation';
  import { getStrategies, getConfig, getDataInfo } from './lib/api/client';

  let logCollapsed = false;
  let configCollapsed = false;
  let resultsCollapsed = false;

  // ---- Keyboard shortcuts ----

  function handleKeydown(event: KeyboardEvent) {
    // Skip if user is typing in an input/textarea/select
    const tag = (event.target as HTMLElement)?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

    // Ctrl+Enter / Cmd+Enter = trigger RUN
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      event.preventDefault();
      // Dispatch a custom event the Toolbar listens for
      window.dispatchEvent(new CustomEvent('pf-run'));
      return;
    }

    // Don't process single-key shortcuts when modifier keys are held
    if (event.ctrlKey || event.metaKey || event.altKey) return;

    switch (event.key) {
      case '[':
        event.preventDefault();
        configCollapsed = !configCollapsed;
        break;
      case ']':
        event.preventDefault();
        resultsCollapsed = !resultsCollapsed;
        break;
      case '\\':
        event.preventDefault();
        logCollapsed = !logCollapsed;
        break;
      case '1':
      case '2':
      case '3':
      case '4': {
        event.preventDefault();
        const idx = parseInt(event.key) - 1;
        const strats = $strategies;
        if (idx < strats.length) {
          toggleStrategy(strats[idx].id);
        }
        break;
      }
    }
  }

  function errorMessage(err: unknown): string {
    if (err instanceof TypeError && (err.message === 'Failed to fetch' || err.message.includes('NetworkError'))) {
      return 'Network error -- is the backend running? (make web-dev)';
    }
    return err instanceof Error ? err.message : String(err);
  }

  onMount(async () => {
    // Load strategies
    try {
      const strats = await getStrategies();
      strategies.set(strats);
      addLog(`Loaded ${strats.length} strategies`);
    } catch (err) {
      addLog(`Failed to load strategies: ${errorMessage(err)}`, 'error');
    }

    // Load config
    try {
      const configResp = await getConfig();
      systemConfig.set({
        battery: configResp.battery,
        grid: configResp.grid,
        tariff: configResp.tariff,
      });
      addLog('Config loaded');
    } catch (err) {
      addLog(`Failed to load config: ${errorMessage(err)}`, 'error');
    }

    // Load data info (point counts and date range)
    try {
      const info = await getDataInfo();
      dataInfo.set(info);
      addLog(`Data loaded: ${info.solar_point_count + info.load_point_count} points`);
    } catch (err) {
      addLog(`Failed to load data info: ${errorMessage(err)}`, 'error');
    }
  });
</script>

<svelte:window on:keydown={handleKeydown} />

<div
  class="shell"
  class:log-collapsed={logCollapsed}
  class:config-collapsed={configCollapsed}
  class:results-collapsed={resultsCollapsed}
>
  <!-- Toolbar -->
  <header class="panel toolbar">
    <Toolbar />
  </header>

  <!-- Config sidebar -->
  <aside class="panel config">
    {#if configCollapsed}
      <button class="expand-handle expand-config" on:click={() => (configCollapsed = false)} title="Expand config [">
        &#x25B6;
      </button>
    {:else}
      <ConfigPanel collapsed={configCollapsed} on:toggle={() => (configCollapsed = !configCollapsed)} />
    {/if}
  </aside>

  <!-- Main canvas (charts) -->
  <main class="panel canvas">
    <Chart />
  </main>

  <!-- Results sidebar -->
  <aside class="panel results">
    {#if resultsCollapsed}
      <button class="expand-handle expand-results" on:click={() => (resultsCollapsed = false)} title="Expand results ]">
        &#x25C0;
      </button>
    {:else}
      <ResultsPanel collapsed={resultsCollapsed} on:toggle={() => (resultsCollapsed = !resultsCollapsed)} />
    {/if}
  </aside>

  <!-- Log panel -->
  <section class="panel log">
    <LogPanel bind:collapsed={logCollapsed} />
  </section>

  <!-- Status bar -->
  <footer class="panel statusbar">
    <StatusBar />
  </footer>
</div>

<style>
  .shell {
    height: 100%;
    display: grid;
    grid-template-columns: 220px 1fr 260px;
    grid-template-rows: 44px 1fr 100px 24px;
    grid-template-areas:
      'toolbar  toolbar  toolbar'
      'config   canvas   results'
      'log      log      log'
      'status   status   status';
    transition: grid-template-columns 0.2s ease, grid-template-rows 0.2s ease;
  }

  .shell.log-collapsed {
    grid-template-rows: 44px 1fr 28px 24px;
  }

  /* ---- Panel collapse grid overrides ---- */

  .shell.config-collapsed {
    grid-template-columns: 24px 1fr 260px;
  }

  .shell.results-collapsed {
    grid-template-columns: 220px 1fr 24px;
  }

  .shell.config-collapsed.results-collapsed {
    grid-template-columns: 24px 1fr 24px;
  }

  .panel {
    border: 1px solid var(--border);
    overflow: hidden;
  }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 28px;
    padding: 0 var(--spacing-sm);
    font-size: var(--font-size-sm);
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid var(--border);
    background: var(--bg-panel);
    flex-shrink: 0;
  }

  /* ---- Grid areas ---- */

  .toolbar {
    grid-area: toolbar;
    display: flex;
    align-items: center;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
  }

  .config {
    grid-area: config;
    background: var(--bg-panel);
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    min-width: 0;
  }

  .canvas {
    grid-area: canvas;
    background: var(--bg-primary);
  }

  .results {
    grid-area: results;
    background: var(--bg-panel);
    overflow-y: auto;
    min-width: 0;
  }

  .log {
    grid-area: log;
    background: var(--bg-panel);
    display: flex;
    flex-direction: column;
  }

  .statusbar {
    grid-area: status;
    display: flex;
    align-items: center;
    padding: 0 var(--spacing-sm);
    background: var(--bg-panel);
    font-size: var(--font-size-sm);
    border-top: 1px solid var(--border);
  }

  /* ---- Expand handles for collapsed panels ---- */

  .expand-handle {
    all: unset;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    cursor: pointer;
    font-size: 10px;
    color: var(--text-dim);
    background: var(--bg-panel);
    transition: color 0.1s;
  }

  .expand-handle:hover {
    color: var(--text-secondary);
  }
</style>
