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

  import { strategies } from './lib/stores/simulation';
  import { systemConfig, dataInfo } from './lib/stores/config';
  import { addLog } from './lib/stores/simulation';
  import { getStrategies, getConfig, getDataInfo } from './lib/api/client';

  let logCollapsed = false;

  onMount(async () => {
    // Load strategies
    try {
      const strats = await getStrategies();
      strategies.set(strats);
      addLog(`Loaded ${strats.length} strategies`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      addLog(`Failed to load strategies: ${message}`, 'error');
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
      const message = err instanceof Error ? err.message : String(err);
      addLog(`Failed to load config: ${message}`, 'error');
    }

    // Load data info (point counts and date range)
    try {
      const info = await getDataInfo();
      dataInfo.set(info);
      addLog(`Data loaded: ${info.solar_point_count + info.load_point_count} points`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      addLog(`Failed to load data info: ${message}`, 'error');
    }
  });
</script>

<div class="shell" class:log-collapsed={logCollapsed}>
  <!-- Toolbar -->
  <header class="panel toolbar">
    <Toolbar />
  </header>

  <!-- Config sidebar -->
  <aside class="panel config">
    <ConfigPanel />
  </aside>

  <!-- Main canvas (charts) -->
  <main class="panel canvas">
    <Chart />
  </main>

  <!-- Results sidebar -->
  <aside class="panel results">
    <ResultsPanel />
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
  }

  .shell.log-collapsed {
    grid-template-rows: 44px 1fr 28px 24px;
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
  }

  .canvas {
    grid-area: canvas;
    background: var(--bg-primary);
  }

  .results {
    grid-area: results;
    background: var(--bg-panel);
    overflow-y: auto;
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

</style>
