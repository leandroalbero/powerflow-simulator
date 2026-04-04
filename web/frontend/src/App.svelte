<script lang="ts">
  import './styles/fonts.css';
  import './styles/theme.css';

  let logCollapsed = false;
</script>

<div class="shell" class:log-collapsed={logCollapsed}>
  <!-- Toolbar -->
  <header class="panel toolbar">
    <span class="toolbar-title">POWERFLOW SIMULATOR</span>
  </header>

  <!-- Config sidebar -->
  <aside class="panel config">
    <div class="panel-header">CONFIG</div>
  </aside>

  <!-- Main canvas (charts) -->
  <main class="panel canvas">
    <div class="panel-header">CANVAS</div>
  </main>

  <!-- Results sidebar -->
  <aside class="panel results">
    <div class="panel-header">RESULTS</div>
  </aside>

  <!-- Log panel -->
  <section class="panel log">
    <div class="panel-header">
      LOG
      <button class="log-toggle" on:click={() => (logCollapsed = !logCollapsed)}>
        {logCollapsed ? '\u25B2' : '\u25BC'}
      </button>
    </div>
  </section>

  <!-- Status bar -->
  <footer class="panel statusbar">
    <span class="text-dim">Ready</span>
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
    padding: 0 var(--spacing-md);
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
  }

  .toolbar-title {
    font-family: var(--font-mono);
    font-size: var(--font-size-lg);
    font-weight: 500;
    letter-spacing: 0.08em;
    color: var(--text-primary);
  }

  .config {
    grid-area: config;
    background: var(--bg-panel);
    overflow-y: auto;
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

  .log-toggle {
    all: unset;
    cursor: pointer;
    font-size: 10px;
    color: var(--text-dim);
    padding: 0 var(--spacing-xs);
  }

  .log-toggle:hover {
    color: var(--text-secondary);
  }
</style>
