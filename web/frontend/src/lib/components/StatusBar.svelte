<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { currentRun } from '../stores/simulation';
  import { dataInfo } from '../stores/config';
  import { healthCheck } from '../api/client';

  let connected = false;
  let healthTimer: ReturnType<typeof setInterval> | null = null;

  async function checkHealth() {
    try {
      await healthCheck();
      connected = true;
    } catch {
      connected = false;
    }
  }

  onMount(() => {
    checkHealth();
    healthTimer = setInterval(checkHealth, 10000);
  });

  onDestroy(() => {
    if (healthTimer) clearInterval(healthTimer);
  });

  $: pointCount = $dataInfo
    ? $dataInfo.solar_point_count + $dataInfo.load_point_count
    : 0;

  $: runStatus = (() => {
    if (!$currentRun) return '';
    if ($currentRun.status === 'running') return 'Running...';
    if ($currentRun.status === 'completed') {
      const total = $currentRun.results.size;
      let ok = 0;
      $currentRun.results.forEach((s) => {
        if (s.status === 'completed') ok++;
      });
      return `Last run: ${ok}/${total} strategies completed`;
    }
    return `Run: ${$currentRun.status}`;
  })();
</script>

<div class="statusbar-content">
  <span class="status-group">
    <span class="dot" class:dot-connected={connected} class:dot-disconnected={!connected}></span>
    <span class="status-label">{connected ? 'Connected' : 'Disconnected'}</span>
  </span>

  <span class="separator"></span>

  {#if pointCount > 0}
    <span class="status-item">{pointCount.toLocaleString()} data points</span>
    <span class="separator"></span>
  {/if}

  {#if runStatus}
    <span class="status-item">{runStatus}</span>
    <span class="separator"></span>
  {/if}

  <span class="version">v0.1</span>
</div>

<style>
  .statusbar-content {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    width: 100%;
    height: 100%;
    font-size: var(--font-size-sm);
    color: var(--text-dim);
  }

  .status-group {
    display: flex;
    align-items: center;
    gap: var(--spacing-xs);
  }

  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .dot-connected {
    background: var(--color-success);
  }

  .dot-disconnected {
    background: var(--color-error);
  }

  .status-label {
    font-size: var(--font-size-sm);
  }

  .separator {
    width: 1px;
    height: 12px;
    background: var(--border);
    flex-shrink: 0;
  }

  .status-item {
    font-family: var(--font-mono);
    font-size: var(--font-size-sm);
  }

  .version {
    margin-left: auto;
    font-family: var(--font-mono);
    font-size: var(--font-size-sm);
    color: var(--text-dim);
  }
</style>
