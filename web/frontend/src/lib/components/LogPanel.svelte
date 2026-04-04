<script lang="ts">
  import { afterUpdate } from 'svelte';
  import { logs, currentRun } from '../stores/simulation';

  export let collapsed = false;

  let scrollContainer: HTMLDivElement;

  // Auto-scroll to bottom on new log entries
  afterUpdate(() => {
    if (scrollContainer && !collapsed) {
      scrollContainer.scrollTop = scrollContainer.scrollHeight;
    }
  });

  function formatTime(ts: number): string {
    const d = new Date(ts);
    return (
      String(d.getHours()).padStart(2, '0') +
      ':' +
      String(d.getMinutes()).padStart(2, '0') +
      ':' +
      String(d.getSeconds()).padStart(2, '0')
    );
  }

  function levelClass(level: string): string {
    if (level === 'warn') return 'level-warn';
    if (level === 'error') return 'level-error';
    return 'level-info';
  }

  // Build progress bar string for running strategies
  function progressBar(percent: number, width: number = 12): string {
    const filled = Math.round((percent / 100) * width);
    const empty = width - filled;
    return '\u2588'.repeat(filled) + '\u2591'.repeat(empty);
  }

  $: runningStrategies = (() => {
    if (!$currentRun || $currentRun.status !== 'running') return [];
    const entries: Array<{ id: string; progress: number }> = [];
    $currentRun.results.forEach((state, id) => {
      if (state.status === 'running') {
        entries.push({ id, progress: state.progress });
      }
    });
    return entries;
  })();
</script>

<div class="panel-header">
  LOG
  <button class="collapse-toggle" on:click={() => (collapsed = !collapsed)}>
    {collapsed ? '\u25B2' : '\u25BC'}
  </button>
</div>

{#if !collapsed}
  <div class="log-body" bind:this={scrollContainer}>
    {#each $logs as entry (entry.timestamp + entry.message)}
      <div class="log-entry {levelClass(entry.level)}">
        <span class="log-time">{formatTime(entry.timestamp)}</span>
        <span class="log-msg">{entry.message}</span>
      </div>
    {/each}

    {#if runningStrategies.length > 0}
      <div class="progress-section">
        {#each runningStrategies as rs (rs.id)}
          <div class="progress-line">
            <span class="progress-name">{rs.id}</span>
            <span class="progress-bar">{progressBar(rs.progress)}</span>
            <span class="progress-pct">{Math.round(rs.progress)}%</span>
          </div>
        {/each}
      </div>
    {/if}
  </div>
{/if}

<style>
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

  .collapse-toggle {
    all: unset;
    cursor: pointer;
    font-size: 10px;
    color: var(--text-dim);
    padding: 0 var(--spacing-xs);
  }

  .collapse-toggle:hover {
    color: var(--text-secondary);
  }

  .log-body {
    flex: 1;
    overflow-y: auto;
    padding: var(--spacing-xs) var(--spacing-sm);
    font-family: var(--font-mono);
    font-size: var(--font-size-sm);
    line-height: 1.6;
  }

  .log-entry {
    display: flex;
    gap: var(--spacing-sm);
    white-space: nowrap;
  }

  .log-time {
    color: var(--text-dim);
    flex-shrink: 0;
  }

  .log-msg {
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .level-info .log-msg {
    color: var(--text-dim);
  }

  .level-warn .log-msg {
    color: var(--color-warning);
  }

  .level-error .log-msg {
    color: var(--color-error);
  }

  .progress-section {
    margin-top: var(--spacing-xs);
    padding-top: var(--spacing-xs);
    border-top: 1px solid var(--border);
  }

  .progress-line {
    display: flex;
    gap: var(--spacing-sm);
    align-items: center;
    color: var(--text-secondary);
  }

  .progress-name {
    width: 120px;
    overflow: hidden;
    text-overflow: ellipsis;
    flex-shrink: 0;
  }

  .progress-bar {
    color: var(--color-selected);
    letter-spacing: -0.05em;
  }

  .progress-pct {
    width: 36px;
    text-align: right;
    flex-shrink: 0;
  }
</style>
