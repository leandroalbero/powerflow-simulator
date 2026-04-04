<script lang="ts">
  import {
    strategies,
    selectedStrategies,
    toggleStrategy,
    selectedDateRange,
    currentRun,
    addLog,
  } from '../stores/simulation';
  import { dataInfo } from '../stores/config';
  import { startSimulation } from '../api/client';
  import { RunWebSocket } from '../api/websocket';
  import type { DateRange } from '../types/index';

  let dateStart = '';
  let dateEnd = '';
  let ws: RunWebSocket | null = null;

  // Sync date inputs from dataInfo when it loads
  $: if ($dataInfo?.date_range && !dateStart && !dateEnd) {
    dateStart = $dataInfo.date_range.start;
    dateEnd = $dataInfo.date_range.end;
  }

  $: isRunning = $currentRun?.status === 'running';
  $: canRun = $selectedStrategies.size > 0 && !isRunning;

  function buildDateRange(): DateRange | undefined {
    if (dateStart && dateEnd) {
      return { start: dateStart, end: dateEnd };
    }
    return undefined;
  }

  async function handleRun() {
    if (!canRun) return;

    const strategyIds = [...$selectedStrategies];
    const dateRange = buildDateRange();

    addLog(`Starting simulation: ${strategyIds.join(', ')}`);

    try {
      const resp = await startSimulation({
        strategies: strategyIds,
        date_range: dateRange,
      });

      const runId = resp.run_id;
      addLog(`Run started: ${runId}`);

      // Initialize currentRun in store
      const results = new Map<string, { progress: number; status: string }>();
      for (const sid of strategyIds) {
        results.set(sid, { progress: 0, status: 'running' });
      }
      currentRun.set({ runId, status: 'running', results });

      // Close any previous WS
      if (ws) ws.close();

      // Open WebSocket for progress
      ws = new RunWebSocket(runId, {
        onProgress(msg) {
          currentRun.update((run) => {
            if (!run) return run;
            const next = new Map(run.results);
            const existing = next.get(msg.strategy) || { progress: 0, status: 'running' };
            next.set(msg.strategy, { ...existing, progress: msg.percent });
            return { ...run, results: next };
          });
        },
        onStrategyDone(msg) {
          addLog(
            msg.status === 'completed'
              ? `Strategy ${msg.strategy} completed`
              : `Strategy ${msg.strategy} failed: ${msg.error || 'unknown'}`,
            msg.status === 'completed' ? 'info' : 'error',
          );
          currentRun.update((run) => {
            if (!run) return run;
            const next = new Map(run.results);
            next.set(msg.strategy, {
              progress: 100,
              status: msg.status,
              metrics: msg.metrics,
              error: msg.error,
            });
            return { ...run, results: next };
          });
        },
        onRunComplete() {
          addLog('Simulation run complete', 'info');
          currentRun.update((run) => {
            if (!run) return run;
            return { ...run, status: 'completed' };
          });
        },
        onError(err) {
          const message = err instanceof Error ? err.message : 'WebSocket error';
          addLog(`WS error: ${message}`, 'error');
        },
        onClose() {
          addLog('WebSocket closed', 'info');
        },
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      addLog(`Failed to start simulation: ${message}`, 'error');
    }
  }
</script>

<div class="toolbar-content">
  <span class="toolbar-title">POWERFLOW SIMULATOR</span>

  <div class="separator"></div>

  <div class="strategy-group">
    {#each $strategies as strat (strat.id)}
      <button
        class="strategy-btn"
        class:selected={$selectedStrategies.has(strat.id)}
        on:click={() => toggleStrategy(strat.id)}
        title={strat.description}
      >
        {strat.name}
      </button>
    {/each}
  </div>

  <div class="separator"></div>

  <div class="date-group">
    <input
      type="date"
      class="date-input"
      bind:value={dateStart}
      on:change={() => selectedDateRange.set(buildDateRange() || null)}
    />
    <span class="date-dash">&ndash;</span>
    <input
      type="date"
      class="date-input"
      bind:value={dateEnd}
      on:change={() => selectedDateRange.set(buildDateRange() || null)}
    />
  </div>

  <button class="run-btn" disabled={!canRun} on:click={handleRun}>
    {#if isRunning}
      RUNNING...
    {:else}
      RUN
    {/if}
  </button>
</div>

<style>
  .toolbar-content {
    display: flex;
    align-items: center;
    gap: var(--spacing-md);
    width: 100%;
    height: 100%;
    padding: 0 var(--spacing-md);
  }

  .toolbar-title {
    font-family: var(--font-mono);
    font-size: var(--font-size-lg);
    font-weight: 500;
    letter-spacing: 0.08em;
    color: var(--text-primary);
    white-space: nowrap;
  }

  .separator {
    width: 1px;
    height: 20px;
    background: var(--border);
    flex-shrink: 0;
  }

  .strategy-group {
    display: flex;
    gap: var(--spacing-xs);
    flex-wrap: nowrap;
    overflow-x: auto;
  }

  .strategy-btn {
    height: 28px;
    padding: 0 var(--spacing-sm);
    font-size: var(--font-size-sm);
    color: var(--text-secondary);
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 2px;
    cursor: pointer;
    white-space: nowrap;
    transition: border-color 0.1s, color 0.1s;
  }

  .strategy-btn:hover {
    border-color: var(--border-active);
    color: var(--text-primary);
  }

  .strategy-btn.selected {
    border-color: var(--color-selected);
    color: var(--text-primary);
  }

  .date-group {
    display: flex;
    align-items: center;
    gap: var(--spacing-xs);
    flex-shrink: 0;
  }

  .date-input {
    height: 28px;
    width: 120px;
    font-size: var(--font-size-sm);
    font-family: var(--font-mono);
    padding: 0 var(--spacing-xs);
    color: var(--text-primary);
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 2px;
    color-scheme: dark;
  }

  .date-input:focus {
    border-color: var(--border-active);
  }

  .date-dash {
    color: var(--text-dim);
    font-size: var(--font-size-sm);
  }

  .run-btn {
    margin-left: auto;
    height: 30px;
    padding: 0 var(--spacing-lg);
    font-size: var(--font-size-sm);
    font-weight: 500;
    letter-spacing: 0.05em;
    color: #fff;
    background: var(--color-selected);
    border: 1px solid var(--color-selected);
    border-radius: 2px;
    cursor: pointer;
    white-space: nowrap;
    transition: opacity 0.1s;
  }

  .run-btn:hover:not(:disabled) {
    opacity: 0.9;
  }

  .run-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
</style>
