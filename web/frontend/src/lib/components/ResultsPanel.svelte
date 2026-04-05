<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { currentRun, strategies } from '../stores/simulation';
  import type { StrategyMetrics, StrategyRunState } from '../types/index';

  export let collapsed = false;

  const dispatch = createEventDispatcher();

  // ---- Types ----

  interface MetricDef {
    key: keyof StrategyMetrics;
    label: string;
    unit: string;
    format: (v: number) => string;
    /** For comparison: 'low' means lowest is best, 'high' means highest is best. */
    best: 'low' | 'high';
  }

  // ---- Metric definitions ----

  const METRICS: MetricDef[] = [
    { key: 'total_cost', label: 'Total Cost', unit: '\u20ac', format: v => v.toFixed(2), best: 'low' },
    { key: 'total_grid_imported', label: 'Grid Import', unit: 'kWh', format: v => formatInt(v), best: 'low' },
    { key: 'total_solar_generated', label: 'Solar Gen.', unit: 'kWh', format: v => formatInt(v), best: 'high' },
    { key: 'total_solar_consumed', label: 'Solar Cons.', unit: 'kWh', format: v => formatInt(v), best: 'high' },
    { key: 'total_solar_exported', label: 'Solar Exp.', unit: 'kWh', format: v => formatInt(v), best: 'high' },
    { key: 'self_consumption_rate', label: 'Self Cons.', unit: '%', format: v => (v * 100).toFixed(1), best: 'high' },
    { key: 'solar_fraction', label: 'Solar Frac.', unit: '%', format: v => (v * 100).toFixed(1), best: 'high' },
    { key: 'total_battery_in', label: 'Battery In', unit: 'kWh', format: v => formatInt(v), best: 'high' },
    { key: 'total_battery_out', label: 'Battery Out', unit: 'kWh', format: v => formatInt(v), best: 'high' },
  ];

  /** Shorter list for comparison mode to fit the narrow sidebar. */
  const COMPARISON_METRICS: MetricDef[] = [
    { key: 'total_cost', label: 'Cost', unit: '\u20ac', format: v => v.toFixed(0), best: 'low' },
    { key: 'solar_fraction', label: 'Solar', unit: '%', format: v => (v * 100).toFixed(1), best: 'high' },
    { key: 'self_consumption_rate', label: 'Self', unit: '%', format: v => (v * 100).toFixed(1), best: 'high' },
    { key: 'total_grid_imported', label: 'Grid', unit: 'kWh', format: v => formatInt(v), best: 'low' },
    { key: 'total_solar_exported', label: 'Export', unit: 'kWh', format: v => formatInt(v), best: 'high' },
    { key: 'total_battery_in', label: 'Bat In', unit: 'kWh', format: v => formatInt(v), best: 'high' },
    { key: 'total_battery_out', label: 'Bat Out', unit: 'kWh', format: v => formatInt(v), best: 'high' },
  ];

  // ---- Helpers ----

  function formatInt(v: number): string {
    return Math.round(v).toLocaleString('en-US');
  }

  function getStrategyName(id: string): string {
    const s = $strategies.find(st => st.id === id);
    return s ? s.name : id;
  }

  function getStrategyAbbrev(id: string): string {
    const name = getStrategyName(id);
    const words = name.split(/[\s_-]+/);
    if (words.length === 1) return name.substring(0, 4).toUpperCase();
    return words.map(w => w[0]).join('').toUpperCase();
  }

  // ---- Reactive data ----

  $: results = $currentRun?.results ?? new Map<string, StrategyRunState>();

  $: completedStrategies = (() => {
    const completed: Array<{ id: string; metrics: StrategyMetrics }> = [];
    for (const [id, state] of results.entries()) {
      if (state.status === 'completed' && state.metrics) {
        completed.push({ id, metrics: state.metrics });
      }
    }
    return completed;
  })();

  $: isComparison = completedStrategies.length > 1;
  $: isSingle = completedStrategies.length === 1;

  // ---- Best value detection for comparison mode ----

  function isBestValue(metric: MetricDef, strategyMetrics: StrategyMetrics): boolean {
    if (completedStrategies.length < 2) return false;

    const currentVal = strategyMetrics[metric.key] as number;
    const allVals = completedStrategies.map(s => s.metrics[metric.key] as number);

    if (metric.best === 'low') {
      return currentVal <= Math.min(...allVals);
    } else {
      return currentVal >= Math.max(...allVals);
    }
  }

  // ---- Running strategies ----

  $: runningStrategies = (() => {
    const running: Array<{ id: string; progress: number }> = [];
    for (const [id, state] of results.entries()) {
      if (state.status === 'running') {
        running.push({ id, progress: state.progress });
      }
    }
    return running;
  })();
</script>

<div class="results-container">
  <div class="panel-header">
    RESULTS
    <button class="collapse-toggle" on:click={() => dispatch('toggle')} title="Toggle results panel ]">
      &#x25B6;
    </button>
  </div>

  <div class="results-body">
    {#if completedStrategies.length === 0 && runningStrategies.length === 0}
      <div class="empty-state">
        <span class="empty-text">No results yet</span>
      </div>
    {/if}

    <!-- Running strategies progress -->
    {#if runningStrategies.length > 0}
      <div class="progress-section">
        {#each runningStrategies as { id, progress } (id)}
          <div class="progress-item">
            <div class="progress-label">{getStrategyName(id)}</div>
            <div class="progress-bar-track">
              <div class="progress-bar-fill" style="width: {progress}%"></div>
            </div>
            <div class="progress-pct mono">{progress}%</div>
          </div>
        {/each}
      </div>
    {/if}

    <!-- Single strategy view -->
    {#if isSingle}
      {@const strat = completedStrategies[0]}
      <div class="single-results">
        <div class="strategy-name">{getStrategyName(strat.id)}</div>
        <table class="metrics-table">
          <tbody>
            {#each METRICS as metric (metric.key)}
              <tr>
                <td class="metric-label">{metric.label}</td>
                <td class="metric-value mono">
                  {#if metric.unit === '\u20ac'}
                    {metric.unit}{metric.format(Number(strat.metrics[metric.key]))}
                  {:else}
                    {metric.format(Number(strat.metrics[metric.key]))}{metric.unit === '%' ? '%' : ''}
                  {/if}
                  {#if metric.unit !== '\u20ac' && metric.unit !== '%'}
                    <span class="metric-unit">{metric.unit}</span>
                  {/if}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}

    <!-- Comparison view -->
    {#if isComparison}
      <div class="comparison-results">
        <table class="comparison-table">
          <thead>
            <tr>
              <th class="metric-col">Metric</th>
              {#each completedStrategies as strat (strat.id)}
                <th class="strat-col" title={getStrategyName(strat.id)}>
                  {getStrategyAbbrev(strat.id)}
                </th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each COMPARISON_METRICS as metric (metric.key)}
              <tr>
                <td class="metric-label">
                  {metric.label}
                  <span class="metric-unit-label">({metric.unit})</span>
                </td>
                {#each completedStrategies as strat (strat.id)}
                  <td
                    class="metric-value mono"
                    class:best-value={isBestValue(metric, strat.metrics)}
                  >
                    {metric.format(Number(strat.metrics[metric.key]))}
                  </td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {/if}

    <!-- Errors -->
    {#each [...results.entries()] as [id, state] (id)}
      {#if state.status === 'error'}
        <div class="error-item">
          <div class="error-label">{getStrategyName(id)}</div>
          <div class="error-message">{state.error || 'Unknown error'}</div>
        </div>
      {/if}
    {/each}
  </div>
</div>

<style>
  .results-container {
    display: flex;
    flex-direction: column;
    height: 100%;
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

  .results-body {
    flex: 1;
    overflow-y: auto;
    padding: var(--spacing-sm);
  }

  .empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 60px;
  }

  .empty-text {
    color: var(--text-dim);
    font-size: var(--font-size-sm);
  }

  /* ---- Progress ---- */

  .progress-section {
    margin-bottom: var(--spacing-md);
  }

  .progress-item {
    display: flex;
    align-items: center;
    gap: var(--spacing-xs);
    margin-bottom: var(--spacing-xs);
  }

  .progress-label {
    font-size: var(--font-size-sm);
    color: var(--text-secondary);
    width: 80px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .progress-bar-track {
    flex: 1;
    height: 4px;
    background: var(--bg-input);
    border-radius: 2px;
    overflow: hidden;
  }

  .progress-bar-fill {
    height: 100%;
    background: var(--color-selected);
    transition: width 0.3s ease;
  }

  .progress-pct {
    font-size: 10px;
    color: var(--text-dim);
    width: 30px;
    text-align: right;
    flex-shrink: 0;
  }

  .mono {
    font-family: var(--font-mono);
  }

  /* ---- Single strategy ---- */

  .single-results {
    margin-bottom: var(--spacing-md);
  }

  .strategy-name {
    font-size: var(--font-size-base);
    color: var(--text-primary);
    font-weight: 500;
    margin-bottom: var(--spacing-sm);
    padding-bottom: var(--spacing-xs);
    border-bottom: 1px solid var(--border);
  }

  .metrics-table {
    width: 100%;
    border-collapse: collapse;
  }

  .metrics-table td {
    padding: 3px 0;
    font-size: var(--font-size-sm);
    border-bottom: none;
  }

  .metrics-table .metric-label {
    color: var(--text-dim);
    text-align: left;
  }

  .metrics-table .metric-value {
    text-align: right;
    color: var(--text-primary);
  }

  .metric-unit {
    color: var(--text-dim);
    font-size: 10px;
    margin-left: 2px;
  }

  /* ---- Comparison ---- */

  .comparison-results {
    margin-bottom: var(--spacing-md);
  }

  .comparison-table {
    width: 100%;
    border-collapse: collapse;
    font-size: var(--font-size-sm);
  }

  .comparison-table th {
    padding: 3px var(--spacing-xs);
    font-size: 10px;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }

  .comparison-table .metric-col {
    text-align: left;
  }

  .comparison-table .strat-col {
    text-align: right;
    font-family: var(--font-mono);
  }

  .comparison-table td {
    padding: 3px var(--spacing-xs);
    border-bottom: 1px solid var(--border);
  }

  .comparison-table .metric-label {
    color: var(--text-dim);
    text-align: left;
    white-space: nowrap;
  }

  .metric-unit-label {
    color: var(--text-dim);
    font-size: 10px;
  }

  .comparison-table .metric-value {
    text-align: right;
    color: var(--text-primary);
    font-size: var(--font-size-sm);
  }

  .best-value {
    color: var(--color-success);
  }

  /* ---- Errors ---- */

  .error-item {
    margin-bottom: var(--spacing-sm);
    padding: var(--spacing-xs) var(--spacing-sm);
    border-left: 2px solid var(--color-error);
  }

  .error-label {
    font-size: var(--font-size-sm);
    color: var(--color-error);
    font-weight: 500;
    margin-bottom: 2px;
  }

  .error-message {
    font-size: var(--font-size-sm);
    color: var(--text-dim);
    font-family: var(--font-mono);
    word-break: break-all;
  }
</style>
