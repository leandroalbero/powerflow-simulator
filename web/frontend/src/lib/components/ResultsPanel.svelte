<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { currentRun, strategies, chartVisibleStrategies, toggleChartVisibility } from '../stores/simulation';
  import { dataInfo } from '../stores/config';
  import type { StrategyMetrics, StrategyRunState } from '../types/index';

  export let collapsed = false;

  const dispatch = createEventDispatcher();

  // ---- Helpers ----

  function getStrategyName(id: string): string {
    const s = $strategies.find(st => st.id === id);
    return s ? s.name : id;
  }

  function formatEur(v: number): string {
    return v.toFixed(0);
  }

  function formatKwh(v: number): string {
    return Math.round(v).toLocaleString('en-US');
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
    // Sort by total cost ascending (best first)
    completed.sort((a, b) => a.metrics.total_cost - b.metrics.total_cost);
    return completed;
  })();

  $: isComparison = completedStrategies.length > 1;
  $: isSingle = completedStrategies.length === 1;

  // Initialize chart visibility when strategies complete
  $: {
    const ids = completedStrategies.map(s => s.id);
    if (ids.length > 0 && $chartVisibleStrategies === null) {
      chartVisibleStrategies.set(new Set(ids));
    }
    // Add newly completed strategies
    if ($chartVisibleStrategies !== null) {
      let changed = false;
      const next = new Set($chartVisibleStrategies);
      for (const id of ids) {
        if (!next.has(id)) {
          next.add(id);
          changed = true;
        }
      }
      if (changed) chartVisibleStrategies.set(next);
    }
  }

  // ---- Comparison helpers ----

  /** Compute simulation period in years from data info. */
  $: periodYears = (() => {
    if (!$dataInfo?.date_range) return null;
    const start = new Date($dataInfo.date_range.start);
    const end = new Date($dataInfo.date_range.end);
    const days = (end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24);
    return days / 365.25;
  })();

  /** Use the most expensive strategy as baseline for savings calculation. */
  $: baselineCost = completedStrategies.length > 0
    ? Math.max(...completedStrategies.map(s => s.metrics.total_cost))
    : 0;

  $: bestCost = completedStrategies.length > 0
    ? Math.min(...completedStrategies.map(s => s.metrics.total_cost))
    : 0;

  $: costRange = baselineCost - bestCost;

  function isVisible(id: string): boolean {
    return $chartVisibleStrategies === null || $chartVisibleStrategies.has(id);
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

  // ---- Detail metrics for single view ----

  interface MetricDef {
    key: keyof StrategyMetrics;
    label: string;
    unit: string;
    format: (v: number) => string;
  }

  const METRICS: MetricDef[] = [
    { key: 'total_cost', label: 'Total Cost', unit: '\u20ac', format: v => v.toFixed(2) },
    { key: 'total_grid_imported', label: 'Grid Import', unit: 'kWh', format: v => formatKwh(v) },
    { key: 'total_solar_generated', label: 'Solar Gen.', unit: 'kWh', format: v => formatKwh(v) },
    { key: 'total_solar_consumed', label: 'Solar Cons.', unit: 'kWh', format: v => formatKwh(v) },
    { key: 'total_solar_exported', label: 'Solar Exp.', unit: 'kWh', format: v => formatKwh(v) },
    { key: 'self_consumption_rate', label: 'Self Cons.', unit: '%', format: v => v.toFixed(1) },
    { key: 'solar_fraction', label: 'Solar Frac.', unit: '%', format: v => v.toFixed(1) },
    { key: 'total_battery_in', label: 'Battery In', unit: 'kWh', format: v => formatKwh(v) },
    { key: 'total_battery_out', label: 'Battery Out', unit: 'kWh', format: v => formatKwh(v) },
  ];
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
      <div class="comparison-section">
        <div class="comparison-header">
          <span class="section-title">Cost Ranking</span>
          {#if periodYears}
            <span class="period-info">{periodYears.toFixed(1)} yr</span>
          {/if}
        </div>

        {#each completedStrategies as strat, i (strat.id)}
          {@const cost = strat.metrics.total_cost}
          {@const savings = baselineCost - cost}
          {@const perYear = periodYears ? cost / periodYears : null}
          {@const savingsPerYear = periodYears ? savings / periodYears : null}
          {@const barPct = costRange > 0 ? ((cost - bestCost) / costRange) * 100 : 0}
          {@const visible = isVisible(strat.id)}

          <button
            class="strat-row"
            class:best={i === 0}
            class:dimmed={!visible}
            on:click={() => toggleChartVisibility(strat.id)}
            title="{visible ? 'Hide' : 'Show'} on chart"
          >
            <div class="strat-header">
              <span class="strat-eye">{visible ? '\u25C9' : '\u25CB'}</span>
              <span class="strat-rank">#{i + 1}</span>
              <span class="strat-name">{getStrategyName(strat.id)}</span>
            </div>

            <div class="strat-numbers">
              <span class="strat-cost mono">{formatEur(cost)}€</span>
              {#if perYear}
                <span class="strat-per-year mono">{formatEur(perYear)}€/yr</span>
              {/if}
              {#if savings > 0}
                <span class="strat-savings mono positive">+{formatEur(savings)}€</span>
              {:else}
                <span class="strat-savings mono baseline">base</span>
              {/if}
            </div>

            <div class="cost-bar-track">
              <div
                class="cost-bar-fill"
                class:bar-best={i === 0}
                class:bar-worst={i === completedStrategies.length - 1}
                style="width: {100 - barPct}%"
              ></div>
            </div>
          </button>
        {/each}

        <!-- Detail metrics table -->
        <div class="detail-section">
          <div class="section-title">Detail Comparison</div>
          <table class="detail-table">
            <thead>
              <tr>
                <th class="detail-metric-col"></th>
                {#each completedStrategies as strat (strat.id)}
                  <th class="detail-strat-col" title={getStrategyName(strat.id)}>
                    {getStrategyName(strat.id).split(/[\s]+/).map(w => w[0]).join('')}
                  </th>
                {/each}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td class="detail-label">Grid<span class="detail-unit">kWh</span></td>
                {#each completedStrategies as strat (strat.id)}
                  {@const val = strat.metrics.total_grid_imported}
                  {@const allVals = completedStrategies.map(s => s.metrics.total_grid_imported)}
                  <td class="detail-val mono" class:best-val={val <= Math.min(...allVals)}>
                    {formatKwh(val)}
                  </td>
                {/each}
              </tr>
              <tr>
                <td class="detail-label">Self<span class="detail-unit">%</span></td>
                {#each completedStrategies as strat (strat.id)}
                  {@const val = strat.metrics.self_consumption_rate}
                  {@const allVals = completedStrategies.map(s => s.metrics.self_consumption_rate)}
                  <td class="detail-val mono" class:best-val={val >= Math.max(...allVals)}>
                    {val.toFixed(1)}
                  </td>
                {/each}
              </tr>
              <tr>
                <td class="detail-label">Solar<span class="detail-unit">%</span></td>
                {#each completedStrategies as strat (strat.id)}
                  {@const val = strat.metrics.solar_fraction}
                  {@const allVals = completedStrategies.map(s => s.metrics.solar_fraction)}
                  <td class="detail-val mono" class:best-val={val >= Math.max(...allVals)}>
                    {val.toFixed(1)}
                  </td>
                {/each}
              </tr>
              <tr>
                <td class="detail-label">Export<span class="detail-unit">kWh</span></td>
                {#each completedStrategies as strat (strat.id)}
                  {@const val = strat.metrics.total_solar_exported}
                  {@const allVals = completedStrategies.map(s => s.metrics.total_solar_exported)}
                  <td class="detail-val mono" class:best-val={val >= Math.max(...allVals)}>
                    {formatKwh(val)}
                  </td>
                {/each}
              </tr>
              <tr>
                <td class="detail-label">Bat In<span class="detail-unit">kWh</span></td>
                {#each completedStrategies as strat (strat.id)}
                  <td class="detail-val mono">
                    {formatKwh(strat.metrics.total_battery_in)}
                  </td>
                {/each}
              </tr>
              <tr>
                <td class="detail-label">Bat Out<span class="detail-unit">kWh</span></td>
                {#each completedStrategies as strat (strat.id)}
                  <td class="detail-val mono">
                    {formatKwh(strat.metrics.total_battery_out)}
                  </td>
                {/each}
              </tr>
            </tbody>
          </table>
        </div>
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

  .mono {
    font-family: var(--font-mono);
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

  .comparison-section {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
  }

  .comparison-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
  }

  .section-title {
    font-size: 10px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .period-info {
    font-size: 10px;
    color: var(--text-dim);
    font-family: var(--font-mono);
  }

  /* ---- Strategy row ---- */

  .strat-row {
    all: unset;
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding: 6px var(--spacing-sm);
    border: 1px solid var(--border);
    border-radius: 3px;
    cursor: pointer;
    transition: border-color 0.1s, opacity 0.15s;
  }

  .strat-row:hover {
    border-color: var(--border-active);
  }

  .strat-row.best {
    border-color: rgba(34, 197, 94, 0.3);
  }

  .strat-row.dimmed {
    opacity: 0.4;
  }

  .strat-header {
    display: flex;
    align-items: center;
    gap: var(--spacing-xs);
  }

  .strat-eye {
    font-size: 11px;
    color: var(--text-dim);
    flex-shrink: 0;
  }

  .strat-rank {
    font-size: 10px;
    color: var(--text-dim);
    font-family: var(--font-mono);
    flex-shrink: 0;
  }

  .strat-name {
    font-size: var(--font-size-sm);
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .strat-numbers {
    display: flex;
    align-items: baseline;
    gap: var(--spacing-sm);
    padding-left: 26px; /* align with name */
  }

  .strat-cost {
    font-size: var(--font-size-base);
    color: var(--text-primary);
    font-weight: 500;
  }

  .strat-cost {
    min-width: 55px;
  }

  .strat-per-year {
    font-size: 10px;
    color: var(--text-dim);
  }

  .strat-savings {
    font-size: 10px;
    margin-left: auto;
  }

  .strat-savings.positive {
    color: var(--color-success, #22c55e);
  }

  .strat-savings.baseline {
    color: var(--text-dim);
  }

  /* Cost bar */

  .cost-bar-track {
    height: 3px;
    background: var(--bg-input);
    border-radius: 1px;
    overflow: hidden;
    margin-left: 26px;
  }

  .cost-bar-fill {
    height: 100%;
    background: var(--text-dim);
    border-radius: 1px;
    transition: width 0.3s ease;
  }

  .cost-bar-fill.bar-best {
    background: var(--color-success, #22c55e);
  }

  .cost-bar-fill.bar-worst {
    background: var(--color-error, #ef4444);
  }

  /* ---- Detail comparison table ---- */

  .detail-section {
    margin-top: var(--spacing-sm);
    padding-top: var(--spacing-sm);
    border-top: 1px solid var(--border);
  }

  .detail-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 10px;
    margin-top: var(--spacing-xs);
  }

  .detail-table th {
    padding: 2px 3px;
    font-weight: 500;
    color: var(--text-secondary);
    text-align: right;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
    max-width: 36px;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .detail-metric-col {
    text-align: left !important;
  }

  .detail-strat-col {
    font-family: var(--font-mono);
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }

  .detail-table td {
    padding: 2px 3px;
    border-bottom: 1px solid var(--border);
  }

  .detail-label {
    color: var(--text-dim);
    white-space: nowrap;
  }

  .detail-unit {
    color: var(--text-dim);
    font-size: 9px;
    margin-left: 2px;
    opacity: 0.6;
  }

  .detail-val {
    text-align: right;
    color: var(--text-primary);
  }

  .best-val {
    color: var(--color-success, #22c55e);
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
