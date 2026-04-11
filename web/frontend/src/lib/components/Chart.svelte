<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import uPlot from 'uplot';
  import 'uplot/dist/uPlot.min.css';

  import { currentRun, strategies, addLog, chartVisibleStrategies, addChartVisibility } from '../stores/simulation';
  import { systemConfig } from '../stores/config';
  import { getTimeseries } from '../api/client';
  import type { TimeseriesResponse, StrategyRunState } from '../types/index';

  // ---- Constants ----

  const SERIES_COLORS: Record<string, string> = {
    solar_power: '#f59e0b',
    house_consumption: '#06b6d4',
    grid_import: '#ef4444',
    grid_export: '#22c55e',
    battery_level: '#8b5cf6',
  };

  const SERIES_LABELS: Record<string, string> = {
    solar_power: 'Solar',
    house_consumption: 'Load',
    grid_import: 'Grid Import',
    grid_export: 'Grid Export',
    battery_level: 'Battery SoC',
  };

  const DASH_PATTERNS: number[][] = [
    [],           // solid
    [8, 4],       // dashed
    [2, 4],       // dotted
    [8, 4, 2, 4], // dash-dot
  ];

  const TARIFF_ZONES: Array<{ startHour: number; endHour: number; type: 'valley' | 'peak' }> = [
    { startHour: 0, endHour: 8, type: 'valley' },
    { startHour: 10, endHour: 14, type: 'peak' },
    { startHour: 18, endHour: 22, type: 'peak' },
  ];

  const BAND_COLORS: Record<string, string> = {
    valley: 'rgba(34, 197, 94, 0.05)',
    peak: 'rgba(239, 68, 68, 0.05)',
  };

  const DEFAULT_MAX_POINTS = 4000;
  const ZOOM_THRESHOLD_DAYS = 30;

  // ---- State ----

  let containerEl: HTMLDivElement;
  let powerChartEl: HTMLDivElement;
  let batteryChartEl: HTMLDivElement;

  let powerChart: uPlot | null = null;
  let batteryChart: uPlot | null = null;

  /** Map of strategy ID -> loaded timeseries data. */
  let loadedData: Map<string, TimeseriesResponse> = new Map();

  /** Track which strategies we've already started loading for. */
  let loadingStrategies: Set<string> = new Set();

  /** Track the first auto-loaded strategy. */
  let firstLoadedStrategy: string | null = null;

  /** Whether we're syncing zoom between charts. */
  let syncing = false;

  /** Debounce timer for zoom-triggered re-fetch. */
  let zoomDebounce: ReturnType<typeof setTimeout> | null = null;

  /** ResizeObserver for chart container. */
  let resizeObserver: ResizeObserver | null = null;

  /** Prevent re-fetch for the same zoom range. */
  let lastZoomRange = '';

  /** Full data x-range for zoom reset. */
  let fullXMin: number | null = null;
  let fullXMax: number | null = null;

  // ---- Reactive: watch for completed strategies ----

  $: runResults = $currentRun?.results ?? new Map<string, StrategyRunState>();
  $: runId = $currentRun?.runId ?? null;

  $: {
    // When new strategies complete, load their timeseries
    if (runId) {
      for (const [stratId, state] of runResults.entries()) {
        if (state.status === 'completed' && !loadedData.has(stratId) && !loadingStrategies.has(stratId)) {
          loadTimeseries(runId, stratId);
        }
      }
    }
  }

  // ---- Reactive: rebuild charts when data or visibility changes ----

  $: loadedKeys = [...loadedData.keys()].sort().join(',');
  $: visibleKeys = [...$chartVisibleStrategies].sort().join(',');
  $: if (loadedKeys && powerChartEl && batteryChartEl && visibleKeys !== undefined) {
    rebuildCharts();
  }

  // ---- Tariff zone helpers ----

  function getTariffZones(): Array<{ startHour: number; endHour: number; type: 'valley' | 'peak' }> {
    if (!$systemConfig?.tariff?.rates) return TARIFF_ZONES;

    const importRates = $systemConfig.tariff.rates.filter(r => r.direction === 'import');
    if (importRates.length === 0) return TARIFF_ZONES;

    const prices = importRates.map(r => r.price);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    if (minPrice === maxPrice) return [];

    const zones: Array<{ startHour: number; endHour: number; type: 'valley' | 'peak' }> = [];
    for (const rate of importRates) {
      if (rate.price === minPrice) {
        zones.push({ startHour: rate.start_hour, endHour: rate.end_hour, type: 'valley' });
      } else if (rate.price === maxPrice) {
        zones.push({ startHour: rate.start_hour, endHour: rate.end_hour, type: 'peak' });
      }
    }
    return zones;
  }

  // ---- Data loading ----

  async function loadTimeseries(rid: string, strategyId: string, opts?: { start?: string; end?: string; maxPoints?: number }) {
    loadingStrategies.add(strategyId);
    loadingStrategies = loadingStrategies;

    try {
      const data = await getTimeseries(rid, strategyId, {
        maxPoints: opts?.maxPoints ?? DEFAULT_MAX_POINTS,
        start: opts?.start,
        end: opts?.end,
      });

      loadedData.set(strategyId, data);
      loadedData = new Map(loadedData); // trigger reactivity
      addChartVisibility(strategyId);

      if (!firstLoadedStrategy) {
        firstLoadedStrategy = strategyId;
      }

      addLog(`Loaded timeseries for ${strategyId}: ${data.point_count} points`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      addLog(`Failed to load timeseries for ${strategyId}: ${message}`, 'error');
    } finally {
      loadingStrategies.delete(strategyId);
      loadingStrategies = loadingStrategies;
    }
  }

  // ---- Chart building ----

  function parseTimestamps(ts: string[]): number[] {
    return ts.map(t => new Date(t).getTime() / 1000);
  }

  function buildTariffDrawHook(zones: Array<{ startHour: number; endHour: number; type: 'valley' | 'peak' }>) {
    return (u: uPlot) => {
      const ctx = u.ctx;
      const { left, top, width, height } = u.bbox;
      const xMin = u.scales.x.min!;
      const xMax = u.scales.x.max!;

      if (xMin == null || xMax == null) return;

      const startDate = new Date(xMin * 1000);
      startDate.setHours(0, 0, 0, 0);
      const endDate = new Date(xMax * 1000);
      endDate.setDate(endDate.getDate() + 1);

      const current = new Date(startDate);
      while (current <= endDate) {
        for (const zone of zones) {
          const zoneStart = new Date(current);
          zoneStart.setHours(zone.startHour, 0, 0, 0);
          const zoneEnd = new Date(current);
          zoneEnd.setHours(zone.endHour, 0, 0, 0);

          const zStartSec = zoneStart.getTime() / 1000;
          const zEndSec = zoneEnd.getTime() / 1000;

          // Skip if completely outside view
          if (zEndSec < xMin || zStartSec > xMax) continue;

          const x0 = Math.max(u.valToPos(Math.max(zStartSec, xMin), 'x', true), left);
          const x1 = Math.min(u.valToPos(Math.min(zEndSec, xMax), 'x', true), left + width);

          if (x1 <= x0) continue;

          ctx.fillStyle = BAND_COLORS[zone.type];
          ctx.fillRect(x0, top, x1 - x0, height);
        }
        current.setDate(current.getDate() + 1);
      }
    };
  }

  function getStrategyName(stratId: string): string {
    const strat = $strategies.find(s => s.id === stratId);
    if (!strat) return stratId;
    // Abbreviate: take first 2 chars of each word
    const words = strat.name.split(/[\s_-]+/);
    if (words.length === 1) return strat.name.substring(0, 8);
    return words.map(w => w.substring(0, 3)).join('');
  }

  function getContainerWidth(): number {
    if (!containerEl) return 800;
    return containerEl.clientWidth - 2; // account for border
  }

  function resetZoom() {
    if (!powerChart || !batteryChart || fullXMin == null || fullXMax == null) return;

    // Reset scale to full range
    syncing = true;
    powerChart.setScale('x', { min: fullXMin, max: fullXMax });
    batteryChart.setScale('x', { min: fullXMin, max: fullXMax });
    syncing = false;

    // Reload full-resolution data if we were zoomed in
    if (isZoomed && runId) {
      lastZoomRange = '';
      isZoomed = false;
      const promises = [...loadedData.keys()].map(stratId =>
        loadTimeseries(runId!, stratId, { maxPoints: DEFAULT_MAX_POINTS })
      );
      Promise.all(promises).then(() => {
        const arrays = buildChartArrays();
        if (arrays && powerChart && batteryChart) {
          powerChart.setData(arrays.powerData);
          batteryChart.setData(arrays.batteryData);
        }
      });
    }
  }

  function getVisibleStrategyIds(): string[] {
    const all = [...loadedData.keys()];
    return all.filter(id => $chartVisibleStrategies.has(id));
  }

  function rebuildCharts() {
    destroyCharts();

    const strategyIds = getVisibleStrategyIds();
    if (strategyIds.length === 0) return;

    const firstData = loadedData.get(strategyIds[0])!;
    const timestamps = parseTimestamps(firstData.timestamps);

    // Store full range for zoom reset
    fullXMin = timestamps[0];
    fullXMax = timestamps[timestamps.length - 1];

    const tariffZones = getTariffZones();
    const drawHook = buildTariffDrawHook(tariffZones);
    const chartWidth = getContainerWidth();

    // ---- Power Chart ----
    const fmtTime = "{YYYY}-{MM}-{DD} {HH}:{mm}";
    const fmtKw = (self: uPlot, rawValue: number, seriesIdx: number, idx: number | null) => rawValue == null ? '--' : rawValue.toFixed(2) + ' kW';
    const fmtKwh = (self: uPlot, rawValue: number, seriesIdx: number, idx: number | null) => rawValue == null ? '--' : rawValue.toFixed(2) + ' kWh';

    const powerSeries: uPlot.Series[] = [{ label: 'Time', value: fmtTime }];
    const powerData: uPlot.AlignedData = [timestamps];

    // Solar and load are strategy-independent — add only once
    const sharedFields = ['solar_power', 'house_consumption'] as const;
    const perStrategyFields = ['grid_import', 'grid_export'] as const;

    const firstStratData = loadedData.get(strategyIds[0])!;
    for (const field of sharedFields) {
      powerSeries.push({
        label: SERIES_LABELS[field],
        stroke: SERIES_COLORS[field],
        width: 1.5,
        value: fmtKw,
        points: { show: false, size: 0, fill: '' },
      });
      powerData.push(firstStratData[field] as number[]);
    }

    for (let si = 0; si < strategyIds.length; si++) {
      const stratId = strategyIds[si];
      const data = loadedData.get(stratId)!;
      const dashPattern = DASH_PATTERNS[si % DASH_PATTERNS.length];
      const label = strategyIds.length > 1 ? getStrategyName(stratId) : '';

      for (const field of perStrategyFields) {
        const seriesLabel = label ? `${SERIES_LABELS[field]} (${label})` : SERIES_LABELS[field];
        powerSeries.push({
          label: seriesLabel,
          stroke: SERIES_COLORS[field],
          width: 1.5,
          dash: dashPattern.length > 0 ? dashPattern : undefined,
          value: fmtKw,
          points: { show: false, size: 0, fill: '' },
        });
        powerData.push(data[field] as number[]);
      }
    }

    const powerOpts: uPlot.Options = {
      width: chartWidth,
      height: 1, // will be set by resize
      plugins: [],
      hooks: {
        draw: [drawHook],
        setScale: [
          (u: uPlot, key: string) => {
            if (key === 'x' && !syncing && batteryChart) {
              syncing = true;
              batteryChart.setScale('x', {
                min: u.scales.x.min!,
                max: u.scales.x.max!,
              });
              syncing = false;
              handleZoom(u.scales.x.min!, u.scales.x.max!);
            }
          },
        ],
      },
      cursor: {
        show: true,
        sync: { key: 'chart-sync', setSeries: true },
        drag: { x: true, y: false },
        focus: { prox: 30 },
      },
      scales: {
        x: { time: true },
        y: { auto: true },
      },
      axes: [
        {
          grid: { stroke: 'rgba(42, 42, 58, 0.6)', width: 1 },
          ticks: { stroke: 'rgba(42, 42, 58, 0.6)', width: 1 },
          stroke: '#8888a0',
          font: '11px JetBrains Mono, SF Mono, monospace',
          // Hide x-axis labels on power chart (bottom chart shows them)
          show: true,
          values: () => [],
          size: 20,
        },
        {
          label: 'kW',
          labelFont: '11px JetBrains Mono, SF Mono, monospace',
          labelSize: 40,
          grid: { stroke: 'rgba(42, 42, 58, 0.4)', width: 1 },
          ticks: { stroke: 'rgba(42, 42, 58, 0.6)', width: 1 },
          stroke: '#8888a0',
          font: '11px JetBrains Mono, SF Mono, monospace',
          size: 56,
        },
      ],
      series: powerSeries,
      legend: {
        show: true,
        live: true,
      },
    };

    // ---- Battery Chart ----
    const batterySeries: uPlot.Series[] = [{ label: 'Time', value: fmtTime }];
    const batteryData: uPlot.AlignedData = [timestamps];

    for (let si = 0; si < strategyIds.length; si++) {
      const stratId = strategyIds[si];
      const data = loadedData.get(stratId)!;
      const dashPattern = DASH_PATTERNS[si % DASH_PATTERNS.length];
      const label = strategyIds.length > 1 ? getStrategyName(stratId) : '';

      const seriesLabel = label ? `${SERIES_LABELS.battery_level} (${label})` : SERIES_LABELS.battery_level;
      batterySeries.push({
        label: seriesLabel,
        stroke: SERIES_COLORS.battery_level,
        width: 1.5,
        dash: dashPattern.length > 0 ? dashPattern : undefined,
        fill: si === 0 ? 'rgba(139, 92, 246, 0.1)' : undefined,
        value: fmtKwh,
        points: { show: false, size: 0, fill: '' },
      });
      batteryData.push(data.battery_level);
    }

    const batteryOpts: uPlot.Options = {
      width: chartWidth,
      height: 1, // will be set by resize
      plugins: [],
      hooks: {
        draw: [drawHook],
        setScale: [
          (u: uPlot, key: string) => {
            if (key === 'x' && !syncing && powerChart) {
              syncing = true;
              powerChart.setScale('x', {
                min: u.scales.x.min!,
                max: u.scales.x.max!,
              });
              syncing = false;
              // handleZoom only called from power chart hook to avoid duplicate fetches
            }
          },
        ],
      },
      cursor: {
        show: true,
        sync: { key: 'chart-sync', setSeries: true },
        drag: { x: true, y: false },
        focus: { prox: 30 },
      },
      scales: {
        x: { time: true },
        y: { auto: true, range: (u, min, max) => [Math.max(0, min), max] },
      },
      axes: [
        {
          grid: { stroke: 'rgba(42, 42, 58, 0.6)', width: 1 },
          ticks: { stroke: 'rgba(42, 42, 58, 0.6)', width: 1 },
          stroke: '#8888a0',
          font: '11px JetBrains Mono, SF Mono, monospace',
          size: 40,
        },
        {
          label: 'kWh',
          labelFont: '11px JetBrains Mono, SF Mono, monospace',
          labelSize: 40,
          grid: { stroke: 'rgba(42, 42, 58, 0.4)', width: 1 },
          ticks: { stroke: 'rgba(42, 42, 58, 0.6)', width: 1 },
          stroke: '#8888a0',
          font: '11px JetBrains Mono, SF Mono, monospace',
          size: 56,
        },
      ],
      series: batterySeries,
      legend: {
        show: true,
        live: true,
      },
    };

    // Calculate heights — reserve space for legends below each canvas
    const LEGEND_ROOM = 80; // ~50px power legend + ~30px battery legend
    const availableHeight = containerEl ? (containerEl.clientHeight - 28) : 400; // minus panel header
    const chartArea = Math.max(availableHeight - LEGEND_ROOM, 200);
    const powerHeight = Math.floor(chartArea * 0.65);
    const batteryHeight = chartArea - powerHeight;

    powerOpts.height = Math.max(powerHeight, 100);
    batteryOpts.height = Math.max(batteryHeight, 80);

    powerChart = new uPlot(powerOpts, powerData, powerChartEl);
    batteryChart = new uPlot(batteryOpts, batteryData, batteryChartEl);
  }

  function destroyCharts() {
    if (powerChart) {
      powerChart.destroy();
      powerChart = null;
    }
    if (batteryChart) {
      batteryChart.destroy();
      batteryChart = null;
    }
  }

  // ---- Zoom handling ----

  /** Track whether current data is zoomed (filtered) so we can detect zoom-out. */
  let isZoomed = false;

  /** Build uPlot-ready data arrays from loadedData (visible strategies only). */
  function buildChartArrays() {
    const strategyIds = getVisibleStrategyIds();
    if (strategyIds.length === 0) return null;

    const firstData = loadedData.get(strategyIds[0])!;
    const timestamps = parseTimestamps(firstData.timestamps);

    const sharedFields = ['solar_power', 'house_consumption'] as const;
    const perStrategyFields = ['grid_import', 'grid_export'] as const;
    const powerData: uPlot.AlignedData = [timestamps];
    const batteryData: uPlot.AlignedData = [timestamps];

    // Shared fields once from first strategy
    const firstStratData = loadedData.get(strategyIds[0])!;
    for (const field of sharedFields) {
      powerData.push(firstStratData[field] as number[]);
    }

    for (const stratId of strategyIds) {
      const data = loadedData.get(stratId)!;
      for (const field of perStrategyFields) {
        powerData.push(data[field] as number[]);
      }
      batteryData.push(data.battery_level);
    }

    return { powerData, batteryData };
  }

  function handleZoom(xMin: number, xMax: number) {
    if (!runId) return;

    const rangeDays = (xMax - xMin) / 86400;
    const rangeKey = `${xMin.toFixed(0)}-${xMax.toFixed(0)}`;

    // Skip if we just loaded data for this exact range
    if (rangeKey === lastZoomRange) return;

    if (zoomDebounce) clearTimeout(zoomDebounce);

    zoomDebounce = setTimeout(async () => {
      if (rangeDays < ZOOM_THRESHOLD_DAYS && rangeDays > 0) {
        // Scale points: more points for tighter zoom
        const maxPoints = Math.round(4000 + (rangeDays / ZOOM_THRESHOLD_DAYS) * 4000);

        const start = new Date(xMin * 1000).toISOString();
        const end = new Date(xMax * 1000).toISOString();

        lastZoomRange = rangeKey;

        // Fetch all strategies in parallel, then update charts in-place
        const promises = [...loadedData.keys()].map(stratId =>
          loadTimeseries(runId!, stratId, { start, end, maxPoints })
        );
        await Promise.all(promises);

        // Update charts with new higher-res data (no full rebuild)
        const arrays = buildChartArrays();
        if (arrays && powerChart && batteryChart) {
          powerChart.setData(arrays.powerData);
          batteryChart.setData(arrays.batteryData);
          addLog(`Zoom: loaded ${loadedData.values().next().value?.point_count ?? '?'} points for ${rangeDays.toFixed(1)} days`);
        }
        isZoomed = true;
      } else if (isZoomed) {
        // Zoomed back out past threshold — reload full data
        lastZoomRange = '';

        const promises = [...loadedData.keys()].map(stratId =>
          loadTimeseries(runId!, stratId, { maxPoints: DEFAULT_MAX_POINTS })
        );
        await Promise.all(promises);

        const arrays = buildChartArrays();
        if (arrays && powerChart && batteryChart) {
          powerChart.setData(arrays.powerData);
          batteryChart.setData(arrays.batteryData);
        }
        isZoomed = false;
      }
    }, 400);
  }

  // ---- Reset charts when a new run starts ----

  let prevRunId: string | null = null;
  $: {
    if (runId !== prevRunId) {
      prevRunId = runId;
      loadedData = new Map();
      loadingStrategies = new Set();
      firstLoadedStrategy = null;
      isZoomed = false;
      lastZoomRange = '';
      chartVisibleStrategies.set(new Set());
      destroyCharts();
    }
  }

  // ---- Lifecycle ----

  function handleDblClick() {
    resetZoom();
  }

  onMount(() => {
    // Observe container resizes
    resizeObserver = new ResizeObserver(() => {
      if (powerChart && batteryChart && containerEl) {
        const chartWidth = getContainerWidth();
        const LEGEND_ROOM = 80;
        const availableHeight = containerEl.clientHeight - 28;
        const chartArea = Math.max(availableHeight - LEGEND_ROOM, 200);
        const powerHeight = Math.floor(chartArea * 0.65);
        const batteryHeight = chartArea - powerHeight;

        powerChart.setSize({
          width: chartWidth,
          height: Math.max(powerHeight, 100),
        });
        batteryChart.setSize({
          width: chartWidth,
          height: Math.max(batteryHeight, 80),
        });
      }
    });

    if (containerEl) {
      resizeObserver.observe(containerEl);
      containerEl.addEventListener('dblclick', handleDblClick);
    }
  });

  onDestroy(() => {
    destroyCharts();
    if (containerEl) {
      containerEl.removeEventListener('dblclick', handleDblClick);
    }
    if (resizeObserver) {
      resizeObserver.disconnect();
      resizeObserver = null;
    }
    if (zoomDebounce) {
      clearTimeout(zoomDebounce);
    }
  });
</script>

<div class="chart-container" bind:this={containerEl}>
  <div class="panel-header">
    CHARTS
    {#if loadedData.size > 0}
      <span class="chart-info">
        {loadedData.size} strateg{loadedData.size === 1 ? 'y' : 'ies'}
        {#if isZoomed}
          &middot; <button class="reset-zoom-btn" on:click={resetZoom}>Reset Zoom</button>
          <span class="zoom-hint">(or double-click)</span>
        {/if}
      </span>
    {/if}
  </div>

  {#if loadedData.size === 0}
    <div class="chart-placeholder">
      {#if loadingStrategies.size > 0}
        <span class="loading-text">Loading timeseries data...</span>
      {:else if $currentRun && $currentRun.status === 'running'}
        <span class="placeholder-text">Simulation running. Charts will appear when strategies complete.</span>
      {:else}
        <span class="placeholder-text">Select strategies and run a simulation to see charts.</span>
      {/if}
    </div>
  {:else}
    <div class="charts-wrapper">
      <div class="power-chart" bind:this={powerChartEl}></div>
      <div class="battery-chart" bind:this={batteryChartEl}></div>
    </div>
  {/if}
</div>

<style>
  .chart-container {
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

  .chart-info {
    font-size: 10px;
    color: var(--text-dim);
    text-transform: none;
    letter-spacing: normal;
    font-family: var(--font-mono);
  }

  .reset-zoom-btn {
    background: none;
    border: 1px solid var(--border);
    color: var(--text-secondary);
    font-size: 10px;
    font-family: var(--font-mono);
    padding: 1px 6px;
    border-radius: 2px;
    cursor: pointer;
    text-transform: none;
    letter-spacing: normal;
  }

  .reset-zoom-btn:hover {
    border-color: var(--border-active);
    color: var(--text-primary);
  }

  .zoom-hint {
    font-size: 9px;
    color: var(--text-dim);
    text-transform: none;
    letter-spacing: normal;
  }

  .chart-placeholder {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .placeholder-text {
    color: var(--text-dim);
    font-size: var(--font-size-sm);
  }

  .loading-text {
    color: var(--text-secondary);
    font-size: var(--font-size-sm);
    font-family: var(--font-mono);
  }

  .charts-wrapper {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    overflow-x: hidden;
  }

  .power-chart,
  .battery-chart {
    flex-shrink: 0;
  }

  /* Override uPlot styles to match theme */
  .chart-container :global(.u-wrap) {
    background: var(--bg-primary);
  }

  .chart-container :global(.u-legend) {
    font-family: var(--font-mono);
    font-size: 10px;
    padding: 2px var(--spacing-sm);
    background: transparent;
  }

  .chart-container :global(.u-legend .u-series) {
    padding: 0 var(--spacing-xs);
  }

  .chart-container :global(.u-legend .u-label) {
    color: var(--text-secondary);
  }

  .chart-container :global(.u-legend .u-value) {
    color: var(--text-primary);
    font-family: var(--font-mono);
    min-width: 5em;
    text-align: right;
  }

  .chart-container :global(.u-select) {
    background: rgba(59, 130, 246, 0.1);
  }

  .chart-container :global(.u-cursor-x),
  .chart-container :global(.u-cursor-y) {
    border-color: rgba(136, 136, 160, 0.5);
  }

  .chart-container :global(.u-legend) {
    text-align: left;
  }
</style>
