<script lang="ts">
  import { onDestroy } from 'svelte';
  import uPlot from 'uplot';
  import { errorMessage, getTimeseries } from '../../api/client';
  import { socPercent } from '../../api/timeseries';
  import { strategyById } from '../../stores/app';
  import { BASELINE_ID, type Ranked } from '../../stores/run';
  import { theme } from '../../stores/theme';
  import { logActivity } from '../../stores/toasts';
  import Chart, { type ChartSeries, type XRange } from '../../ui/Chart.svelte';
  import Card from '../../ui/Card.svelte';
  import Button from '../../ui/Button.svelte';
  import StrategyDot from '../../ui/StrategyDot.svelte';
  import { flowColor, strategyColor, strategyDash } from '../../ui/colors';
  import { fmtDateTime } from '../../ui/format';
  import type { Timeseries } from '../../types';
  import { focusPinned, focusStrategy, pickFocus } from './state';

  export let runId: string;
  export let rows: Ranked[];
  /** Battery capacity of the run, to express SoC in percent when the API only gives kWh. */
  export let capacityKwh: number;

  const MAX_POINTS = 2000;

  let full = new Map<string, Timeseries>();
  let detail = new Map<string, Timeseries>();
  let errors = new Map<string, string>();
  let pending = new Set<string>();
  let xRange: XRange = null;
  let detailKey = '';
  let detailTimer: ReturnType<typeof setTimeout> | null = null;
  let loadedRun = '';
  let abort = new AbortController();

  $: if (runId !== loadedRun) {
    abort.abort();
    abort = new AbortController();
    loadedRun = runId;
    full = new Map();
    detail = new Map();
    errors = new Map();
    pending = new Set();
    xRange = null;
  }

  $: completed = rows.filter((r) => r.status === 'completed').map((r) => r.id);
  $: batteryIds = completed.filter((id) => id !== BASELINE_ID);
  $: if (completed.length) for (const id of completed) if (!full.has(id) && !pending.has(id) && !errors.has(id)) void load(id);

  $: if (batteryIds.length) {
    const valid = $focusStrategy != null && completed.includes($focusStrategy);
    if (!valid || (!$focusPinned && $focusStrategy !== batteryIds[0])) focusStrategy.set(batteryIds[0]);
  }
  $: focus = $focusStrategy && completed.includes($focusStrategy) ? $focusStrategy : batteryIds[0] ?? null;

  async function load(id: string) {
    pending = new Set(pending).add(id);
    const signal = abort.signal;
    try {
      const ts = await getTimeseries(runId, id, { maxPoints: MAX_POINTS, signal });
      if (signal.aborted) return;
      full = new Map(full).set(id, ts);
    } catch (err) {
      if (signal.aborted) return;
      errors = new Map(errors).set(id, errorMessage(err));
      logActivity(`Timeseries for ${$strategyById.get(id)?.name ?? id}: ${errorMessage(err)}`, 'error');
    } finally {
      const p = new Set(pending);
      p.delete(id);
      pending = p;
    }
  }

  /** Zoomed in far enough that the overview is coarse: fetch the window at full detail. */
  function onZoom(e: CustomEvent<XRange>) {
    xRange = e.detail;
    if (detailTimer) clearTimeout(detailTimer);
    if (!xRange) {
      detail = new Map();
      detailKey = '';
      return;
    }
    const r = xRange;
    detailTimer = setTimeout(() => void loadDetail(r), 250);
  }

  async function loadDetail(r: { min: number; max: number }) {
    const ids = [...full.keys()];
    if (!ids.length) return;
    const key = `${Math.round(r.min)}-${Math.round(r.max)}`;
    detailKey = key;
    const signal = abort.signal;
    try {
      const res = await Promise.all(
        ids.map((id) => getTimeseries(runId, id, { start: r.min * 1000, end: r.max * 1000, maxPoints: MAX_POINTS, signal })),
      );
      if (signal.aborted || detailKey !== key) return;
      detail = new Map(ids.map((id, i) => [id, res[i]]));
    } catch (err) {
      if (!signal.aborted) logActivity(`Zoom detail: ${errorMessage(err)}`, 'warning');
    }
  }

  function pick(id: string, f: Map<string, Timeseries>, d: Map<string, Timeseries>, zoomed: boolean): Timeseries | undefined {
    return (zoomed && d.get(id)) || f.get(id);
  }

  // ---- Power chart ----
  $: focusTs = focus ? pick(focus, full, detail, xRange != null) : undefined;
  $: powerSeries = buildPowerSeries(focusTs, $theme);
  $: powerData = focusTs ? buildPowerData(focusTs) : null;
  $: priceBand = focusTs?.importPrice ? { values: focusTs.importPrice, label: 'Import price' } : null;

  function buildPowerSeries(t: Timeseries | undefined, mode: 'light' | 'dark'): ChartSeries[] {
    if (!t) return [];
    const s: ChartSeries[] = [
      { label: 'Solar', color: flowColor('solar', mode), fill: 0.14, width: 1.5 },
      { label: 'Load', color: flowColor('load', mode), width: 1.5 },
      { label: 'Grid import', color: flowColor('gridImport', mode), width: 1.5 },
      { label: 'Grid export', color: flowColor('gridExport', mode), width: 1.5 },
    ];
    if (t.batteryCharge) s.push({ label: 'Battery charge', color: flowColor('batteryCharge', mode), width: 1.5 });
    if (t.batteryDischarge) s.push({ label: 'Battery discharge', color: flowColor('batteryDischarge', mode), width: 1.5 });
    return s;
  }

  function buildPowerData(t: Timeseries): uPlot.AlignedData {
    const cols: (number | null)[][] = [t.solar, t.load, t.gridImport, t.gridExport];
    if (t.batteryCharge) cols.push(t.batteryCharge);
    if (t.batteryDischarge) cols.push(t.batteryDischarge);
    return [t.x, ...cols] as uPlot.AlignedData;
  }

  // ---- SoC chart ----
  $: socIds = batteryIds.filter((id) => full.has(id));
  $: socSeries = socIds.map(
    (id): ChartSeries => ({
      label: $strategyById.get(id)?.name ?? id,
      color: strategyColor(id, $theme),
      dash: strategyDash(id),
      width: id === focus ? 2.25 : 1.5,
      legendDotId: id,
    }),
  );
  $: socData = buildSocData(socIds, full, detail, xRange != null);

  function buildSocData(ids: string[], f: Map<string, Timeseries>, d: Map<string, Timeseries>, zoomed: boolean): uPlot.AlignedData | null {
    const tables: uPlot.AlignedData[] = [];
    for (const id of ids) {
      const t = pick(id, f, d, zoomed);
      if (!t) return null;
      const soc = socPercent(t, capacityKwh);
      if (!soc) return null;
      // The backend keeps reporting the held battery level through data gaps; blank it where the
      // flows are missing so the gap reads as a gap here too, as it does in the power chart.
      const masked = soc.map((v, i) => (t.load[i] == null ? null : v));
      tables.push([t.x, masked] as uPlot.AlignedData);
    }
    if (!tables.length) return null;
    // Downsampling picks different timestamps per strategy; join aligns them, and the
    // undefined fillers it inserts are spanned while real nulls (data gaps) stay gaps.
    return tables.length === 1 ? tables[0] : uPlot.join(tables);
  }

  $: loading = pending.size > 0 && !powerData;
  const coarse = typeof matchMedia === 'function' && matchMedia('(pointer: coarse)').matches;
  const zoomHint = coarse ? 'Drag sideways across a chart to zoom, tap to read values' : 'Drag across a chart to zoom, double-click to reset';
  $: rangeLabel = xRange ? `${fmtDateTime(new Date(xRange.min * 1000))} – ${fmtDateTime(new Date(xRange.max * 1000))}` : null;
  $: switcher = completed.map((id) => ({ id, name: $strategyById.get(id)?.name ?? id }));

  let switchEl: HTMLDivElement;
  function onSwitchKey(e: KeyboardEvent) {
    const i = switcher.findIndex((s) => s.id === focus);
    let n = -1;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') n = (i + 1) % switcher.length;
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') n = (i - 1 + switcher.length) % switcher.length;
    else if (e.key === 'Home') n = 0;
    else if (e.key === 'End') n = switcher.length - 1;
    if (n < 0 || !switcher[n]) return;
    e.preventDefault();
    pickFocus(switcher[n].id);
    switchEl.querySelectorAll<HTMLButtonElement>('button')[n]?.focus();
  }

  onDestroy(() => {
    abort.abort();
    if (detailTimer) clearTimeout(detailTimer);
  });
</script>

<Card title="Power and battery" subtitle={rangeLabel ? `Zoomed to ${rangeLabel}` : zoomHint}>
  <svelte:fragment slot="actions">
    {#if xRange}
      <Button size="sm" variant="secondary" icon="reset" on:click={() => onZoom(new CustomEvent('zoom', { detail: null }))}>Reset zoom</Button>
    {/if}
  </svelte:fragment>

  <div class="switch" role="radiogroup" tabindex="-1" aria-label="Strategy shown in the power chart" bind:this={switchEl} on:keydown={onSwitchKey}>
    {#each switcher as s (s.id)}
      <button
        type="button"
        role="radio"
        aria-checked={focus === s.id}
        tabindex={focus === s.id || (!focus && s === switcher[0]) ? 0 : -1}
        class="opt"
        on:click={() => pickFocus(s.id)}
      >
        <StrategyDot id={s.id} size={8} />{s.name}
      </button>
    {/each}
  </div>

  <div class="block">
    <h3 class="chart-title">Power <span class="muted">· kW · {focus ? $strategyById.get(focus)?.name ?? focus : ''}</span></h3>
    {#if focus && errors.get(focus)}
      <p class="state err">Couldn't load this strategy's timeseries: {errors.get(focus)}</p>
    {:else if powerData}
      <Chart
        data={powerData}
        series={powerSeries}
        unit="kW"
        height={300}
        syncKey="sim-{runId}"
        {xRange}
        band={priceBand}
        on:zoom={onZoom}
        ariaLabel="Power flows in kilowatts over time for the selected strategy"
      />
    {:else}
      <div class="state skeleton" style="height: 330px">{loading ? 'Loading timeseries…' : 'Waiting for results…'}</div>
    {/if}
  </div>

  <div class="block">
    <h3 class="chart-title">Battery charge <span class="muted">· % of capacity · all strategies</span></h3>
    {#if socData && socSeries.length}
      <Chart
        data={socData}
        series={socSeries}
        unit="%"
        decimals={0}
        height={220}
        yMin={0}
        yMax={100}
        syncKey="sim-{runId}"
        {xRange}
        on:zoom={onZoom}
        ariaLabel="Battery state of charge in percent for each strategy"
      />
    {:else}
      <div class="state skeleton" style="height: 250px">{batteryIds.length ? 'Loading…' : 'Waiting for results…'}</div>
    {/if}
  </div>
</Card>

<style>
  .switch {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin: calc(var(--s-2) * -1) 0 var(--s-4);
  }
  .opt {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 30px;
    padding: 0 10px;
    border-radius: var(--r-pill);
    border: 1px solid transparent;
    background: transparent;
    color: var(--text-2);
    font-size: var(--fs-xs);
    font-weight: 600;
  }
  .opt:hover {
    background: var(--surface-3);
  }
  .opt[aria-checked='true'] {
    border-color: var(--border-strong);
    background: var(--surface);
    color: var(--text);
    box-shadow: var(--shadow-1);
  }
  .block + .block {
    margin-top: var(--s-5);
    padding-top: var(--s-4);
    border-top: 1px solid var(--border);
  }
  .chart-title {
    font-size: var(--fs-sm);
    font-weight: 650;
    margin-bottom: var(--s-1);
  }
  .chart-title .muted {
    font-weight: 500;
  }
  .state {
    display: grid;
    place-items: center;
    color: var(--text-3);
    font-size: var(--fs-sm);
  }
  .skeleton {
    border-radius: var(--r-md);
    background: var(--surface-2);
  }
  .err {
    padding: var(--s-5);
    color: var(--negative);
  }
  @media (max-width: 720px) {
    .opt {
      height: 40px;
    }
  }
</style>
