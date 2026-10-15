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
  import { flowColor, strategyColor, strategyDash, type FlowKey } from '../../ui/colors';
  import { bucketize, bucketLabel, bucketWord, pickBucket } from '../../ui/envelope';
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
  // Without any finished battery strategy (e.g. a cancelled run), the baseline's flows are still worth showing.
  $: focus = $focusStrategy && completed.includes($focusStrategy) ? $focusStrategy : batteryIds[0] ?? completed[0] ?? null;
  $: settled = rows.every((r) => r.status === 'completed' || r.status === 'failed' || r.status === 'cancelled');

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

  // ---- Range view ----
  $: fullSpan = spanOf(focus ? full.get(focus) : full.values().next().value);
  $: visibleSpan = xRange ? xRange.max - xRange.min : fullSpan;
  $: bucket = visibleSpan > 0 ? pickBucket(visibleSpan) : null;
  $: xLabel = bucket ? (x: number) => bucketLabel(x, bucket ?? 3600) : null;

  // Explicit full-run bounds keep the three charts on one time axis: in the range view the price
  // strip plots bucket starts while the others plot bucket centres, so their own data extents differ.
  $: runBounds = boundsOf(full);
  $: shownRange = xRange ?? runBounds;

  function boundsOf(m: Map<string, Timeseries>): XRange {
    let min = Infinity;
    let max = -Infinity;
    for (const t of m.values()) {
      if (!t.x.length) continue;
      min = Math.min(min, t.x[0]);
      max = Math.max(max, t.x[t.x.length - 1]);
    }
    return Number.isFinite(min) && max > min ? { min, max } : null;
  }

  function spanOf(t: Timeseries | undefined): number {
    return t && t.x.length > 1 ? t.x[t.x.length - 1] - t.x[0] : 0;
  }

  // ---- Power chart ----
  interface Flow {
    key: FlowKey;
    label: string;
    values: (number | null)[];
    fill?: number;
    dash?: number[];
  }

  function flowsOf(t: Timeseries, withBattery: boolean): Flow[] {
    const f: Flow[] = [
      { key: 'solar', label: 'Solar', values: t.solar, fill: 0.14 },
      { key: 'load', label: 'Load', values: t.load },
      { key: 'gridImport', label: 'Grid import', values: t.gridImport },
      { key: 'gridExport', label: 'Grid export', values: t.gridExport },
    ];
    if (!withBattery) return f;
    if (t.batteryCharge) f.push({ key: 'batteryCharge', label: 'Battery charge', values: t.batteryCharge, dash: [4, 3] });
    if (t.batteryDischarge) f.push({ key: 'batteryDischarge', label: 'Battery discharge', values: t.batteryDischarge });
    return f;
  }

  $: focusTs = focus ? pick(focus, full, detail, xRange != null) : undefined;
  $: power = focusTs ? buildPower(focusTs, bucket, $theme, focus !== BASELINE_ID) : null;

  function buildPower(
    t: Timeseries,
    b: number | null,
    mode: 'light' | 'dark',
    withBattery: boolean,
  ): { series: ChartSeries[]; data: uPlot.AlignedData } {
    const flows = flowsOf(t, withBattery);
    if (!b) {
      return {
        series: flows.map((f) => ({ label: f.label, color: flowColor(f.key, mode), fill: f.fill, dash: f.dash, width: 1.5 })),
        data: [t.x, ...flows.map((f) => f.values)] as uPlot.AlignedData,
      };
    }
    const g = bucketize(t.x, flows.map((f) => f.values), b);
    const series: ChartSeries[] = [];
    const cols: (number | null)[][] = [];
    flows.forEach((f, i) => {
      const color = flowColor(f.key, mode);
      series.push({ label: f.label, color, dash: f.dash, width: 1.25, bandFill: f.key === 'solar' ? 0.15 : 0 });
      series.push({ label: `${f.label} (min)`, color, lowerOf: f.label });
      cols.push(g.hi[i], g.lo[i]);
    });
    return { series, data: [g.x, ...cols] as uPlot.AlignedData };
  }

  // ---- Price strip ----
  $: price = focusTs?.importPrice ? buildPrice(focusTs, bucket, $theme) : null;

  function buildPrice(
    t: Timeseries,
    b: number | null,
    mode: 'light' | 'dark',
  ): { series: ChartSeries[]; data: uPlot.AlignedData; centre?: Map<number, number> } {
    const lines: { label: string; key: FlowKey; values: (number | null)[] }[] = [];
    if (t.importPrice) lines.push({ label: 'Import', key: 'gridImport', values: t.importPrice });
    if (t.exportPrice) lines.push({ label: 'Export', key: 'gridExport', values: t.exportPrice });
    if (!b) {
      return {
        series: lines.map((l) => ({ label: l.label, color: flowColor(l.key, mode), width: 1.25, step: true })),
        data: [t.x, ...lines.map((l) => l.values)] as uPlot.AlignedData,
      };
    }
    const g = bucketize(t.x, lines.map((l) => l.values), b);
    // Each bucket is drawn flat from its start to its end (two points), so the strip reads as
    // steps on the same time axis as the other charts, and a cursor at a bucket's centre (where
    // the power chart puts its points) lands on that bucket rather than the next one.
    const xs: number[] = [];
    const centre = new Map<number, number>();
    for (let i = 0; i < g.start.length; i++) {
      xs.push(g.start[i], g.start[i] + b - 1);
      centre.set(g.start[i], g.x[i]).set(g.start[i] + b - 1, g.x[i]);
    }
    const twice = (v: (number | null)[]) => v.flatMap((y) => [y, y]);
    const series: ChartSeries[] = [];
    const cols: (number | null)[][] = [];
    lines.forEach((l, i) => {
      const color = flowColor(l.key, mode);
      series.push({ label: l.label, color, width: 1.25, bandFill: 0.1 });
      series.push({ label: `${l.label} (min)`, color, lowerOf: l.label });
      cols.push(twice(g.hi[i]), twice(g.lo[i]));
    });
    return { series, data: [xs, ...cols] as uPlot.AlignedData, centre };
  }
  $: priceXLabel = bucket && price?.centre ? (x: number) => bucketLabel(price?.centre?.get(x) ?? x, bucket ?? 3600) : null;

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
  $: socData = buildSocData(socIds, full, detail, xRange != null, bucket);

  function buildSocData(
    ids: string[],
    f: Map<string, Timeseries>,
    d: Map<string, Timeseries>,
    zoomed: boolean,
    b: number | null,
  ): uPlot.AlignedData | null {
    const tables: uPlot.AlignedData[] = [];
    for (const id of ids) {
      const t = pick(id, f, d, zoomed);
      if (!t) return null;
      const soc = socPercent(t, capacityKwh);
      if (!soc) return null;
      // The backend keeps reporting the held battery level through data gaps; blank it where the
      // flows are missing so the gap reads as a gap here too, as it does in the power chart.
      const masked = soc.map((v, i) => (t.load[i] == null ? null : v));
      if (b) {
        const g = bucketize(t.x, [masked], b);
        tables.push([g.x, g.mean[0]] as uPlot.AlignedData);
      } else tables.push([t.x, masked] as uPlot.AlignedData);
    }
    if (!tables.length) return null;
    // Downsampling picks different timestamps per strategy; join aligns them, and the
    // undefined fillers it inserts are spanned while real nulls (data gaps) stay gaps.
    return tables.length === 1 ? tables[0] : uPlot.join(tables);
  }

  $: loading = pending.size > 0 && !power;
  // The server keeps timeseries for its most recent runs only; an older run answers 410.
  $: allFailed = completed.length > 0 && completed.every((id) => errors.has(id));
  $: expired = allFailed && completed.some((id) => /expired/i.test(errors.get(id) ?? ''));
  $: socFailed = batteryIds.length > 0 && batteryIds.every((id) => errors.has(id));
  $: rangeNote = bucket ? `${bucketWord(bucket)} min–max` : null;
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

<Card title="Power and battery" subtitle={allFailed ? null : rangeLabel ? `Zoomed to ${rangeLabel}` : zoomHint}>
  <svelte:fragment slot="actions">
    {#if xRange}
      <Button size="sm" variant="secondary" icon="reset" on:click={() => onZoom(new CustomEvent('zoom', { detail: null }))}>Reset zoom</Button>
    {/if}
  </svelte:fragment>

  {#if allFailed}
    <div class="state gone">
      {#if expired}
        <p>The server no longer has the minute-by-minute data for this run; it keeps it for the most recent runs only.</p>
        <p class="muted">Run the simulation again to see these charts. The costs, ranking and monthly figures are unaffected.</p>
      {:else}
        <p class="err-text">Couldn't load the timeseries: {errors.get(completed[0])}</p>
      {/if}
    </div>
  {:else}
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
    <h3 class="chart-title">
      Power <span class="muted">· kW{focus ? ` · ${$strategyById.get(focus)?.name ?? focus}` : ''}{rangeNote ? ` · ${rangeNote}` : ''}</span>
    </h3>
    {#if focus && errors.get(focus)}
      <p class="state err">Couldn't load this strategy's timeseries: {errors.get(focus)}</p>
    {:else if power}
      <Chart
        data={power.data}
        series={power.series}
        unit="kW"
        decimals={bucket ? 1 : 2}
        height={300}
        syncKey="sim-{runId}"
        xRange={shownRange}
        {xLabel}
        on:zoom={onZoom}
        ariaLabel="Power flows in kilowatts over time for the selected strategy"
      />
      {#if price}
        <div class="price">
          <Chart
            data={price.data}
            series={price.series}
            unit="Price · €/kWh"
            decimals={3}
            axisDecimals={2}
            height={96}
            syncKey="sim-{runId}"
            xRange={shownRange}
            xLabel={priceXLabel}
            compact
            on:zoom={onZoom}
            ariaLabel="Energy prices in euros per kilowatt-hour over time"
          />
        </div>
      {/if}
    {:else}
      <div class="state skeleton" style="height: 330px">{loading ? 'Loading timeseries…' : 'Waiting for results…'}</div>
    {/if}
  </div>

  <div class="block">
    <h3 class="chart-title">
      Battery charge <span class="muted">· % of capacity · all strategies{bucket ? ` · ${bucketWord(bucket)} average` : ''}</span>
    </h3>
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
        xRange={shownRange}
        {xLabel}
        on:zoom={onZoom}
        ariaLabel="Battery state of charge in percent for each strategy"
      />
    {:else}
      <div class="state skeleton" style="height: {(!batteryIds.length && settled) || socFailed ? 96 : 250}px">
        {socFailed
          ? "Couldn't load the battery timeseries"
          : batteryIds.length
            ? 'Loading…'
            : settled
              ? 'No battery strategy finished in this run'
              : 'Waiting for results…'}
      </div>
    {/if}
  </div>
  {/if}
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
  .price {
    margin-top: var(--s-2);
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
  .gone {
    display: grid;
    gap: var(--s-1);
    justify-items: start;
    place-items: start;
    text-align: left;
    padding: var(--s-5);
    border-radius: var(--r-md);
    background: var(--surface-2);
    color: var(--text-2);
  }
  .err-text {
    color: var(--negative);
  }
  .err {
    padding: var(--s-5);
    color: var(--negative);
  }
  @media (max-width: 720px), (pointer: coarse) {
    .opt {
      height: 40px;
    }
  }
</style>
