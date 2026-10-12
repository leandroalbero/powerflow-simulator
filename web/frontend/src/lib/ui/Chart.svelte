<script lang="ts" context="module">
  export interface ChartSeries {
    label: string;
    color: string;
    width?: number;
    dash?: number[];
    /** Fill alpha under the line (0 = none). */
    fill?: number;
    show?: boolean;
    /** Marker shown in the legend instead of a line swatch. */
    legendDotId?: string;
  }
  export type XRange = { min: number; max: number } | null;
</script>

<script lang="ts">
  import { createEventDispatcher, onDestroy, onMount } from 'svelte';
  import uPlot from 'uplot';
  import 'uplot/dist/uPlot.min.css';
  import { theme } from '../stores/theme';
  import { alpha } from './colors';
  import { fmtDateTime, num } from './format';
  import StrategyDot from './StrategyDot.svelte';

  /** [x (epoch s), ...series]. */
  export let data: uPlot.AlignedData;
  export let series: ChartSeries[];
  export let unit: string;
  export let decimals = 2;
  export let height = 280;
  export let yMin: number | null = null;
  export let yMax: number | null = null;
  export let syncKey: string | null = null;
  /** Controlled x zoom shared between charts; null = full range. */
  export let xRange: XRange = null;
  /** Optional background shading by price (same x as data). */
  export let band: { values: (number | null)[]; label: string } | null = null;
  export let ariaLabel: string;

  const dispatch = createEventDispatcher<{ zoom: XRange }>();

  let host: HTMLDivElement;
  let plot: uPlot | null = null;
  let ro: ResizeObserver | null = null;
  let cursorIdx: number | null = null;
  /** Per-series data index under the cursor, from uPlot's scanner (skips join fillers). */
  let cursorIdxs: (number | null)[] = [];
  let hidden = new Set<number>();
  let width = 600;

  function cssVar(name: string): string {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
  }

  const pad = (n: number) => String(n).padStart(2, '0');
  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  function xTicks(_u: uPlot, splits: number[], _ax: number, _space: number, incr: number): string[] {
    return splits.map((s) => {
      const d = new Date(s * 1000);
      const midnight = d.getHours() === 0 && d.getMinutes() === 0;
      if (incr >= 86400 || midnight) return `${d.getDate()} ${MONTHS[d.getMonth()]}`;
      return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
    });
  }

  function bandHook(u: uPlot) {
    if (!band) return;
    const vals = band.values;
    const xs = u.data[0];
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of vals) {
      if (v == null) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!Number.isFinite(lo) || hi <= lo) return;
    const { ctx } = u;
    const { top, height: h } = u.bbox;
    const base = cssVar('--warning');
    ctx.save();
    for (let i = 0; i < xs.length - 1; i++) {
      const v = vals[i];
      if (v == null) continue;
      const t = (v - lo) / (hi - lo);
      if (t < 0.05) continue;
      const x0 = u.valToPos(xs[i], 'x', true);
      const x1 = u.valToPos(xs[i + 1], 'x', true);
      ctx.fillStyle = alpha(base.startsWith('#') ? base : '#d99400', 0.03 + t * 0.13);
      ctx.fillRect(x0, top, Math.max(1, x1 - x0), h);
    }
    ctx.restore();
  }

  function build() {
    plot?.destroy();
    plot = null;
    if (!host) return;
    const axisColor = cssVar('--chart-axis');
    const gridColor = cssVar('--chart-grid');
    const font = `12px ${cssVar('--font')}`;

    const opts: uPlot.Options = {
      width,
      height,
      legend: { show: false },
      padding: [12, 12, 0, 4],
      cursor: {
        sync: syncKey ? { key: syncKey, setSeries: false } : undefined,
        drag: { x: true, y: false, setScale: false },
        points: { size: 7, width: 2 },
        // A finite proximity makes uPlot pick each series' nearest point by time (pixel distance)
        // rather than by array index, which drifts when per-strategy timestamps are merged.
        hover: { prox: 1e6 },
        // Browsers follow a tap with compatibility mouse events ending in mouseleave, which would
        // hide the values the tap just showed.
        bind: {
          mouseleave: (_u, _t, handler) => (e) => (performance.now() - lastTouch < 800 ? null : handler(e)),
        },
      },
      select: { show: true, left: 0, top: 0, width: 0, height: 0 },
      scales: {
        x: { time: true },
        y: {
          range: (_u, dmin, dmax) => {
            const lo = yMin ?? Math.min(0, dmin);
            const hi = yMax ?? (dmax <= lo ? lo + 1 : dmax * 1.08);
            return [lo, hi];
          },
        },
      },
      axes: [
        {
          stroke: axisColor,
          font,
          grid: { stroke: gridColor, width: 1 },
          ticks: { show: false },
          values: xTicks,
          space: 64,
          gap: 6,
          size: 30,
        },
        {
          stroke: axisColor,
          font,
          grid: { stroke: gridColor, width: 1 },
          ticks: { show: false },
          size: 48,
          gap: 6,
          values: (_u, splits) => splits.map((v) => `${num(v, Math.abs(v) < 10 && v % 1 !== 0 ? 1 : 0)}`),
        },
      ],
      series: [
        {},
        ...series.map((s, i) => ({
          label: s.label,
          stroke: s.color,
          width: s.width ?? 1.75,
          dash: s.dash,
          fill: s.fill ? alpha(s.color, s.fill) : undefined,
          show: !hidden.has(i + 1) && s.show !== false,
          points: { show: false },
        })),
      ],
      hooks: {
        setCursor: [
          (u) => {
            cursorIdx = u.cursor.idx ?? null;
            cursorIdxs = u.cursor.idxs ? [...u.cursor.idxs] : [];
          },
        ],
        setSelect: [
          (u) => {
            if (u.select.width > 4) {
              const min = u.posToVal(u.select.left, 'x');
              const max = u.posToVal(u.select.left + u.select.width, 'x');
              dispatch('zoom', { min, max });
            }
            u.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
          },
        ],
        drawAxes: [bandHook],
      },
    };
    plot = new uPlot(opts, data, host);
    applyRange();
  }

  function applyRange() {
    if (!plot) return;
    const xs = plot.data[0];
    if (!xs.length) return;
    const min = xRange?.min ?? xs[0];
    const max = xRange?.max ?? xs[xs.length - 1];
    plot.setScale('x', { min, max });
  }

  function toggle(i: number) {
    const next = new Set(hidden);
    if (next.has(i)) next.delete(i);
    else next.add(i);
    hidden = next;
    plot?.setSeries(i, { show: !next.has(i) });
  }

  // Rebuild when the series definition or theme changes; otherwise just swap data.
  $: seriesKey = JSON.stringify(series.map((s) => [s.label, s.color, s.dash, s.fill, s.width])) + $theme + (band ? 'b' : '') + height;
  let builtKey = '';
  $: if (host && seriesKey !== builtKey) {
    builtKey = seriesKey;
    hidden = new Set();
    build();
  }
  let lastData: uPlot.AlignedData | null = null;
  $: if (plot && data !== lastData) {
    lastData = data;
    plot.setData(data, false);
    applyRange();
  }
  $: if (plot) {
    void xRange;
    applyRange();
  }

  onMount(() => {
    width = Math.max(200, host.clientWidth);
    lastData = data;
    build();
    builtKey = seriesKey;
    ro = new ResizeObserver(() => {
      const w = Math.max(200, Math.floor(host.clientWidth));
      if (w !== width) {
        width = w;
        plot?.setSize({ width, height });
      }
    });
    ro.observe(host);
  });

  onDestroy(() => {
    ro?.disconnect();
    plot?.destroy();
  });

  // uPlot has no touch support; with touch-action: pan-y the browser keeps vertical scrolling and
  // hands horizontal drags to us, which become the same select-to-zoom as a mouse drag.
  const TOUCH_SLOP = 8;
  let touch: { id: number; x0: number; dragging: boolean } | null = null;
  let lastTouch = -Infinity;

  function overPos(e: PointerEvent): { x: number; y: number } | null {
    if (!plot) return null;
    const r = plot.over.getBoundingClientRect();
    return { x: Math.min(Math.max(e.clientX - r.left, 0), r.width), y: Math.min(Math.max(e.clientY - r.top, 0), r.height) };
  }

  // uPlot's third setCursor argument publishes to synced charts; its typings omit it.
  type SetCursorPub = (opts: { left: number; top: number }, fire?: boolean, pub?: boolean) => void;
  function setCursorSynced(p: { x: number; y: number }) {
    if (plot) (plot.setCursor as SetCursorPub)({ left: p.x, top: p.y }, true, true);
  }

  function onPointerDown(e: PointerEvent) {
    if (e.pointerType !== 'touch' || !e.isPrimary || !plot || !plot.over.contains(e.target as Node)) return;
    const p = overPos(e);
    if (p) touch = { id: e.pointerId, x0: p.x, dragging: false };
  }

  function onPointerMove(e: PointerEvent) {
    if (!touch || e.pointerId !== touch.id || !plot) return;
    const p = overPos(e);
    if (!p) return;
    if (!touch.dragging && Math.abs(p.x - touch.x0) < TOUCH_SLOP) return;
    touch.dragging = true;
    plot.setSelect({ left: Math.min(touch.x0, p.x), width: Math.abs(p.x - touch.x0), top: 0, height: plot.over.clientHeight }, false);
    setCursorSynced(p);
  }

  function onPointerUp(e: PointerEvent) {
    if (!touch || e.pointerId !== touch.id || !plot) return;
    lastTouch = performance.now();
    const t = touch;
    touch = null;
    const p = overPos(e);
    if (!p) return;
    if (t.dragging) {
      const left = Math.min(t.x0, p.x);
      const w = Math.abs(p.x - t.x0);
      plot.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
      if (w > 4) dispatch('zoom', { min: plot.posToVal(left, 'x'), max: plot.posToVal(left + w, 'x') });
    } else {
      setCursorSynced(p);
    }
  }

  function onPointerCancel(e: PointerEvent) {
    if (!touch || e.pointerId !== touch.id) return;
    touch = null;
    plot?.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
  }

  function valueAt(d: uPlot.AlignedData, idx: number | null, idxs: (number | null)[], si: number): string {
    if (idx == null) return '';
    const j = idxs[si] ?? idx;
    const v = d[si]?.[j];
    return v == null ? '–' : num(v, decimals);
  }

  $: legendValues = series.map((_, i) => valueAt(data, cursorIdx, cursorIdxs, i + 1));

  $: cursorTime = cursorIdx != null && data[0][cursorIdx] != null ? fmtDateTime(new Date(data[0][cursorIdx] * 1000)) : null;
</script>

<div class="chart">
  <div class="legend" role="group" aria-label="Series">
    <span class="when num" aria-live="off">{cursorTime ?? unit}</span>
    {#each series as s, i (s.label)}
      <button
        type="button"
        class="item"
        class:off={hidden.has(i + 1)}
        aria-pressed={!hidden.has(i + 1)}
        title={hidden.has(i + 1) ? `Show ${s.label}` : `Hide ${s.label}`}
        on:click={() => toggle(i + 1)}
      >
        {#if s.legendDotId}
          <StrategyDot id={s.legendDotId} size={9} />
        {:else}
          <span class="swatch" class:dashed={!!s.dash} style="--c: {s.color}" />
        {/if}
        <span class="name">{s.label}</span>
        <span class="val num" class:empty={cursorIdx == null}>{cursorIdx != null ? legendValues[i] : ''}</span>
      </button>
    {/each}
    {#if band}
      <span class="band-key"><span class="band-swatch" />{band.label}</span>
    {/if}
  </div>
  <div class="plot-wrap">
    {#if cursorTime}<span class="when-over num" aria-hidden="true">{cursorTime}</span>{/if}
    <div
      class="plot"
      bind:this={host}
      role="img"
      aria-label={ariaLabel}
      on:dblclick={() => dispatch('zoom', null)}
      on:pointerdown={onPointerDown}
      on:pointermove={onPointerMove}
      on:pointerup={onPointerUp}
      on:pointercancel={onPointerCancel}
    />
  </div>
</div>

<style>
  .chart {
    min-width: 0;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 2px 4px;
    min-height: 30px;
    margin-bottom: var(--s-1);
  }
  .when {
    font-size: var(--fs-xs);
    color: var(--text-3);
    margin-right: var(--s-2);
    min-width: 92px;
  }
  .item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 28px;
    padding: 0 8px;
    border: 0;
    border-radius: var(--r-sm);
    background: transparent;
    color: var(--text-2);
    font-size: var(--fs-xs);
    font-weight: 550;
  }
  .item:hover {
    background: var(--surface-3);
  }
  .item.off {
    opacity: 0.45;
  }
  .item.off .name {
    text-decoration: line-through;
  }
  .swatch {
    width: 14px;
    height: 0;
    border-top: 2.5px solid var(--c);
    border-radius: 2px;
  }
  .swatch.dashed {
    border-top-style: dashed;
  }
  .val {
    color: var(--text);
    min-width: 3.5ch;
    text-align: right;
  }
  .val.empty {
    display: none;
  }
  .plot-wrap {
    position: relative;
  }
  .when-over {
    display: none;
  }
  .band-key {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0 8px;
    font-size: var(--fs-xs);
    color: var(--text-3);
  }
  .band-swatch {
    width: 14px;
    height: 10px;
    border-radius: 2px;
    background: linear-gradient(90deg, transparent, var(--warning-soft), var(--warning-border));
  }
  .plot {
    width: 100%;
    min-width: 0;
    touch-action: pan-y;
  }
  .plot :global(.u-select) {
    background: var(--accent-soft);
    border-left: 1px solid var(--accent);
    border-right: 1px solid var(--accent);
  }
  .plot :global(.u-cursor-x) {
    border-right: 1px dashed var(--text-3) !important;
  }
  .plot :global(.u-cursor-y) {
    display: none;
  }
  .plot :global(.u-over) {
    cursor: crosshair;
  }
  @media (max-width: 720px) {
    .legend {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0 4px;
    }
    /* On phones the cursor time floats over the plot instead of taking a legend row: a row that
       only holds the unit (already in the chart title) pushes the plot down, and one that appears
       on hover shifts everything below it. */
    .when {
      display: none;
    }
    .when-over {
      display: block;
      position: absolute;
      z-index: 2;
      top: 0;
      left: 56px;
      padding: 1px 6px;
      border-radius: var(--r-sm);
      background: color-mix(in srgb, var(--surface) 88%, transparent);
      color: var(--text-2);
      font-size: var(--fs-xs);
      pointer-events: none;
    }
    .item {
      height: 40px;
      min-width: 0;
    }
    /* Wrapped lines instead of an ellipsis: cut names leave only the dot colour to tell series apart. */
    .item {
      text-align: left;
      line-height: 1.2;
      height: auto;
      min-height: 40px;
      padding-top: 4px;
      padding-bottom: 4px;
    }
    .name {
      overflow: hidden;
      display: -webkit-box;
      -webkit-box-orient: vertical;
      -webkit-line-clamp: 3;
      line-clamp: 3;
      overflow-wrap: anywhere;
    }
    /* Reserved even without a cursor so names don't rewrap (and the chart below jump) on hover. */
    .val,
    .val.empty {
      display: inline-block;
      min-width: 5ch;
      margin-left: auto;
    }
    .band-key {
      grid-column: 1 / -1;
      height: 28px;
    }
  }
</style>
