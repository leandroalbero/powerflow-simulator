<script lang="ts">
  import type { DayRow } from './analysis';

  /** Measured (a) against simulated (b), one group per row. */
  export let rows: DayRow[];
  export let mode: 'bars' | 'lines' = 'bars';
  export let aLabel: string;
  export let bLabel: string;
  export let aColor: string;
  export let bColor: string;
  /** Draw the simulated bars as outlines (hollow strategies and the no-battery reference). */
  export let bHollow = false;
  export let bDash: number[] | undefined = undefined;
  export let fmt: (v: number) => string;
  export let axisFmt: (v: number) => string;
  export let diffLabel = 'Simulated − measured';
  export let ariaLabel: string;
  export let height = 240;

  let width = 600;
  const M = { top: 12, right: 8, bottom: 28, left: 52 };

  $: values = rows.flatMap((r) => [r.a, r.b]).filter((v): v is number => v != null);
  $: lo = Math.min(0, ...values);
  $: hi = Math.max(0, ...values);
  $: ticks = niceTicks(lo, hi === lo ? lo + 1 : hi, 4);
  $: yMin = ticks[0];
  $: yMax = ticks[ticks.length - 1];
  $: plotW = Math.max(60, width - M.left - M.right);
  $: plotH = height - M.top - M.bottom;
  $: y = (v: number) => M.top + ((yMax - v) / (yMax - yMin || 1)) * plotH;
  $: groupW = plotW / Math.max(1, rows.length);
  $: gap = groupW >= 14 ? 2 : groupW >= 6 ? 1 : 0;
  $: inner = Math.min(groupW * 0.76, 2 * 26 + gap);
  $: barW = Math.max(1, (inner - gap) / 2);
  $: labelEvery = Math.max(1, Math.ceil(rows.length / Math.max(1, Math.floor(plotW / 58))));
  $: cx = (i: number) => M.left + i * groupW + groupW / 2;

  function niceTicks(a: number, b: number, n: number): number[] {
    const span = b - a || 1;
    const raw = span / n;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= raw) ?? raw;
    const start = Math.floor(a / step) * step;
    const out: number[] = [];
    for (let v = start; v <= b + step * 0.5; v += step) out.push(Math.round(v * 1e6) / 1e6);
    if (out[out.length - 1] < b) out.push(out[out.length - 1] + step);
    return out;
  }

  /** Rounded on the data end only, so negative values (export-heavy days) hang below zero. */
  function barPath(x: number, v: number, w: number): string {
    const y0 = y(0);
    const y1 = y(v);
    const h = Math.abs(y1 - y0);
    if (h < 0.5) return '';
    const r = Math.min(3, w / 2, h);
    if (v >= 0) return `M${x},${y0}V${y1 + r}Q${x},${y1} ${x + r},${y1}H${x + w - r}Q${x + w},${y1} ${x + w},${y1 + r}V${y0}Z`;
    return `M${x},${y0}V${y1 - r}Q${x},${y1} ${x + r},${y1}H${x + w - r}Q${x + w},${y1} ${x + w},${y1 - r}V${y0}Z`;
  }

  function linePath(vals: (number | null)[]): string {
    let d = '';
    let pen = false;
    vals.forEach((v, i) => {
      if (v == null) {
        pen = false;
        return;
      }
      d += `${pen ? 'L' : 'M'}${cx(i).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  }

  $: aPath = mode === 'lines' ? linePath(rows.map((r) => r.a)) : '';
  $: bPath = mode === 'lines' ? linePath(rows.map((r) => r.b)) : '';

  // ---- Hover / tap ----
  let hover: number | null = null;
  let svgEl: SVGSVGElement;
  let tipW = 0;
  let tipH = 0;

  function idxAt(e: PointerEvent): number | null {
    const r = svgEl.getBoundingClientRect();
    const x = e.clientX - r.left - M.left;
    if (x < 0 || x > plotW) return null;
    return Math.min(rows.length - 1, Math.max(0, Math.floor(x / groupW)));
  }
  function onMove(e: PointerEvent) {
    hover = idxAt(e);
  }
  function onLeave(e: PointerEvent) {
    if (e.pointerType !== 'touch') hover = null;
  }

  $: hRow = hover != null ? rows[hover] : null;
  $: anchorY = hRow ? y(Math.max(0, hRow.a ?? 0, hRow.b ?? 0)) : 0;
  $: tipLeft = hover != null ? Math.min(Math.max(cx(hover) - tipW / 2, 0), Math.max(0, width - tipW)) : 0;
  $: tipTop = hRow ? (anchorY - tipH - 10 >= 0 ? anchorY - tipH - 10 : Math.min(anchorY + 10, height - tipH)) : 0;
  $: diff = hRow && hRow.a != null && hRow.b != null ? hRow.b - hRow.a : null;
</script>

<div class="pc">
  <div class="legend" aria-hidden="true">
    <span class="key"><span class="sw" class:line={mode === 'lines'} style="--c: {aColor}" />{aLabel}</span>
    <span class="key"><span class="sw" class:line={mode === 'lines'} class:hollow={bHollow && mode === 'bars'} class:dashed={!!bDash && mode === 'lines'} style="--c: {bColor}" />{bLabel}</span>
  </div>
  <div class="wrap" bind:clientWidth={width}>
    <svg
      bind:this={svgEl}
      {width}
      {height}
      role="img"
      aria-label={ariaLabel}
      on:pointermove={onMove}
      on:pointerdown={onMove}
      on:pointerleave={onLeave}
    >
      {#each ticks as t}
        <line class="grid" class:zero={t === 0} x1={M.left} x2={M.left + plotW} y1={y(t)} y2={y(t)} />
        <text class="axis" x={M.left - 8} y={y(t)} dy="0.32em" text-anchor="end">{axisFmt(t)}</text>
      {/each}
      {#if hover != null}
        <rect class="hl" x={M.left + hover * groupW} y={M.top} width={groupW} height={plotH} />
      {/if}
      {#each rows as r, i (r.key)}
        {#if i % labelEvery === 0}
          <text class="axis" x={cx(i)} y={height - 8} text-anchor="middle">{r.label}</text>
        {/if}
        {#if mode === 'bars'}
          {#if r.a == null && r.b == null}
            <rect class="nodata" x={M.left + i * groupW + 1} y={M.top} width={Math.max(1, groupW - 2)} height={plotH} />
          {:else}
            {@const x0 = cx(i) - inner / 2}
            {#if r.a != null}<path d={barPath(x0, r.a, barW)} fill={aColor} class:dim={hover != null && hover !== i} />{/if}
            {#if r.b != null}
              <path
                d={barPath(x0 + barW + gap, r.b, barW)}
                fill={bHollow ? 'none' : bColor}
                stroke={bHollow ? bColor : 'none'}
                stroke-width={bHollow ? 1.5 : 0}
                class:dim={hover != null && hover !== i}
              />
            {/if}
          {/if}
        {/if}
      {/each}
      {#if mode === 'lines'}
        <path class="ln" d={aPath} stroke={aColor} />
        <path class="ln" d={bPath} stroke={bColor} stroke-dasharray={bDash?.join(' ')} />
        {#if hRow}
          {#if hRow.a != null}<circle cx={cx(hover ?? 0)} cy={y(hRow.a)} r="3.5" fill={aColor} />{/if}
          {#if hRow.b != null}<circle cx={cx(hover ?? 0)} cy={y(hRow.b)} r="3.5" fill={bColor} />{/if}
        {/if}
      {/if}
    </svg>
    {#if hRow}
      <div class="tip" style="left: {tipLeft}px; top: {tipTop}px" bind:offsetWidth={tipW} bind:offsetHeight={tipH}>
        <span class="tip-h">{hRow.tipLabel}</span>
        {#if hRow.a == null && hRow.b == null}
          <span class="tip-row muted">No measured data</span>
        {:else}
          <span class="tip-row"><span class="sw" style="--c: {aColor}" />{aLabel}<strong class="num">{hRow.a == null ? '—' : fmt(hRow.a)}</strong></span>
          <span class="tip-row"><span class="sw" class:hollow={bHollow} style="--c: {bColor}" />{bLabel}<strong class="num">{hRow.b == null ? '—' : fmt(hRow.b)}</strong></span>
          {#if diff != null}<span class="tip-row diff">{diffLabel}<strong class="num">{diff >= 0 ? '+' : '−'}{fmt(Math.abs(diff))}</strong></span>{/if}
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 16px;
    margin-bottom: var(--s-2);
  }
  .key {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: var(--fs-xs);
    font-weight: 550;
    color: var(--text-2);
  }
  .sw {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    background: var(--c);
    flex: none;
  }
  .sw.hollow {
    background: transparent;
    box-shadow: inset 0 0 0 1.5px var(--c);
  }
  .sw.line {
    width: 14px;
    height: 0;
    border-radius: 0;
    background: none;
    border-top: 2.5px solid var(--c);
  }
  .sw.line.dashed {
    border-top-style: dashed;
  }
  .wrap {
    position: relative;
    width: 100%;
    overflow: hidden;
  }
  svg {
    display: block;
    touch-action: pan-y;
  }
  .grid {
    stroke: var(--chart-grid);
    stroke-width: 1;
  }
  .grid.zero {
    stroke: var(--border-strong);
  }
  .axis {
    fill: var(--chart-axis);
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }
  .hl {
    fill: var(--surface-3);
    opacity: 0.7;
  }
  .nodata {
    fill: var(--surface-2);
  }
  path {
    transition: opacity var(--dur) var(--ease);
  }
  path.dim {
    opacity: 0.45;
  }
  .ln {
    fill: none;
    stroke-width: 2;
    stroke-linejoin: round;
    stroke-linecap: round;
  }
  .tip {
    position: absolute;
    pointer-events: none;
    display: grid;
    gap: 3px;
    min-width: 180px;
    padding: 8px 10px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r-md);
    box-shadow: var(--shadow-2);
    font-size: var(--fs-xs);
    white-space: nowrap;
    z-index: 2;
  }
  .tip-h {
    color: var(--text-3);
    margin-bottom: 1px;
  }
  .tip-row {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--text-2);
  }
  .tip-row strong {
    margin-left: auto;
    padding-left: var(--s-3);
    color: var(--text);
    font-size: var(--fs-sm);
  }
  .tip-row.diff {
    padding-top: 3px;
    border-top: 1px solid var(--border);
  }
</style>
