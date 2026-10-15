<script lang="ts">
  import { strategyById } from '../../stores/app';
  import type { Ranked } from '../../stores/run';
  import { theme } from '../../stores/theme';
  import Card from '../../ui/Card.svelte';
  import StrategyDot from '../../ui/StrategyDot.svelte';
  import { strategyColor, strategyHollow } from '../../ui/colors';
  import { eur, fmtMonth } from '../../ui/format';

  export let rows: Ranked[];

  let width = 800;
  const H = 260;
  const M = { top: 12, right: 8, bottom: 28, left: 56 };

  $: series = rows.filter((r) => r.status === 'completed' && r.metrics?.monthly?.length);
  $: months = [...new Set(series.flatMap((r) => r.metrics?.monthly?.map((m) => m.month) ?? []))].sort();
  $: lookup = new Map(series.map((r) => [r.id, new Map((r.metrics?.monthly ?? []).map((m) => [m.month, m.cost]))]));
  $: values = series.flatMap((r) => months.map((m) => lookup.get(r.id)?.get(m) ?? 0));
  $: lo = Math.min(0, ...values);
  $: hi = Math.max(0, ...values) || 1;
  $: ticks = niceTicks(lo, hi, 4);
  $: yMin = ticks[0];
  $: yMax = ticks[ticks.length - 1];
  $: plotW = Math.max(100, width - M.left - M.right);
  $: plotH = H - M.top - M.bottom;
  $: y = (v: number) => M.top + ((yMax - v) / (yMax - yMin || 1)) * plotH;
  $: groupW = plotW / Math.max(1, months.length);
  $: inner = Math.min(groupW * 0.78, series.length * 26);
  $: barW = Math.max(2, (inner - (series.length - 1) * 2) / Math.max(1, series.length));
  $: labelEvery = Math.ceil(months.length / Math.max(1, Math.floor(plotW / 56)));

  let hover: { month: string; id: string; v: number; x: number; y: number } | null = null;
  let tipW = 0;
  let tipH = 0;
  const GAP = 8;
  // .wrap clips overflow, so the tooltip is kept inside it: below the bar top when there is no
  // room above, and clamped at the left and right edges.
  $: tipLeft = hover ? Math.min(Math.max(hover.x - tipW / 2, 0), Math.max(0, width - tipW)) : 0;
  $: tipTop = hover ? (hover.y - tipH - GAP >= 0 ? hover.y - tipH - GAP : Math.min(hover.y + GAP, H - tipH)) : 0;

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

  /** Bar path with the 4px rounding on the data end only (top for positive, bottom for negative). */
  function barPath(x: number, v: number): string {
    const y0 = y(0);
    const y1 = y(v);
    const h = Math.abs(y1 - y0);
    const r = Math.min(4, barW / 2, h);
    if (h < 0.5) return '';
    if (v >= 0) {
      return `M${x},${y0}V${y1 + r}Q${x},${y1} ${x + r},${y1}H${x + barW - r}Q${x + barW},${y1} ${x + barW},${y1 + r}V${y0}Z`;
    }
    return `M${x},${y0}V${y1 - r}Q${x},${y1} ${x + r},${y1}H${x + barW - r}Q${x + barW},${y1} ${x + barW},${y1 - r}V${y0}Z`;
  }
</script>

{#if months.length >= 2}
  <Card title="Cost per month" subtitle="Energy cost each month, before the value of energy left in the battery">
    <div class="legend">
      {#each series as r (r.id)}
        <span class="key"><StrategyDot id={r.id} size={9} />{$strategyById.get(r.id)?.name ?? r.id}</span>
      {/each}
    </div>
    <div class="wrap" bind:clientWidth={width}>
      <svg {width} height={H} role="img" aria-label="Monthly energy cost per strategy" on:mouseleave={() => (hover = null)}>
        {#each ticks as t}
          <line class="grid" class:zero={t === 0} x1={M.left} x2={M.left + plotW} y1={y(t)} y2={y(t)} />
          <text class="axis" x={M.left - 8} y={y(t)} dy="0.32em" text-anchor="end">{eur(t, 0)}</text>
        {/each}
        {#each months as m, mi (m)}
          {@const gx = M.left + mi * groupW + (groupW - inner) / 2}
          {#if mi % labelEvery === 0}
            <text class="axis" x={M.left + mi * groupW + groupW / 2} y={H - 8} text-anchor="middle">
              {fmtMonth(m, months.length > 8)}{months.length > 8 && (mi === 0 || m.endsWith('-01')) ? ` ’${m.slice(2, 4)}` : ''}
            </text>
          {/if}
          {#each series as r, si (r.id)}
            {@const v = lookup.get(r.id)?.get(m) ?? 0}
            {@const c = strategyColor(r.id, $theme)}
            {@const hollow = strategyHollow(r.id) || r.isBaseline}
            {@const bx = gx + si * (barW + 2)}
            <path
              d={barPath(bx, v)}
              fill={hollow ? 'none' : c}
              stroke={hollow ? c : 'none'}
              stroke-width={hollow ? 1.5 : 0}
              class:dim={hover != null && (hover.id !== r.id || hover.month !== m)}
            />
            <rect
              class="hit"
              x={bx - 1}
              y={M.top}
              width={barW + 2}
              height={plotH}
              on:mouseenter={() => (hover = { month: m, id: r.id, v, x: bx + barW / 2, y: y(Math.max(0, v)) })}
              role="presentation"
            />
          {/each}
        {/each}
      </svg>
      {#if hover}
        <div class="tip" style="left: {tipLeft}px; top: {tipTop}px" bind:offsetWidth={tipW} bind:offsetHeight={tipH}>
          <span class="tip-m">{fmtMonth(hover.month)}</span>
          <span class="tip-row"><StrategyDot id={hover.id} size={8} />{$strategyById.get(hover.id)?.name ?? hover.id}</span>
          <strong class="num">{eur(hover.v)}</strong>
        </div>
      {/if}
    </div>
  </Card>
{/if}

<style>
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 16px;
    margin: calc(var(--s-2) * -1) 0 var(--s-3);
  }
  .key {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: var(--fs-xs);
    font-weight: 550;
    color: var(--text-2);
  }
  .wrap {
    position: relative;
    width: 100%;
    overflow: hidden;
  }
  svg {
    display: block;
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
  path {
    transition: opacity var(--dur) var(--ease);
  }
  path.dim {
    opacity: 0.35;
  }
  .hit {
    fill: transparent;
  }
  .tip {
    position: absolute;
    pointer-events: none;
    display: grid;
    gap: 2px;
    padding: 8px 10px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r-md);
    box-shadow: var(--shadow-2);
    font-size: var(--fs-xs);
    white-space: nowrap;
  }
  .tip-m {
    color: var(--text-3);
  }
  .tip-row {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--text-2);
  }
  .tip strong {
    font-size: var(--fs-sm);
  }
</style>
