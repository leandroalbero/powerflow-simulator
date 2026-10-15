<script lang="ts">
  import { theme } from '../../stores/theme';
  import Card from '../../ui/Card.svelte';
  import StrategyDot from '../../ui/StrategyDot.svelte';
  import { strategyColor } from '../../ui/colors';
  import { eur, eurSigned, numTrim } from '../../ui/format';
  import type { SweepCase } from '../../types';
  import { bestInCell, shownSavings, type SweepModel } from './derive';
  import { capLabel, stratNames, tariffNames } from './names';

  export let m: SweepModel;

  const H = 280;
  const M = { top: 16, right: 16, bottom: 34, left: 60 };
  /** Tariffs are told apart by line style and a direct label; colour is reserved for strategies. */
  const DASH = ['', '7 5', '2 5'];
  const INK = ['var(--text)', 'var(--text-2)', 'var(--text-3)'];

  let width = 720;

  interface Pt {
    cap: number;
    v: number;
    c: SweepCase | null;
  }

  $: tn = $tariffNames;
  $: dec = m.perYear ? 0 : 2;
  $: unit = m.perYear ? '/yr' : '';
  $: lines = m.tariffs.map((t) => {
    const pts: Pt[] = [{ cap: 0, v: 0, c: null }];
    for (const cap of m.capacities) {
      const c = bestInCell(m, t, cap);
      const v = shownSavings(m, c);
      if (c && v != null) pts.push({ cap, v, c });
    }
    return { tariff: t, pts };
  });
  $: usedStrategies = [...new Set(lines.flatMap((l) => l.pts.flatMap((p) => (p.c ? [p.c.strategy_id] : []))))];
  $: vals = lines.flatMap((l) => l.pts.map((p) => p.v));
  $: ticks = niceTicks(Math.min(0, ...vals), Math.max(0, ...vals) || 1, 4);
  $: tickDec = ticks.length > 1 && ticks[1] - ticks[0] < 1 ? 2 : 0;
  $: yMin = ticks[0];
  $: yMax = ticks[ticks.length - 1];
  $: xMax = Math.max(1, ...m.capacities);
  $: labelRoom = width < 520 ? 0 : 92;
  $: plotW = Math.max(120, width - M.left - M.right - labelRoom);
  $: plotH = H - M.top - M.bottom;
  $: x = (v: number) => M.left + (v / xMax) * plotW;
  $: y = (v: number) => M.top + ((yMax - v) / (yMax - yMin || 1)) * plotH;
  $: xTicks = thinTicks([0, ...m.capacities], x);
  $: endLabels = placeEndLabels(lines, y);

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

  /** Drops x labels that would overlap the previous one. */
  function thinTicks(vals: number[], xf: (v: number) => number): number[] {
    const out: number[] = [];
    let last = -Infinity;
    for (const v of vals) {
      const px = xf(v);
      if (px - last < 38) continue;
      out.push(v);
      last = px;
    }
    return out;
  }

  /** End-of-line labels, nudged apart vertically so they never overlap. */
  function placeEndLabels(ls: typeof lines, yf: (v: number) => number) {
    const items = ls
      .filter((l) => l.pts.length > 1)
      .map((l, i) => ({ tariff: l.tariff, i, y: yf(l.pts[l.pts.length - 1].v), x: l.pts[l.pts.length - 1].cap }))
      .sort((a, b) => a.y - b.y);
    for (let k = 1; k < items.length; k++) if (items[k].y - items[k - 1].y < 16) items[k].y = items[k - 1].y + 16;
    return items;
  }

  $: path = (pts: Pt[]) => pts.map((p, i) => `${i ? 'L' : 'M'}${x(p.cap).toFixed(1)},${y(p.v).toFixed(1)}`).join('');

  // ---- Hover: nearest capacity column ----
  let hoverCap: number | null = null;
  let tipW = 0;
  function onMove(e: PointerEvent) {
    const svg = e.currentTarget as SVGSVGElement;
    const r = svg.getBoundingClientRect();
    const px = e.clientX - r.left;
    let best: number | null = null;
    let bd = Infinity;
    for (const c of m.capacities) {
      const d = Math.abs(x(c) - px);
      if (d < bd) {
        bd = d;
        best = c;
      }
    }
    hoverCap = bd < 60 ? best : null;
  }
  $: hoverRows = hoverCap == null
    ? []
    : lines.map((l, i) => {
        const k = l.pts.findIndex((p) => p.cap === hoverCap);
        const p = k >= 0 ? l.pts[k] : null;
        const prev = k > 0 ? l.pts[k - 1] : null;
        return { tariff: l.tariff, i, p, gain: p && prev ? p.v - prev.v : null, prevCap: prev?.cap ?? null };
      });
  $: tipLeft = hoverCap == null ? 0 : Math.min(Math.max(x(hoverCap) + 12, 0), Math.max(0, width - tipW));
  $: tipFlip = hoverCap != null && x(hoverCap) + 12 + tipW > width;
</script>

<Card title="Diminishing returns" subtitle="Savings{m.perYear ? ' per year' : ''} by battery size, using the best strategy at each size">
  <div class="legend">
    {#each lines as l, i (l.tariff)}
      <span class="key"><svg width="22" height="8" aria-hidden="true"><line x1="1" x2="21" y1="4" y2="4" stroke={INK[i % 3]} stroke-width="2" stroke-dasharray={DASH[i % 3]} /></svg>{tn(l.tariff)}</span>
    {/each}
    <span class="sep" aria-hidden="true" />
    {#each usedStrategies as s (s)}
      <span class="key"><StrategyDot id={s} size={9} />{$stratNames(s)}</span>
    {/each}
  </div>
  <div class="wrap" bind:clientWidth={width}>
    <svg
      {width}
      height={H}
      role="img"
      aria-label="Savings by battery size for each tariff. The table below lists every value."
      on:pointermove={onMove}
      on:pointerdown={onMove}
      on:pointerleave={() => (hoverCap = null)}
    >
      {#each ticks as t}
        <line class="grid" class:zero={t === 0} x1={M.left} x2={M.left + plotW} y1={y(t)} y2={y(t)} />
        <text class="axis" x={M.left - 8} y={y(t)} dy="0.32em" text-anchor="end">{eur(t, tickDec)}</text>
      {/each}
      {#each xTicks as t}
        <text class="axis" x={x(t)} y={H - M.bottom + 18} text-anchor="middle">{t === 0 ? 'None' : numTrim(t)}{t === xTicks[xTicks.length - 1] ? ' kWh' : ''}</text>
      {/each}

      {#if hoverCap != null}
        <line class="cross" x1={x(hoverCap)} x2={x(hoverCap)} y1={M.top} y2={M.top + plotH} />
      {/if}

      {#each lines as l, i (l.tariff)}
        <path class="line" d={path(l.pts)} stroke={INK[i % 3]} stroke-dasharray={DASH[i % 3]} />
      {/each}
      {#each lines as l (l.tariff)}
        {#each l.pts as p (p.cap)}
          {#if p.c}
            <circle
              cx={x(p.cap)}
              cy={y(p.v)}
              r={hoverCap === p.cap ? 6.5 : 5}
              fill={strategyColor(p.c.strategy_id, $theme)}
              class="pt"
            />
          {:else}
            <circle cx={x(0)} cy={y(0)} r="3.5" class="origin" />
          {/if}
        {/each}
      {/each}
      {#each endLabels as e (e.tariff)}
        {#if labelRoom}
          <text class="end" x={x(e.x) + 12} y={e.y} dy="0.32em">{tn(e.tariff)}</text>
        {/if}
      {/each}
    </svg>

    {#if hoverCap != null}
      <div class="tip" class:flip={tipFlip} style="left: {tipFlip ? x(hoverCap) - 12 - tipW : tipLeft}px" bind:clientWidth={tipW} role="tooltip">
        <p class="tip-title num">{capLabel(hoverCap)}</p>
        {#each hoverRows as r (r.tariff)}
          <div class="tip-row">
            <span class="tip-name">{tn(r.tariff)}</span>
            {#if r.p?.c}
              <span class="tip-strat"><StrategyDot id={r.p.c.strategy_id} size={8} />{$stratNames(r.p.c.strategy_id)}</span>
              <span class="num tip-v">{eur(r.p.v, dec)}{unit}</span>
              {#if r.gain != null}
                <span class="num tip-gain">{eurSigned(r.gain, dec)} vs {r.prevCap === 0 ? 'no battery' : capLabel(r.prevCap ?? 0)}</span>
              {/if}
            {:else}
              <span class="tip-strat muted">No result</span>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  </div>
</Card>

<style>
  .legend {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--s-2) var(--s-4);
    margin-bottom: var(--s-3);
    font-size: var(--fs-xs);
    color: var(--text-2);
    font-weight: 550;
  }
  .key {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .sep {
    width: 1px;
    height: 14px;
    background: var(--border);
  }
  .wrap {
    position: relative;
    min-width: 0;
  }
  svg {
    display: block;
    overflow: visible;
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
  .line {
    fill: none;
    stroke-width: 2;
    stroke-linejoin: round;
    stroke-linecap: round;
  }
  .pt {
    stroke: var(--surface);
    stroke-width: 2;
    transition: r var(--dur) var(--ease);
  }
  .origin {
    fill: var(--text-3);
  }
  .cross {
    stroke: var(--text-3);
    stroke-dasharray: 3 3;
  }
  .end {
    fill: var(--text-2);
    font-size: 12px;
    font-weight: 600;
  }
  .tip {
    position: absolute;
    top: 8px;
    z-index: 5;
    width: max-content;
    max-width: 280px;
    padding: var(--s-3);
    border-radius: var(--r-md);
    background: var(--surface);
    border: 1px solid var(--border);
    box-shadow: var(--shadow-2);
    pointer-events: none;
    font-size: var(--fs-xs);
    display: grid;
    gap: var(--s-2);
  }
  .tip-title {
    font-weight: 650;
    font-size: var(--fs-sm);
  }
  .tip-row {
    display: grid;
    grid-template-columns: auto auto;
    gap: 1px var(--s-3);
  }
  .tip-name {
    grid-column: 1 / -1;
    font-weight: 600;
    color: var(--text);
  }
  .tip-strat {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--text-2);
  }
  .tip-v {
    text-align: right;
    font-weight: 650;
  }
  .tip-gain {
    grid-column: 1 / -1;
    color: var(--text-3);
  }
</style>
