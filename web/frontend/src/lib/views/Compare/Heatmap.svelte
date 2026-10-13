<script lang="ts">
  import { theme } from '../../stores/theme';
  import Card from '../../ui/Card.svelte';
  import Chip from '../../ui/Chip.svelte';
  import Icon from '../../ui/Icon.svelte';
  import StrategyDot from '../../ui/StrategyDot.svelte';
  import { eur, eurSigned, num, pct } from '../../ui/format';
  import type { SweepCase } from '../../types';
  import { bestInCell, cheapest, getCase, savingsDomain, shownCost, shownSavings, type SweepModel } from './derive';
  import { capLabel, stratNames, tariffNames } from './names';
  import { cellColors, legendGradient, negativeSwatch } from './scale';

  export let m: SweepModel;
  /** The sweep has stopped: cases still pending will never run. */
  export let settled = false;

  let view = 'best';
  $: if (view !== 'best' && !m.strategies.includes(view)) view = 'best';
  $: if (m.strategies.length === 1) view = m.strategies[0];

  $: domain = savingsDomain(m);
  $: tn = $tariffNames;
  $: unit = m.perYear ? '/yr' : '';
  $: dec = m.perYear ? 0 : 2;
  $: winner = cheapest(m, view === 'best' ? null : view);
  $: hasNegative = domain.min < -0.005;
  $: winners = view === 'best'
    ? [...new Set(m.tariffs.flatMap((t) => m.capacities.map((cap) => bestInCell(m, t, cap)?.strategy_id ?? '')))].filter(Boolean)
    : [];

  // Reactive function values, so cells re-render when the model, the sweep state or the strategy names change.
  $: caseFor = (tariff: string, cap: number, v: string): SweepCase | undefined => {
    if (v === 'best') {
      const b = bestInCell(m, tariff, cap);
      if (b) return b;
      // Nothing finished yet: show the cell's state from any of its cases.
      return m.strategies.map((s) => getCase(m, tariff, cap, s)).find((c) => c) ?? undefined;
    }
    return getCase(m, tariff, cap, v);
  };

  $: stateOf = (c: SweepCase | undefined): 'done' | 'pending' | 'failed' | 'cancelled' | 'notrun' | 'none' => {
    if (!c) return 'none';
    if (c.status === 'completed') return c.savings_vs_baseline == null ? 'pending' : 'done';
    if (c.status === 'failed') return 'failed';
    if (c.status === 'cancelled') return 'cancelled';
    return settled ? 'notrun' : 'pending';
  };

  // ---- Tooltip ----
  let wrap: HTMLDivElement;
  let tip: { c: SweepCase; x: number; y: number; below: boolean } | null = null;
  let tipW = 0;
  let wrapW = 0;

  function show(e: Event, c: SweepCase | undefined) {
    if (!c || !wrap) return;
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const w = wrap.getBoundingClientRect();
    const top = r.top - w.top;
    // Room for the tooltip above the cell, below the sticky header.
    tip = { c, x: r.left - w.left + r.width / 2, y: top, below: r.top < 330 };
    if (tip.below) tip.y = top + r.height;
  }
  function hide() {
    tip = null;
  }
  $: tipLeft = tip ? Math.min(Math.max(tip.x - tipW / 2, 0), Math.max(0, wrapW - tipW)) : 0;

  $: ariaFor = (t: string, cap: number, c: SweepCase | undefined): string => {
    const st = stateOf(c);
    const head = `${tn(t)}, ${capLabel(cap)}`;
    if (!c || st === 'none') return `${head}: not part of this comparison`;
    if (st !== 'done') return `${head}: ${st === 'pending' ? 'running' : st === 'notrun' ? 'not run' : st}`;
    return `${head}, ${$stratNames(c.strategy_id)}: saves ${eur(shownSavings(m, c), dec)}${m.perYear ? ' per year' : ''}`;
  };

  function onCellKey(e: KeyboardEvent) {
    if (e.key === 'Escape' && tip) {
      e.stopPropagation();
      hide();
    }
  }
</script>

<Card
  title="Savings by tariff and battery size"
  subtitle="Each cell: {m.perYear ? 'savings per year' : 'savings over the period'} against the same tariff with no battery. Outlined: the cheapest combination overall, by net cost."
>
  {#if m.strategies.length > 1}
    <div class="switch" role="group" aria-label="Strategy shown in the matrix">
      <Chip selected={view === 'best'} on:click={() => (view = 'best')}>Best strategy per cell</Chip>
      {#each m.strategies as s (s)}
        <Chip dotId={s} selected={view === s} on:click={() => (view = s)}>{$stratNames(s)}</Chip>
      {/each}
    </div>
  {/if}

  <div class="wrap" bind:this={wrap} bind:clientWidth={wrapW}>
    <div class="scroller">
      <table class="matrix" style="max-width: {200 + m.capacities.length * 168}px">
        <caption class="sr-only">
          {view === 'best' ? 'Best strategy' : $stratNames(view)}: savings per tariff (rows) and battery size (columns)
        </caption>
        <thead>
          <tr>
            <th scope="col" class="corner">Tariff<span class="sub muted">no battery{unit}</span></th>
            {#each m.capacities as cap (cap)}
              <th scope="col" class="num">{capLabel(cap)}</th>
            {/each}
          </tr>
        </thead>
        <tbody>
          {#each m.tariffs as t (t)}
            {@const base = m.baselines.get(t)}
            <tr>
              <th scope="row" class="rowhead">
                <span class="tname">{tn(t)}</span>
                <span class="base num">
                  {#if base?.status === 'completed'}{eur(shownCost(m, base), dec)}{unit}{:else if base?.status === 'failed'}Baseline failed{:else}…{/if}
                </span>
              </th>
              {#each m.capacities as cap (cap)}
                {@const c = caseFor(t, cap, view)}
                {@const st = stateOf(c)}
                {@const v = st === 'done' ? shownSavings(m, c) : null}
                {@const col = cellColors(v, domain, $theme)}
                {@const isBest = !!winner && c === winner}
                <td>
                  <button
                    type="button"
                    class="cell {st}"
                    class:best={isBest}
                    style={st === 'done' ? `--bg: ${col.bg}; --fg: ${col.fg}` : undefined}
                    aria-label={ariaFor(t, cap, c) + (isBest ? ', cheapest overall' : '')}
                    aria-describedby={tip && tip.c === c ? 'hm-tip' : undefined}
                    on:mouseenter={(e) => show(e, c)}
                    on:mouseleave={hide}
                    on:focus={(e) => show(e, c)}
                    on:blur={hide}
                    on:keydown={onCellKey}
                  >
                    {#if st === 'done'}
                      <span class="val num">{eur(v, dec)}</span>
                      {#if view === 'best' && c}<span class="who"><StrategyDot id={c.strategy_id} size={10} /></span>{/if}
                      {#if isBest}<span class="best-tag">Cheapest</span>{/if}
                    {:else if st === 'pending'}
                      <span class="shimmer" aria-hidden="true" />
                    {:else if st === 'failed'}
                      <span class="state"><Icon name="alert" size={13} /> Failed</span>
                    {:else if st === 'cancelled'}
                      <span class="state">Cancelled</span>
                    {:else}
                      <span class="state">Not run</span>
                    {/if}
                  </button>
                </td>
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>

    {#if tip}
      {@const c = tip.c}
      <div
        id="hm-tip"
        class="tip"
        class:below={tip.below}
        role="tooltip"
        bind:clientWidth={tipW}
        style="left: {tipLeft}px; top: {tip.y}px"
      >
        <p class="tip-title">{tn(c.tariff)} · {capLabel(c.capacity_kwh)}</p>
        <p class="tip-strat"><StrategyDot id={c.strategy_id} size={8} />{$stratNames(c.strategy_id)}</p>
        {#if c.status === 'completed'}
          <dl>
            <dt>Net cost{m.perYear ? ' per year' : ''}</dt><dd class="num">{eur(shownCost(m, c), dec)}</dd>
            {#if m.perYear}<dt>Net cost, period</dt><dd class="num">{eur(c.adjusted_cost)}</dd>{/if}
            <dt>Savings, period</dt><dd class="num">{eurSigned(c.savings_vs_baseline)}</dd>
            {#if m.perYear}<dt>Savings per year</dt><dd class="num strong">{eurSigned(c.annualised_savings, 0)}</dd>{/if}
            <dt>Full cycles</dt><dd class="num">{num(c.equivalent_full_cycles, 1)}</dd>
            <dt>Solar used at home</dt><dd class="num">{pct(c.self_consumption_rate)}</dd>
          </dl>
        {:else if c.status === 'failed'}
          <p class="tip-err">{c.error ?? 'This case failed'}</p>
        {:else}
          <p class="tip-muted">{c.status === 'cancelled' ? 'Cancelled' : settled ? 'Not run' : 'Running…'}</p>
        {/if}
      </div>
    {/if}
  </div>

  <div class="legend" aria-hidden="true">
    {#if domain.max > 0.005}
      <span class="scale">
        <span class="num">{eur(0, 0)}</span>
        <span class="bar" style="background: {legendGradient($theme)}" />
        <span class="num">{eur(domain.max, dec)}{unit}</span>
      </span>
    {/if}
    {#if hasNegative}
      <span class="key"><span class="sw" style="background: {negativeSwatch($theme)}" />Costs more than no battery</span>
    {/if}
    {#if winner}<span class="key"><span class="sw outline" />Cheapest overall</span>{/if}
    {#if view === 'best' && winners.length}
      <span class="key winners">Dot, winning strategy:
        {#each winners as w (w)}<span class="key"><StrategyDot id={w} size={8} />{$stratNames(w)}</span>{/each}
      </span>
    {/if}
  </div>
</Card>

<style>
  .switch {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-2);
    margin-bottom: var(--s-4);
  }
  .wrap {
    position: relative;
  }
  .scroller {
    overflow-x: auto;
    margin: 0 calc(var(--s-2) * -1);
    padding: 0 var(--s-2) var(--s-1);
  }
  .matrix {
    border-collapse: separate;
    border-spacing: 4px;
    width: 100%;
    font-variant-numeric: tabular-nums;
  }
  thead th {
    padding: 0 4px 4px;
    font-size: var(--fs-xs);
    font-weight: 600;
    color: var(--text-2);
    text-align: center;
    white-space: nowrap;
  }
  .corner {
    text-align: left;
  }
  .corner .sub {
    display: block;
    font-weight: 500;
  }
  thead th {
    vertical-align: bottom;
  }
  .rowhead {
    text-align: left;
    padding: 0 var(--s-3) 0 0;
    white-space: nowrap;
    width: 1%;
  }
  .tname {
    display: block;
    font-size: var(--fs-md);
    font-weight: 650;
  }
  .base {
    display: block;
    font-size: var(--fs-xs);
    font-weight: 500;
    color: var(--text-3);
  }
  td {
    padding: 0;
  }
  .cell {
    position: relative;
    display: grid;
    place-items: center;
    width: 100%;
    min-width: 76px;
    height: 60px;
    border: 0;
    border-radius: var(--r-md);
    background: var(--surface-2);
    color: var(--text-3);
    font-size: var(--fs-sm);
  }
  .cell.done {
    background: var(--bg);
    color: var(--fg);
  }
  .cell:hover {
    box-shadow: 0 0 0 2px var(--surface), 0 0 0 4px var(--text-3);
  }
  .cell.best {
    box-shadow: 0 0 0 2px var(--surface), 0 0 0 4px var(--text);
  }
  /* After .best so the winning cell also shows when it has focus. */
  .cell:focus-visible {
    outline: none;
    z-index: 2;
    box-shadow: 0 0 0 2px var(--surface), 0 0 0 4px var(--accent);
  }
  .cell.best:focus-visible {
    box-shadow: 0 0 0 2px var(--surface), 0 0 0 4px var(--text), 0 0 0 6px var(--accent);
  }
  .val {
    font-size: var(--fs-md);
    font-weight: 650;
    letter-spacing: -0.01em;
  }
  /* A square tile, not a round ring: inside a round surface ring, a fill close to the cell's own
     green read as the cell showing through, i.e. as a hollow (advanced-strategy) dot. */
  .who {
    position: absolute;
    left: 6px;
    top: 6px;
    display: inline-grid;
    place-items: center;
    width: 18px;
    height: 18px;
    border-radius: 5px;
    background: var(--surface);
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.16);
  }
  .best-tag {
    position: absolute;
    right: 6px;
    top: 5px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.02em;
    text-transform: uppercase;
  }
  .state {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: var(--fs-xs);
  }
  .cell.failed {
    color: var(--negative);
    background: var(--negative-soft);
  }
  .shimmer {
    width: 44%;
    height: 10px;
    border-radius: 5px;
    background: linear-gradient(90deg, var(--surface-3), var(--border), var(--surface-3));
    background-size: 200% 100%;
    animation: sh 1.2s linear infinite;
  }
  @keyframes sh {
    to {
      background-position: -200% 0;
    }
  }
  .tip {
    position: absolute;
    z-index: 20;
    width: max-content;
    max-width: 260px;
    padding: var(--s-3);
    border-radius: var(--r-md);
    background: var(--surface);
    border: 1px solid var(--border);
    box-shadow: var(--shadow-2);
    transform: translateY(calc(-100% - 8px));
    pointer-events: none;
    font-size: var(--fs-xs);
  }
  .tip.below {
    transform: translateY(8px);
  }
  .tip-title {
    font-weight: 650;
    font-size: var(--fs-sm);
  }
  .tip-strat {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--text-2);
    margin-bottom: var(--s-2);
  }
  dl {
    display: grid;
    grid-template-columns: auto auto;
    gap: 3px var(--s-4);
  }
  dt {
    color: var(--text-3);
  }
  dd {
    text-align: right;
    color: var(--text);
  }
  dd.strong {
    font-weight: 650;
  }
  .tip-err {
    color: var(--negative);
  }
  .tip-muted {
    color: var(--text-3);
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--s-2) var(--s-5);
    margin-top: var(--s-4);
    font-size: var(--fs-xs);
    color: var(--text-3);
  }
  .scale {
    display: inline-flex;
    align-items: center;
    gap: var(--s-2);
  }
  .bar {
    width: 120px;
    height: 10px;
    border-radius: 3px;
  }
  .key {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .sw {
    width: 14px;
    height: 14px;
    border-radius: 3px;
  }
  .sw.outline {
    box-shadow: inset 0 0 0 2px var(--text);
  }
  .winners {
    gap: var(--s-3);
    flex-wrap: wrap;
  }
  @media (max-width: 720px) {
    .cell {
      min-width: 64px;
      height: 52px;
    }
    /* Narrow cells leave no room for a corner badge clear of the centred value, so it sits inline. */
    .cell.done {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 5px;
    }
    .val {
      font-size: var(--fs-sm);
    }
    .who {
      position: static;
      order: -1;
      flex: none;
      width: 14px;
      height: 14px;
      border-radius: 4px;
    }
    /* The scroller's bleed would leave a strip left of the sticky column where scrolled cells show through. */
    .scroller {
      margin: 0;
      padding: 0 0 var(--s-1);
    }
    .rowhead,
    .corner {
      position: sticky;
      left: 0;
      z-index: 3;
      background: var(--surface);
      padding-right: var(--s-2);
      /* Cover the border-spacing on both sides so cells sliding under the column never peek out. */
      box-shadow: -4px 0 0 var(--surface), 4px 0 0 var(--surface);
    }
    .best-tag {
      display: none;
    }
  }
</style>
