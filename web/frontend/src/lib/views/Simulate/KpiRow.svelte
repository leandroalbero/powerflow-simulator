<script lang="ts">
  import { strategyName } from '../../stores/app';
  import { MIN_HOURS_FOR_ANNUAL, verdict, type Ranked } from '../../stores/run';
  import KPI from '../../ui/KPI.svelte';
  import StrategyDot from '../../ui/StrategyDot.svelte';
  import { eur, num, pct } from '../../ui/format';

  export let rows: Ranked[];
  export let days: number;
  /** True while some strategies are still running: the figures describe the finished ones only. */
  export let provisional = false;

  $: done = rows.filter((r) => r.status === 'completed' || r.status === 'failed' || r.status === 'cancelled').length;
  $: baseline = rows.find((r) => r.isBaseline && r.cost != null) ?? null;
  $: v = verdict(rows);
  $: best = v.best;
  $: beatsBaseline = v.outcome === 'win';
  $: bestName = best ? strategyName(best.id) : '';
  $: annual = best?.validHours != null && best.validHours >= MIN_HOURS_FOR_ANNUAL;
  let savingsTone: 'default' | 'positive' | 'negative' = 'default';
  $: savingsTone = v.outcome === 'win' ? 'positive' : v.outcome === 'loss' ? 'negative' : 'default';
  $: periodWord = days === 1 ? 'this day' : `these ${days} days`;
  $: soFar = provisional && best ? ' so far' : '';
  $: settled = rows.length > 0 && done === rows.length;
  $: idle = !settled
    ? 'Waiting for results'
    : rows.some((r) => !r.isBaseline && r.status === 'cancelled')
      ? 'Cancelled before a battery strategy finished'
      : 'No battery strategy finished';
</script>

{#if provisional}
  <p class="provisional num" role="status">Provisional: {done} of {rows.length} strategies finished. Figures can still change.</p>
{/if}
<div class="kpis" aria-live="polite">
  <KPI label="Best strategy{soFar}" value={best ? (beatsBaseline ? bestName : 'No battery') : '—'}>
    {#if best && beatsBaseline}
      <StrategyDot id={best.id} size={9} />
      <span class="num">{eur(best.cost)} net cost</span>
    {:else if best && v.outcome === 'tie'}
      <span class="num">No strategy beat it. {bestName} breaks even</span>
    {:else if best && best.savings != null}
      <span class="num">No strategy beat it. {bestName} costs {eur(-best.savings)} more</span>
    {:else}
      <span>{idle}</span>
    {/if}
  </KPI>

  <KPI
    hero
    label="Savings vs no battery{soFar}"
    tone={savingsTone}
    value={best?.savings == null ? '—' : annual ? `${eur(best.savingsPerYear, 0)}/yr` : eur(best.savings)}
  >
    {#if best?.savings != null}
      {#if !beatsBaseline}
        <span class="with-dot">
          <StrategyDot id={best.id} size={9} />
          <span class="num">{bestName}, the cheapest strategy{annual ? `: ${eur(best.savings)}` : ''} over {periodWord}</span>
        </span>
      {:else if annual}
        <span class="num">{eur(best.savings)} over {periodWord}</span>
      {:else}
        <span class="num">over {periodWord}, too short to annualise</span>
      {/if}
    {:else if baseline}
      <span class="num">No battery costs {eur(baseline.cost)}</span>
    {:else}
      <span>{idle}</span>
    {/if}
  </KPI>

  <KPI label="Self-sufficiency" value={best ? pct(best.metrics?.solar_fraction) : '—'}>
    {#if best && baseline}
      <span class="num">{beatsBaseline ? '' : `${bestName} · `}{pct(baseline.metrics?.solar_fraction)} without a battery</span>
    {:else}
      <span>Share of load not bought from the grid</span>
    {/if}
  </KPI>

  <KPI label="Cycles per day" value={best?.cyclesPerDay != null ? num(best.cyclesPerDay, 2) : '—'}>
    {#if best?.metrics?.equivalent_full_cycles != null}
      <span class="num">{beatsBaseline ? '' : `${bestName} · `}{num(best.metrics.equivalent_full_cycles, 1)} full cycles in total</span>
    {:else}
      <span>Equivalent full cycles</span>
    {/if}
  </KPI>
</div>

<style>
  .provisional {
    margin-bottom: calc(var(--s-2) * -1);
    font-size: var(--fs-sm);
    color: var(--text-3);
  }
  /* Keeps the dot beside the first line when the text wraps, instead of alone on its own line. */
  .with-dot {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    min-width: 0;
  }
  .with-dot > :global(.dot) {
    margin-top: calc((1.45em - 9px) / 2);
  }
  .with-dot > .num {
    min-width: 0;
  }
  .kpis {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: var(--s-3);
  }
  @media (max-width: 1080px) {
    .kpis {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
  @media (max-width: 420px) {
    .kpis {
      gap: var(--s-2);
    }
  }
</style>
