<script lang="ts">
  import KPI from '../../ui/KPI.svelte';
  import StrategyDot from '../../ui/StrategyDot.svelte';
  import { eur, num, pct } from '../../ui/format';
  import { capLabel, stratNames, tariffNames } from './names';
  import { cheapest, cheapestBaseline, mostSaving, shownCost, shownSavings, type SweepModel } from './derive';

  export let m: SweepModel;
  export let days: number;
  export let provisional = false;
  export let bestSuffix = ' so far';
  /** Why the figures are incomplete, shown above the cards. */
  export let provisionalText = 'Provisional: figures cover the finished cases only and can still change.';

  $: best = cheapest(m);
  $: saver = mostSaving(m);
  $: baseBest = cheapestBaseline(m);
  $: tn = $tariffNames;
  $: unit = m.perYear ? '/yr' : '';
  $: dataDays = m.hours != null && days * 24 - m.hours > 1.5 ? m.hours / 24 : null;
  $: periodWord = (days === 1 ? 'this day' : `these ${days} days`) + (dataDays != null ? ` (${num(dataDays, 1)} d with data)` : '');
  $: bestBase = best ? m.baselines.get(best.tariff) : undefined;
  $: bestSavings = shownSavings(m, best ?? undefined);
  $: wins = bestSavings != null && bestSavings > 0.005;
  $: tone = (bestSavings == null ? 'default' : wins ? 'positive' : 'negative') as 'default' | 'positive' | 'negative';
  // The biggest saving against its own tariff can belong to a combination that still costs more
  // overall, when that tariff's no-battery cost is higher to begin with.
  $: divergent = best && saver && saver !== best && (saver.savings_vs_baseline ?? 0) > (best.savings_vs_baseline ?? 0) + 0.005;
  $: tariffSpread = (() => {
    if (m.baselines.size < 2 || !baseBest) return null;
    let worst = baseBest;
    for (const b of m.baselines.values()) if ((b.adjusted_cost ?? -Infinity) > (worst.adjusted_cost ?? -Infinity)) worst = b;
    const diff = (shownCost(m, worst) ?? 0) - (shownCost(m, baseBest) ?? 0);
    return { worst, diff };
  })();
  $: cyclesPerYear = best?.equivalent_full_cycles != null && m.hours ? (best.equivalent_full_cycles * 8760) / m.hours : null;
</script>

{#if provisional}
  <p class="provisional" role="status">{provisionalText}</p>
{/if}
<div class="kpis" aria-live="polite">
  <KPI label="Cheapest combination{provisional ? bestSuffix : ''}" value={best ? `${tn(best.tariff)} · ${capLabel(best.capacity_kwh)}` : '—'}>
    {#if best}
      <span class="with-dot"><StrategyDot id={best.strategy_id} size={9} /><span>{$stratNames(best.strategy_id)}</span></span>
    {:else}
      <span>Waiting for results</span>
    {/if}
  </KPI>

  <KPI hero label="Savings, cheapest combination" {tone} value={bestSavings == null ? '—' : `${eur(bestSavings, m.perYear ? 0 : 2)}${unit}`}>
    {#if best && bestSavings != null}
      <span class="num">
        On {tn(best.tariff)}{m.perYear ? `, ${eur(best.savings_vs_baseline)} over ${periodWord}` : ` over ${periodWord}${m.hours != null ? ', too short to annualise' : ''}`}
      </span>
    {:else}
      <span>Compared with the same tariff without a battery</span>
    {/if}
  </KPI>

  <KPI label="Net cost{m.perYear ? ' per year' : ''}" value={best ? `${eur(shownCost(m, best), m.perYear ? 0 : 2)}${unit}` : '—'}>
    {#if bestBase}
      <span class="num">{eur(shownCost(m, bestBase), m.perYear ? 0 : 2)}{unit} with no battery</span>
    {:else}
      <span>Energy cost minus the value left in the battery</span>
    {/if}
  </KPI>

  {#if tariffSpread && baseBest}
    <KPI label="Cheapest tariff without a battery" value={tn(baseBest.tariff)}>
      <span class="num">{eur(tariffSpread.diff, m.perYear ? 0 : 2)}{unit} less than {tn(tariffSpread.worst.tariff)}</span>
    </KPI>
  {:else}
    <KPI label="Battery use" value={cyclesPerYear != null && m.perYear ? `${num(cyclesPerYear, 0)} cycles/yr` : best ? `${num(best.equivalent_full_cycles, 1)} cycles` : '—'}>
      {#if best}<span class="num">{pct(best.self_consumption_rate)} of solar used at home</span>{:else}<span>Equivalent full cycles</span>{/if}
    </KPI>
  {/if}
</div>
{#if divergent && saver}
  <p class="note">
    <strong>{tn(saver.tariff)} · {capLabel(saver.capacity_kwh)} · {$stratNames(saver.strategy_id)}</strong> saves more against its own
    tariff ({eur(shownSavings(m, saver), m.perYear ? 0 : 2)}{unit}), but its tariff costs more to begin with, so the total is
    {eur((shownCost(m, saver) ?? 0) - (shownCost(m, best ?? undefined) ?? 0), m.perYear ? 0 : 2)}{unit} higher.
  </p>
{/if}

<style>
  .provisional {
    margin-bottom: calc(var(--s-2) * -1);
    font-size: var(--fs-sm);
    color: var(--text-3);
  }
  .kpis {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: var(--s-3);
  }
  .with-dot {
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }
  /* Same callout as Backtest's summary. */
  .note {
    font-size: var(--fs-sm);
    color: var(--text-2);
    padding: var(--s-3) var(--s-4);
    border-radius: var(--r-lg);
    background: var(--accent-soft);
    border: 1px solid color-mix(in srgb, var(--accent) 22%, transparent);
  }
  .note strong {
    font-weight: 600;
    color: var(--text);
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
