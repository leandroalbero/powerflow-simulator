<script lang="ts">
  import { strategyName } from '../../stores/app';
  import { theme } from '../../stores/theme';
  import type { BacktestResult } from '../../types';
  import Badge from '../../ui/Badge.svelte';
  import KPI from '../../ui/KPI.svelte';
  import StrategyDot from '../../ui/StrategyDot.svelte';
  import { flowColor, strategyColor, strategyHollow } from '../../ui/colors';
  import { eur, kwh, num } from '../../ui/format';
  import { costPctMeaningful, exportIsTiny, realVsSim, totals } from './analysis';

  export let result: BacktestResult;
  export let days: number;
  export let noBattery: BacktestResult | null = null;

  $: t = totals(result);
  $: id = result.strategy_id;
  $: name = strategyName(id);
  $: m = result.metrics;
  $: diff = realVsSim(result);
  $: same = Math.abs(diff.eur) < Math.max(0.05 * Math.abs(m.sim_cost), 0.1);
  $: tiny = exportIsTiny(t);
  $: pctOk = costPctMeaningful(result);
  $: earning = m.measured_cost < 0 || m.sim_cost < 0;

  $: bars = [
    { key: 'measured', label: 'Your system (measured)', value: m.measured_cost, color: flowColor('load', $theme), hollow: false, dot: null as string | null },
    { key: 'sim', label: `${name} (simulated)`, value: m.sim_cost, color: strategyColor(id, $theme), hollow: strategyHollow(id) || id === 'no_battery', dot: id },
    ...(noBattery && id !== 'no_battery'
      ? [{ key: 'nb', label: 'No battery (simulated)', value: noBattery.metrics.sim_cost, color: strategyColor('no_battery', $theme), hollow: true, dot: 'no_battery' }]
      : []),
  ];
  $: maxAbs = Math.max(...bars.map((b) => Math.abs(b.value)), 0.01);
  const signed = (v: number) => `${v >= 0 ? '+' : '−'}${num(Math.abs(v), 1)}%`;
</script>

<div class="kpis">
  <section class="cost" aria-label="Cost, measured against simulated">
    <p class="label">Cost over {days === 1 ? 'this day' : `these ${days} days`}</p>
    <ul class="bars">
      {#each bars as b (b.key)}
        <li class:ref={b.key === 'nb'}>
          <span class="bl">
            {#if b.dot}<StrategyDot id={b.dot} size={9} />{:else}<span class="ink" style="--c: {b.color}" />{/if}
            {b.label}
          </span>
          <span class="bv num">{eur(b.value)}</span>
          <span class="track" aria-hidden="true">
            <span class="fill" class:hollow={b.hollow} class:neg={b.value < 0} style="--c: {b.color}; width: {(Math.abs(b.value) / maxAbs) * 100}%" />
          </span>
        </li>
      {/each}
    </ul>
    <div class="verdict">
      {#if same}
        <Badge>About the same</Badge>
        <span class="num">Real {eur(m.measured_cost)} vs simulated {eur(m.sim_cost)}</span>
      {:else if diff.eur > 0}
        <Badge tone="positive">{earning ? `Real ${eur(diff.eur)} better off` : `Real paid ${eur(diff.eur)} less`}</Badge>
        {#if diff.pctOfSim != null}<span class="num">{num(diff.pctOfSim, 1)}% below {id === 'no_battery' ? 'no battery' : name}’s cost</span>{/if}
      {:else}
        <Badge tone="negative">{earning ? `Real ${eur(-diff.eur)} worse off` : `Real paid ${eur(-diff.eur)} more`}</Badge>
        {#if diff.pctOfSim != null}<span class="num">{num(-diff.pctOfSim, 1)}% above {id === 'no_battery' ? 'no battery' : name}’s cost</span>{/if}
      {/if}
    </div>
    <p class="foot num">
      {#if !pctOk}
        Measured cost is near zero or negative, so no % {id === 'no_battery' ? 'difference' : 'error'} is shown
      {:else if id === 'no_battery'}
        <!-- Not a simulator error here: the gap is what the real battery is worth. -->
        Without a battery the cost would be {num(Math.abs(m.cost_error_pct), 1)}% {m.cost_error_pct >= 0 ? 'higher' : 'lower'} than measured
      {:else}
        Simulator cost error {signed(m.cost_error_pct)} of the measured cost
      {/if}
    </p>
  </section>

  <KPI label="Grid import error" value={signed(m.import_error_pct)}>
    <span class="num">{kwh(t.measuredImport, 0)} measured · {kwh(t.simImport, 0)} simulated</span>
  </KPI>

  {#if tiny}
    <KPI label="Grid export" value="{num(t.measuredExport, 1)} vs {num(t.simExport, 1)} kWh">
      <span>Measured vs simulated. Too little export for a % error to mean much</span>
    </KPI>
  {:else}
    <KPI label="Grid export error" value={signed(m.export_error_pct)}>
      <span class="num">{kwh(t.measuredExport, 0)} measured · {kwh(t.simExport, 0)} simulated</span>
    </KPI>
  {/if}

  <KPI label="Daily import error" value={kwh(m.daily_import_mae_kwh, 2)}>
    <span>Average miss per day, either direction (MAE)</span>
  </KPI>

  <KPI label="Battery charge error" value={id === 'no_battery' ? '—' : `${num(m.soc_rmse_pct, 1)} pp`}>
    <span>{id === 'no_battery' ? 'No simulated battery to compare' : 'Typical gap in charge level, percentage points (RMSE)'}</span>
  </KPI>
</div>

<style>
  .kpis {
    display: grid;
    grid-template-columns: minmax(0, 1.35fr) repeat(2, minmax(0, 1fr));
    grid-template-rows: auto auto;
    gap: var(--s-3);
  }
  .cost {
    grid-row: span 2;
    display: flex;
    flex-direction: column;
    gap: var(--s-3);
    min-width: 0;
    padding: var(--s-5);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    box-shadow: var(--shadow-1);
  }
  .label {
    font-size: var(--fs-sm);
    font-weight: 600;
    color: var(--text-2);
  }
  .bars {
    display: grid;
    gap: var(--s-3);
  }
  .bars li {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 4px var(--s-3);
    align-items: baseline;
  }
  .bl {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
    font-size: var(--fs-sm);
    color: var(--text-2);
    font-weight: 550;
  }
  .ink {
    width: 9px;
    height: 9px;
    border-radius: 2px;
    background: var(--c);
    flex: none;
  }
  .bv {
    font-size: var(--fs-xl);
    font-weight: 680;
    letter-spacing: -0.02em;
  }
  li:first-child .bv {
    font-size: var(--fs-2xl);
  }
  li.ref .bv {
    font-size: var(--fs-lg);
    color: var(--text-2);
    font-weight: 600;
  }
  .track {
    grid-column: 1 / -1;
    height: 8px;
    border-radius: 4px;
    background: var(--surface-3);
    overflow: hidden;
  }
  li.ref .track {
    height: 6px;
  }
  .fill {
    display: block;
    height: 100%;
    border-radius: 4px;
    background: var(--c);
    transition: width 300ms var(--ease);
  }
  .fill.hollow {
    background: repeating-linear-gradient(135deg, var(--c) 0 2px, transparent 2px 5px);
  }
  .verdict {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px 10px;
    margin-top: auto;
    padding-top: var(--s-3);
    border-top: 1px solid var(--border);
    font-size: var(--fs-sm);
    color: var(--text-2);
  }
  .verdict :global(.badge) {
    height: 24px;
    padding: 0 10px;
    font-size: var(--fs-sm);
  }
  .foot {
    font-size: var(--fs-xs);
    color: var(--text-3);
  }
  @media (max-width: 1080px) {
    .kpis {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .cost {
      grid-column: 1 / -1;
      grid-row: auto;
    }
  }
  @media (max-width: 720px) {
    .cost {
      padding: var(--s-4);
    }
    .kpis {
      gap: var(--s-2);
    }
  }
</style>
