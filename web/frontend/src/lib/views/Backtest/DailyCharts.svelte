<script lang="ts">
  import { strategyName } from '../../stores/app';
  import { theme } from '../../stores/theme';
  import type { BacktestResult } from '../../types';
  import Card from '../../ui/Card.svelte';
  import Segmented from '../../ui/Segmented.svelte';
  import { flowColor, strategyColor, strategyDash, strategyHollow } from '../../ui/colors';
  import { eur, num } from '../../ui/format';
  import { cumulativeRows, dailyRows } from './analysis';
  import PairChart from './PairChart.svelte';

  export let result: BacktestResult;

  let costMode: 'period' | 'cumulative' = 'period';

  $: id = result.strategy_id;
  $: imp = dailyRows(result.daily, (d) => [d.measured_import_kwh, d.sim_import_kwh]);
  $: cost = dailyRows(result.daily, (d) => [d.measured_cost, d.sim_cost]);
  $: cum = cumulativeRows(result.daily);
  $: per = imp.weekly ? 'week' : 'day';
  $: common = {
    aLabel: 'Measured',
    bLabel: `${strategyName(id)} (simulated)`,
    aColor: flowColor('load', $theme),
    bColor: strategyColor(id, $theme),
    bHollow: strategyHollow(id) || id === 'no_battery',
    bDash: strategyDash(id),
  };
  const kwhFmt = (v: number) => `${num(v, 1)} kWh`;
  const kwhAxis = (v: number) => num(v, Math.abs(v) < 10 && v % 1 !== 0 ? 1 : 0);
  const eurFmt = (v: number) => eur(v);
  const eurAxis = (v: number) => eur(v, Math.abs(v) < 10 && v % 1 !== 0 ? 2 : 0);
</script>

<div class="pair">
  <Card title="Grid import per {per} · kWh" subtitle={imp.weekly ? 'Summed per week (Monday to Sunday) to stay readable' : 'Energy bought from the grid each day'}>
    <PairChart
      rows={imp.rows}
      {...common}
      fmt={kwhFmt}
      axisFmt={kwhAxis}
      ariaLabel="Grid import per {per} in kWh, measured against simulated"
    />
  </Card>

  <Card title={costMode === 'cumulative' ? 'Cumulative cost · €' : `Cost per ${per} · €`} subtitle={costMode === 'cumulative' ? 'Running total, where the gap opens up' : 'Imports minus export revenue'}>
    <svelte:fragment slot="actions">
      <Segmented
        label="Cost view"
        bind:value={costMode}
        options={[
          { value: 'period', label: imp.weekly ? 'Weekly' : 'Daily' },
          { value: 'cumulative', label: 'Cumulative' },
        ]}
      />
    </svelte:fragment>
    {#if costMode === 'cumulative'}
      <PairChart
        rows={cum}
        mode="lines"
        {...common}
        fmt={eurFmt}
        axisFmt={eurAxis}
        ariaLabel="Cumulative cost in euros, measured against simulated"
      />
    {:else}
      <PairChart
        rows={cost.rows}
        {...common}
        fmt={eurFmt}
        axisFmt={eurAxis}
        ariaLabel="Cost per {per} in euros, measured against simulated"
      />
    {/if}
  </Card>
</div>

<style>
  .pair {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--s-4);
  }
  @media (max-width: 1080px) {
    .pair {
      grid-template-columns: minmax(0, 1fr);
    }
  }
</style>
