<script lang="ts">
  import type uPlot from 'uplot';
  import { strategyName } from '../../stores/app';
  import { theme } from '../../stores/theme';
  import type { BacktestResult } from '../../types';
  import Button from '../../ui/Button.svelte';
  import Card from '../../ui/Card.svelte';
  import Chart, { type ChartSeries, type XRange } from '../../ui/Chart.svelte';
  import { flowColor, strategyColor, strategyDash } from '../../ui/colors';
  import { fmtDateTime } from '../../ui/format';
  import { socSeries } from './analysis';

  export let result: BacktestResult;

  let xRange: XRange = null;
  let shownFor: BacktestResult | null = null;
  $: if (result !== shownFor) {
    shownFor = result;
    xRange = null;
  }

  $: s = socSeries(result);
  // A house without a battery has no charge level; the backend's flat sim_pct is only the start value.
  $: noBattery = result.strategy_id === 'no_battery';
  $: data = (noBattery ? [s.x, s.measured] : [s.x, s.measured, s.sim]) as uPlot.AlignedData;
  $: series = [
    { label: 'Measured', color: flowColor('load', $theme), width: 1.75 },
    ...(noBattery
      ? []
      : [
          {
            label: `${strategyName(result.strategy_id)} (simulated)`,
            color: strategyColor(result.strategy_id, $theme),
            dash: strategyDash(result.strategy_id),
            width: 1.75,
          },
        ]),
  ] satisfies ChartSeries[];

  const coarse = typeof matchMedia === 'function' && matchMedia('(pointer: coarse)').matches;
  const hint = coarse ? 'Drag sideways to zoom, tap to read values' : 'Drag across the chart to zoom, double-click to reset';
  $: subtitle = xRange
    ? `Zoomed to ${fmtDateTime(new Date(xRange.min * 1000))} – ${fmtDateTime(new Date(xRange.max * 1000))}`
    : noBattery
      ? `Measured only: without a battery there is no simulated charge. ${hint}`
      : hint;
</script>

<Card title="Battery charge · % of capacity" {subtitle}>
  <svelte:fragment slot="actions">
    {#if xRange}
      <Button size="sm" variant="secondary" icon="reset" on:click={() => (xRange = null)}>Reset zoom</Button>
    {/if}
  </svelte:fragment>
  <Chart
    {data}
    {series}
    unit="% of capacity"
    decimals={0}
    height={260}
    yMin={0}
    yMax={100}
    {xRange}
    on:zoom={(e) => (xRange = e.detail)}
    ariaLabel={noBattery
      ? 'Battery state of charge in percent as measured by the inverter; no battery is simulated'
      : 'Battery state of charge in percent: measured by the inverter and simulated by the selected strategy'}
  />
</Card>
