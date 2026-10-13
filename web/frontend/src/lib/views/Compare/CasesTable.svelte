<script lang="ts">
  import Table from '../../ui/Table.svelte';
  import StrategyDot from '../../ui/StrategyDot.svelte';
  import Badge from '../../ui/Badge.svelte';
  import Button from '../../ui/Button.svelte';
  import Icon from '../../ui/Icon.svelte';
  import { DASH, eur, eurSigned, num, numTrim, pct } from '../../ui/format';
  import type { SweepCase } from '../../types';
  import { cheapest, isBaseline, rawCost, shownCost, type SweepModel } from './derive';
  import { stratNames, tariffNames } from './names';

  export let m: SweepModel;
  export let cases: SweepCase[];
  export let settled = false;

  type Key = 'tariff' | 'cap' | 'strategy' | 'cost' | 'raw' | 'savings' | 'year' | 'cycles' | 'self';
  const LIMIT = 12;

  $: COLS = [
    { key: 'tariff', label: 'Tariff' },
    { key: 'cap', label: 'Size', unit: 'kWh' },
    { key: 'strategy', label: 'Strategy' },
    { key: 'cost', label: m.perYear ? 'Net cost / yr' : 'Net cost', title: 'Energy cost minus the value of the energy left in the battery. Used for ranking.' },
    { key: 'raw', label: m.perYear ? 'Raw cost / yr' : 'Raw cost', title: 'Imports minus export revenue, ignoring the battery level at the end' },
    { key: 'savings', label: 'Savings', title: 'Over the period, against the same tariff with no battery' },
    ...(m.perYear ? [{ key: 'year', label: 'Per year', title: 'Savings scaled to a year using the hours with data' }] : []),
    { key: 'cycles', label: 'Cycles', title: 'Equivalent full cycles over the period' },
    { key: 'self', label: 'Solar at home', title: 'Share of solar production used in the house' },
  ] as { key: Key; label: string; unit?: string; title?: string }[];

  let sortKey: Key = 'cost';
  let asc = true;
  let showAll = false;

  $: tn = $tariffNames;

  function value(c: SweepCase, k: Key): number | string | null {
    switch (k) {
      case 'tariff': return tn(c.tariff);
      case 'cap': return c.capacity_kwh;
      case 'strategy': return $stratNames(c.strategy_id);
      case 'cost': return c.adjusted_cost;
      case 'raw': return c.total_cost;
      case 'savings': return isBaseline(c) ? null : c.savings_vs_baseline;
      case 'year': return isBaseline(c) ? null : c.annualised_savings;
      case 'cycles': return isBaseline(c) ? null : c.equivalent_full_cycles;
      case 'self': return c.self_consumption_rate;
    }
  }

  function sortBy(k: Key) {
    if (sortKey === k) asc = !asc;
    else {
      sortKey = k;
      asc = k === 'tariff' || k === 'cap' || k === 'strategy' || k === 'cost' || k === 'raw';
    }
  }

  $: sorted = [...cases].sort((a, b) => {
    const va = value(a, sortKey);
    const vb = value(b, sortKey);
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    const c = typeof va === 'string' ? va.localeCompare(String(vb)) : va - (vb as number);
    return asc ? c : -c;
  });
  $: shown = showAll ? sorted : sorted.slice(0, LIMIT);
  $: winner = cheapest(m);
  $: span = COLS.length - 3;
  $: dec = m.perYear ? 0 : 2;
</script>

<Table label="All cases table, scrollable">
  <table>
    <thead>
      <tr>
        {#each COLS as c (c.key)}
          <th aria-sort={sortKey === c.key ? (asc ? 'ascending' : 'descending') : 'none'} scope="col" title={c.title} class:left={c.key === 'tariff' || c.key === 'strategy'}>
            <button type="button" class="sort" class:active={sortKey === c.key} on:click={() => sortBy(c.key)}>
              {c.label}{#if c.unit}<span class="unit">{c.unit}</span>{/if}
              <span class="arrow" aria-hidden="true">
                {#if sortKey === c.key}<Icon name={asc ? 'arrowUp' : 'arrowDown'} size={12} stroke={2.2} />{/if}
              </span>
            </button>
          </th>
        {/each}
      </tr>
    </thead>
    <tbody>
      {#each shown as c (`${c.tariff}|${c.capacity_kwh}|${c.strategy_id}`)}
        {@const base = isBaseline(c)}
        <tr class:baseline={base}>
          <td class="left strong-name">{tn(c.tariff)}</td>
          <td>{base ? 'None' : numTrim(c.capacity_kwh)}</td>
          <td class="left">
            <span class="name">
              <StrategyDot id={c.strategy_id} />
              {$stratNames(c.strategy_id)}
              {#if base}<Badge>Baseline</Badge>{/if}
              {#if c === winner}<Badge tone="positive">Cheapest</Badge>{/if}
            </span>
          </td>
          {#if c.status === 'completed'}
            <td class="strong">{eur(shownCost(m, c), dec)}</td>
            <td>{eur(rawCost(m, c), dec)}</td>
            <td class:pos={!base && (c.savings_vs_baseline ?? 0) > 0.005} class:neg={!base && (c.savings_vs_baseline ?? 0) < -0.005}>
              {base ? DASH : eurSigned(c.savings_vs_baseline)}
            </td>
            {#if m.perYear}
              <td class:pos={!base && (c.annualised_savings ?? 0) > 0.5} class:neg={!base && (c.annualised_savings ?? 0) < -0.5}>
                {base ? DASH : eurSigned(c.annualised_savings, 0)}
              </td>
            {/if}
            <td>{base ? DASH : num(c.equivalent_full_cycles, 1)}</td>
            <td>{pct(c.self_consumption_rate)}</td>
          {:else}
            <td colspan={span} class="msg">
              {#if c.status === 'failed'}
                <span class="err"><Icon name="alert" size={14} />{c.error ?? 'Failed'}</span>
              {:else if c.status === 'cancelled'}
                <span class="muted">Cancelled</span>
              {:else if settled}
                <span class="muted">Not run</span>
              {:else}
                <span class="muted">{c.status === 'running' ? 'Running…' : 'Queued'}</span>
              {/if}
            </td>
          {/if}
        </tr>
      {/each}
    </tbody>
  </table>
</Table>
{#if sorted.length > LIMIT}
  <div class="more">
    <Button variant="ghost" size="sm" on:click={() => (showAll = !showAll)}>
      {showAll ? 'Show fewer' : `Show all ${sorted.length} cases`}
    </Button>
  </div>
{/if}

<style>
  .sort {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    border: 0;
    background: transparent;
    padding: 0;
    color: inherit;
    font: inherit;
    font-weight: 600;
    white-space: nowrap;
  }
  .sort:hover,
  .sort.active {
    color: var(--text);
  }
  .unit {
    margin-left: 3px;
    font-weight: 500;
    color: var(--text-3);
  }
  .arrow {
    display: inline-block;
    width: 12px;
    height: 12px;
  }
  th.left,
  td.left {
    text-align: left;
  }
  td {
    color: var(--text-2);
  }
  .strong-name {
    font-weight: 600;
    color: var(--text);
  }
  .name {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: var(--text);
    font-weight: 550;
  }
  .strong {
    font-weight: 650;
    color: var(--text);
  }
  .pos {
    color: var(--positive);
    font-weight: 600;
  }
  .neg {
    color: var(--negative);
    font-weight: 600;
  }
  tr.baseline td {
    background: var(--surface-2);
  }
  tbody tr:hover td {
    background: var(--surface-2);
  }
  .msg {
    text-align: left !important;
  }
  .err {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--negative);
    white-space: normal;
  }
  .more {
    display: flex;
    justify-content: center;
    padding: var(--s-2);
    border-top: 1px solid var(--border);
  }
  @media (max-width: 720px), (pointer: coarse) {
    .sort {
      min-height: 40px;
    }
  }
</style>
