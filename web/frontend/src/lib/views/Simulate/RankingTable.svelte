<script lang="ts">
  import { strategyById } from '../../stores/app';
  import { MIN_HOURS_FOR_ANNUAL, verdict, type Ranked } from '../../stores/run';
  import Table from '../../ui/Table.svelte';
  import StrategyDot from '../../ui/StrategyDot.svelte';
  import Badge from '../../ui/Badge.svelte';
  import Icon from '../../ui/Icon.svelte';
  import { DASH, eur, eurSigned, num } from '../../ui/format';
  import { focusStrategy, pickFocus } from './state';

  export let rows: Ranked[];
  /** Calendar hours of the run, to decide on the "Per year" column before any result has valid_hours. */
  export let hours = 0;

  type Key = 'name' | 'cost' | 'raw' | 'savings' | 'year' | 'import' | 'export' | 'gridCharge' | 'overLimit' | 'cycles';
  const COLS: { key: Key; label: string; unit?: string; title?: string }[] = [
    { key: 'name', label: 'Strategy' },
    { key: 'cost', label: 'Net cost', title: 'Energy cost minus the value of the energy left in the battery. Used for ranking.' },
    { key: 'raw', label: 'Raw cost', title: 'Imports minus export revenue, ignoring the battery level at the end' },
    { key: 'savings', label: 'Savings', title: 'Net cost of no battery minus this strategy' },
    { key: 'year', label: 'Per year', title: 'Savings scaled to a year using the hours simulated' },
    { key: 'import', label: 'Import', unit: 'kWh' },
    { key: 'export', label: 'Export', unit: 'kWh' },
    { key: 'gridCharge', label: 'Grid-charged', unit: 'kWh', title: 'Energy charged into the battery from the grid' },
    {
      key: 'overLimit',
      label: 'Import above grid limit',
      unit: 'kWh',
      title: 'Informational, not a penalty: grid import above the connection limit, mostly EV charging at night. Not used for ranking.',
    },
    { key: 'cycles', label: 'Cycles', title: 'Equivalent full cycles' },
  ];

  let sortKey: Key = 'cost';
  let asc = true;

  function value(r: Ranked, k: Key): number | string | null {
    const m = r.metrics;
    switch (k) {
      case 'name': return $strategyById.get(r.id)?.name ?? r.id;
      case 'cost': return r.cost;
      case 'raw': return r.rawCost;
      case 'savings': return r.savings;
      case 'year': return r.savingsPerYear;
      case 'import': return m?.total_grid_imported ?? null;
      case 'export': return m?.total_solar_exported ?? null;
      case 'gridCharge': return m?.grid_charge_kwh ?? null;
      case 'overLimit': return m?.over_limit_kwh ?? null;
      case 'cycles': return m?.equivalent_full_cycles ?? null;
    }
  }

  function sortBy(k: Key) {
    if (sortKey === k) asc = !asc;
    else {
      sortKey = k;
      asc = k === 'name' || k === 'cost' || k === 'raw' || k === 'import' || k === 'overLimit';
    }
  }

  $: sorted = [...rows].sort((a, b) => {
    const va = value(a, sortKey);
    const vb = value(b, sortKey);
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    const c = typeof va === 'string' ? va.localeCompare(String(vb)) : va - (vb as number);
    return asc ? c : -c;
  });
  $: bestId = verdict(rows).winnerId;
  $: known = rows.filter((r) => r.validHours != null);
  $: showYear = known.length ? known.some((r) => (r.validHours ?? 0) >= MIN_HOURS_FOR_ANNUAL) : hours >= MIN_HOURS_FOR_ANNUAL;
  $: cols = showYear ? COLS : COLS.filter((c) => c.key !== 'year');
  $: span = cols.length - 1;
</script>

<Table label="Strategy ranking">
  <table>
    <thead>
      <tr>
        {#each cols as c (c.key)}
          <th aria-sort={sortKey === c.key ? (asc ? 'ascending' : 'descending') : 'none'} scope="col" title={c.title}>
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
      {#each sorted as r (r.id)}
        {@const info = $strategyById.get(r.id)}
        <tr class:baseline={r.isBaseline} class:focused={$focusStrategy === r.id && !r.isBaseline} class:failed={r.status === 'failed'}>
          <td class="name-cell">
            <span class="name">
              <StrategyDot id={r.id} />
              {#if r.status === 'completed' && !r.isBaseline}
                <button type="button" class="link" on:click={() => pickFocus(r.id)} title="Show in the power chart">
                  {info?.name ?? r.id}
                </button>
              {:else}
                <span>{info?.name ?? r.id}</span>
              {/if}
              {#if r.isBaseline}<Badge>Baseline</Badge>{/if}
              {#if r.id === bestId}<Badge tone="positive">Best</Badge>{/if}
              {#if info?.group === 'experimental' && !/experimental/i.test(info.name)}<Badge tone="warning">Experimental</Badge>{/if}
            </span>
          </td>
          {#if r.status === 'failed' || r.status === 'cancelled'}
            <td colspan={span} class="msg-cell">
              <span class="err" class:cancelled={r.status === 'cancelled'}>
                {#if r.status === 'failed'}<Icon name="alert" size={14} />{/if}
                {r.status === 'cancelled' ? 'Cancelled' : r.error ?? 'Failed'}
              </span>
            </td>
          {:else if r.status !== 'completed'}
            <td colspan={span} class="msg-cell">
              <span class="pending">
                <span class="num state">{r.status === 'running' ? `${Math.round(r.progress)}%` : 'Queued'}</span>
                <span class="bar"><span style="width: {r.status === 'running' ? Math.max(3, r.progress) : 0}%" /></span>
              </span>
            </td>
          {:else}
            <td class="strong">{eur(r.cost)}</td>
            <td>{eur(r.rawCost)}</td>
            <td class:pos={r.savings != null && r.savings > 0.005} class:neg={r.savings != null && r.savings < -0.005}>
              {r.isBaseline ? DASH : eurSigned(r.savings)}
            </td>
            {#if showYear}
              <td class:pos={r.savingsPerYear != null && r.savingsPerYear > 0.5} class:neg={r.savingsPerYear != null && r.savingsPerYear < -0.5}>
                {r.isBaseline || r.savingsPerYear == null ? DASH : eurSigned(r.savingsPerYear, 0)}
              </td>
            {/if}
            <td>{num(r.metrics?.total_grid_imported, 1)}</td>
            <td>{num(r.metrics?.total_solar_exported, 1)}</td>
            <td>{num(r.metrics?.grid_charge_kwh, 1)}</td>
            <td>{num(r.metrics?.over_limit_kwh, 1)}</td>
            <td>{r.isBaseline ? DASH : num(r.metrics?.equivalent_full_cycles, 1)}</td>
          {/if}
        </tr>
      {/each}
    </tbody>
  </table>
</Table>


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
  th:first-child .sort {
    justify-content: flex-start;
  }
  .name {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
    color: var(--text);
  }
  .link {
    border: 0;
    background: transparent;
    padding: 0;
    font: inherit;
    color: inherit;
    text-align: left;
    text-decoration: underline;
    text-decoration-color: transparent;
    text-underline-offset: 3px;
  }
  .link:hover {
    text-decoration-color: var(--text-3);
  }
  .strong {
    font-weight: 650;
    color: var(--text);
  }
  td {
    color: var(--text-2);
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
  tr.baseline .name {
    color: var(--text-2);
    font-weight: 550;
  }
  tr.focused td:first-child {
    box-shadow: inset 3px 0 0 var(--accent);
  }
  tbody tr:hover td {
    background: var(--surface-2);
  }
  .msg-cell {
    text-align: left !important;
  }
  .err {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--negative);
    white-space: normal;
  }
  .err.cancelled {
    color: var(--text-3);
  }
  .pending {
    display: inline-flex;
    align-items: center;
    gap: var(--s-3);
    color: var(--text-3);
  }
  .state {
    min-width: 4.5ch;
  }
  .bar {
    width: 160px;
    height: 4px;
    border-radius: 2px;
    background: var(--surface-3);
    overflow: hidden;
  }
  .bar span {
    display: block;
    height: 100%;
    background: var(--accent);
    transition: width 300ms var(--ease);
  }
  @media (max-width: 720px) {
    .bar {
      width: 64px;
    }
  }
  @media (max-width: 720px), (pointer: coarse) {
    .sort {
      min-height: 40px;
    }
    .link {
      min-height: 40px;
      display: inline-flex;
      align-items: center;
    }
  }
</style>
