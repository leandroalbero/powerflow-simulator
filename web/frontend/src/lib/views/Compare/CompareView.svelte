<script lang="ts">
  import { onMount } from 'svelte';
  import { get } from 'svelte/store';
  import { sweepExportUrl } from '../../api/client';
  import { capabilities, dataBounds, hasEv, presets, strategies as allStrategies } from '../../stores/app';
  import Card from '../../ui/Card.svelte';
  import Button from '../../ui/Button.svelte';
  import { dayCount, fmtDateTime, fmtRange, num } from '../../ui/format';
  import { rangeFor } from '../Simulate/state';
  import SweepForm from './SweepForm.svelte';
  import EmptyState from './EmptyState.svelte';
  import Headline from './Headline.svelte';
  import Heatmap from './Heatmap.svelte';
  import SavingsCurve from './SavingsCurve.svelte';
  import CasesTable from './CasesTable.svelte';
  import RecentSweeps from './RecentSweeps.svelte';
  import { buildModel, estimateSeconds, fmtDuration } from './derive';
  import {
    activeRange,
    applyQuickPreset,
    capacities,
    customRange,
    DEFAULT_CAPACITIES,
    DEFAULT_TARIFFS,
    excludeEv,
    hardware,
    label,
    openingSweep,
    runSweep,
    strategies,
    sweep,
    sweepBusy,
    syncSweepFromHash,
    tariffs,
  } from './state';

  onMount(() => {
    syncSweepFromHash();
    window.addEventListener('hashchange', syncSweepFromHash);
    return () => window.removeEventListener('hashchange', syncSweepFromHash);
  });

  function exportCsv() {
    if (!s?.id) return;
    const a = document.createElement('a');
    a.href = sweepExportUrl(s.id);
    a.download = `comparison-${s.id}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  function go() {
    const range = get(activeRange);
    if (!range) return;
    const usable = new Set(get(allStrategies).filter((s) => s.available !== false).map((s) => s.id));
    const known = get(presets)?.tariffs.map((t) => t.id);
    const l = get(label).trim();
    void runSweep({
      date_range: range,
      strategies: get(strategies).filter((s) => usable.has(s)),
      tariffs: known ? get(tariffs).filter((t) => known.includes(t)) : get(tariffs),
      capacities_kwh: [...get(capacities)],
      hardware: get(hardware),
      exclude_ev: get(hasEv) ? get(excludeEv) : false,
      label: l || undefined,
    });
  }

  function quickRun() {
    applyQuickPreset();
    go();
  }

  $: s = $sweep;
  $: m = s ? buildModel(s.cases, s.range) : null;
  $: days = s ? dayCount(s.range.start, s.range.end) : 0;
  // As on Simulate: per-year figures divide by the hours with data, which gaps and a partial last
  // day make fewer than the calendar hours.
  $: dataDays = m?.hours != null && days * 24 - m.hours > 1.5 ? m.hours / 24 : null;
  $: settled = !!s && s.status !== 'running' && s.status !== 'starting';
  $: anyDone = !!m && m.completed.some((c) => c.strategy_id !== 'no_battery');
  $: finished = s ? s.cases.filter((c) => c.status === 'completed').length : 0;
  $: provisional = !!s && (s.status === 'running' || s.status === 'cancelled') && anyDone && finished < s.cases.length;
  $: failedStart = s?.status === 'failed' && !s.id;
  $: quickDays = $dataBounds ? (() => {
    const r = rangeFor('last12m', $dataBounds, $customRange);
    return dayCount(r.start, r.end);
  })() : 365;
  $: quickEstimate = fmtDuration(estimateSeconds(quickDays, DEFAULT_TARIFFS.length, DEFAULT_CAPACITIES.length, $strategies));
  $: summary = m
    ? `${m.tariffs.length} ${m.tariffs.length === 1 ? 'tariff' : 'tariffs'} × ${m.capacities.length} ${m.capacities.length === 1 ? 'size' : 'sizes'} × ${m.strategies.length} ${m.strategies.length === 1 ? 'strategy' : 'strategies'}`
    : '';
  $: canExport = !!s?.id && $capabilities.sweep !== 'unavailable' && anyDone;
</script>

<div class="view">
  <SweepForm onRun={go} />

  {#if $openingSweep && $openingSweep !== s?.id}
    <div class="opening" role="status">
      <span class="spinner" aria-hidden="true" />
      <span>Opening comparison <code>{$openingSweep}</code>…</span>
    </div>
  {:else if !s}
    <EmptyState onQuickRun={quickRun} disabled={$sweepBusy} estimate={quickEstimate} />
  {:else if failedStart}
    <div class="start-error" role="alert">
      <h2>The comparison didn't start</h2>
      <p>{s.error}</p>
    </div>
  {:else if m}
    <section class="results" aria-labelledby="cmp-results-title" aria-busy={$sweepBusy}>
      <header class="results-head">
        <div class="titles">
          <h1 id="cmp-results-title">Results{#if s.label}<span class="label-sep" aria-hidden="true">{' · '}</span><span class="run-label">{s.label}</span>{/if}</h1>
          <p class="meta num">
            {fmtRange(s.range.start, s.range.end)} · {days} {days === 1 ? 'day' : 'days'}{#if dataDays != null}{' '}<span
                title="Per-year figures use the hours with data. Hours with missing measurements are left out."
              >({num(dataDays, 1)} d with data)</span>{/if} · {summary}
            {#if s.createdAt} · ran {fmtDateTime(s.createdAt)}{/if}
            {#if s.status === 'cancelled'} · <span class="warn">cancelled, {finished} of {s.total} cases finished</span>{/if}
            {#if s.status === 'failed' && s.error} · <span class="err">failed: {s.error}</span>{/if}
          </p>
        </div>
        {#if canExport && s.id}
          <Button size="sm" icon="download" on:click={exportCsv}>Export CSV</Button>
        {/if}
      </header>

      {#if anyDone}
        <Headline
          {m}
          {days}
          {provisional}
          bestSuffix={s.status === 'cancelled' ? ' of those finished' : ' so far'}
          provisionalText={s.status === 'cancelled'
            ? `Partial: ${finished} of ${s.total} cases finished before the comparison was cancelled.`
            : 'Provisional: figures cover the finished cases only and can still change.'}
        />
      {:else if s.status === 'running' || s.status === 'starting'}
        <p class="waiting" role="status">
          Running the no-battery baselines first. The matrix fills in as cases finish.
          {#if m.hoursPending}Totals are for the whole period until the first battery case gives the hours with data for per-year figures.{/if}
        </p>
      {:else if m.hoursPending}
        <p class="waiting">
          Totals are for the whole period: per-year figures need the hours with data, which come with the first finished battery case.
        </p>
      {/if}

      {#if m.tariffs.length && m.capacities.length}
        <Heatmap {m} {settled} />
      {/if}

      {#if anyDone}
        <SavingsCurve {m} />
      {/if}

      {#if s.cases.length}
        <Card title="All cases" subtitle="Sorted by net cost. Baselines are the same tariff with no battery." pad={false}>
          <CasesTable {m} cases={s.cases} {settled} />
        </Card>
      {/if}
    </section>
  {/if}

  <RecentSweeps />
</div>

<style>
  .view {
    display: grid;
    gap: var(--s-5);
  }
  .results {
    display: grid;
    gap: var(--s-4);
    min-width: 0;
  }
  .results-head {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: var(--s-3);
    flex-wrap: wrap;
    margin-top: var(--s-2);
  }
  .titles {
    min-width: 0;
  }
  h1 {
    font-size: var(--fs-xl);
    font-weight: 700;
    letter-spacing: -0.02em;
    overflow-wrap: anywhere;
  }
  .label-sep {
    color: var(--text-3);
    font-weight: 500;
  }
  .run-label {
    font-weight: 600;
    color: var(--text-2);
  }
  .meta {
    margin-top: 2px;
    color: var(--text-3);
    font-size: var(--fs-sm);
  }
  .warn {
    color: var(--warning);
  }
  .err {
    color: var(--negative);
  }
  .waiting {
    color: var(--text-3);
    font-size: var(--fs-sm);
  }
  .start-error {
    padding: var(--s-5);
    border-radius: var(--r-lg);
    background: var(--negative-soft);
    color: var(--negative);
  }
  .start-error h2 {
    font-size: var(--fs-lg);
    margin-bottom: var(--s-1);
  }
  .opening {
    display: flex;
    align-items: center;
    gap: var(--s-3);
    padding: 10px var(--s-4);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    background: var(--surface);
    box-shadow: var(--shadow-1);
    color: var(--text-2);
    font-size: var(--fs-sm);
  }
  .opening code {
    font-family: var(--font-mono);
    font-size: 0.95em;
  }
  .spinner {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    border: 2px solid currentColor;
    border-right-color: transparent;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
