<script lang="ts">
  import { get } from 'svelte/store';
  import { strategies } from '../../stores/app';
  import { draft, draftValid } from '../../stores/draft';
  import { ranked, run, runBusy, startRun } from '../../stores/run';
  import { dayCount, fmtRange, fmtTime, num } from '../../ui/format';
  import Card from '../../ui/Card.svelte';
  import ControlsBar from './ControlsBar.svelte';
  import EmptyState from './EmptyState.svelte';
  import KpiRow from './KpiRow.svelte';
  import RankingTable from './RankingTable.svelte';
  import FlowCharts from './FlowCharts.svelte';
  import MonthlyChart from './MonthlyChart.svelte';
  import ExportMenu from './ExportMenu.svelte';
  import { activeRange, focusStrategy, period, resetFocus, selected } from './state';

  function go() {
    const range = get(activeRange);
    const cfg = get(draft);
    if (!range || !cfg) return;
    const usable = new Set(get(strategies).filter((s) => s.available !== false).map((s) => s.id));
    resetFocus();
    void startRun({ range, strategies: get(selected).filter((s) => usable.has(s)), config: structuredClone(cfg) });
  }

  function quickRun() {
    period.set('last7');
    go();
  }

  $: r = $run;
  $: days = r ? dayCount(r.range.start, r.range.end) : 0;
  $: hasMonthly = $ranked.some((x) => (x.metrics?.monthly?.length ?? 0) >= 2);
  $: anyDone = $ranked.some((x) => x.status === 'completed');
  $: failedStart = r?.status === 'failed' && !r.runId;
  // Rates (per year, per day) divide by the hours with data, which can be fewer than the calendar
  // hours; say so, or the figures can't be reproduced from what is on screen.
  $: validHours = $ranked.find((x) => x.isBaseline)?.validHours ?? $ranked.find((x) => x.validHours != null)?.validHours ?? null;
  $: dataDays = validHours != null && days * 24 - validHours > 1.5 ? validHours / 24 : null;
</script>

<div class="view">
  <ControlsBar onRun={go} />

  {#if !r}
    <EmptyState onQuickRun={quickRun} disabled={$runBusy || !$draftValid} />
  {:else if failedStart}
    <div class="start-error" role="alert">
      <h2>The simulation didn't start</h2>
      <p>{r.error}</p>
    </div>
  {:else}
    <section class="results" aria-labelledby="results-title" aria-busy={$runBusy}>
      <header class="results-head">
        <div>
          <h1 id="results-title">Results</h1>
          <p class="meta num">
            {fmtRange(r.range.start, r.range.end)} · {days} {days === 1 ? 'day' : 'days'}{#if dataDays != null}{' '}<span
                title="Per-year and per-day figures use the hours with data. Hours with missing measurements are left out."
              >({num(dataDays, 1)} d with data)</span>{/if} · {r.strategyIds.length - 1}
            {r.strategyIds.length === 2 ? 'strategy' : 'strategies'} vs no battery{r.finishedAt ? ` · ran at ${fmtTime(new Date(r.startedAt))}` : ''}
          </p>
        </div>
        <ExportMenu runId={r.finishedAt ? r.runId : null} focusId={$focusStrategy} {hasMonthly} />
      </header>

      <KpiRow rows={$ranked} {days} provisional={$runBusy && anyDone} />

      <Card title="Ranking" subtitle="Ranked by net cost: what you pay, minus the value of energy left in the battery" pad={false}>
        <RankingTable rows={$ranked} hours={days * 24} />
      </Card>

      {#if r.runId && anyDone}
        <FlowCharts runId={r.runId} rows={$ranked} capacityKwh={r.config.battery.capacity} />
      {/if}

      <MonthlyChart rows={$ranked} />
    </section>
  {/if}
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
  h1 {
    font-size: var(--fs-xl);
    font-weight: 700;
    letter-spacing: -0.02em;
  }
  .meta {
    margin-top: 2px;
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
</style>
