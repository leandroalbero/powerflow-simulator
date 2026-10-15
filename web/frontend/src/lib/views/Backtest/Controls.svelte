<script lang="ts">
  import { onDestroy } from 'svelte';
  import { hardwareSummary } from '../../stores/draft';
  import { capabilities, dataBounds, dataInfo, partialLastDay, presets, strategies } from '../../stores/app';
  import Button from '../../ui/Button.svelte';
  import DateRange from '../../ui/DateRange.svelte';
  import Icon from '../../ui/Icon.svelte';
  import Segmented from '../../ui/Segmented.svelte';
  import Select from '../../ui/Select.svelte';
  import { dayCount, fmtDate, fmtRange, fmtTime, num } from '../../ui/format';
  import {
    activeRange,
    busy,
    comparing,
    customRange,
    hardware,
    MAX_DAYS,
    period,
    PERIOD_OPTIONS,
    runningId,
    session,
    shortenToYear,
    strategyId,
    tooLong,
    UPGRADE_DATE,
    type BtPeriod,
  } from './state';

  export let onRun: () => void;
  export let onCompare: () => void;
  export let coreIds: string[];

  const GROUP_ORDER = { core: 0, advanced: 1, experimental: 2 } as const;

  $: range = $activeRange;
  $: days = range ? dayCount(range.start, range.end) : 0;
  $: long = tooLong(range);
  $: partial = partialLastDay($dataBounds);
  $: partialUntil = partial ? fmtTime(new Date(partial.until)) : '';
  $: capMissing = $capabilities.backtest === 'unavailable';
  $: noMeasured = $dataInfo?.has_measured === false;

  $: strategyOptions = [...$strategies]
    .sort((a, b) => GROUP_ORDER[a.group] - GROUP_ORDER[b.group])
    .map((s) => ({
      value: s.id,
      label: s.id === 'no_battery' ? `${s.name} (reference)` : s.available === false ? `${s.name} (unavailable)` : s.name,
      disabled: s.available === false,
    }));
  $: if ($strategies.length && !$strategies.some((s) => s.id === $strategyId && s.available !== false)) strategyId.set('self_consume');
  $: strategyDesc = $strategies.find((s) => s.id === $strategyId)?.description ?? null;

  $: hwOptions = [
    { value: 'auto', label: 'Auto (by date)' },
    ...($presets?.hardware ?? []).map((h) => ({
      value: h.id,
      label: `${h.name} · ${num(h.battery.max_charge_rate, 1)}/${num(h.battery.max_discharge_rate, 1)} kW`,
    })),
  ];
  $: if ($presets && !hwOptions.some((o) => o.value === $hardware)) hardware.set('auto');
  $: hwHint =
    $hardware === 'auto'
      ? range && range.start < UPGRADE_DATE && range.end >= UPGRADE_DATE
        ? 'This range spans the upgrade: 2.2/2.1 kW before 7 Apr 2026, 4.5/4.6 kW from then on.'
        : range && range.end < UPGRADE_DATE
          ? 'Before the upgrade: 2.2 kW charge, 2.1 kW discharge.'
          : 'After the upgrade: 4.5 kW charge, 4.6 kW discharge.'
      : (() => {
          const h = $presets?.hardware.find((x) => x.id === $hardware);
          return h ? hardwareSummary(h) : null;
        })();

  $: blocker = capMissing
    ? 'Backtesting needs a backend update'
    : noMeasured
      ? 'The loaded data has no measured grid and battery readings'
      : !range
        ? 'Waiting for the data range'
        : long
          ? `Backtests cover at most one year (${MAX_DAYS} days)`
          : null;
  $: canRun = !$busy && !blocker;

  function onPeriod(e: CustomEvent<string>) {
    const p = e.detail as BtPeriod;
    if (p === 'custom' && !$customRange && range) customRange.set({ ...range });
  }

  function shorten() {
    if (!range) return;
    customRange.set(shortenToYear(range));
    period.set('custom');
  }

  // Elapsed time for the synchronous request; a year of MPC can take minutes.
  let now = Date.now();
  let started = 0;
  let timer: ReturnType<typeof setInterval> | null = null;
  $: if ($busy && !timer) {
    started = Date.now();
    now = started;
    timer = setInterval(() => (now = Date.now()), 1000);
  } else if (!$busy && timer) {
    clearInterval(timer);
    timer = null;
  }
  onDestroy(() => timer && clearInterval(timer));
  $: elapsed = Math.round((now - started) / 1000);

  let announcement = '';
  let wasBusy = false;
  $: if ($busy !== wasBusy) {
    wasBusy = $busy;
    announcement = $busy ? ($comparing ? 'Comparing strategies' : 'Backtest started') : 'Backtest finished. Results are below.';
  }
  $: runningName = $runningId ? $strategies.find((s) => s.id === $runningId)?.name ?? $runningId : null;
  $: cmpCount = coreIds.length;
  $: cached = $session && range && $session.key === `${range.start}|${range.end}|${$hardware}` ? coreIds.filter((id) => $session?.entries[id]?.status === 'done').length : 0;
</script>

<section class="controls" aria-label="Backtest setup">
  <div class="top">
    <div class="period">
      <span class="label" id="bt-period-label">Period</span>
      <div class="period-row">
        <Segmented label="Period" bind:value={$period} options={PERIOD_OPTIONS} disabled={$busy} on:change={onPeriod} />
        {#if range && $period !== 'custom'}
          <span class="range-text num">{fmtRange(range.start, range.end)} <span class="muted">· {days} {days === 1 ? 'day' : 'days'}</span></span>
        {/if}
      </div>
      {#if $period === 'custom' && range && $dataBounds}
        <div class="custom">
          <DateRange
            start={range.start}
            end={range.end}
            min={$dataBounds.min}
            max={$dataBounds.max}
            disabled={$busy}
            on:change={(e) => customRange.set(e.detail)}
          />
          <span class="hint num">{days} {days === 1 ? 'day' : 'days'}, both dates included</span>
        </div>
      {/if}
      {#if partial && range}
        {#if range.end === partial.day}
          <p class="partial warn num"><Icon name="alert" size={14} />{fmtDate(partial.day)} has data only to {partialUntil} so far, yet it counts as a whole day in the totals and daily averages.</p>
        {:else if $period !== 'custom'}
          <p class="partial num">{fmtDate(partial.day)} is left out: it has data only to {partialUntil} so far.</p>
        {/if}
      {/if}
      {#if long && range}
        <div class="too-long" role="alert">
          <Icon name="alert" size={16} />
          <span>
            Backtests cover at most one year. This range is {days} days ({fmtDate(range.start)} to {fmtDate(range.end)}).
          </span>
          <Button size="sm" variant="secondary" on:click={shorten}>Use the last 12 months of it</Button>
        </div>
      {/if}
    </div>

    <div class="selects">
      <div class="col">
        <Select label="Strategy to compare with" bind:value={$strategyId} options={strategyOptions} disabled={$busy} />
        {#if strategyDesc}<p class="hint">{strategyDesc}</p>{/if}
      </div>
      <div class="col">
        <Select label="Battery hardware" bind:value={$hardware} options={hwOptions} disabled={$busy} />
        {#if hwHint}<p class="hint">{hwHint}</p>{/if}
      </div>
    </div>
  </div>

  <div class="actions">
    <p class="note">
      <Icon name="info" size={14} />
      <span>Uses the server’s default tariff and strategy settings. Changes in Settings don’t apply here.</span>
    </p>
    <span class="spacer" />
    {#if blocker}<span class="blocker" role="status">{blocker}</span>{/if}
    {#if $busy}
      <span class="status num" aria-hidden="true">{runningName ?? 'Starting'} · {elapsed} s</span>
    {:else if days > 31 && !blocker}
      <span class="blocker">A long range can take a few minutes per strategy</span>
    {/if}
    <span class="sr-only" role="status" aria-live="polite">{announcement}</span>
    <Button
      variant="secondary"
      size="lg"
      disabled={!canRun || cmpCount < 2}
      loading={$busy && $comparing}
      keepFocus
      title="Backtest each core strategy in turn ({cmpCount}){cached ? `, reusing ${cached} already done` : ''}"
      on:click={() => canRun && onCompare()}
    >
      Compare all core strategies
    </Button>
    <Button variant="primary" size="lg" icon="play" loading={$busy && !$comparing} disabled={!canRun} keepFocus on:click={() => canRun && onRun()}>
      {$busy && !$comparing ? 'Backtesting' : 'Run backtest'}
    </Button>
  </div>
</section>

<style>
  .controls {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    box-shadow: var(--shadow-1);
  }
  .top {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 460px);
    gap: var(--s-5);
    padding: var(--s-5);
  }
  .label {
    display: block;
    font-size: var(--fs-xs);
    font-weight: 600;
    color: var(--text-2);
    margin-bottom: 6px;
  }
  .period-row {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--s-2) var(--s-4);
  }
  .range-text {
    font-size: var(--fs-sm);
    font-weight: 550;
  }
  .custom {
    margin-top: var(--s-3);
    display: flex;
    align-items: flex-end;
    gap: var(--s-2) var(--s-4);
    flex-wrap: wrap;
    max-width: 600px;
  }
  .custom > :global(.range) {
    flex: 1 1 320px;
  }
  .custom .hint {
    padding-bottom: 10px;
  }
  .too-long {
    margin-top: var(--s-3);
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--s-2) var(--s-3);
    padding: var(--s-2) var(--s-3);
    border-radius: var(--r-md);
    background: var(--warning-soft);
    border: 1px solid var(--warning-border);
    color: var(--warning);
    font-size: var(--fs-sm);
    max-width: 640px;
  }
  .too-long span {
    flex: 1 1 260px;
  }
  .selects {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--s-2) var(--s-3);
    align-content: start;
  }
  .col {
    display: grid;
    gap: 6px;
    align-content: start;
    min-width: 0;
  }
  .actions {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: var(--s-3);
    padding: var(--s-3) var(--s-5);
    border-top: 1px solid var(--border);
    background: var(--surface-2);
    border-radius: 0 0 calc(var(--r-lg) - 1px) calc(var(--r-lg) - 1px);
  }
  .note {
    display: inline-flex;
    align-items: flex-start;
    gap: 6px;
    font-size: var(--fs-xs);
    color: var(--text-3);
    max-width: 52ch;
  }
  .note :global(.icon) {
    margin-top: 2px;
  }
  .spacer {
    flex: 1;
  }
  .partial {
    display: flex;
    align-items: flex-start;
    gap: 6px;
    margin-top: var(--s-2);
    font-size: var(--fs-xs);
    color: var(--text-3);
  }
  .partial.warn {
    color: var(--warning);
  }
  .partial :global(.icon) {
    flex: none;
    margin-top: 1px;
  }
  .blocker {
    font-size: var(--fs-sm);
    color: var(--text-3);
  }
  .status {
    font-size: var(--fs-sm);
    color: var(--text-2);
  }
  @media (max-width: 960px) {
    .top {
      grid-template-columns: 1fr;
      gap: var(--s-4);
    }
    /* The note takes its own line so the buttons stay right-aligned, as on the other tabs. */
    .note {
      flex: 1 1 100%;
      max-width: none;
    }
  }
  @media (max-width: 720px) {
    .top {
      padding: var(--s-4);
    }
    .actions {
      padding: var(--s-3) var(--s-4);
    }
    .selects {
      grid-template-columns: 1fr;
    }
    .actions :global(.btn) {
      flex: 1 1 100%;
    }
    .actions :global(.btn.primary) {
      order: 10;
    }
    .spacer {
      display: none;
    }
  }
</style>
