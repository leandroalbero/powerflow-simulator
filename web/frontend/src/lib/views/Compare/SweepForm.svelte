<script lang="ts">
  import { tick } from 'svelte';
  import { capabilities, dataBounds, hasEv, presets, strategies as allStrategies } from '../../stores/app';
  import Segmented from '../../ui/Segmented.svelte';
  import DateRange from '../../ui/DateRange.svelte';
  import Select from '../../ui/Select.svelte';
  import Switch from '../../ui/Switch.svelte';
  import Chip from '../../ui/Chip.svelte';
  import Button from '../../ui/Button.svelte';
  import Icon from '../../ui/Icon.svelte';
  import { dayCount, fmtDate, fmtRange, num } from '../../ui/format';
  import { PERIOD_OPTIONS, type PeriodPreset } from '../Simulate/state';
  import CapacityChips from './CapacityChips.svelte';
  import StrategyChips from './StrategyChips.svelte';
  import { estimateSeconds, fmtDuration } from './derive';
  import {
    activeRange,
    cancelCurrentSweep,
    capacities,
    customRange,
    excludeEv,
    hardware,
    label,
    period,
    strategies,
    sweep,
    sweepBusy,
    tariffs,
    toggleIn,
    MAX_CASES,
    MAX_DAYS,
    WARN_CASES,
  } from './state';

  export let onRun: () => void;

  /** Names used when the presets endpoint can't be reached, so the form still reads well. */
  const FALLBACK_TARIFFS = [
    { id: '2_0td', name: '2.0TD', description: 'Fixed time-of-use rates' },
    { id: 'pvpc', name: 'PVPC', description: 'Hourly regulated price' },
  ];

  $: busy = $sweepBusy;
  $: range = $activeRange;
  $: days = range ? dayCount(range.start, range.end) : 0;
  $: tariffList = $presets?.tariffs ?? FALLBACK_TARIFFS;
  $: hwList = $presets?.hardware ?? [];
  $: hw = hwList.find((h) => h.id === $hardware) ?? null;
  $: usableIds = new Set($allStrategies.filter((s) => s.available !== false).map((s) => s.id));
  $: chosenStrategies = $strategies.filter((s) => usableIds.has(s));
  $: chosenTariffs = $tariffs.filter((t) => tariffList.some((x) => x.id === t));
  $: runs = chosenTariffs.length * $capacities.length * chosenStrategies.length;
  $: total = runs + chosenTariffs.length;
  $: eta = range ? estimateSeconds(days, chosenTariffs.length, $capacities.length, chosenStrategies) : 0;
  $: sweepAvailable = $capabilities.sweep !== 'unavailable';
  $: blocker = !sweepAvailable
    ? 'Comparisons need a backend update'
    : !range
      ? 'Waiting for the data range'
      : chosenTariffs.length === 0
        ? 'Pick at least one tariff'
        : $capacities.length === 0
          ? 'Add at least one battery size'
          : chosenStrategies.length === 0
            ? 'Pick at least one strategy'
            : days > MAX_DAYS
              ? `Pick at most ${MAX_DAYS} days`
              : total > MAX_CASES
                ? `${total} cases is over the limit of ${MAX_CASES}; remove some sizes or strategies`
                : null;
  $: canRun = !busy && !blocker;

  $: hwOptions = hwList.length
    ? hwList.map((h) => ({ value: h.id, label: h.name }))
    : [{ value: $hardware, label: $capabilities.presets === 'unavailable' ? 'Current' : 'Loading…' }];

  // A preset describes the hardware of one period; say so when the comparison reaches outside it.
  $: hwNote =
    hw && range
      ? hw.valid_from && range.start < hw.valid_from
        ? `${hw.name} applies from ${fmtDate(hw.valid_from)}; the whole period is simulated with it.`
        : hw.valid_until && range.end > hw.valid_until
          ? `${hw.name} applies until ${fmtDate(hw.valid_until)}; the whole period is simulated with it.`
          : null
      : null;
  // The backend scales the preset's rates by size / 15.36 kWh, whatever the preset's own capacity.
  $: hwPower = hw ? `${num(hw.battery.max_charge_rate, 1)} / ${num(hw.battery.max_discharge_rate, 1)} kW per 15.36 kWh` : null;

  const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;

  function onPeriod(e: CustomEvent<string>) {
    const p = e.detail as PeriodPreset;
    if (p === 'custom' && !$customRange && range) customRange.set({ ...range });
  }

  function runClick() {
    if (canRun) onRun();
  }

  // Cancel disappears when the comparison stops; hand keyboard focus to the Run button instead of <body>.
  let runBtn: Button;
  let cancelPressed = false;
  function cancelClick() {
    cancelPressed = true;
    void cancelCurrentSweep();
  }
  $: if (!busy && cancelPressed) {
    cancelPressed = false;
    void tick().then(() => {
      if (!document.activeElement || document.activeElement === document.body) runBtn?.focus();
    });
  }

  // Persistent live region: one inserted with its text already in it is often not announced.
  let announcement = '';
  let wasBusy = false;
  $: if (busy !== wasBusy) {
    wasBusy = busy;
    if (busy) announcement = 'Comparison started';
    else if ($sweep) {
      const st = $sweep.status;
      announcement =
        st === 'completed' ? 'Comparison finished. Results are below.'
        : st === 'cancelled' ? 'Comparison cancelled'
        : st === 'failed' ? `Comparison failed${$sweep.error ? `: ${$sweep.error}` : ''}`
        : '';
    }
  }

  $: progress = $sweep && $sweep.total ? ($sweep.done / $sweep.total) * 100 : 0;
  $: remaining = (() => {
    const s = $sweep;
    if (!s || s.status !== 'running') return null;
    const seen = s.done - s.doneAtAttach;
    if (seen < 2) return null;
    const elapsed = (Date.now() - s.attachedAt) / 1000;
    return (elapsed * (s.total - s.done)) / seen;
  })();
</script>

<section class="form" aria-label="Comparison setup">
  <div class="row top">
    <div class="field">
      <span class="label">Period</span>
      <div class="period-row">
        <Segmented label="Period" bind:value={$period} options={PERIOD_OPTIONS} disabled={busy} on:change={onPeriod} />
        {#if range && $period !== 'custom'}
          <span class="range-text num">{fmtRange(range.start, range.end)} <span class="muted">· {plural(days, 'day', 'days')}</span></span>
        {/if}
      </div>
      {#if $period === 'custom' && range && $dataBounds}
        <div class="custom">
          <DateRange
            start={range.start}
            end={range.end}
            min={$dataBounds.min}
            max={$dataBounds.max}
            disabled={busy}
            on:change={(e) => customRange.set(e.detail)}
          />
          <span class="hint num">{plural(days, 'day', 'days')}, both dates included</span>
        </div>
      {/if}
    </div>
    <div class="field hw">
      <Select label="Hardware" bind:value={$hardware} options={hwOptions} disabled={busy || !hwList.length} />
      <p class="hint">
        {#if hwNote}{hwNote}{:else}Charge and discharge power scale with size{hwPower ? ` (${hwPower})` : ''}.{/if}
      </p>
    </div>
  </div>

  <div class="row grid2">
    <div class="field">
      <span class="label" id="cmp-tariffs">Tariffs</span>
      <div class="chips" role="group" aria-labelledby="cmp-tariffs">
        {#each tariffList as t (t.id)}
          <Chip selected={$tariffs.includes(t.id)} disabled={busy} title={t.description} on:click={() => toggleIn(tariffs, t.id)}>
            {t.name}
          </Chip>
        {/each}
      </div>
    </div>
    <div class="field">
      <span class="label">Battery sizes <span class="muted">· no battery is always included</span></span>
      <CapacityChips disabled={busy} current={hw?.battery.capacity ?? null} />
    </div>
  </div>

  <div class="row">
    <span class="label">Strategies</span>
    <StrategyChips disabled={busy} />
  </div>

  <div class="row extras">
    <div class="label-field">
      <label for="cmp-label" class="label">Label <span class="muted">· optional</span></label>
      <input id="cmp-label" type="text" bind:value={$label} maxlength="120" placeholder="e.g. Should I double the battery?" disabled={busy} />
    </div>
    {#if $hasEv}
      <div class="ev">
        <Switch bind:checked={$excludeEv} label="Exclude EV charging" description="Leave the car's charging out of the house load" disabled={busy} />
      </div>
    {/if}
    <p class="note">
      <Icon name="info" size={14} />
      <span>Uses the tariffs and hardware picked here with the server’s default strategy settings. Changes in Settings don’t apply here.</span>
    </p>
  </div>

  <div class="actions">
    <div class="count">
      {#if blocker && !busy}
        <span class="blocker" role="status">{blocker}</span>
      {:else if busy && $sweep}
        <span class="num status">
          {$sweep.status === 'starting' ? 'Starting…' : `${$sweep.done} of ${$sweep.total} cases`}
          {#if remaining != null}<span class="muted"> · {fmtDuration(remaining)} left</span>{/if}
        </span>
      {:else}
        <span class="num formula">
          {plural(chosenTariffs.length, 'tariff', 'tariffs')} × {plural($capacities.length, 'size', 'sizes')} × {plural(chosenStrategies.length, 'strategy', 'strategies')}
          = <strong>{plural(runs, 'run', 'runs')}</strong> + {plural(chosenTariffs.length, 'baseline', 'baselines')}
        </span>
        <span class="eta num" class:warn={total > WARN_CASES}>
          {#if total > WARN_CASES}<Icon name="alert" size={14} />{/if}
          {fmtDuration(eta)}{total > WARN_CASES ? ` · ${total} cases is a lot; fewer sizes or strategies run much faster` : ', rough estimate'}
        </span>
      {/if}
    </div>
    <span class="sr-only" role="status" aria-live="polite">{announcement}</span>
    {#if busy && $capabilities.cancel !== 'unavailable'}
      <Button variant="secondary" icon="stop" loading={$sweep?.cancelling} disabled={$sweep?.status !== 'running'} keepFocus on:click={cancelClick}>
        Cancel
      </Button>
    {/if}
    <Button bind:this={runBtn} variant="primary" size="lg" icon="play" loading={busy} disabled={!canRun} keepFocus on:click={runClick}>
      {busy ? 'Comparing' : 'Run comparison'}
    </Button>
  </div>
  {#if busy}
    <div class="progress-clip" aria-hidden="true"><div class="progress"><span style="width: {progress}%" /></div></div>
  {/if}
</section>

<style>
  .form {
    position: relative;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    box-shadow: var(--shadow-1);
  }
  .row {
    padding: var(--s-5) var(--s-5) 0;
  }
  .top {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 300px);
    gap: var(--s-5);
  }
  .grid2 {
    display: grid;
    grid-template-columns: minmax(0, 300px) minmax(0, 1fr);
    gap: var(--s-5);
  }
  .field {
    min-width: 0;
  }
  .hw {
    display: grid;
    gap: 6px;
    align-content: start;
  }
  .label {
    display: block;
    font-size: var(--fs-xs);
    font-weight: 600;
    color: var(--text-2);
    margin-bottom: 6px;
  }
  .label .muted {
    font-weight: 500;
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
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-2);
  }
  .extras {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    gap: var(--s-4) var(--s-6);
    padding-bottom: var(--s-5);
  }
  .label-field {
    flex: 0 1 360px;
    min-width: 0;
  }
  .note {
    display: inline-flex;
    align-items: flex-start;
    gap: 6px;
    margin-left: auto;
    max-width: 46ch;
    font-size: var(--fs-xs);
    color: var(--text-3);
  }
  .note :global(.icon) {
    flex: none;
    margin-top: 2px;
  }
  .label-field input {
    width: 100%;
    height: var(--control-h);
    padding: 0 var(--s-3);
    border: 1px solid var(--border-strong);
    border-radius: var(--r-md);
    background: var(--surface);
    font-size: var(--fs-md);
  }
  .label-field input:hover:not(:disabled) {
    border-color: var(--text-3);
  }
  .label-field input::placeholder {
    color: var(--text-3);
  }
  .ev {
    flex: 0 1 340px;
    padding-bottom: 6px;
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
  .count {
    flex: 1;
    min-width: 0;
    display: grid;
    gap: 2px;
    font-size: var(--fs-sm);
  }
  .formula {
    color: var(--text-2);
  }
  .formula strong {
    color: var(--text);
    font-weight: 650;
  }
  .eta {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: var(--fs-xs);
    color: var(--text-3);
  }
  .eta.warn {
    color: var(--warning);
    font-weight: 550;
  }
  .blocker {
    color: var(--text-3);
  }
  .status {
    color: var(--text-2);
    font-weight: 550;
  }
  .progress-clip {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: var(--r-lg);
    border-radius: 0 0 calc(var(--r-lg) - 1px) calc(var(--r-lg) - 1px);
    overflow: hidden;
    pointer-events: none;
  }
  .progress {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 3px;
    background: var(--accent-soft);
  }
  .progress span {
    display: block;
    height: 100%;
    background: var(--accent);
    transition: width 300ms var(--ease);
  }
  @media (max-width: 960px) {
    .top,
    .grid2 {
      grid-template-columns: 1fr;
      gap: var(--s-4);
    }
  }
  @media (max-width: 720px) {
    .row {
      padding: var(--s-4) var(--s-4) 0;
    }
    .extras {
      padding-bottom: var(--s-4);
    }
    .label-field,
    .ev,
    .note {
      flex: 1 1 100%;
      max-width: none;
    }
    .label-field input {
      height: 42px;
    }
    .actions {
      padding: var(--s-3) var(--s-4);
    }
    .actions :global(.btn.primary) {
      flex: 1 1 100%;
    }
    .count {
      flex: 1 1 100%;
    }
  }
</style>
