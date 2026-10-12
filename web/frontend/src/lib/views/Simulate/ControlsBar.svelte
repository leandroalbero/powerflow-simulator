<script lang="ts">
  import { capabilities, dataBounds, presets } from '../../stores/app';
  import { applyHardwarePreset, applyTariffPreset, draft, draftModified, draftValid, hardwarePresetId, tariffPresetId } from '../../stores/draft';
  import { cancelCurrentRun, run, runBusy, runProgress } from '../../stores/run';
  import { settingsOpen } from '../../stores/ui';
  import Segmented from '../../ui/Segmented.svelte';
  import DateRange from '../../ui/DateRange.svelte';
  import Select from '../../ui/Select.svelte';
  import Button from '../../ui/Button.svelte';
  import Icon from '../../ui/Icon.svelte';
  import { dayCount, fmtRange } from '../../ui/format';
  import StrategyPicker from './StrategyPicker.svelte';
  import { activeRange, customRange, period, PERIOD_OPTIONS, selected, type PeriodPreset } from './state';
  import { strategies } from '../../stores/app';

  export let onRun: () => void;

  $: busy = $runBusy;
  $: range = $activeRange;
  $: days = range ? dayCount(range.start, range.end) : 0;
  $: presetsReady = $capabilities.presets === 'available' && $presets;
  $: presetsMissing = $capabilities.presets === 'unavailable';
  $: availableIds = new Set($strategies.filter((s) => s.available !== false).map((s) => s.id));
  $: runnable = $selected.filter((id) => availableIds.has(id));
  $: skipped = $strategies.filter((s) => s.available === false && $selected.includes(s.id)).map((s) => s.name);
  $: canRun = !busy && !!range && runnable.length > 0 && $draftValid && !!$draft;
  $: blocker = !range
    ? 'Waiting for the data range'
    : runnable.length === 0
      ? 'Pick at least one strategy'
      : !$draftValid
        ? 'Fix the highlighted settings first'
        : null;

  $: hwOptions = presetsReady && $presets
    ? [
        ...$presets.hardware.map((h) => ({ value: h.id, label: h.name })),
        ...($hardwarePresetId ? [] : [{ value: '', label: 'Custom (from Settings)' }]),
      ]
    : [{ value: '', label: presetsMissing ? 'From Settings' : 'Loading…' }];
  $: tariffOptions = presetsReady && $presets
    ? [
        ...$presets.tariffs.map((t) => ({ value: t.id, label: t.name })),
        ...($tariffPresetId ? [] : [{ value: '', label: 'Custom (from Settings)' }]),
      ]
    : [{ value: '', label: presetsMissing ? 'From Settings' : 'Loading…' }];

  $: hwValue = $hardwarePresetId ?? '';
  $: tariffValue = $tariffPresetId ?? '';

  function onHw(e: Event) {
    const id = (e.currentTarget as HTMLSelectElement).value;
    const p = $presets?.hardware.find((h) => h.id === id);
    if (p) applyHardwarePreset(p);
  }
  function onTariff(e: Event) {
    const id = (e.currentTarget as HTMLSelectElement).value;
    const p = $presets?.tariffs.find((t) => t.id === id);
    if (p) applyTariffPreset(p);
  }

  function onPeriod(e: CustomEvent<string>) {
    const p = e.detail as PeriodPreset;
    if (p === 'custom' && !$customRange && range) customRange.set({ ...range });
  }

  // Persistent live region: one inserted with its text already in it is often not announced.
  let announcement = '';
  let wasBusy = false;
  $: if (busy !== wasBusy) {
    wasBusy = busy;
    if (busy) announcement = 'Simulation started';
    else if ($run) {
      const st = $run.status;
      announcement =
        st === 'completed' ? 'Simulation finished. Results are below.'
        : st === 'cancelled' ? 'Simulation cancelled'
        : st === 'failed' ? `Simulation failed${$run.error ? `: ${$run.error}` : ''}`
        : 'Simulation finished';
    }
  }

  function runClick() {
    if (canRun) onRun();
  }

  $: hwDescription = presetsReady && $presets ? $presets.hardware.find((h) => h.id === hwValue)?.description ?? null : null;
</script>

<section class="controls" aria-label="Simulation setup">
  <div class="top">
    <div class="period">
      <span class="label" id="period-label">Period</span>
      <div class="period-row">
        <Segmented label="Period" bind:value={$period} options={PERIOD_OPTIONS} disabled={busy} on:change={onPeriod} />
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
            disabled={busy}
            on:change={(e) => customRange.set(e.detail)}
          />
          <span class="hint num">{days} {days === 1 ? 'day' : 'days'}, both dates included</span>
        </div>
      {/if}
    </div>

    <div class="presets">
      <Select
        label="Hardware"
        value={hwValue}
        options={hwOptions}
        disabled={!presetsReady || busy}
        on:change={onHw}
      />
      <Select
        label="Tariff"
        value={tariffValue}
        options={tariffOptions}
        disabled={!presetsReady || busy}
        on:change={onTariff}
      />
      {#if presetsMissing}
        <p class="hint span">Presets need a backend update. Battery and tariff come from Settings.</p>
      {:else if hwDescription}
        <p class="hint span">{hwDescription}</p>
      {/if}
    </div>
  </div>

  <div class="strategies">
    <span class="label">Strategies <span class="muted">· compared against no battery</span></span>
    <StrategyPicker disabled={busy} />
  </div>

  <div class="actions">
    <Button variant="ghost" on:click={() => settingsOpen.set(true)}>
      <span class="settings-btn">
        <Icon name="sliders" size={16} />
        Settings
        {#if $draftModified}<span class="mdot" title="Settings differ from the defaults" /><span class="sr-only">(modified)</span>{/if}
      </span>
    </Button>
    {#if blocker && !busy}<span class="blocker" role="status">{blocker}</span>{/if}
    {#if skipped.length && !busy && !blocker}
      <span class="blocker warn" role="status">
        <Icon name="alert" size={14} />
        {skipped.join(', ')} {skipped.length === 1 ? 'is' : 'are'} unavailable and will be skipped
      </span>
    {/if}
    <span class="spacer" />
    {#if busy}
      <span class="status num" aria-hidden="true">
        {$run?.status === 'starting' ? 'Starting…' : `Running · ${Math.round($runProgress)}%`}
      </span>
      {#if $capabilities.cancel === 'available'}
        <Button variant="secondary" icon="stop" loading={$run?.cancelling} on:click={cancelCurrentRun}>Cancel</Button>
      {/if}
    {/if}
    <span class="sr-only" role="status" aria-live="polite">{announcement}</span>
    <Button variant="primary" size="lg" icon="play" loading={busy} disabled={!canRun} keepFocus on:click={runClick}>
      {busy ? 'Simulating' : 'Run simulation'}
    </Button>
  </div>
  {#if busy}
    <div class="progress-clip" aria-hidden="true"><div class="progress"><span style="width: {$runProgress}%" /></div></div>
  {/if}
</section>

<style>
  .controls {
    position: relative;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    box-shadow: var(--shadow-1);
  }
  .top {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 420px);
    gap: var(--s-5);
    padding: var(--s-5) var(--s-5) 0;
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
    color: var(--text);
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
  /* Without a basis the date group shrinks for the hint beside it and its two fields stack. */
  .custom > :global(.range) {
    flex: 1 1 320px;
  }
  .custom .hint {
    padding-bottom: 10px;
  }
  .presets {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--s-2) var(--s-3);
    align-content: start;
  }
  .span {
    grid-column: 1 / -1;
  }
  .strategies {
    padding: var(--s-5) var(--s-5) var(--s-4);
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
  /* Pull the ghost button back so its label lines up with the content edge above it. */
  .actions > :global(.btn.ghost:first-child) {
    margin-left: calc(var(--s-4) * -1 + 4px);
  }
  .settings-btn {
    display: inline-flex;
    align-items: center;
    gap: var(--s-2);
  }
  .mdot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--accent);
  }
  .blocker {
    font-size: var(--fs-sm);
    color: var(--text-3);
  }
  .blocker.warn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--warning);
  }
  .spacer {
    flex: 1;
  }
  .status {
    font-size: var(--fs-sm);
    color: var(--text-2);
  }
  /* A 3px bar can't carry the card's corner radius (it gets scaled down to square), so a
     taller clip box with the card's inner radius rounds the bar's ends instead. The card
     itself can't clip: the More strategies popover overflows it. */
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
    .top {
      grid-template-columns: 1fr;
      gap: var(--s-4);
    }
  }
  @media (max-width: 720px) {
    .top {
      padding: var(--s-4) var(--s-4) 0;
    }
    .strategies {
      padding: var(--s-4);
    }
    .actions {
      padding: var(--s-3) var(--s-4);
    }
    .actions :global(.btn.primary) {
      flex: 1 1 100%;
      order: 10;
    }
    .spacer {
      display: none;
    }
    .period-row {
      gap: var(--s-2);
    }
  }
</style>
