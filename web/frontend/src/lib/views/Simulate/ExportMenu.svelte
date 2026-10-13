<script lang="ts">
  import { errorMessage, exportCsv, isUnavailable } from '../../api/client';
  import { capabilities, setCapability, strategyName } from '../../stores/app';
  import { toast } from '../../stores/toasts';
  import Popover from '../../ui/Popover.svelte';
  import Icon from '../../ui/Icon.svelte';
  import type { ExportKind } from '../../types';

  export let runId: string | null;
  export let focusId: string | null;
  export let hasMonthly: boolean;

  let open = false;
  let busy: ExportKind | null = null;

  $: cap = $capabilities.export;
  $: available = cap === 'available' && !!runId;

  async function download(kind: ExportKind) {
    if (!runId) return;
    busy = kind;
    try {
      const { blob, filename } = await exportCsv(runId, kind, kind === 'timeseries' ? focusId ?? undefined : undefined);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename ?? `powerflow-${runId}-${kind}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      open = false;
    } catch (err) {
      if (isUnavailable(err)) {
        setCapability('export', 'unavailable');
        toast('info', 'CSV export needs a backend update');
      } else toast('error', 'Export failed', errorMessage(err));
    } finally {
      busy = null;
    }
  }
</script>

<Popover {open} label="Export CSV" width={300} on:close={() => (open = false)}>
  <button
    slot="trigger"
    type="button"
    class="trigger"
    aria-expanded={open}
    aria-haspopup="dialog"
    disabled={!runId}
    on:click|stopPropagation={() => (open = !open)}
  >
    <Icon name="download" size={15} />
    Export
    <Icon name="chevronDown" size={14} />
  </button>
  <div class="menu">
    {#if !available}
      <p class="note">{cap === 'unknown' ? 'Checking export support…' : 'CSV export needs a backend update.'}</p>
    {/if}
    <button type="button" class="item" disabled={!available || busy != null} on:click={() => download('summary')}>
      <span class="t">Summary</span><span class="d">One row per strategy, all totals</span>
    </button>
    <button type="button" class="item" disabled={!available || busy != null || !hasMonthly} on:click={() => download('monthly')}>
      <span class="t">Monthly</span><span class="d">Cost and energy per strategy per month</span>
    </button>
    <button type="button" class="item" disabled={!available || busy != null || !focusId} on:click={() => download('timeseries')}>
      <span class="t">Timeseries</span>
      <span class="d">Every minute for {focusId ? strategyName(focusId) : 'the selected strategy'}</span>
    </button>
  </div>
</Popover>

<style>
  .trigger {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 32px;
    padding: 0 10px 0 12px;
    border-radius: var(--r-md);
    border: 1px solid var(--border-strong);
    background: var(--surface);
    font-size: var(--fs-sm);
    font-weight: 550;
  }
  .trigger:hover:not(:disabled) {
    background: var(--surface-2);
  }
  .trigger:disabled {
    opacity: 0.5;
  }
  .menu {
    padding: var(--s-2);
    display: grid;
    gap: 2px;
  }
  .note {
    padding: var(--s-2);
    font-size: var(--fs-xs);
    color: var(--text-3);
  }
  .item {
    display: grid;
    gap: 2px;
    text-align: left;
    padding: var(--s-2) var(--s-3);
    border: 0;
    border-radius: var(--r-md);
    background: transparent;
  }
  .item:hover:not(:disabled) {
    background: var(--surface-2);
  }
  .item:disabled {
    opacity: 0.5;
  }
  .t {
    font-weight: 600;
    font-size: var(--fs-sm);
  }
  .d {
    font-size: var(--fs-xs);
    color: var(--text-3);
  }
  @media (max-width: 720px), (pointer: coarse) {
    .trigger {
      height: 40px;
    }
    .item {
      min-height: 48px;
    }
  }
</style>
