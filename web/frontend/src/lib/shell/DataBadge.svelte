<script lang="ts">
  import { onDestroy } from 'svelte';
  import { capabilities, dataInfo, probeCapabilities, refreshDataInfo } from '../stores/app';
  import { refreshJob, startRefresh } from '../stores/refresh';
  import { errorMessage, reloadData } from '../api/client';
  import { toast } from '../stores/toasts';
  import Popover from '../ui/Popover.svelte';
  import Button from '../ui/Button.svelte';
  import { fmtAgo, fmtDate, fmtDateTime, fmtHours, num } from '../ui/format';

  let open = false;
  let now = Date.now();
  let reloading = false;
  const tick = setInterval(() => (now = Date.now()), 60_000);
  onDestroy(() => clearInterval(tick));

  $: info = $dataInfo;
  $: until = info?.data_until ?? info?.date_range?.end ?? null;
  $: from = info?.data_from ?? info?.date_range?.start ?? null;
  $: stale = until ? now - Date.parse(until) > 24 * 3600_000 : false;
  $: coverage = info?.coverage != null ? Math.round(info.coverage * 100) : null;
  $: gapHours = info?.gaps.reduce((a, g) => a + g.hours, 0) ?? 0;
  $: job = $refreshJob;
  $: running = job?.status === 'running';
  $: refreshCap = $capabilities.refresh;

  function toggle() {
    open = !open;
    if (open) {
      void refreshDataInfo();
      void probeCapabilities();
    }
  }

  async function reload() {
    reloading = true;
    try {
      dataInfo.set(await reloadData());
      toast('success', 'Data reloaded from disk');
    } catch (err) {
      toast('error', 'Could not reload data', errorMessage(err));
    } finally {
      reloading = false;
    }
  }
</script>

<Popover {open} label="Data source" width={360} on:close={() => (open = false)}>
  <button
    slot="trigger"
    type="button"
    class="badge"
    class:stale
    aria-expanded={open}
    aria-haspopup="dialog"
    on:click|stopPropagation={toggle}
    title={stale ? 'Data is more than 24 hours old' : 'Data source'}
  >
    <span class="dot" class:running aria-hidden="true" />
    {#if info && until}
      <span class="long">Data to {fmtDateTime(until)}{coverage != null ? ` · ${coverage}% coverage` : ''}</span>
      <span class="short">{fmtDateTime(until)}</span>
    {:else}
      <span>Loading data…</span>
    {/if}
  </button>

  {#if info}
    <div class="pop">
      <div class="head">
        <h3>Measured data</h3>
        {#if stale}<span class="warn">Last reading {fmtAgo(until ?? '', now)}</span>{/if}
      </div>
      <dl>
        <div><dt>Source</dt><dd>{info.source === 'store' ? 'Parquet store (InfluxDB export)' : info.source === 'csv' ? 'CSV files' : '—'}</dd></div>
        <div><dt>Range</dt><dd class="num">{fmtDate(from)} – {fmtDateTime(until, true)}</dd></div>
        <div><dt>Coverage</dt><dd class="num">{coverage != null ? `${coverage}% of minutes have house load` : '—'}</dd></div>
        <div><dt>Gaps</dt><dd class="num">{info.gaps.length ? `${num(info.gaps.length, 0)} outages over 1 h, ${fmtHours(gapHours)} in total` : 'None over 1 h'}</dd></div>
        {#if info.manifest_updated_at}<div><dt>Store updated</dt><dd class="num">{fmtDateTime(info.manifest_updated_at, true)}</dd></div>{/if}
        {#if info.uploads.length}<div><dt>Uploaded</dt><dd>{info.uploads.join(', ')}</dd></div>{/if}
      </dl>

      {#if running && job}
        <div class="progress" role="status" aria-live="polite">
          <div class="bar"><span style="width: {job.progress_pct}%" /></div>
          <p class="hint">{job.message} · {Math.round(job.progress_pct)}%</p>
        </div>
      {/if}

      <div class="actions">
        <Button
          variant="primary"
          size="sm"
          icon="refresh"
          loading={running}
          disabled={refreshCap !== 'available'}
          on:click={() => startRefresh(false)}
        >
          {running ? 'Refreshing' : 'Refresh from InfluxDB'}
        </Button>
        <Button variant="ghost" size="sm" loading={reloading} on:click={reload}>Reload from disk</Button>
      </div>
      {#if refreshCap === 'unavailable'}
        <p class="hint">Refreshing from InfluxDB needs a backend update.</p>
      {/if}
    </div>
  {/if}
</Popover>

<style>
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    height: 32px;
    padding: 0 12px;
    border-radius: var(--r-pill);
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--text-2);
    font-size: var(--fs-sm);
    font-weight: 500;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .badge:hover {
    border-color: var(--border-strong);
    color: var(--text);
  }
  .badge.stale {
    background: var(--warning-soft);
    border-color: var(--warning-border);
    color: var(--warning);
  }
  .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--positive);
  }
  .stale .dot {
    background: var(--warning);
  }
  .dot.running {
    background: var(--accent);
    animation: pulse 1.2s ease-in-out infinite;
  }
  .short {
    display: none;
  }
  .pop {
    padding: var(--s-4);
  }
  .head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: var(--s-2);
    margin-bottom: var(--s-3);
  }
  h3 {
    font-size: var(--fs-md);
    font-weight: 650;
  }
  .warn {
    font-size: var(--fs-xs);
    color: var(--warning);
    font-weight: 600;
  }
  dl {
    display: grid;
    gap: var(--s-2);
    margin: 0 0 var(--s-4);
  }
  dl div {
    display: grid;
    grid-template-columns: 88px 1fr;
    gap: var(--s-2);
    font-size: var(--fs-sm);
  }
  dt {
    color: var(--text-3);
  }
  dd {
    margin: 0;
    color: var(--text);
  }
  .actions {
    display: flex;
    gap: var(--s-2);
    flex-wrap: wrap;
  }
  .actions + .hint {
    margin-top: var(--s-2);
  }
  .progress {
    margin-bottom: var(--s-3);
  }
  .bar {
    height: 4px;
    border-radius: 2px;
    background: var(--surface-3);
    overflow: hidden;
    margin-bottom: 6px;
  }
  .bar span {
    display: block;
    height: 100%;
    background: var(--accent);
    transition: width 300ms var(--ease);
  }
  @keyframes pulse {
    50% {
      opacity: 0.35;
    }
  }
  @media (max-width: 900px) {
    .long {
      display: none;
    }
    .short {
      display: inline;
    }
  }
  @media (max-width: 720px), (pointer: coarse) {
    .badge {
      height: 40px;
    }
  }
</style>
