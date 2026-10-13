<script lang="ts">
  import { createEventDispatcher, tick } from 'svelte';
  import { deleteRun, errorMessage, isUnavailable } from '../../api/client';
  import { capabilities, setCapability, strategyById, strategyName } from '../../stores/app';
  import { forgetRun, rerun, run, runBusy } from '../../stores/run';
  import { toast } from '../../stores/toasts';
  import Popover from '../../ui/Popover.svelte';
  import Icon from '../../ui/Icon.svelte';
  import StrategyDot from '../../ui/StrategyDot.svelte';
  import { dayCount, eur, eurSigned, fmtAgo, fmtDateTime, fmtRange } from '../../ui/format';
  import type { ExportKind, RunListItem } from '../../types';
  import { STATUS_LABEL, digestHue, digestRun, downloadRunCsv, hashHref, runTitle } from './history';

  export let item: RunListItem;
  /** Shared clock so relative times tick together. */
  export let now: number;
  export let digestActive = false;

  const dispatch = createEventDispatcher<{ deleted: string; digest: string }>();

  let menuOpen = false;
  let confirming = false;
  let busy: ExportKind | 'delete' | null = null;

  $: d = digestRun(item);
  $: title = runTitle(item, d.strategyIds, (id) => $strategyById.get(id)?.name ?? strategyName(id));
  $: days = dayCount(item.date_range.start, item.date_range.end);
  $: shown = $run?.runId === item.run_id;
  $: exportCap = $capabilities.export;
  $: canExport = exportCap === 'available';
  $: openHref = hashHref('simulate', 'run', item.run_id);
  $: hue = digestHue(item.config_digest);

  let rowEl: HTMLLIElement;
  let keepBtn: HTMLButtonElement | null = null;
  let deleteItem: HTMLButtonElement | null = null;

  // Swapping the menu for the question destroys the focused button; move focus into what replaced it.
  async function askDelete() {
    confirming = true;
    await tick();
    keepBtn?.focus();
  }

  async function keep() {
    confirming = false;
    await tick();
    deleteItem?.focus();
  }

  function closeMenu() {
    menuOpen = false;
    confirming = false;
  }

  async function doExport(kind: ExportKind) {
    busy = kind;
    try {
      await downloadRunCsv(item.run_id, kind);
      closeMenu();
    } catch (err) {
      if (isUnavailable(err)) {
        setCapability('export', 'unavailable');
        toast('info', 'CSV export needs a backend update');
      } else toast('error', 'Export failed', errorMessage(err));
    } finally {
      busy = null;
    }
  }

  async function doDelete() {
    busy = 'delete';
    try {
      await deleteRun(item.run_id);
      forgetRun(item.run_id);
      // This row is about to go: pick the neighbour's menu button (or the search field) to take focus.
      const neighbour = (rowEl.nextElementSibling ?? rowEl.previousElementSibling)?.querySelector<HTMLElement>('button.more');
      closeMenu();
      toast('success', 'Run deleted', title);
      dispatch('deleted', item.run_id);
      await tick();
      (neighbour?.isConnected ? neighbour : document.getElementById('history-search'))?.focus();
    } catch (err) {
      toast('error', 'Could not delete the run', errorMessage(err));
      void tick().then(() => keepBtn?.focus());
    } finally {
      busy = null;
    }
  }

  function doRerun() {
    closeMenu();
    void rerun(item.run_id);
  }
</script>

<li class="row" class:shown aria-label={title} bind:this={rowEl}>
  <div class="cell main">
    <a class="title" href={openHref} title="Open on Simulate">{title}</a>
    <div class="sub">
      <span class="dots" aria-hidden="true">
        {#each d.strategyIds.slice(0, 8) as id (id)}<StrategyDot id={id} size={8} />{/each}
      </span>
      <span class="count">{d.strategyIds.length} {d.strategyIds.length === 1 ? 'strategy' : 'strategies'}</span>
      {#if shown}<span class="now-showing">On Simulate</span>{/if}
    </div>
  </div>

  <div class="cell period">
    <span class="k">Period</span>
    <span class="v num">{fmtRange(item.date_range.start, item.date_range.end)}</span>
    <span class="s num">{days} {days === 1 ? 'day' : 'days'}</span>
  </div>

  <div class="cell created">
    <span class="k">Created</span>
    <span class="v">{fmtAgo(item.created_at, now)}</span>
    <span class="s num">{fmtDateTime(item.created_at, true)}</span>
  </div>

  <div class="cell status">
    <span class="k">Status</span>
    <span class="state {item.status}" class:partial={item.status === 'completed' && d.failed > 0}>
      {#if item.status === 'running'}<span class="pulse" aria-hidden="true" />{:else}<span class="mark" aria-hidden="true" />{/if}
      {STATUS_LABEL[item.status]}{#if d.failed > 0 && item.status === 'completed'}, {d.failed} failed{/if}
    </span>
  </div>

  <div class="cell best">
    <span class="k">Best strategy</span>
    {#if d.bestId}
      <span class="v best-name"><StrategyDot id={d.bestId} size={9} />{$strategyById.get(d.bestId)?.name ?? strategyName(d.bestId)}</span>
      <span class="s num">{eur(d.bestCost)} net cost</span>
    {:else}
      <span class="v muted">—</span>
    {/if}
  </div>

  <div class="cell savings">
    <span class="k">Savings</span>
    <span class="v num money" class:pos={d.outcome === 'win'} class:neg={d.outcome === 'loss'}>{eurSigned(d.savings)}</span>
    <span class="s">{d.savings == null ? '' : 'vs no battery'}</span>
  </div>

  <div class="cell config">
    <span class="k">Settings</span>
    <button
      type="button"
      class="digest"
      class:active={digestActive}
      style="--h: {hue}"
      aria-pressed={digestActive}
      title={digestActive
        ? 'Showing only runs with these settings. Click to show all.'
        : 'Settings fingerprint: runs with the same code used identical settings. Click to show only those.'}
      on:click={() => dispatch('digest', item.config_digest)}
    >
      <span class="swatch" aria-hidden="true" />
      <span class="code">{item.config_digest}</span>
    </button>
  </div>

  <div class="cell actions">
    <a class="open" href={openHref}>Open</a>
    <Popover open={menuOpen} label="Run actions" width={260} on:close={closeMenu}>
      <button
        slot="trigger"
        type="button"
        class="more"
        aria-label="More actions for {title}"
        aria-expanded={menuOpen}
        aria-haspopup="dialog"
        on:click|stopPropagation={() => (menuOpen ? closeMenu() : (menuOpen = true))}
      >
        <Icon name="more" size={18} stroke={2.6} />
      </button>
      <!-- Buttons that swap this content stop propagation: the popover's outside-click check would
           otherwise see the removed button as outside and close. -->
      {#if confirming}
        <div class="confirm" role="group" aria-labelledby="del-q-{item.run_id}" aria-describedby="del-d-{item.run_id}">
          <p class="q" id="del-q-{item.run_id}">Delete this run?</p>
          <p class="d" id="del-d-{item.run_id}">Its results and settings are removed from history. This can't be undone.</p>
          <div class="confirm-actions">
            <button type="button" class="mbtn" bind:this={keepBtn} on:click|stopPropagation={keep} disabled={busy === 'delete'}>Keep</button>
            <button type="button" class="mbtn danger" on:click={doDelete} disabled={busy === 'delete'}>
              {busy === 'delete' ? 'Deleting…' : 'Delete run'}
            </button>
          </div>
        </div>
      {:else}
        <div class="menu">
          <button type="button" class="item" on:click={doRerun} disabled={$runBusy || item.status === 'running'}>
            <Icon name="refresh" size={15} />
            <span><span class="t">Re-run</span><span class="dsc">Same period, strategies and settings</span></span>
          </button>
          <div class="sep" role="separator" />
          {#if !canExport}
            <p class="note">{exportCap === 'unknown' ? 'Checking export support…' : 'CSV export needs a backend update.'}</p>
          {/if}
          <button type="button" class="item" disabled={!canExport || busy != null} on:click={() => doExport('summary')}>
            <Icon name="download" size={15} />
            <span><span class="t">{busy === 'summary' ? 'Exporting…' : 'Summary CSV'}</span><span class="dsc">One row per strategy, all totals</span></span>
          </button>
          <button type="button" class="item" disabled={!canExport || busy != null} on:click={() => doExport('monthly')}>
            <Icon name="download" size={15} />
            <span><span class="t">{busy === 'monthly' ? 'Exporting…' : 'Monthly CSV'}</span><span class="dsc">Cost and energy per strategy per month</span></span>
          </button>
          <div class="sep" role="separator" />
          <button
            type="button"
            class="item del"
            disabled={item.status === 'running'}
            title={item.status === 'running' ? 'Wait for the run to finish or cancel it first' : undefined}
            bind:this={deleteItem}
            on:click|stopPropagation={askDelete}
          >
            <Icon name="trash" size={15} />
            <span class="t">Delete…</span>
          </button>
        </div>
      {/if}
    </Popover>
  </div>
</li>

<style>
  .row {
    display: grid;
    grid-template-columns: var(--history-cols);
    align-items: center;
    gap: var(--s-4);
    padding: 14px var(--s-5);
    border-bottom: 1px solid var(--border);
    position: relative;
  }
  .row:last-child {
    border-bottom: 0;
  }
  .row.shown {
    background: linear-gradient(90deg, var(--accent-soft), transparent 40%);
  }
  .row.shown::before {
    content: '';
    position: absolute;
    left: 0;
    top: 10px;
    bottom: 10px;
    width: 3px;
    border-radius: 0 3px 3px 0;
    background: var(--accent);
  }
  .cell {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
  }
  /* Column labels: visible on the stacked phone layout, read out by screen readers on desktop. */
  .k {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
  }
  .v {
    font-size: var(--fs-sm);
    color: var(--text);
    white-space: nowrap;
  }
  .s {
    font-size: var(--fs-xs);
    color: var(--text-3);
    white-space: nowrap;
  }
  .title {
    font-weight: 620;
    color: var(--text);
    text-decoration: none;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .title:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
    text-decoration-color: var(--text-3);
  }
  .sub {
    display: flex;
    align-items: center;
    gap: var(--s-2);
    font-size: var(--fs-xs);
    color: var(--text-3);
  }
  .dots {
    display: inline-flex;
    gap: 3px;
  }
  .now-showing {
    color: var(--accent);
    font-weight: 600;
  }
  .period .v {
    white-space: normal;
  }
  .best-name {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .money {
    font-size: var(--fs-md);
    font-weight: 650;
    text-align: right;
  }
  .savings {
    align-items: flex-end;
  }
  .pos {
    color: var(--positive);
  }
  .neg {
    color: var(--negative);
  }

  .state {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: var(--fs-sm);
    color: var(--text-2);
    white-space: nowrap;
  }
  .mark {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--text-3);
  }
  .state.completed .mark {
    background: var(--positive);
  }
  .state.partial {
    color: var(--warning);
  }
  .state.partial .mark {
    background: var(--warning);
  }
  .state.failed {
    color: var(--negative);
  }
  .state.failed .mark {
    background: var(--negative);
  }
  .state.cancelled .mark {
    background: transparent;
    box-shadow: inset 0 0 0 1.5px var(--text-3);
  }
  .state.running {
    color: var(--accent);
    font-weight: 600;
  }
  .pulse {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 1.4s ease-in-out infinite;
  }
  @keyframes pulse {
    50% {
      opacity: 0.3;
    }
  }

  .digest {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 26px;
    padding: 0 8px 0 6px;
    border-radius: var(--r-sm);
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--text-2);
    font-family: var(--font-mono);
    font-size: 11.5px;
    align-self: flex-start;
  }
  .digest:hover {
    border-color: var(--border-strong);
    color: var(--text);
  }
  .digest.active {
    border-color: var(--accent);
    background: var(--accent-soft);
    color: var(--accent);
  }
  .swatch {
    width: 10px;
    height: 10px;
    border-radius: 3px;
    background: hsl(var(--h) 62% 52%);
  }

  .actions {
    flex-direction: row;
    align-items: center;
    justify-content: flex-end;
    gap: var(--s-1);
  }
  .open {
    display: inline-flex;
    align-items: center;
    height: 30px;
    padding: 0 var(--s-3);
    border-radius: var(--r-md);
    border: 1px solid var(--border-strong);
    background: var(--surface);
    color: var(--text);
    font-size: var(--fs-sm);
    font-weight: 550;
    text-decoration: none;
  }
  .open:hover {
    background: var(--surface-2);
  }
  .more {
    display: inline-grid;
    place-items: center;
    width: 30px;
    height: 30px;
    border: 0;
    border-radius: var(--r-md);
    background: transparent;
    color: var(--text-2);
  }
  .more:hover,
  .more[aria-expanded='true'] {
    background: var(--surface-3);
    color: var(--text);
  }

  .menu {
    padding: var(--s-2);
    display: grid;
    gap: 2px;
  }
  .item {
    display: flex;
    align-items: flex-start;
    gap: var(--s-3);
    text-align: left;
    padding: var(--s-2) var(--s-3);
    border: 0;
    border-radius: var(--r-md);
    background: transparent;
    color: var(--text-2);
  }
  .item > :global(svg) {
    margin-top: 2px;
  }
  .item > span {
    display: grid;
    gap: 1px;
  }
  .item:hover:not(:disabled) {
    background: var(--surface-2);
    color: var(--text);
  }
  .item:disabled {
    opacity: 0.5;
  }
  .t {
    font-weight: 600;
    font-size: var(--fs-sm);
    color: var(--text);
  }
  .dsc {
    font-size: var(--fs-xs);
    color: var(--text-3);
  }
  .del,
  .del .t {
    color: var(--negative);
  }
  .sep {
    height: 1px;
    margin: 4px var(--s-2);
    background: var(--border);
  }
  .note {
    padding: var(--s-1) var(--s-3);
    font-size: var(--fs-xs);
    color: var(--text-3);
  }
  .confirm {
    padding: var(--s-4);
    display: grid;
    gap: var(--s-2);
  }
  .q {
    font-weight: 650;
  }
  .d {
    font-size: var(--fs-sm);
    color: var(--text-2);
  }
  .confirm-actions {
    display: flex;
    justify-content: flex-end;
    gap: var(--s-2);
    margin-top: var(--s-2);
  }
  .mbtn {
    height: 32px;
    padding: 0 var(--s-3);
    border-radius: var(--r-md);
    border: 1px solid var(--border-strong);
    background: var(--surface);
    font-size: var(--fs-sm);
    font-weight: 550;
  }
  .mbtn:hover:not(:disabled) {
    background: var(--surface-2);
  }
  .mbtn.danger {
    background: var(--negative);
    border-color: var(--negative);
    color: var(--surface);
  }
  .mbtn.danger:hover:not(:disabled) {
    filter: brightness(1.08);
    background: var(--negative);
  }

  /* Phone and narrow tablet: each run becomes a compact card with labelled facts. */
  @media (max-width: 960px) {
    .row {
      grid-template-columns: 1fr 1fr;
      gap: var(--s-3) var(--s-4);
      padding: var(--s-4);
    }
    .main {
      grid-column: 1 / -1;
    }
    .title {
      white-space: normal;
    }
    .v,
    .s {
      white-space: normal;
    }
    .k {
      position: static;
      width: auto;
      height: auto;
      overflow: visible;
      clip: auto;
      font-size: var(--fs-xs);
      color: var(--text-3);
    }
    .status {
      order: -1;
      position: absolute;
      top: var(--s-4);
      right: var(--s-4);
    }
    .status .k {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
    }
    .main {
      padding-right: 110px;
    }
    .savings {
      align-items: flex-start;
    }
    .money {
      text-align: left;
    }
    .actions {
      grid-column: 1 / -1;
      justify-content: flex-start;
    }
    .config {
      grid-column: 1 / -1;
      flex-direction: row;
      align-items: center;
      gap: var(--s-2);
    }
    .actions {
      gap: var(--s-2);
    }
    .open {
      height: 40px;
      flex: 1;
      justify-content: center;
    }
    .more {
      width: 40px;
      height: 40px;
      border: 1px solid var(--border-strong);
    }
    .item {
      min-height: 48px;
      align-items: center;
    }
    .item > :global(svg) {
      margin-top: 0;
    }
    .mbtn {
      height: 40px;
    }
    .digest {
      height: 40px;
      padding: 0 12px 0 10px;
    }
  }
  /* Phone: the settings code shares the actions' line, keeping each card short. */
  @media (max-width: 720px) {
    .config {
      grid-column: 1 / 2;
    }
    .config .k {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
    }
    .actions {
      grid-column: 2 / -1;
    }
  }
  /* Tablet: the facts fit on one line, and the settings share a line with the actions. */
  @media (min-width: 721px) and (max-width: 960px) {
    .row {
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }
    .config {
      grid-column: 1 / 3;
    }
    .actions {
      grid-column: 3 / -1;
      justify-content: flex-end;
    }
    .open {
      flex: none;
      padding: 0 var(--s-5);
    }
  }
  /* Finger-sized hit area without changing the row's layout; the row's Open button is the main target. */
  @media (pointer: coarse) {
    .title {
      padding-block: 10px;
      margin-block: -10px;
    }
  }
</style>
