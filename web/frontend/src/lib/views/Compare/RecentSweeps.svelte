<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { errorMessage, isUnavailable, listSweeps } from '../../api/client';
  import Card from '../../ui/Card.svelte';
  import Badge from '../../ui/Badge.svelte';
  import Button from '../../ui/Button.svelte';
  import { dayCount, fmtAgo, fmtRange } from '../../ui/format';
  import type { SweepListItem } from '../../types';
  import { loadSweep, sweep } from './state';

  /** The API has no paging; its maximum page size keeps every saved comparison reachable. */
  const LIMIT = 200;
  const PAGE = 6;

  let items: SweepListItem[] = [];
  let shown = PAGE;
  let listEl: HTMLElement;

  // The Show more button can disappear with the last page; keep keyboard focus on the first new item.
  async function showMore() {
    const from = shown;
    shown += PAGE * 2;
    await tick();
    listEl?.querySelectorAll<HTMLElement>('button.item')[from]?.focus();
  }
  let state: 'loading' | 'ready' | 'error' | 'unavailable' = 'loading';
  let error = '';

  async function load() {
    try {
      items = await listSweeps(LIMIT);
      state = 'ready';
    } catch (err) {
      if (isUnavailable(err)) state = 'unavailable';
      else {
        state = 'error';
        error = errorMessage(err);
      }
    }
  }

  onMount(load);

  // Refresh when a sweep starts or ends, so the list shows it.
  let lastKey = '';
  $: key = `${$sweep?.id ?? ''}:${$sweep?.status ?? ''}`;
  $: if (key !== lastKey) {
    lastKey = key;
    if (state !== 'loading') void load();
  }

  const STATUS: Record<string, { tone: 'accent' | 'neutral' | 'negative'; text: string }> = {
    running: { tone: 'accent', text: 'Running' },
    cancelled: { tone: 'neutral', text: 'Cancelled' },
    failed: { tone: 'negative', text: 'Failed' },
  };
</script>

<Card title="Recent comparisons" subtitle="Saved on the server. Pick one to open it." pad={false}>
  {#if state === 'loading'}
    <p class="msg">Loading…</p>
  {:else if state === 'unavailable'}
    <p class="msg">Saved comparisons need a backend update.</p>
  {:else if state === 'error'}
    <div class="msg row">
      <span>Couldn't load comparisons: {error}</span>
      <Button size="sm" variant="ghost" icon="refresh" on:click={load}>Retry</Button>
    </div>
  {:else if !items.length}
    <p class="msg">No comparisons yet. They appear here once you run one.</p>
  {:else}
    <ul class="list" bind:this={listEl}>
      {#each items.slice(0, shown) as it (it.sweep_id)}
        {@const st = STATUS[it.status]}
        {@const days = dayCount(it.date_range.start, it.date_range.end)}
        <li>
          <button type="button" class="item" class:current={$sweep?.id === it.sweep_id} aria-current={$sweep?.id === it.sweep_id || undefined} on:click={() => loadSweep(it.sweep_id)}>
            <span class="main">
              <span class="title">{it.label || fmtRange(it.date_range.start, it.date_range.end)}</span>
              <span class="meta num">
                {#if it.label}{fmtRange(it.date_range.start, it.date_range.end)} · {/if}{days} {days === 1 ? 'day' : 'days'} · {it.total} cases
              </span>
            </span>
            <span class="side">
              {#if st}<Badge tone={st.tone}>{st.text}</Badge>{/if}
              <span class="ago num">{fmtAgo(it.created_at)}</span>
            </span>
          </button>
        </li>
      {/each}
    </ul>
    {#if items.length > shown}
      <div class="more">
        <span>Showing the newest {shown} of {items.length >= LIMIT ? `${items.length}+` : items.length}</span>
        <Button size="sm" variant="secondary" on:click={showMore}>Show {Math.min(PAGE * 2, items.length - shown)} more</Button>
      </div>
    {/if}
  {/if}
</Card>

<style>
  .msg {
    padding: 0 var(--s-5) var(--s-5);
    color: var(--text-3);
    font-size: var(--fs-sm);
  }
  .row {
    display: flex;
    align-items: center;
    gap: var(--s-3);
    flex-wrap: wrap;
  }
  .list {
    border-top: 1px solid var(--border);
  }
  .more {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: var(--s-2) var(--s-4);
    padding: var(--s-3) var(--s-5);
    border-top: 1px solid var(--border);
    font-size: var(--fs-xs);
    color: var(--text-3);
  }
  li + li {
    border-top: 1px solid var(--border);
  }
  .item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--s-3);
    width: 100%;
    padding: var(--s-3) var(--s-5);
    border: 0;
    background: transparent;
    text-align: left;
  }
  .item:hover {
    background: var(--surface-2);
  }
  .item.current {
    box-shadow: inset 3px 0 0 var(--accent);
    background: var(--surface-2);
  }
  .item:focus-visible {
    outline-offset: -2px;
  }
  .main {
    display: grid;
    gap: 1px;
    min-width: 0;
  }
  .title {
    font-weight: 600;
    font-size: var(--fs-sm);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .meta {
    font-size: var(--fs-xs);
    color: var(--text-3);
  }
  .side {
    display: inline-flex;
    align-items: center;
    gap: var(--s-2);
    flex: none;
  }
  .ago {
    font-size: var(--fs-xs);
    color: var(--text-3);
  }
  @media (max-width: 720px) {
    .msg {
      padding: 0 var(--s-4) var(--s-4);
    }
    .item {
      padding: var(--s-3) var(--s-4);
      min-height: 52px;
    }
    .more {
      padding: var(--s-3) var(--s-4);
    }
  }
</style>
