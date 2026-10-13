<script lang="ts">
  import { sweepExportUrl } from '../../api/client';
  import Icon from '../../ui/Icon.svelte';
  import { dayCount, fmtAgo, fmtDateTime, fmtRange } from '../../ui/format';
  import type { SweepListItem } from '../../types';
  import { STATUS_LABEL, hashHref } from './history';

  export let items: SweepListItem[];
  export let now: number;
</script>

<ul class="list">
  {#each items as s (s.sweep_id)}
    {@const days = dayCount(s.date_range.start, s.date_range.end)}
    {@const title = s.label?.trim() || fmtRange(s.date_range.start, s.date_range.end)}
    {@const href = hashHref('compare', 'sweep', s.sweep_id)}
    <li class="row" aria-label={title}>
      <div class="main">
        <a class="title" {href} title="Open in Compare">{title}</a>
        <span class="sub num">{s.total} {s.total === 1 ? 'case' : 'cases'}</span>
      </div>
      <div class="cell">
        <span class="v num">{fmtRange(s.date_range.start, s.date_range.end)}</span>
        <span class="s num">{days} {days === 1 ? 'day' : 'days'}</span>
      </div>
      <div class="cell">
        <span class="v">{fmtAgo(s.created_at, now)}</span>
        <span class="s num">{fmtDateTime(s.created_at, true)}</span>
      </div>
      <div class="cell status">
        <span class="state {s.status}"><span class="mark" aria-hidden="true" />{STATUS_LABEL[s.status]}</span>
      </div>
      <div class="actions">
        <a class="btn" {href}>Open in Compare</a>
        <a
          class="icon-btn"
          href={sweepExportUrl(s.sweep_id)}
          download
          aria-label="Download CSV for {title}"
          title="Download CSV"
        >
          <Icon name="download" size={16} />
        </a>
      </div>
    </li>
  {/each}
</ul>

<style>
  .row {
    display: grid;
    grid-template-columns: minmax(0, 2.2fr) minmax(0, 1.3fr) minmax(0, 1.2fr) minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--s-4);
    padding: 12px var(--s-5);
    border-bottom: 1px solid var(--border);
  }
  .row:last-child {
    border-bottom: 0;
  }
  .main,
  .cell {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
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
  .sub,
  .s {
    font-size: var(--fs-xs);
    color: var(--text-3);
    white-space: nowrap;
  }
  .v {
    font-size: var(--fs-sm);
    white-space: nowrap;
  }
  .state {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: var(--fs-sm);
    color: var(--text-2);
  }
  .mark {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--text-3);
  }
  .cancelled .mark {
    background: transparent;
    box-shadow: inset 0 0 0 1.5px var(--text-3);
  }
  .completed .mark {
    background: var(--positive);
  }
  .failed {
    color: var(--negative);
  }
  .failed .mark {
    background: var(--negative);
  }
  .running {
    color: var(--accent);
    font-weight: 600;
  }
  .running .mark {
    background: var(--accent);
  }
  .actions {
    display: flex;
    align-items: center;
    gap: var(--s-1);
    justify-content: flex-end;
  }
  .btn {
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
    white-space: nowrap;
  }
  .btn:hover {
    background: var(--surface-2);
  }
  .icon-btn {
    display: inline-grid;
    place-items: center;
    width: 30px;
    height: 30px;
    border-radius: var(--r-md);
    color: var(--text-2);
  }
  .icon-btn:hover {
    background: var(--surface-3);
    color: var(--text);
  }
  @media (max-width: 960px) {
    .row {
      position: relative;
      grid-template-columns: 1fr 1fr;
      gap: var(--s-2) var(--s-4);
      padding: var(--s-4);
    }
    .main {
      grid-column: 1 / -1;
      padding-right: 110px;
    }
    .title {
      white-space: normal;
    }
    .status {
      position: absolute;
      top: var(--s-4);
      right: var(--s-4);
    }
    .actions {
      grid-column: 1 / -1;
      justify-content: flex-start;
      gap: var(--s-2);
      margin-top: var(--s-1);
    }
    .btn {
      height: 40px;
      flex: 1;
      justify-content: center;
    }
    .icon-btn {
      width: 40px;
      height: 40px;
      border: 1px solid var(--border-strong);
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
