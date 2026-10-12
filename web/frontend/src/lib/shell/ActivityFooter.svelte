<script lang="ts">
  import { activity } from '../stores/toasts';
  import { fmtTime } from '../ui/format';
  import { MOCK_MODE } from '../api/client';
</script>

<footer class="foot">
  <div class="inner">
    <details>
      <summary>Activity <span class="count num">{$activity.length || ''}</span></summary>
      {#if $activity.length}
        <ol class="log">
          {#each $activity as e (e.id)}
            <li class={e.kind}><time class="num">{fmtTime(new Date(e.at))}</time><span>{e.text}</span></li>
          {/each}
        </ol>
      {:else}
        <p class="hint">Nothing yet.</p>
      {/if}
    </details>
    <p class="hint">
      Energy costs in € from the tariff in Settings{MOCK_MODE ? '. Mock mode: new endpoints are simulated in the browser' : ''}.
    </p>
  </div>
</footer>

<style>
  .foot {
    border-top: 1px solid var(--border);
  }
  .inner {
    max-width: var(--content-max);
    margin: 0 auto;
    padding: var(--s-4) var(--gutter) var(--s-6);
    display: flex;
    gap: var(--s-4);
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
  }
  details {
    min-width: 0;
    flex: 1 1 320px;
  }
  summary {
    cursor: pointer;
    color: var(--text-2);
    font-size: var(--fs-sm);
    font-weight: 600;
    width: max-content;
  }
  .count {
    color: var(--text-3);
    font-weight: 500;
  }
  .log {
    margin-top: var(--s-3);
    max-height: 220px;
    overflow: auto;
    display: grid;
    gap: 4px;
    font-size: var(--fs-xs);
  }
  .log li {
    display: flex;
    gap: var(--s-3);
    color: var(--text-2);
  }
  .log time {
    color: var(--text-3);
    flex: none;
  }
  .log .error span {
    color: var(--negative);
  }
  .log .warning span {
    color: var(--warning);
  }
  .foot > .inner > .hint {
    flex: 0 1 auto;
  }
  @media (max-width: 720px) {
    summary {
      padding: 11px 0;
    }
  }
</style>
