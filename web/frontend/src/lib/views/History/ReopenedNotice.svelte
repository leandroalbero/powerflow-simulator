<script lang="ts">
  /*
   * Shown on Simulate for runs reopened from history. `banner` goes above the results;
   * `charts` stands in for the charts when the server no longer has the per-minute data.
   */
  import { openingRun, rerun, run, runBusy } from '../../stores/run';
  import Button from '../../ui/Button.svelte';
  import Icon from '../../ui/Icon.svelte';
  import { fmtAgo, fmtDateTime } from '../../ui/format';

  export let part: 'banner' | 'charts' = 'banner';

  $: r = $run;
  $: opening = $openingRun != null && r?.runId !== $openingRun;
  $: canRerun = !!r?.runId && !$runBusy;

  function again() {
    if (r?.runId) void rerun(r.runId);
  }
</script>

{#if part === 'banner'}
  {#if opening}
    <div class="bar opening" role="status">
      <span class="spinner" aria-hidden="true" />
      <span>Opening run <code>{$openingRun}</code>…</span>
    </div>
  {:else if r?.reopened}
    <div class="bar" role="note">
      <span class="ic" aria-hidden="true"><Icon name="clock" size={16} /></span>
      <p class="txt">
        <span class="lead">{r.label ? r.label : 'Saved run'}</span>
        <span class="sep" aria-hidden="true" />
        <span class="when">
          Created {r.createdAt ? fmtDateTime(r.createdAt, true) : 'earlier'}{#if r.createdAt}{' '}<span class="ago">({fmtAgo(r.createdAt)})</span>{/if}
        </span>
        {#if r.configFromRun === false}
          <span class="warn">The server didn't return this run's settings; the battery capacity shown comes from your current settings.</span>
        {/if}
      </p>
      <div class="acts">
        <a class="back" href="#/history">All runs</a>
        <Button size="sm" icon="refresh" disabled={!canRerun} on:click={again} title="Run again with the same period, strategies and settings">
          Re-run
        </Button>
      </div>
    </div>
  {/if}
{:else if r?.timeseriesExpired}
  <section class="expired" aria-labelledby="expired-title">
    <div class="art" aria-hidden="true">
      <svg viewBox="0 0 120 56" width="120" height="56">
        <path class="axis" d="M4 50H116" />
        <path class="ghost" d="M4 44C20 44 26 14 44 14S64 44 80 40 100 22 116 26" />
      </svg>
    </div>
    <div class="body">
      <h2 id="expired-title">Charts aren't available for this run</h2>
      <p>
        The server keeps minute-by-minute data only for the most recent runs. Costs, the ranking and the monthly figures
        are complete; run it again to see the power and battery charts.
      </p>
    </div>
    <Button variant="primary" icon="refresh" disabled={!canRerun} on:click={again}>Re-run</Button>
  </section>
{/if}

<style>
  .bar {
    display: flex;
    align-items: center;
    gap: var(--s-3);
    padding: 10px var(--s-3) 10px var(--s-4);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    background: var(--surface);
    box-shadow: var(--shadow-1);
    font-size: var(--fs-sm);
    color: var(--text-2);
  }
  .opening {
    color: var(--text-3);
  }
  .ic {
    display: grid;
    place-items: center;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: var(--accent-soft);
    color: var(--accent);
    flex: none;
  }
  .txt {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 2px var(--s-2);
    min-width: 0;
    flex: 1;
  }
  .lead {
    font-weight: 620;
    color: var(--text);
  }
  .sep {
    width: 1px;
    height: 14px;
    background: var(--border-strong);
  }
  .ago {
    color: var(--text-3);
  }
  .warn {
    flex-basis: 100%;
    color: var(--warning);
    font-size: var(--fs-xs);
  }
  .acts {
    display: flex;
    align-items: center;
    gap: var(--s-2);
    flex: none;
  }
  .back {
    font-size: var(--fs-sm);
    font-weight: 550;
    color: var(--text-2);
    text-decoration: none;
    padding: 0 var(--s-2);
  }
  .back:hover {
    color: var(--text);
    text-decoration: underline;
    text-underline-offset: 3px;
  }
  code {
    font-family: var(--font-mono);
    font-size: 12px;
  }
  .spinner {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    border: 2px solid currentColor;
    border-right-color: transparent;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .expired {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    align-items: center;
    gap: var(--s-5);
    padding: var(--s-5);
    border: 1px dashed var(--border-strong);
    border-radius: var(--r-lg);
    background: var(--surface);
  }
  .expired h2 {
    font-size: var(--fs-lg);
    font-weight: 650;
    margin-bottom: 4px;
  }
  .expired p {
    color: var(--text-2);
    font-size: var(--fs-sm);
    max-width: 70ch;
  }
  svg {
    display: block;
    fill: none;
    stroke-linecap: round;
  }
  .axis {
    stroke: var(--border-strong);
    stroke-width: 1;
  }
  .ghost {
    stroke: var(--text-3);
    stroke-width: 2;
    stroke-dasharray: 3 5;
  }

  @media (max-width: 720px) {
    .sep {
      display: none;
    }
    .txt {
      flex-direction: column;
      align-items: flex-start;
    }
    .bar {
      flex-wrap: wrap;
      padding: var(--s-3);
    }
    .acts {
      width: 100%;
    }
    .acts :global(.btn) {
      flex: 1;
    }
    .back {
      display: inline-flex;
      align-items: center;
      height: 40px;
    }
    .expired {
      grid-template-columns: 1fr;
      gap: var(--s-3);
      padding: var(--s-4);
    }
    .art svg {
      width: 96px;
      height: 44px;
    }
  }
  @media (pointer: coarse) {
    .back {
      display: inline-flex;
      align-items: center;
      height: 40px;
    }
  }
</style>
