<script lang="ts">
  import { onDestroy } from 'svelte';
  import { strategyById } from '../../stores/app';
  import Badge from '../../ui/Badge.svelte';
  import Button from '../../ui/Button.svelte';
  import Card from '../../ui/Card.svelte';
  import Icon from '../../ui/Icon.svelte';
  import Segmented from '../../ui/Segmented.svelte';
  import StrategyDot from '../../ui/StrategyDot.svelte';
  import Table from '../../ui/Table.svelte';
  import { DASH, eur, eurSigned, num } from '../../ui/format';
  import { costPctMeaningful, realVsSim } from './analysis';
  import { comparing, runningId, session, shownId, stopCompare, type Entry } from './state';

  let rankBy: 'soc' | 'cost' = 'soc';
  let stopping = false;
  $: if (!$comparing) stopping = false;

  let now = Date.now();
  let tick: ReturnType<typeof setInterval> | null = null;
  $: if ($comparing && !tick) tick = setInterval(() => (now = Date.now()), 1000);
  else if (!$comparing && tick) {
    clearInterval(tick);
    tick = null;
  }
  onDestroy(() => tick && clearInterval(tick));

  const REF = 'no_battery';
  const name = (id: string) => $strategyById.get(id)?.name ?? id.replace(/_/g, ' ');

  $: ids = $session?.compareIds ?? [];
  $: entries = $session?.entries ?? {};
  $: done = ids.filter((id) => entries[id]?.status === 'done' && id !== REF);
  $: score = (id: string): number => {
    const r = entries[id]?.result;
    if (!r) return Infinity;
    return rankBy === 'soc' ? r.metrics.soc_rmse_pct : Math.abs(realVsSim(r).eur);
  };
  $: ranked = [...done].sort((a, b) => score(a) - score(b));
  $: rest = ids.filter((id) => !done.includes(id) && id !== REF);
  $: rows = [...ranked, ...(ids.includes(REF) ? [REF] : []), ...rest];

  $: bySoc = [...done].sort((a, b) => (entries[a]?.result?.metrics.soc_rmse_pct ?? 0) - (entries[b]?.result?.metrics.soc_rmse_pct ?? 0))[0] ?? null;
  $: byCost = [...done].sort((a, b) => Math.abs(realVsSim(entries[a].result!).eur) - Math.abs(realVsSim(entries[b].result!).eur))[0] ?? null;
  $: refResult = entries[REF]?.status === 'done' ? entries[REF].result ?? null : null;
  $: finished = ids.filter((id) => ['done', 'failed', 'skipped'].includes(entries[id]?.status ?? '')).length;
  $: running = $runningId && ids.includes($runningId) ? $runningId : null;
  $: runningSecs = running && entries[running]?.startedAt ? Math.max(0, Math.round((now - (entries[running].startedAt ?? now)) / 1000)) : 0;

  function cell(e: Entry | undefined) {
    return e?.result ?? null;
  }
  const signedPct = (v: number) => `${v >= 0 ? '+' : '−'}${num(Math.abs(v), 1)}%`;
  function secs(ms: number | undefined): string {
    if (ms == null) return DASH;
    return ms < 1000 ? `${num(ms / 1000, 1)} s` : ms < 60_000 ? `${num(ms / 1000, ms < 10_000 ? 1 : 0)} s` : `${Math.floor(ms / 60_000)} min ${Math.round((ms % 60_000) / 1000)} s`;
  }
  function stop() {
    stopping = true;
    stopCompare();
  }
</script>

<Card title="Which strategy behaves most like your system?" pad={false}>
  <svelte:fragment slot="actions">
    {#if $comparing}
      <Button size="sm" variant="secondary" icon="stop" loading={stopping} on:click={stop}>{stopping ? 'Stopping after this one' : 'Stop'}</Button>
    {:else if done.length >= 2}
      <Segmented
        label="Rank by"
        bind:value={rankBy}
        options={[
          { value: 'soc', label: 'Battery behaviour', short: 'Behaviour' },
          { value: 'cost', label: 'Cost', short: 'Cost' },
        ]}
      />
    {/if}
  </svelte:fragment>

  <div class="summary">
    {#if $comparing}
      <p class="progress num" role="status">
        <span class="spin" aria-hidden="true" />
        {finished} of {ids.length} done{#if running} · running {name(running)} · {runningSecs} s{/if}
      </p>
      <div class="bar" aria-hidden="true"><span style="width: {(finished / Math.max(1, ids.length)) * 100}%" /></div>
    {/if}
    {#if done.length >= 2 && bySoc && byCost}
      {@const soc = entries[bySoc].result}
      {@const cst = entries[byCost].result}
      <ul class="facts">
        {#if soc}
          <li>
            <span class="k">Most similar battery behaviour</span>
            <span class="v"><StrategyDot id={bySoc} size={9} /><strong>{name(bySoc)}</strong><span class="muted num">charge level {num(soc.metrics.soc_rmse_pct, 0)} pp apart on average</span></span>
          </li>
        {/if}
        {#if cst}
          {@const d = realVsSim(cst)}
          <li>
            <span class="k">Closest on cost</span>
            <span class="v"><StrategyDot id={byCost} size={9} /><strong>{name(byCost)}</strong><span class="muted num">{eur(cst.metrics.sim_cost)} simulated vs {eur(cst.metrics.measured_cost)} measured{costPctMeaningful(cst) && cst.metrics.cost_error_pct != null ? ` (${num(Math.abs(cst.metrics.cost_error_pct), 0)}% ${d.eur >= 0 ? 'above' : 'below'} reality)` : ''}</span></span>
          </li>
        {/if}
        {#if refResult}
          {@const saved = refResult.metrics.sim_cost - refResult.metrics.measured_cost}
          <li>
            <span class="k">Your battery vs none</span>
            <span class="v">
              {#if Math.abs(saved) < 0.05}
                <span class="num">About the same as a simulated house without a battery</span>
              {:else if saved > 0}
                <strong class="pos num">Saved about {eur(saved)}</strong><span class="muted num">against {eur(refResult.metrics.sim_cost)} simulated without a battery</span>
              {:else}
                <strong class="neg num">{eur(-saved)} more</strong><span class="muted num">than {eur(refResult.metrics.sim_cost)} simulated without a battery</span>
              {/if}
            </span>
          </li>
        {/if}
      </ul>
    {:else if !$comparing && done.length < 2}
      <p class="muted">Fewer than two strategies finished, so there is nothing to rank yet.</p>
    {/if}
  </div>

  <div class="wide">
  <Table label="Strategies compared with your measured data">
    <table>
      <thead>
        <tr>
          <th scope="col">Strategy</th>
          <th scope="col" title="Cost the simulator gives this strategy for the same days">Simulated cost</th>
          <th scope="col" title="Measured cost minus simulated cost: negative means your real system paid less than this strategy would have">Real − simulated</th>
          <th scope="col" title="(simulated − measured) / measured cost">Cost error</th>
          <th scope="col" title="(simulated − measured) / measured grid import">Import error</th>
          <th scope="col" title="Mean absolute daily import error">Daily MAE<span class="unit">kWh</span></th>
          <th scope="col" class:sorted={rankBy === 'soc'} title="Root-mean-square gap between measured and simulated charge level">Charge error<span class="unit">pp</span></th>
          <th scope="col">Took</th>
        </tr>
      </thead>
      <tbody>
        {#each rows as id (id)}
          {@const e = entries[id]}
          {@const r = cell(e)}
          {@const rank = ranked.indexOf(id)}
          <tr class:ref={id === REF} class:shown={$shownId === id} class:failed={e?.status === 'failed'}>
            <td class="name-cell">
              <span class="name">
                {#if rank >= 0}<span class="rank num" aria-label="Rank {rank + 1}">{rank + 1}</span>{:else}<span class="rank" />{/if}
                <StrategyDot id={id} />
                {#if r}
                  <button type="button" class="link" aria-pressed={$shownId === id} on:click={() => shownId.set(id)} title="Show this strategy's charts">{name(id)}</button>
                {:else}
                  <span>{name(id)}</span>
                {/if}
                {#if id === REF}<Badge>Reference</Badge>{/if}
                {#if id === bySoc && rank >= 0 && done.length >= 2}<Badge tone="accent">Most similar</Badge>{/if}
              </span>
            </td>
            {#if r}
              {@const d = realVsSim(r)}
              <td class="strong">{eur(r.metrics.sim_cost)}</td>
              <td class:pos={d.eur > 0.005} class:neg={d.eur < -0.005}>{eurSigned(-d.eur)}</td>
              <td>{costPctMeaningful(r) ? signedPct(r.metrics.cost_error_pct) : DASH}</td>
              <td>{signedPct(r.metrics.import_error_pct)}</td>
              <td>{num(r.metrics.daily_import_mae_kwh, 2)}</td>
              <td class:strong={rankBy === 'soc'}>{id === REF ? DASH : num(r.metrics.soc_rmse_pct, 1)}</td>
              <td class="muted">{secs(e?.ms)}</td>
            {:else}
              <td colspan="7" class="msg-cell">
                {#if e?.status === 'failed'}
                  <span class="err"><Icon name="alert" size={14} />{e.error ?? 'Failed'}</span>
                {:else if e?.status === 'running'}
                  <span class="pending"><span class="spin" aria-hidden="true" />Running…</span>
                {:else if e?.status === 'skipped'}
                  <span class="muted">Skipped (stopped)</span>
                {:else}
                  <span class="muted">Queued</span>
                {/if}
              </td>
            {/if}
          </tr>
        {/each}
      </tbody>
    </table>
  </Table>
  </div>
  <ol class="list" aria-label="Strategies compared with your measured data">
    {#each rows as id (id)}
      {@const e = entries[id]}
      {@const r = cell(e)}
      {@const rank = ranked.indexOf(id)}
      <li class:ref={id === REF} class:shown={$shownId === id}>
        <div class="lhead">
          <span class="rank num">{rank >= 0 ? rank + 1 : ''}</span>
          <StrategyDot id={id} />
          {#if r}
            <button type="button" class="link" aria-pressed={$shownId === id} on:click={() => shownId.set(id)}>{name(id)}</button>
          {:else}
            <span class="lname">{name(id)}</span>
          {/if}
          {#if id === REF}<Badge>Reference</Badge>{/if}
        </div>
        {#if r}
          {@const d = realVsSim(r)}
          <dl class="lstats num">
            <div><dt>Simulated</dt><dd>{eur(r.metrics.sim_cost)}</dd></div>
            <div><dt>Real − sim</dt><dd class:pos={d.eur > 0.005} class:neg={d.eur < -0.005}>{eurSigned(-d.eur)}</dd></div>
            <div><dt>Charge error</dt><dd>{id === REF ? DASH : `${num(r.metrics.soc_rmse_pct, 1)} pp`}</dd></div>
            <div><dt>Import error</dt><dd>{signedPct(r.metrics.import_error_pct)}</dd></div>
          </dl>
        {:else if e?.status === 'failed'}
          <p class="err lmsg"><Icon name="alert" size={14} />{e.error ?? 'Failed'}</p>
        {:else}
          <p class="muted lmsg">{e?.status === 'running' ? 'Running…' : e?.status === 'skipped' ? 'Skipped (stopped)' : 'Queued'}</p>
        {/if}
      </li>
    {/each}
  </ol>
  <p class="foot">
    Ranked by {rankBy === 'soc' ? 'how closely the simulated battery charge follows the measured one' : 'how close the simulated cost is to the measured cost'}. Select a strategy to see its charts.
    Each one is backtested from the first measured charge level with the server’s default tariff.
  </p>
</Card>

<style>
  .summary {
    padding: 0 var(--s-5) var(--s-4);
    display: grid;
    gap: var(--s-3);
  }
  .progress {
    display: inline-flex;
    align-items: center;
    gap: var(--s-2);
    font-size: var(--fs-sm);
    color: var(--text-2);
  }
  .bar {
    height: 4px;
    border-radius: 2px;
    background: var(--surface-3);
    overflow: hidden;
  }
  .bar span {
    display: block;
    height: 100%;
    background: var(--accent);
    transition: width 300ms var(--ease);
  }
  .facts {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: var(--s-3);
  }
  .facts li {
    display: grid;
    gap: 4px;
    align-content: start;
    padding: var(--s-3) var(--s-4);
    border-radius: var(--r-md);
    background: var(--surface-2);
    border: 1px solid var(--border);
    min-width: 0;
  }
  .k {
    font-size: var(--fs-xs);
    font-weight: 600;
    color: var(--text-3);
  }
  .v {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 2px 8px;
    font-size: var(--fs-sm);
  }
  .v strong {
    font-weight: 650;
  }
  .v .muted {
    flex-basis: 100%;
    font-size: var(--fs-xs);
  }
  .pos {
    color: var(--positive);
  }
  .neg {
    color: var(--negative);
  }
  .unit {
    margin-left: 3px;
    font-weight: 500;
  }
  th.sorted {
    color: var(--text);
  }
  .name {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
    color: var(--text);
  }
  .rank {
    display: inline-block;
    width: 1.6ch;
    text-align: right;
    color: var(--text-3);
    font-weight: 600;
    font-size: var(--fs-xs);
  }
  .link {
    border: 0;
    background: transparent;
    padding: 0;
    font: inherit;
    color: inherit;
    text-align: left;
    text-decoration: underline;
    text-decoration-color: transparent;
    text-underline-offset: 3px;
  }
  .link:hover {
    text-decoration-color: var(--text-3);
  }
  td {
    color: var(--text-2);
  }
  td.strong {
    color: var(--text);
    font-weight: 650;
  }
  td.pos {
    color: var(--positive);
    font-weight: 600;
  }
  td.neg {
    color: var(--negative);
    font-weight: 600;
  }
  td.muted {
    color: var(--text-3);
  }
  tr.ref td {
    background: var(--surface-2);
  }
  tr.ref .name {
    color: var(--text-2);
    font-weight: 550;
  }
  tr.shown td:first-child {
    box-shadow: inset 3px 0 0 var(--accent);
  }
  tbody tr:hover td {
    background: var(--surface-2);
  }
  .msg-cell {
    text-align: left !important;
  }
  .err {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--negative);
    white-space: normal;
  }
  .pending {
    display: inline-flex;
    align-items: center;
    gap: var(--s-2);
    color: var(--text-2);
  }
  .spin {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    border: 2px solid var(--accent);
    border-right-color: transparent;
    animation: spin 0.8s linear infinite;
    flex: none;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
  .foot {
    padding: var(--s-3) var(--s-5) var(--s-4);
    border-top: 1px solid var(--border);
    font-size: var(--fs-xs);
    color: var(--text-3);
  }
  .list {
    display: none;
  }
  @media (max-width: 720px) {
    .wide {
      display: none;
    }
    .list {
      display: block;
      border-top: 1px solid var(--border);
    }
    .list li {
      padding: var(--s-2) var(--s-4) var(--s-3);
      border-bottom: 1px solid var(--border);
    }
    .list li:last-child {
      border-bottom: 0;
    }
    .list li.ref {
      background: var(--surface-2);
    }
    .list li.shown {
      box-shadow: inset 3px 0 0 var(--accent);
    }
    .lhead {
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 600;
    }
    .lstats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 2px var(--s-3);
      margin: 2px 0 0 calc(1.6ch + 26px);
      font-size: var(--fs-sm);
    }
    .lstats div {
      display: flex;
      justify-content: space-between;
      gap: var(--s-2);
    }
    .lstats dt {
      color: var(--text-3);
    }
    .lstats dd {
      margin: 0;
      color: var(--text);
      font-weight: 600;
    }
    .lstats dd.pos {
      color: var(--positive);
    }
    .lstats dd.neg {
      color: var(--negative);
    }
    .lmsg {
      margin: 2px 0 0 calc(1.6ch + 26px);
      font-size: var(--fs-sm);
    }
  }
  @media (max-width: 960px) {
    .facts {
      grid-template-columns: minmax(0, 1fr);
    }
  }
  @media (max-width: 720px) {
    .summary {
      padding: 0 var(--s-4) var(--s-3);
    }
    .foot {
      padding: var(--s-3) var(--s-4);
    }
    .link {
      min-height: 40px;
      display: inline-flex;
      align-items: center;
    }
  }
</style>
