<script lang="ts">
  import { get } from 'svelte/store';
  import { bootState, strategies, strategyName } from '../../stores/app';
  import Button from '../../ui/Button.svelte';
  import Icon from '../../ui/Icon.svelte';
  import StrategyDot from '../../ui/StrategyDot.svelte';
  import { dayCount, fmtDate, fmtRange } from '../../ui/format';
  import { readResult } from './analysis';
  import CompareCard from './CompareCard.svelte';
  import Controls from './Controls.svelte';
  import DailyCharts from './DailyCharts.svelte';
  import HardwareStrip from './HardwareStrip.svelte';
  import KpiCards from './KpiCards.svelte';
  import SocChart from './SocChart.svelte';
  import { activeRange, busy, compareAll, hardware, period, runSingle, session, shownId, strategyId, tooLong, type Entry } from './state';

  $: coreIds = $strategies.filter((s) => s.group === 'core' && s.available !== false).map((s) => s.id);

  function run() {
    const r = get(activeRange);
    if (!r || tooLong(r)) return;
    void runSingle(r, get(hardware), get(strategyId));
  }

  function compare() {
    const r = get(activeRange);
    if (!r || tooLong(r)) return;
    const picked = get(strategyId);
    // A non-core pick joins the comparison, so it stays on screen and gets ranked against the core set.
    const ids = coreIds.includes(picked) ? coreIds : [picked, ...coreIds];
    void compareAll(r, get(hardware), ids, picked);
  }

  function quick() {
    period.set('last7');
    run();
  }

  $: s = $session;
  $: days = s ? dayCount(s.range.start, s.range.end) : 0;
  $: entry = s && $shownId ? s.entries[$shownId] ?? null : null;
  $: result = entry?.status === 'done' ? entry.result ?? null : null;
  $: noBattery = s?.entries.no_battery?.status === 'done' ? s.entries.no_battery.result ?? null : null;
  // $strategies is read so the text is rebuilt once names load (strategyName itself isn't reactive).
  $: reading = result && $strategies ? readResult(result, strategyName, { days, noBattery }) : null;
  $: doneIds = s ? Object.keys(s.entries).filter((id) => s?.entries[id]?.status === 'done') : [];
  $: switchIds = s?.compareIds.length ? s.compareIds.filter((id) => doneIds.includes(id)) : doneIds;
  $: hwLabel = s?.hardware === 'auto' ? 'auto hardware' : s?.hardware.replace(/_/g, ' ');

  function failure(e: Entry, range: { start: string; end: string }): { title: string; body: string } {
    const msg = e.error ?? 'Unknown error';
    if (e.httpStatus === 422 && /no measured data/i.test(msg))
      return {
        title: 'No measured data for this period',
        body: `The meters recorded no grid readings between ${fmtDate(range.start)} and ${fmtDate(range.end)}, so there is nothing to compare against. Pick a period with data; the data badge in the top bar lists the gaps.`,
      };
    if (e.httpStatus === 422 && /exceeds 1 year/i.test(msg))
      return { title: 'This range is longer than a year', body: 'Backtests cover at most one year. Shorten the range and run it again.' };
    if (e.httpStatus === 422) return { title: 'The backend rejected this backtest', body: msg };
    if (e.httpStatus === 0) return { title: 'Can’t reach the server', body: msg };
    return { title: 'The backtest failed on the server', body: `${msg}${e.httpStatus ? ` (HTTP ${e.httpStatus})` : ''}` };
  }

  let switchEl: HTMLDivElement;
  function onSwitchKey(e: KeyboardEvent) {
    const i = switchIds.indexOf($shownId ?? '');
    let n = -1;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') n = (i + 1) % switchIds.length;
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') n = (i - 1 + switchIds.length) % switchIds.length;
    if (n < 0) return;
    e.preventDefault();
    shownId.set(switchIds[n]);
    switchEl.querySelectorAll<HTMLButtonElement>('button')[n]?.focus();
  }
</script>

<div class="view">
  <Controls onRun={run} onCompare={compare} {coreIds} />

  {#if !s}
    <div class="empty">
      <div class="art" aria-hidden="true">
        <svg viewBox="0 0 220 120" width="220" height="120">
          <path class="grid" d="M10 100H210M10 70H210M10 40H210" />
          <path class="meas" d="M10 82L40 76L70 52L100 34L130 38L160 58L190 74L210 80" />
          <path class="sim" d="M10 86L40 80L70 60L100 40L130 34L160 50L190 70L210 76" />
        </svg>
      </div>
      <div class="text">
        <h1>How close is the simulator to what your house really did?</h1>
        <p>
          A backtest replays a period with the battery limits your system actually had, then lines the simulated grid
          import, cost and battery charge up against the meters. Compare all core strategies to see which one behaves most
          like your real controller.
        </p>
        <div class="cta">
          <Button variant="secondary" size="lg" icon="play" disabled={$busy || $bootState !== 'ready'} on:click={quick}>Backtest last 7 days</Button>
          <span class="hint">Takes a second or two for a week.</span>
        </div>
      </div>
    </div>
  {:else}
    <!-- A session restored after a reload can render before the strategy list; re-render once names arrive. -->
    {#key $strategies}
    <section class="results" aria-labelledby="bt-results-title" aria-busy={$busy}>
      <header class="results-head">
        <div>
          <h1 id="bt-results-title">Results<span class="label-sep" aria-hidden="true">{' · '}</span><span class="run-label">real vs simulated</span></h1>
          <p class="meta num">{fmtRange(s.range.start, s.range.end)} · {days} {days === 1 ? 'day' : 'days'} · {hwLabel}</p>
        </div>
        {#if switchIds.length > 1}
          <div class="switch" role="radiogroup" tabindex="-1" aria-label="Strategy shown" bind:this={switchEl} on:keydown={onSwitchKey}>
            {#each switchIds as id (id)}
              <button type="button" role="radio" aria-checked={$shownId === id} tabindex={$shownId === id ? 0 : -1} class="opt" on:click={() => shownId.set(id)}>
                <StrategyDot {id} size={8} />{strategyName(id)}
              </button>
            {/each}
          </div>
        {/if}
      </header>

      {#if result}
        <HardwareStrip used={result.hardware_used} />
      {/if}

      {#if entry?.status === 'failed'}
        {@const f = failure(entry, s.range)}
        <div class="error" role="alert">
          <Icon name="alert" size={20} />
          <div>
            <h2>{f.title}</h2>
            <p>{f.body}</p>
          </div>
        </div>
      {:else if !result}
        <div class="loading" role="status">
          <span class="spin" aria-hidden="true" />
          Backtesting {strategyName($shownId ?? '')} against {days} {days === 1 ? 'day' : 'days'} of measurements…
        </div>
      {/if}

      {#if result && reading}
        <div class="reading" aria-live="polite">
          <p class="headline">{reading.headline}</p>
          {#if reading.details.length}
            <ul>
              {#each reading.details as d}<li>{d}</li>{/each}
            </ul>
          {/if}
        </div>
        <KpiCards {result} {days} {noBattery} />
      {/if}

      {#if s.compareIds.length}
        <CompareCard />
      {/if}

      {#if result}
        <DailyCharts {result} />
        <SocChart {result} />
      {/if}
    </section>
    {/key}
  {/if}
</div>

<style>
  .view {
    display: grid;
    gap: var(--s-5);
  }
  .results {
    display: grid;
    gap: var(--s-4);
    min-width: 0;
  }
  .results-head {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: var(--s-3);
    flex-wrap: wrap;
    margin-top: var(--s-2);
  }
  h1 {
    font-size: var(--fs-xl);
    font-weight: 700;
    letter-spacing: -0.02em;
  }
  .label-sep {
    color: var(--text-3);
    font-weight: 500;
  }
  .run-label {
    font-weight: 600;
    color: var(--text-2);
  }
  .meta {
    margin-top: 2px;
    color: var(--text-3);
    font-size: var(--fs-sm);
  }
  .switch {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  .opt {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 30px;
    padding: 0 10px;
    border-radius: var(--r-pill);
    border: 1px solid transparent;
    background: transparent;
    color: var(--text-2);
    font-size: var(--fs-xs);
    font-weight: 600;
  }
  .opt:hover {
    background: var(--surface-3);
  }
  .opt[aria-checked='true'] {
    border-color: var(--border-strong);
    background: var(--surface);
    color: var(--text);
    box-shadow: var(--shadow-1);
  }
  .reading {
    padding: var(--s-4) var(--s-5);
    border-radius: var(--r-lg);
    background: var(--accent-soft);
    border: 1px solid color-mix(in srgb, var(--accent) 22%, transparent);
  }
  .headline {
    font-size: var(--fs-lg);
    font-weight: 650;
    letter-spacing: -0.01em;
  }
  .reading ul {
    margin-top: var(--s-2);
    display: grid;
    gap: 4px;
  }
  .reading li {
    position: relative;
    padding-left: 14px;
    color: var(--text-2);
    max-width: 110ch;
  }
  .reading li::before {
    content: '';
    position: absolute;
    left: 2px;
    top: 0.62em;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: var(--accent);
  }
  .error {
    display: flex;
    gap: var(--s-3);
    align-items: flex-start;
    padding: var(--s-4) var(--s-5);
    border-radius: var(--r-lg);
    background: var(--negative-soft);
    color: var(--negative);
  }
  .error :global(.icon) {
    margin-top: 2px;
  }
  .error h2 {
    font-size: var(--fs-lg);
    margin-bottom: 2px;
  }
  .error p {
    color: var(--text-2);
    max-width: 90ch;
  }
  .loading {
    display: flex;
    align-items: center;
    gap: var(--s-3);
    padding: var(--s-6) var(--s-5);
    border-radius: var(--r-lg);
    background: var(--surface-2);
    border: 1px dashed var(--border-strong);
    color: var(--text-2);
  }
  .spin {
    width: 16px;
    height: 16px;
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
  .empty {
    display: grid;
    grid-template-columns: auto minmax(0, 68ch);
    justify-content: center;
    gap: var(--s-6);
    align-items: center;
    padding: var(--s-7) var(--s-6);
    border: 1px dashed var(--border-strong);
    border-radius: var(--r-lg);
  }
  .empty h1 {
    font-weight: 680;
    letter-spacing: -0.015em;
    margin-bottom: var(--s-2);
  }
  .empty p {
    color: var(--text-2);
    max-width: 64ch;
  }
  .cta {
    margin-top: var(--s-5);
    display: flex;
    align-items: center;
    gap: var(--s-4);
    flex-wrap: wrap;
  }
  svg {
    display: block;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  .grid {
    stroke: var(--border);
    stroke-width: 1;
  }
  .meas {
    stroke: var(--text-2);
    stroke-width: 2.5;
  }
  .sim {
    stroke: var(--accent);
    stroke-width: 2.5;
    stroke-dasharray: 6 5;
  }
  @media (pointer: coarse) {
    .opt {
      height: 40px;
    }
  }
  @media (max-width: 720px) {
    .empty {
      grid-template-columns: 1fr;
      padding: var(--s-5) var(--s-4);
      gap: var(--s-4);
    }
    .art svg {
      width: 160px;
      height: 88px;
    }
    .reading {
      padding: var(--s-3) var(--s-4);
    }
    .headline {
      font-size: var(--fs-md);
    }
    .opt {
      height: 40px;
      flex: none;
    }
    .results-head {
      display: grid;
    }
    /* One scrollable row instead of four wrapped rows of pills. */
    .switch {
      flex-wrap: nowrap;
      overflow-x: auto;
      margin: 0 calc(var(--gutter) * -1);
      padding: 2px var(--gutter);
      scrollbar-width: none;
    }
  }
</style>
