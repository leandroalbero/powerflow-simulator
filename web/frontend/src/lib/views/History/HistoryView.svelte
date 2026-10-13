<script lang="ts">
  import { onDestroy, onMount, tick } from 'svelte';
  import { errorMessage, isUnavailable, listRuns, listSweeps } from '../../api/client';
  import { capabilities, setCapability, strategyById, strategyName } from '../../stores/app';
  import Card from '../../ui/Card.svelte';
  import Button from '../../ui/Button.svelte';
  import Icon from '../../ui/Icon.svelte';
  import Select from '../../ui/Select.svelte';
  import { fmtRange } from '../../ui/format';
  import type { RunListItem, SweepListItem } from '../../types';
  import RunRow from './RunRow.svelte';
  import SweepList from './SweepList.svelte';
  import { STATUS_LABEL, digestRun, runTitle } from './history';

  type Load = 'loading' | 'ready' | 'error' | 'unavailable';

  /** The API has no paging; these are its own maximum page sizes, so everything it keeps can be reached. */
  const RUN_LIMIT = 500;
  const SWEEP_LIMIT = 200;
  const PAGE = 20;
  const SWEEP_PAGE = 10;

  let runs: RunListItem[] = [];
  let runsState: Load = 'loading';
  let runsError = '';
  let sweeps: SweepListItem[] = [];
  let sweepsState: Load = 'loading';

  let query = '';
  let status = 'all';
  let digest: string | null = null;
  let now = Date.now();

  const STATUS_OPTIONS = [
    { value: 'all', label: 'All statuses' },
    ...(['completed', 'running', 'failed', 'cancelled'] as const).map((s) => ({ value: s, label: STATUS_LABEL[s] })),
  ];

  let inflight = false;
  async function loadRuns(quiet = false) {
    if (inflight) return;
    inflight = true;
    if (!quiet && !runs.length) runsState = 'loading';
    try {
      runs = await listRuns(RUN_LIMIT);
      runsState = 'ready';
      setCapability('runs', 'available');
    } catch (err) {
      if (isUnavailable(err)) {
        runsState = 'unavailable';
        setCapability('runs', 'unavailable');
      } else if (!quiet || !runs.length) {
        runsState = 'error';
        runsError = errorMessage(err);
      }
    } finally {
      inflight = false;
    }
  }

  async function loadSweeps() {
    try {
      sweeps = await listSweeps(SWEEP_LIMIT);
      sweepsState = 'ready';
    } catch (err) {
      sweepsState = isUnavailable(err) ? 'unavailable' : 'error';
    }
  }

  function reload() {
    void loadRuns();
    void loadSweeps();
  }

  let clock: ReturnType<typeof setInterval> | null = null;
  let poll: ReturnType<typeof setInterval> | null = null;
  const onVisible = () => document.visibilityState === 'visible' && reload();

  onMount(() => {
    reload();
    clock = setInterval(() => (now = Date.now()), 30_000);
    document.addEventListener('visibilitychange', onVisible);
  });
  onDestroy(() => {
    if (clock) clearInterval(clock);
    if (poll) clearInterval(poll);
    document.removeEventListener('visibilitychange', onVisible);
  });

  // Follow running runs and sweeps until they finish, without polling an idle list.
  $: anyRunning = runs.some((r) => r.status === 'running') || sweeps.some((s) => s.status === 'running');
  $: syncPoll(anyRunning);
  function syncPoll(on: boolean) {
    if (on && !poll) {
      poll = setInterval(() => {
        void loadRuns(true);
        if (sweeps.some((s) => s.status === 'running')) void loadSweeps();
      }, 3000);
    } else if (!on && poll) {
      clearInterval(poll);
      poll = null;
    }
  }

  const nameOf = (id: string) => $strategyById.get(id)?.name ?? strategyName(id);

  function haystack(r: RunListItem): string {
    const d = digestRun(r);
    return [
      runTitle(r, d.strategyIds, nameOf),
      r.label ?? '',
      ...d.strategyIds.map(nameOf),
      fmtRange(r.date_range.start, r.date_range.end),
      r.date_range.start,
      r.date_range.end,
      r.config_digest,
      r.run_id,
    ]
      .join(' ')
      .toLowerCase();
  }

  $: terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  $: filtered = runs.filter((r) => {
    if (status !== 'all' && r.status !== status) return false;
    if (digest && r.config_digest !== digest) return false;
    if (!terms.length) return true;
    const h = haystack(r);
    return terms.every((t) => h.includes(t));
  });
  $: filtering = terms.length > 0 || status !== 'all' || digest != null;
  $: digestCount = digest ? runs.filter((r) => r.config_digest === digest).length : 0;

  let shownRuns = PAGE;
  let shownSweeps = SWEEP_PAGE;
  // A new search or filter starts from the top of its matches again.
  $: resetPage(terms, status, digest);
  function resetPage(..._deps: unknown[]) {
    shownRuns = PAGE;
  }
  $: visible = filtered.slice(0, shownRuns);

  const count = (n: number, limit: number, one: string, many: string) =>
    `${n >= limit ? `${n}+` : n} ${n === 1 ? one : many}`;

  let sweepsCard: HTMLElement;
  function jumpToSweeps() {
    sweepsCard?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    // Focus the card's heading so screen readers announce where the jump landed.
    const h = sweepsCard?.querySelector<HTMLElement>('h2');
    if (h) {
      h.tabIndex = -1;
      h.style.outline = 'none';
      h.focus({ preventScroll: true });
    }
  }

  // The Show more button can disappear with the last page; keep keyboard focus on the first new row.
  let runsList: HTMLElement;
  let sweepsWrap: HTMLElement;
  async function showMore(kind: 'runs' | 'sweeps') {
    const from = kind === 'runs' ? shownRuns : shownSweeps;
    if (kind === 'runs') shownRuns += PAGE;
    else shownSweeps += SWEEP_PAGE;
    await tick();
    const root = kind === 'runs' ? runsList : sweepsWrap;
    root?.querySelectorAll<HTMLElement>('li a.title')[from]?.focus();
  }

  function clearFilters() {
    query = '';
    status = 'all';
    digest = null;
  }

  function toggleDigest(e: CustomEvent<string>) {
    digest = digest === e.detail ? null : e.detail;
  }

  function onDeleted(e: CustomEvent<string>) {
    runs = runs.filter((r) => r.run_id !== e.detail);
  }

  $: sweepCap = $capabilities.sweep;
  $: runCount = runsState === 'ready' && runs.length ? count(runs.length, RUN_LIMIT, 'run', 'runs') : null;
  $: sweepCount = sweepsState === 'ready' && sweeps.length ? count(sweeps.length, SWEEP_LIMIT, 'comparison', 'comparisons') : null;
</script>

<div class="view">
  <header class="head">
    <div>
      <h1>History</h1>
      <p class="meta">
        {#if runCount}
          {runCount}{#if sweepCount}, <button type="button" class="linkish jump" on:click={jumpToSweeps}>{sweepCount}</button>{/if}, newest first
        {:else}
          Past simulations and comparisons
        {/if}
      </p>
    </div>
    <Button variant="ghost" size="sm" icon="refresh" on:click={reload} loading={inflight}>Refresh</Button>
  </header>

  <Card pad={false}>
    <div class="toolbar">
      <div class="search">
        <label for="history-search" class="sr-only">Search runs</label>
        <span class="search-icon" aria-hidden="true">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="6.5" /><path d="m20 20-4.2-4.2" /></svg>
        </span>
        <input
          id="history-search"
          type="search"
          placeholder="Search label, strategy, date or settings code"
          autocomplete="off"
          spellcheck="false"
          bind:value={query}
        />
      </div>
      <div class="status-filter">
        <Select label="Status" hideLabel bind:value={status} options={STATUS_OPTIONS} />
      </div>
      {#if digest}
        <button type="button" class="digest-filter" on:click={() => (digest = null)} aria-label="Stop filtering by settings {digest}">
          Settings <code>{digest}</code>
          <span class="n">{digestCount} {digestCount === 1 ? 'run' : 'runs'}</span>
          <Icon name="x" size={13} stroke={2.2} />
        </button>
      {/if}
    </div>

    {#if runsState === 'loading'}
      <ul class="skeleton" aria-label="Loading runs" aria-busy="true">
        {#each [0, 1, 2, 3] as i (i)}
          <li><span style="width: {38 - i * 4}%" /><span style="width: 14%" /><span style="width: 12%" /><span style="width: 16%" /></li>
        {/each}
      </ul>
    {:else if runsState === 'unavailable'}
      <div class="empty">
        <p class="e-title">Run history needs a backend update</p>
        <p class="e-text">This server doesn't keep past runs yet. Results stay visible on Simulate until you start another run.</p>
      </div>
    {:else if runsState === 'error'}
      <div class="empty" role="alert">
        <p class="e-title">Couldn't load the run history</p>
        <p class="e-text">{runsError}</p>
        <Button size="sm" icon="refresh" on:click={() => loadRuns()}>Try again</Button>
      </div>
    {:else if !runs.length}
      <div class="empty">
        <span class="e-icon" aria-hidden="true"><Icon name="clock" size={22} /></span>
        <p class="e-title">No runs yet</p>
        <p class="e-text">
          Every simulation is saved here with its period and settings, so you can reopen it, run it again or export it as CSV.
        </p>
        <a class="cta" href="#/simulate">Go to Simulate</a>
      </div>
    {:else}
      <div class="cols" aria-hidden="true">
        <span>Run</span><span>Period</span><span>Created</span><span>Status</span><span>Best strategy</span>
        <span class="r">Savings</span><span>Settings</span><span />
      </div>
      {#if filtered.length}
        <ul class="runs" aria-label="Runs" bind:this={runsList}>
          {#each visible as item (item.run_id)}
            <RunRow {item} {now} digestActive={digest === item.config_digest} on:deleted={onDeleted} on:digest={toggleDigest} />
          {/each}
        </ul>
      {:else}
        <div class="empty">
          <p class="e-title">No runs match these filters</p>
          <Button size="sm" on:click={clearFilters}>Clear filters</Button>
        </div>
      {/if}
      {#if filtered.length > visible.length || (filtering && filtered.length)}
        <div class="foot">
          <span>
            {#if filtering}Showing {visible.length} of {filtered.length} matching runs.{:else}Showing the newest {visible.length} of {filtered.length} runs.{/if}
            {#if filtering}<button type="button" class="linkish" on:click={clearFilters}>Clear filters</button>{/if}
          </span>
          {#if filtered.length > visible.length}
            <Button size="sm" variant="secondary" on:click={() => showMore('runs')}>
              Show {Math.min(PAGE, filtered.length - visible.length)} more
            </Button>
          {/if}
        </div>
      {/if}
    {/if}
  </Card>

  {#if sweepCap !== 'unavailable' && sweepsState !== 'unavailable'}
    <div class="sweeps" bind:this={sweepsCard}>
    <Card title="Comparisons" subtitle="Saved from the Compare tab: tariffs × battery sizes × strategies" pad={false}>
      {#if sweepsState === 'loading'}
        <p class="pad muted">Loading comparisons…</p>
      {:else if sweepsState === 'error'}
        <p class="pad muted">Couldn't load comparisons. <button type="button" class="linkish" on:click={loadSweeps}>Try again</button></p>
      {:else if !sweeps.length}
        <p class="pad muted">No comparisons yet. <a href="#/compare">Build one in Compare</a> to see tariffs and battery sizes side by side.</p>
      {:else}
        <div bind:this={sweepsWrap}><SweepList items={sweeps.slice(0, shownSweeps)} {now} /></div>
        {#if sweeps.length > shownSweeps}
          <div class="foot">
            <span>Showing the newest {shownSweeps} of {sweeps.length} comparisons.</span>
            <Button size="sm" variant="secondary" on:click={() => showMore('sweeps')}>
              Show {Math.min(SWEEP_PAGE, sweeps.length - shownSweeps)} more
            </Button>
          </div>
        {/if}
      {/if}
    </Card>
    </div>
  {/if}
</div>

<style>
  .view {
    display: grid;
    gap: var(--s-4);
    /* Shared by the header row and every run row so the columns line up. */
    /* The last track is fixed: as `auto` it sized to an empty header cell but to Open + menu in rows, so every fr column drifted. */
    --history-cols: minmax(0, 2.1fr) minmax(0, 1.65fr) minmax(0, 1.1fr) minmax(0, 1fr) minmax(0, 1.5fr) minmax(0, 1fr) minmax(0, 0.95fr) 96px;
  }
  .head {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: var(--s-3);
    margin-top: var(--s-2);
  }
  h1 {
    font-size: var(--fs-xl);
    font-weight: 700;
    letter-spacing: -0.02em;
  }
  .meta {
    margin-top: 2px;
    color: var(--text-3);
    font-size: var(--fs-sm);
  }

  .toolbar {
    display: flex;
    align-items: center;
    gap: var(--s-2);
    flex-wrap: wrap;
    padding: var(--s-4) var(--s-5);
    border-bottom: 1px solid var(--border);
  }
  .search {
    position: relative;
    flex: 1 1 280px;
    max-width: 440px;
  }
  .search-icon {
    position: absolute;
    left: 11px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--text-3);
    display: flex;
  }
  input[type='search'] {
    width: 100%;
    height: var(--control-h);
    padding: 0 var(--s-3) 0 34px;
    border: 1px solid var(--border-strong);
    border-radius: var(--r-md);
    background: var(--surface);
    font-size: var(--fs-md);
  }
  input[type='search']::placeholder {
    color: var(--text-3);
  }
  input[type='search']:hover {
    border-color: var(--text-3);
  }
  .status-filter {
    width: 160px;
  }
  .digest-filter {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: var(--control-h);
    padding: 0 var(--s-3);
    border-radius: var(--r-pill);
    border: 1px solid var(--accent);
    background: var(--accent-soft);
    color: var(--accent);
    font-size: var(--fs-sm);
    font-weight: 550;
  }
  .digest-filter code {
    font-family: var(--font-mono);
    font-size: 12px;
  }
  .digest-filter .n {
    color: var(--text-2);
    font-weight: 500;
  }

  .cols {
    display: grid;
    grid-template-columns: var(--history-cols);
    gap: var(--s-4);
    padding: 10px var(--s-5);
    border-bottom: 1px solid var(--border);
    font-size: var(--fs-xs);
    font-weight: 600;
    color: var(--text-3);
    white-space: nowrap;
  }
  .cols .r {
    text-align: right;
  }

  .empty {
    display: grid;
    justify-items: center;
    text-align: center;
    gap: var(--s-2);
    padding: var(--s-7) var(--s-5);
  }
  .e-icon {
    display: grid;
    place-items: center;
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: var(--surface-3);
    color: var(--text-2);
    margin-bottom: var(--s-1);
  }
  .e-title {
    font-weight: 650;
    font-size: var(--fs-lg);
  }
  .e-text {
    color: var(--text-2);
    max-width: 52ch;
    margin-bottom: var(--s-2);
  }
  .cta {
    display: inline-flex;
    align-items: center;
    height: var(--control-h);
    padding: 0 var(--s-4);
    border-radius: var(--r-md);
    background: var(--accent);
    color: var(--text-on-accent);
    font-weight: 550;
    text-decoration: none;
  }
  .cta:hover {
    background: var(--accent-hover);
  }
  .foot {
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
  .sweeps {
    scroll-margin-top: calc(var(--header-h) + var(--s-3));
    border-radius: var(--r-lg);
  }
  .jump {
    color: var(--accent);
  }
  .linkish {
    border: 0;
    background: transparent;
    padding: 0;
    color: var(--accent);
    font: inherit;
    text-decoration: underline;
    text-underline-offset: 2px;
  }
  .pad {
    padding: 0 var(--s-5) var(--s-5);
    font-size: var(--fs-sm);
  }

  .skeleton li {
    display: flex;
    gap: var(--s-6);
    padding: 20px var(--s-5);
    border-bottom: 1px solid var(--border);
  }
  .skeleton span {
    height: 12px;
    border-radius: 6px;
    background: var(--surface-3);
    animation: shimmer 1.4s ease-in-out infinite;
  }
  @keyframes shimmer {
    50% {
      opacity: 0.45;
    }
  }

  @media (max-width: 960px) {
    .cols {
      display: none;
    }
  }
  @media (max-width: 720px) {
    .toolbar {
      padding: var(--s-3) var(--s-4);
    }
    .search {
      flex-basis: 100%;
      max-width: none;
    }
    .status-filter {
      flex: 1;
      width: auto;
    }
    .pad {
      padding: 0 var(--s-4) var(--s-4);
    }
    .foot {
      padding: var(--s-3) var(--s-4);
    }
  }
</style>
