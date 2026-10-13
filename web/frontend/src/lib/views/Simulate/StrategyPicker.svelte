<script lang="ts">
  import { strategies } from '../../stores/app';
  import { BASELINE_ID } from '../../stores/run';
  import Chip from '../../ui/Chip.svelte';
  import Popover from '../../ui/Popover.svelte';
  import Badge from '../../ui/Badge.svelte';
  import Icon from '../../ui/Icon.svelte';
  import StrategyDot from '../../ui/StrategyDot.svelte';
  import { selected, toggleStrategy } from './state';
  import type { StrategyInfo } from '../../types';

  export let disabled = false;

  let moreOpen = false;

  const usable = (s: StrategyInfo) => s.available !== false;

  $: list = $strategies.filter((s) => s.id !== BASELINE_ID);
  $: core = list.filter((s) => s.group === 'core');
  $: more = list.filter((s) => s.group !== 'core');
  $: extraSelected = more.filter((s) => $selected.includes(s.id));
  $: moreCount = extraSelected.length;
</script>

<div class="picker" role="group" aria-label="Strategies">
  {#each core as s (s.id)}
    {@const on = $selected.includes(s.id)}
    {@const ok = usable(s)}
    <Chip
      dotId={s.id}
      selected={on}
      unavailable={on && !ok}
      disabled={disabled || (!ok && !on)}
      title={ok ? s.description : `Unavailable: ${s.unavailable_reason ?? 'no reason given'}.${on ? ' Click to remove.' : ''}`}
      on:click={() => toggleStrategy(s.id)}
    >
      {s.name}
    </Chip>
  {/each}
  {#each extraSelected as s (s.id)}
    {@const ok = usable(s)}
    <Chip
      dotId={s.id}
      selected
      unavailable={!ok}
      {disabled}
      title={ok ? `${s.description} Click to remove.` : `Unavailable: ${s.unavailable_reason ?? 'no reason given'}. Click to remove.`}
      on:click={() => toggleStrategy(s.id)}
    >
      {s.name}
    </Chip>
  {/each}

  {#if more.length}
    <Popover open={moreOpen} align="start" width={400} label="More strategies" on:close={() => (moreOpen = false)}>
      <button
        slot="trigger"
        type="button"
        class="more"
        aria-expanded={moreOpen}
        aria-haspopup="dialog"
        {disabled}
        on:click|stopPropagation={() => (moreOpen = !moreOpen)}
      >
        <Icon name="plus" size={14} />
        More{#if moreCount}<span class="count num">{moreCount}</span>{/if}
      </button>
      <div class="menu">
        <p class="menu-title">More strategies</p>
        <ul>
          {#each more as s (s.id)}
            {@const on = $selected.includes(s.id)}
            {@const ok = usable(s)}
            <li>
              <label class="opt" class:disabled={!ok}>
                <input type="checkbox" checked={on} disabled={(!ok && !on) || disabled} on:change={() => toggleStrategy(s.id)} />
                <span class="body">
                  <span class="name">
                    <StrategyDot id={s.id} size={9} />
                    {s.name}
                    {#if s.group === 'experimental' && !/experimental/i.test(s.name)}<Badge tone="warning">Experimental</Badge>{/if}
                  </span>
                  <span class="desc">{ok ? s.description : s.unavailable_reason ?? 'Unavailable'}</span>
                </span>
              </label>
            </li>
          {/each}
        </ul>
      </div>
    </Popover>
  {/if}
</div>

<style>
  .picker {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-2);
    align-items: center;
  }
  .more {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 32px;
    padding: 0 12px;
    border-radius: var(--r-pill);
    border: 1px dashed var(--border-strong);
    background: transparent;
    color: var(--text-2);
    font-size: var(--fs-sm);
    font-weight: 550;
  }
  .more:hover:not(:disabled) {
    border-color: var(--text-3);
    color: var(--text);
  }
  .more:disabled {
    opacity: 0.5;
  }
  .count {
    display: inline-grid;
    place-items: center;
    min-width: 18px;
    height: 18px;
    padding: 0 5px;
    border-radius: var(--r-pill);
    background: var(--accent);
    color: var(--text-on-accent);
    font-size: 11px;
    font-weight: 700;
  }
  .menu {
    padding: var(--s-2);
    max-height: min(480px, 70vh);
    overflow-y: auto;
  }
  .menu-title {
    padding: var(--s-2) var(--s-2) var(--s-1);
    font-size: var(--fs-xs);
    font-weight: 600;
    color: var(--text-3);
  }
  .opt {
    display: flex;
    gap: var(--s-3);
    padding: var(--s-2);
    border-radius: var(--r-md);
    cursor: pointer;
  }
  .opt:hover {
    background: var(--surface-2);
  }
  .opt.disabled {
    cursor: not-allowed;
    opacity: 0.6;
  }
  input {
    margin-top: 3px;
    width: 16px;
    height: 16px;
    accent-color: var(--accent);
    flex: none;
  }
  .body {
    display: grid;
    gap: 2px;
    min-width: 0;
  }
  .name {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-weight: 600;
    font-size: var(--fs-sm);
  }
  .desc {
    font-size: var(--fs-xs);
    color: var(--text-3);
    line-height: 1.4;
  }
  @media (max-width: 720px), (pointer: coarse) {
    .more {
      height: 40px;
    }
  }
</style>
