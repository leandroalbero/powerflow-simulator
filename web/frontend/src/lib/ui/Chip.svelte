<script lang="ts">
  import StrategyDot from './StrategyDot.svelte';

  export let selected = false;
  export let disabled = false;
  export let dotId: string | null = null;
  export let title: string | undefined = undefined;
  /** Selected but can't run right now: styled as such and left clickable so it can be removed. */
  export let unavailable = false;
</script>

<button
  type="button"
  class="chip"
  class:selected={selected && !unavailable}
  class:unavailable
  aria-pressed={selected}
  {disabled}
  {title}
  on:click
>
  {#if dotId}<StrategyDot id={dotId} size={9} />{/if}
  <span class="label"><slot /></span>
  {#if unavailable}<span class="sr-only">(unavailable, will be skipped)</span>{/if}
  {#if $$slots.extra}<slot name="extra" />{/if}
</button>

<style>
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    height: 32px;
    padding: 0 12px;
    border-radius: var(--r-pill);
    border: 1px solid var(--border-strong);
    background: var(--surface);
    color: var(--text-2);
    font-size: var(--fs-sm);
    font-weight: 550;
    white-space: nowrap;
    transition: background var(--dur) var(--ease), border-color var(--dur) var(--ease), color var(--dur) var(--ease);
  }
  .chip:hover:not(:disabled) {
    border-color: var(--text-3);
    color: var(--text);
  }
  .selected {
    background: var(--accent-soft);
    border-color: var(--accent);
    color: var(--text);
  }
  .selected:hover:not(:disabled) {
    border-color: var(--accent);
  }
  .unavailable {
    background: transparent;
    border-style: dashed;
    border-color: var(--border-strong);
    color: var(--text-3);
  }
  .unavailable .label {
    text-decoration: line-through;
  }
  .chip:disabled {
    opacity: 0.5;
  }
  @media (max-width: 720px), (pointer: coarse) {
    .chip {
      height: 40px;
      padding: 0 14px;
    }
  }
</style>
