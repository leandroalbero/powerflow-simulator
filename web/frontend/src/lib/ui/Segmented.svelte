<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  export let label: string;
  export let value: string;
  export let options: { value: string; label: string; short?: string }[];
  export let disabled = false;

  const dispatch = createEventDispatcher<{ change: string }>();
  let root: HTMLDivElement;

  function select(v: string) {
    value = v;
    dispatch('change', v);
  }

  function onKey(e: KeyboardEvent) {
    if (disabled) return;
    const i = options.findIndex((o) => o.value === value);
    let n = -1;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') n = (i + 1) % options.length;
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') n = (i - 1 + options.length) % options.length;
    if (n < 0) return;
    e.preventDefault();
    select(options[n].value);
    root.querySelectorAll<HTMLButtonElement>('button')[n]?.focus();
  }
</script>

<div class="seg" role="radiogroup" tabindex="-1" aria-label={label} aria-disabled={disabled || undefined} bind:this={root} on:keydown={onKey}>
  {#each options as o (o.value)}
    <button
      type="button"
      role="radio"
      aria-checked={o.value === value}
      aria-label={o.short ? o.label : undefined}
      tabindex={o.value === value ? 0 : -1}
      {disabled}
      on:click={() => select(o.value)}
    >
      {#if o.short}<span class="long">{o.label}</span><span class="short">{o.short}</span>{:else}{o.label}{/if}
    </button>
  {/each}
</div>

<style>
  .seg {
    display: inline-flex;
    padding: 3px;
    gap: 2px;
    border-radius: var(--r-md);
    background: var(--surface-3);
    max-width: 100%;
  }
  button {
    height: calc(var(--control-h) - 6px);
    padding: 0 var(--s-3);
    border: 0;
    border-radius: 6px;
    background: transparent;
    color: var(--text-2);
    font-size: var(--fs-sm);
    font-weight: 550;
    white-space: nowrap;
    flex: 1 1 auto;
  }
  button:hover:not(:disabled) {
    color: var(--text);
  }
  button:disabled {
    cursor: not-allowed;
  }
  .seg:has(button:disabled) {
    opacity: 0.6;
  }
  button[aria-checked='true'] {
    background: var(--surface);
    color: var(--text);
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.12), 0 0 0 1px var(--border);
  }
  .short {
    display: none;
  }
  @media (max-width: 720px) {
    .seg {
      display: flex;
      width: 100%;
    }
    button {
      padding: 0 var(--s-2);
    }
    .long {
      display: none;
    }
    .short {
      display: inline;
    }
  }
  @media (max-width: 720px) {
    button {
      height: 40px;
    }
  }
</style>
