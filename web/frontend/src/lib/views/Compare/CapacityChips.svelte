<script lang="ts">
  import { tick } from 'svelte';
  import Icon from '../../ui/Icon.svelte';
  import { numTrim } from '../../ui/format';
  import { addCapacity, capacities, MAX_CAPACITIES, MAX_CAPACITY, removeCapacity } from './state';

  export let disabled = false;
  /** Capacity of the chosen hardware, marked so the user can find their own battery. */
  export let current: number | null = null;

  let text = '';
  let error = '';
  const uid = Math.random().toString(36).slice(2, 8);

  function add() {
    const v = Number(text.replace(',', '.'));
    if (text.trim() === '' || !Number.isFinite(v)) {
      error = 'Enter a size in kWh';
      return;
    }
    if ($capacities.length >= MAX_CAPACITIES) {
      error = `At most ${MAX_CAPACITIES} sizes per comparison`;
      return;
    }
    if (v <= 0 || v > MAX_CAPACITY) {
      error = `Size must be more than 0 and at most ${MAX_CAPACITY} kWh`;
      return;
    }
    if ($capacities.includes(Math.round(v * 100) / 100)) {
      error = `${numTrim(v)} kWh is already in the list`;
      return;
    }
    addCapacity(v);
    text = '';
    error = '';
  }

  function onKey(e: KeyboardEvent) {
    if (e.key === 'Enter') {
      e.preventDefault();
      add();
    }
  }

  let listEl: HTMLUListElement;
  let inputEl: HTMLInputElement;

  /** The removed chip's button disappears with it, so hand focus to the chip that takes its place. */
  async function remove(c: number, i: number) {
    removeCapacity(c);
    await tick();
    const next = [...listEl.querySelectorAll<HTMLButtonElement>('button.rm:not(:disabled)')];
    (next[Math.min(i, next.length - 1)] ?? inputEl)?.focus();
  }

  const fmt = (v: number) => numTrim(v);
</script>

<div class="caps">
  <ul class="list" aria-label="Battery sizes" bind:this={listEl}>
    {#each $capacities as c, i (c)}
      <li class="cap" class:current={current != null && Math.abs(c - current) < 0.005}>
        <span class="num">{fmt(c)} kWh</span>
        {#if current != null && Math.abs(c - current) < 0.005}<span class="tag">current</span>{/if}
        <button
          type="button"
          class="rm"
          aria-label="Remove {fmt(c)} kWh"
          title="Remove"
          disabled={disabled || $capacities.length <= 1}
          on:click={() => remove(c, i)}
        >
          <Icon name="x" size={12} stroke={2.2} />
        </button>
      </li>
    {/each}
    <li class="add">
      <label for="cap-add-{uid}" class="sr-only">Add a battery size in kWh</label>
      <input
        id="cap-add-{uid}"
        type="text"
        inputmode="decimal"
        placeholder="Add kWh"
        bind:value={text}
        bind:this={inputEl}
        {disabled}
        aria-invalid={!!error || undefined}
        aria-describedby={error ? `cap-err-${uid}` : undefined}
        on:keydown={onKey}
        on:input={() => (error = '')}
      />
      <button type="button" class="add-btn" aria-label="Add size" title="Add size" {disabled} on:click={add}>
        <Icon name="plus" size={14} stroke={2.2} />
      </button>
    </li>
  </ul>
  {#if error}<p class="err" id="cap-err-{uid}" role="alert">{error}</p>{/if}
</div>

<style>
  .list {
    display: flex;
    flex-wrap: wrap;
    gap: var(--s-2);
    align-items: center;
  }
  .cap {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 32px;
    padding: 0 4px 0 12px;
    border-radius: var(--r-pill);
    border: 1px solid var(--accent);
    background: var(--accent-soft);
    font-size: var(--fs-sm);
    font-weight: 550;
    white-space: nowrap;
  }
  .tag {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-2);
  }
  .rm {
    display: inline-grid;
    place-items: center;
    width: 24px;
    height: 24px;
    border: 0;
    border-radius: 50%;
    background: transparent;
    color: var(--text-2);
  }
  .rm:hover:not(:disabled) {
    background: var(--surface);
    color: var(--text);
  }
  .rm:disabled {
    opacity: 0.35;
  }
  .add {
    display: inline-flex;
    align-items: center;
    height: 32px;
    border: 1px dashed var(--border-strong);
    border-radius: var(--r-pill);
    padding: 0 4px 0 12px;
  }
  /* The input draws no outline of its own, so the pill takes the global focus ring; a solid
     accent border alone looked like a selected size chip. */
  .add:has(input:focus-visible) {
    border-style: solid;
    border-color: var(--border-strong);
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  input {
    width: 68px;
    border: 0;
    background: transparent;
    font-size: var(--fs-sm);
    font-variant-numeric: tabular-nums;
    outline: none;
  }
  input::placeholder {
    color: var(--text-3);
  }
  .add-btn {
    display: inline-grid;
    place-items: center;
    width: 24px;
    height: 24px;
    border: 0;
    border-radius: 50%;
    background: transparent;
    color: var(--text-2);
  }
  .add-btn:hover:not(:disabled) {
    background: var(--surface-3);
    color: var(--text);
  }
  .err {
    margin-top: 6px;
    font-size: var(--fs-xs);
    color: var(--negative);
  }
  @media (max-width: 720px), (pointer: coarse) {
    /* 40px buttons plus the 1px border. */
    .cap,
    .add {
      height: 42px;
      padding-right: 0;
    }
    .rm,
    .add-btn {
      width: 40px;
      height: 40px;
    }
    input {
      align-self: stretch;
      height: auto;
    }
  }
</style>
