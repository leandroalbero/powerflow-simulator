<script lang="ts">
  export let label: string;
  export let value: string;
  export let options: { value: string; label: string; disabled?: boolean }[];
  export let disabled = false;
  export let hint: string | null = null;
  export let id = `sel-${Math.random().toString(36).slice(2, 8)}`;
  export let hideLabel = false;
</script>

<div class="field">
  <label for={id} class:sr-only={hideLabel}>{label}</label>
  <div class="wrap">
    <select {id} bind:value {disabled} on:change aria-describedby={hint ? `${id}-hint` : undefined}>
      {#each options as o (o.value)}
        <option value={o.value} disabled={o.disabled}>{o.label}</option>
      {/each}
    </select>
    <svg class="chev" width="14" height="14" viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>
  </div>
  {#if hint}<p class="hint" id="{id}-hint">{hint}</p>{/if}
</div>

<style>
  .field {
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 0;
  }
  label {
    font-size: var(--fs-xs);
    font-weight: 600;
    color: var(--text-2);
  }
  .wrap {
    position: relative;
  }
  select {
    appearance: none;
    width: 100%;
    height: var(--control-h);
    padding: 0 32px 0 var(--s-3);
    border: 1px solid var(--border-strong);
    border-radius: var(--r-md);
    background: var(--surface);
    font-size: var(--fs-md);
    color: var(--text);
    text-overflow: ellipsis;
  }
  select:hover:not(:disabled) {
    border-color: var(--text-3);
  }
  select:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }
  .chev {
    position: absolute;
    right: 10px;
    top: 50%;
    transform: translateY(-50%);
    pointer-events: none;
    color: var(--text-3);
  }
</style>
