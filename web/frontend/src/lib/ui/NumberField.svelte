<script lang="ts">
  /**
   * Numeric input. `scale` converts between the stored value and what is shown,
   * e.g. scale=100 shows a 0–1 fraction as a percentage.
   */
  export let label: string;
  export let value: number;
  export let unit = '';
  export let min: number | undefined = undefined;
  export let max: number | undefined = undefined;
  export let step: number | 'any' = 'any';
  export let scale = 1;
  export let error: string | undefined = undefined;
  export let hint: string | undefined = undefined;
  export let id = `num-${Math.random().toString(36).slice(2, 8)}`;
  export let compact = false;
  export let hideLabel = false;

  const round = (v: number) => Math.round(v * 1e6) / 1e6;
  let text = '';
  let lastValue: number | null = null;
  $: if (value !== lastValue) {
    lastValue = value;
    text = Number.isFinite(value) ? String(round(value * scale)) : '';
  }

  function onInput(e: Event) {
    text = (e.currentTarget as HTMLInputElement).value;
    const n = text.trim() === '' ? NaN : Number(text);
    value = Number.isFinite(n) ? round(n / scale) : NaN;
    lastValue = value;
  }
</script>

<div class="field" class:compact>
  <label for={id} class:sr-only={hideLabel}>{label}</label>
  <div class="wrap" class:invalid={!!error}>
    <input
      {id}
      type="number"
      inputmode="decimal"
      value={text}
      min={min != null ? round(min * scale) : undefined}
      max={max != null ? round(max * scale) : undefined}
      {step}
      aria-invalid={!!error}
      aria-describedby={error || hint ? `${id}-msg` : undefined}
      on:input={onInput}
    />
    {#if unit}<span class="unit">{unit}</span>{/if}
  </div>
  {#if error}
    <p class="msg error" id="{id}-msg">{error}</p>
  {:else if hint}
    <p class="msg" id="{id}-msg">{hint}</p>
  {/if}
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
    display: flex;
    align-items: center;
    height: var(--control-h);
    border: 1px solid var(--border-strong);
    border-radius: var(--r-md);
    background: var(--surface);
    overflow: hidden;
  }
  .wrap:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-soft);
  }
  .wrap.invalid {
    border-color: var(--negative);
  }
  input {
    flex: 1;
    min-width: 0;
    height: 100%;
    border: 0;
    background: transparent;
    padding: 0 var(--s-3);
    font-size: var(--fs-md);
    font-variant-numeric: tabular-nums;
    outline: none;
    -moz-appearance: textfield;
    appearance: textfield;
  }
  input::-webkit-outer-spin-button,
  input::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
  .unit {
    padding: 0 var(--s-3) 0 0;
    color: var(--text-3);
    font-size: var(--fs-sm);
    white-space: nowrap;
  }
  .msg {
    font-size: var(--fs-xs);
    color: var(--text-3);
  }
  .msg.error {
    color: var(--negative);
  }
  .compact .wrap {
    height: 32px;
  }
  .compact input {
    padding: 0 var(--s-2);
  }
  .compact .unit {
    padding-right: var(--s-2);
  }
  /* The border sits outside the input, so the wrap is 2px taller to give a 40px tap target. */
  @media (max-width: 720px) {
    .wrap,
    .compact .wrap {
      height: 42px;
    }
  }
</style>
