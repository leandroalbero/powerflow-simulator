<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { fmtDate } from './format';

  /** Inclusive local dates, 'YYYY-MM-DD'. */
  export let start: string;
  export let end: string;
  export let min: string | undefined = undefined;
  export let max: string | undefined = undefined;
  export let disabled = false;

  const dispatch = createEventDispatcher<{ change: { start: string; end: string } }>();
  const uid = Math.random().toString(36).slice(2, 8);

  let fromEl: HTMLInputElement;
  let toEl: HTMLInputElement;

  let bad = { start: false, end: false };

  const inRange = (v: string) => !!v && (!min || v >= min) && (!max || v <= max);

  function clamp(v: string): string {
    if (min && v < min) return min;
    if (max && v > max) return max;
    return v;
  }

  /*
   * A native date input fires change on every digit typed into the year (0002, 0020, 0202, 2026),
   * so while it has focus only in-range values are taken; anything else waits for blur, where it is
   * clamped. Ordering the two dates also waits for blur, so a half-typed date can't move the other one.
   */
  function onInput(which: 'start' | 'end', e: Event) {
    const v = (e.currentTarget as HTMLInputElement).value;
    bad = { ...bad, [which]: !!v && !inRange(v) && /^[1-9]/.test(v) };
    if (!inRange(v)) return;
    const next = which === 'start' ? { start: v, end } : { start, end: v };
    if (next.start > next.end) return;
    if (next.start === start && next.end === end) return;
    start = next.start;
    end = next.end;
    dispatch('change', { start, end });
  }

  function onBlur(which: 'start' | 'end') {
    bad = { start: false, end: false };
    const el = which === 'start' ? fromEl : toEl;
    const typed = el.value;
    let s = start;
    let en = end;
    if (typed) {
      if (which === 'start') s = clamp(typed);
      else en = clamp(typed);
    }
    if (s > en) {
      if (which === 'start') en = s;
      else s = en;
    }
    // One-way value bindings don't rewrite the DOM when the clamped value equals the old one.
    fromEl.value = s;
    toEl.value = en;
    if (s === start && en === end) return;
    start = s;
    end = en;
    dispatch('change', { start, end });
  }
</script>

<div class="range" role="group" aria-label="Custom date range">
  <div class="field">
    <label for="dr-from-{uid}">From</label>
    <input
      id="dr-from-{uid}"
      type="date"
      value={start}
      {min}
      {max}
      {disabled}
      bind:this={fromEl}
      class:bad={bad.start}
      aria-invalid={bad.start || undefined}
      aria-describedby={bad.start ? `dr-msg-${uid}` : undefined}
      on:change={(e) => onInput('start', e)}
      on:blur={() => onBlur('start')}
    />
  </div>
  <div class="field">
    <label for="dr-to-{uid}">To (inclusive)</label>
    <input
      id="dr-to-{uid}"
      type="date"
      value={end}
      {min}
      {max}
      {disabled}
      bind:this={toEl}
      class:bad={bad.end}
      aria-invalid={bad.end || undefined}
      aria-describedby={bad.end ? `dr-msg-${uid}` : undefined}
      on:change={(e) => onInput('end', e)}
      on:blur={() => onBlur('end')}
    />
  </div>
  {#if bad.start || bad.end}
    <p class="msg" id="dr-msg-{uid}">Data covers {fmtDate(min)} to {fmtDate(max)}. The date will be moved into that range.</p>
  {/if}
</div>

<style>
  .range {
    display: flex;
    gap: var(--s-2);
    flex-wrap: wrap;
  }
  .field {
    display: flex;
    flex-direction: column;
    gap: 6px;
    flex: 1 1 140px;
  }
  label {
    font-size: var(--fs-xs);
    font-weight: 600;
    color: var(--text-2);
  }
  input {
    height: var(--control-h);
    padding: 0 var(--s-3);
    border: 1px solid var(--border-strong);
    border-radius: var(--r-md);
    background: var(--surface);
    font-size: var(--fs-md);
    font-variant-numeric: tabular-nums;
    min-width: 0;
  }
  input:hover:not(:disabled) {
    border-color: var(--text-3);
  }
  input.bad {
    border-color: var(--negative);
  }
  .msg {
    flex: 1 1 100%;
    font-size: var(--fs-xs);
    color: var(--negative);
  }
</style>
