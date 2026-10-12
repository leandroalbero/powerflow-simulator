<script lang="ts">
  import { createEventDispatcher, onDestroy, tick } from 'svelte';

  export let open = false;
  export let align: 'start' | 'end' = 'end';
  export let width = 320;
  export let label: string;

  const dispatch = createEventDispatcher<{ close: void }>();
  let root: HTMLDivElement;
  let panel: HTMLDivElement | null = null;
  /** Horizontal shift that keeps the panel inside the viewport. */
  let dx = 0;
  const EDGE = 12;

  /** Shift that keeps a panel spanning [left, left + w] inside the viewport. */
  function shiftFor(left: number, w: number): number {
    const vw = document.documentElement.clientWidth;
    let d = 0;
    if (left + w > vw - EDGE) d = vw - EDGE - (left + w);
    if (left + d < EDGE) d = EDGE - left;
    return d;
  }

  // Computed from the trigger before the panel exists: even one frame off-screen makes mobile
  // browsers widen the layout viewport. The measured pass after render corrects any estimate error.
  function place() {
    if (!root) return;
    const r = root.getBoundingClientRect();
    const vw = document.documentElement.clientWidth;
    const w = Math.min(width, vw - 24);
    dx = shiftFor(align === 'start' ? r.left : r.right - w, w);
  }

  async function fit() {
    place();
    await tick();
    if (!panel) return;
    const r = panel.getBoundingClientRect();
    dx += shiftFor(r.left, r.width);
  }

  /** On a phone the panel can open mostly below the fold, so bring all of it into view. */
  function reveal() {
    if (!panel) return;
    const r = panel.getBoundingClientRect();
    const margin = 16;
    const overflow = r.bottom + margin - window.innerHeight;
    if (overflow <= 0) return;
    // Never scroll the panel's top under the sticky header (html's scroll-padding-top).
    const headerRoom = parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop) || 0;
    const by = Math.min(overflow, r.top - headerRoom);
    if (by <= 0) return;
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    window.scrollBy({ top: by, behavior: reduce ? 'auto' : 'smooth' });
  }

  /** Measured correction only: keeps the current dx and pulls the panel back inside the viewport. */
  async function clampToViewport() {
    await tick();
    if (!panel) return;
    const r = panel.getBoundingClientRect();
    dx += shiftFor(r.left, r.width);
  }

  /*
   * Content changes elsewhere on the page (e.g. a chip inserted before the trigger when an option
   * is ticked) move the trigger without any resize event, so its position is watched while open.
   * The panel is shifted back by the trigger's movement so the option under the pointer stays put,
   * then clamped to the viewport.
   */
  let raf = 0;
  let lastLeft = NaN;
  let lastVw = NaN;
  function watch() {
    raf = requestAnimationFrame(() => {
      if (!open || !root) return;
      const left = root.getBoundingClientRect().left;
      const vw = document.documentElement.clientWidth;
      if (vw !== lastVw) void fit();
      else if (left !== lastLeft) {
        dx -= left - lastLeft;
        void clampToViewport();
      }
      lastLeft = left;
      lastVw = vw;
      watch();
    });
  }

  // A function call, so the reactive statement depends on `open` only, not on `raf`.
  function sync(isOpen: boolean) {
    cancelAnimationFrame(raf);
    if (!isOpen) return;
    void fit().then(reveal);
    lastLeft = root ? root.getBoundingClientRect().left : NaN;
    lastVw = document.documentElement.clientWidth;
    watch();
  }
  $: sync(open);

  onDestroy(() => cancelAnimationFrame(raf));

  function onDocClick(e: MouseEvent) {
    if (open && root && !root.contains(e.target as Node)) dispatch('close');
  }

  function onKey(e: KeyboardEvent) {
    if (open && e.key === 'Escape') {
      dispatch('close');
      root.querySelector<HTMLElement>('[aria-expanded]')?.focus();
    }
  }
</script>

<svelte:document on:click={onDocClick} on:keydown={onKey} />

<div class="pop" bind:this={root}>
  <slot name="trigger" />
  {#if open}
    <div class="panel {align}" role="dialog" aria-label={label} style="--w: {width}px; translate: {dx}px 0" bind:this={panel}>
      <slot />
    </div>
  {/if}
</div>

<style>
  .pop {
    position: relative;
    display: inline-flex;
  }
  .panel {
    position: absolute;
    top: calc(100% + 8px);
    z-index: 30;
    width: min(var(--w), calc(100vw - 24px));
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    box-shadow: var(--shadow-2);
    animation: pop 140ms var(--ease);
  }
  .end {
    right: 0;
  }
  .start {
    left: 0;
  }
  @keyframes pop {
    from {
      opacity: 0;
      transform: translateY(-4px);
    }
  }
</style>
