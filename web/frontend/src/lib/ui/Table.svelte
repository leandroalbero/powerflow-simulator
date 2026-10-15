<script lang="ts">
  /** Scroll container + shared table styling; the caller renders <table>. */
  import { onDestroy, onMount } from 'svelte';

  export let label: string;

  let scroller: HTMLDivElement;
  let moreRight = false;
  let ro: ResizeObserver | null = null;

  function measure() {
    if (!scroller) return;
    moreRight = scroller.scrollLeft + scroller.clientWidth < scroller.scrollWidth - 1;
  }

  onMount(() => {
    ro = new ResizeObserver(measure);
    ro.observe(scroller);
    if (scroller.firstElementChild) ro.observe(scroller.firstElementChild);
    measure();
  });
  onDestroy(() => ro?.disconnect());
</script>

<!-- Focusable so keyboard users can scroll a table wider than the screen. -->
<!-- svelte-ignore a11y-no-noninteractive-tabindex -->
<div class="outer" class:more-right={moreRight}>
  <div class="scroller" role="region" aria-label={label} tabindex="0" bind:this={scroller} on:scroll={measure}>
    <slot />
  </div>
</div>

<style>
  .outer {
    position: relative;
  }
  /* Fade on the right edge while more columns are hidden, so it reads as scrollable. */
  .outer::after {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    bottom: 0;
    width: 28px;
    pointer-events: none;
    background: linear-gradient(90deg, transparent, var(--surface));
    opacity: 0;
    transition: opacity var(--dur) var(--ease);
    border-radius: 0 var(--r-lg) var(--r-lg) 0;
  }
  .outer.more-right::after {
    opacity: 1;
  }
  .scroller {
    overflow-x: auto;
    max-width: 100%;
    -webkit-overflow-scrolling: touch;
  }
  .scroller:focus-visible {
    outline-offset: -2px;
  }
  .scroller :global(table) {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: var(--fs-sm);
    font-variant-numeric: tabular-nums;
  }
  .scroller :global(th) {
    position: sticky;
    top: 0;
    padding: 10px 12px;
    text-align: right;
    font-size: var(--fs-xs);
    font-weight: 600;
    color: var(--text-3);
    white-space: nowrap;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }
  .scroller :global(td) {
    padding: 11px 12px;
    text-align: right;
    white-space: nowrap;
    border-bottom: 1px solid var(--border);
  }
  .scroller :global(tr:last-child td) {
    border-bottom: 0;
  }
  .scroller :global(th:first-child),
  .scroller :global(td:first-child) {
    text-align: left;
    padding-left: var(--s-5);
  }
  .scroller :global(th:last-child),
  .scroller :global(td:last-child) {
    padding-right: var(--s-5);
  }
  /* Sort buttons grow to finger size on touch; the cell's own padding would stack on top. */
  @media (pointer: coarse) {
    .scroller :global(th:has(> button)) {
      padding-top: 0;
      padding-bottom: 0;
    }
  }
  @media (max-width: 720px) {
    .scroller :global(th) {
      padding-top: 0;
      padding-bottom: 0;
    }
    .scroller :global(td) {
      padding-top: 4px;
      padding-bottom: 4px;
    }
    .scroller :global(th:first-child),
    .scroller :global(td:first-child) {
      position: sticky;
      left: 0;
      z-index: 1;
      background: var(--surface);
      box-shadow: 1px 0 0 var(--border);
    }
    .scroller :global(th:first-child),
    .scroller :global(td:first-child) {
      padding-left: var(--s-4);
    }
  }
</style>
