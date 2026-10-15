<script lang="ts">
  import { createEventDispatcher, tick } from 'svelte';
  import Button from './Button.svelte';

  export let open = false;
  export let title: string;
  export let width = 480;

  const dispatch = createEventDispatcher<{ close: void }>();
  let panel: HTMLDivElement;
  let returnFocus: HTMLElement | null = null;

  // Focus is restored on the open→closed transition rather than in close(), so a parent that
  // closes the drawer itself (e.g. a footer "Done" button) still returns focus to the trigger.
  let wasOpen = false;
  $: syncOpen(open);

  function syncOpen(isOpen: boolean) {
    if (isOpen === wasOpen) return;
    wasOpen = isOpen;
    if (isOpen) void onOpen();
    else void onClosed();
  }

  async function onOpen() {
    returnFocus = document.activeElement as HTMLElement | null;
    await tick();
    panel?.querySelector<HTMLElement>('[data-autofocus], button, input, select')?.focus();
  }

  async function onClosed() {
    const target = returnFocus;
    returnFocus = null;
    await tick();
    if (target?.isConnected) target.focus();
  }

  function close() {
    dispatch('close');
  }

  function onKey(e: KeyboardEvent) {
    if (!open) return;
    if (e.key === 'Escape') {
      e.preventDefault();
      close();
    } else if (e.key === 'Tab') {
      const items = [...panel.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled), [tabindex="0"], a[href]')];
      if (!items.length) return;
      const first = items[0];
      const last = items[items.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  }
</script>

<svelte:window on:keydown={onKey} />

{#if open}
  <div class="backdrop" on:click={close} aria-hidden="true" />
  <div class="drawer" role="dialog" aria-modal="true" aria-labelledby="drawer-title" style="--w: {width}px" bind:this={panel}>
    <div class="head">
      <h2 id="drawer-title">{title}</h2>
      {#if $$slots.headerExtra}<slot name="headerExtra" />{/if}
      <Button variant="ghost" icon="x" iconOnly label="Close" on:click={close} />
    </div>
    <div class="body"><slot /></div>
    {#if $$slots.footer}<div class="foot"><slot name="footer" /></div>{/if}
  </div>
{/if}

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    z-index: 40;
    background: var(--overlay);
    animation: fade var(--dur) var(--ease);
  }
  .drawer {
    position: fixed;
    z-index: 41;
    top: 0;
    right: 0;
    bottom: 0;
    width: min(var(--w), 100vw);
    display: flex;
    flex-direction: column;
    background: var(--surface);
    border-left: 1px solid var(--border);
    box-shadow: var(--shadow-3);
    animation: slide 220ms var(--ease);
  }
  .head {
    display: flex;
    align-items: center;
    gap: var(--s-3);
    padding: var(--s-3) var(--s-3) var(--s-3) var(--s-5);
    border-bottom: 1px solid var(--border);
    min-height: var(--topbar-h);
  }
  h2 {
    flex: 1;
    font-size: var(--fs-lg);
    font-weight: 650;
  }
  .body {
    flex: 1;
    overflow-y: auto;
    overscroll-behavior: contain;
  }
  .foot {
    padding: var(--s-3) var(--s-5);
    border-top: 1px solid var(--border);
    display: flex;
    gap: var(--s-2);
    align-items: center;
  }
  @media (max-width: 720px) {
    .drawer {
      width: 100vw;
      border-left: 0;
    }
    .head {
      padding-left: var(--s-4);
    }
    .foot {
      padding: var(--s-3) var(--s-4) calc(var(--s-3) + env(safe-area-inset-bottom));
    }
  }
  @keyframes slide {
    from {
      transform: translateX(24px);
      opacity: 0;
    }
  }
  @keyframes fade {
    from {
      opacity: 0;
    }
  }
</style>
