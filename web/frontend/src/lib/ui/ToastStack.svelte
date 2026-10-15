<script lang="ts">
  import { dismissToast, toasts } from '../stores/toasts';
  import Icon from './Icon.svelte';

  const ICON = { info: 'info', success: 'check', warning: 'alert', error: 'alert' } as const;
</script>

<div class="stack" aria-live="polite" aria-relevant="additions">
  {#each $toasts as t (t.id)}
    <div class="toast {t.kind}" role={t.kind === 'error' ? 'alert' : 'status'}>
      <span class="ic"><Icon name={ICON[t.kind]} size={16} /></span>
      <div class="text">
        <p class="title">{t.title}</p>
        {#if t.message}<p class="msg">{t.message}</p>{/if}
      </div>
      <button class="x" type="button" aria-label="Dismiss" on:click={() => dismissToast(t.id)}>
        <Icon name="x" size={14} />
      </button>
    </div>
  {/each}
</div>

<style>
  .stack {
    position: fixed;
    z-index: 60;
    right: var(--s-5);
    bottom: var(--s-5);
    display: flex;
    flex-direction: column;
    gap: var(--s-2);
    width: min(380px, calc(100vw - 32px));
    pointer-events: none;
  }
  .toast {
    pointer-events: auto;
    display: flex;
    align-items: flex-start;
    gap: var(--s-3);
    padding: var(--s-3) var(--s-3) var(--s-3) var(--s-4);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    box-shadow: var(--shadow-2);
    animation: in 180ms var(--ease);
  }
  .ic {
    margin-top: 2px;
    color: var(--text-3);
  }
  .success .ic {
    color: var(--positive);
  }
  .warning .ic {
    color: var(--warning);
  }
  .error .ic {
    color: var(--negative);
  }
  .text {
    flex: 1;
    min-width: 0;
  }
  .title {
    font-weight: 600;
    font-size: var(--fs-md);
  }
  .msg {
    margin-top: 2px;
    font-size: var(--fs-sm);
    color: var(--text-2);
    overflow-wrap: anywhere;
  }
  .x {
    border: 0;
    background: transparent;
    color: var(--text-3);
    width: 28px;
    height: 28px;
    display: grid;
    place-items: center;
    border-radius: var(--r-sm);
  }
  .x:hover {
    background: var(--surface-3);
    color: var(--text);
  }
  @media (max-width: 720px) {
    .stack {
      left: 16px;
      right: 16px;
      bottom: 16px;
      width: auto;
    }
  }
  @keyframes in {
    from {
      opacity: 0;
      transform: translateY(8px);
    }
  }
  @media (max-width: 720px) {
    .x {
      width: 40px;
      height: 40px;
    }
  }
</style>
