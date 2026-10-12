<script lang="ts">
  import Icon, { type IconName } from './Icon.svelte';

  export let variant: 'primary' | 'secondary' | 'ghost' | 'danger' = 'secondary';
  export let size: 'sm' | 'md' | 'lg' = 'md';
  export let icon: IconName | null = null;
  export let iconOnly = false;
  export let type: 'button' | 'submit' = 'button';
  export let disabled = false;
  export let loading = false;
  export let pressed: boolean | undefined = undefined;
  export let label: string | undefined = undefined;
  export let title: string | undefined = undefined;
  export let full = false;
  /**
   * Stay focusable while disabled (aria-disabled instead of disabled), so keyboard focus isn't
   * dropped to <body> when a button disables itself on click. The click handler must guard.
   */
  export let keepFocus = false;

  $: off = disabled || loading;
</script>

<button
  {type}
  class="btn {variant} {size}"
  class:icon-only={iconOnly}
  class:full
  disabled={off && !keepFocus}
  aria-disabled={off && keepFocus ? 'true' : undefined}
  aria-busy={loading || undefined}
  aria-pressed={pressed}
  aria-label={label}
  title={title ?? (iconOnly ? label : undefined)}
  on:click
  {...$$restProps}
>
  {#if loading}
    <span class="spinner" aria-hidden="true" />
  {:else if icon}
    <Icon name={icon} size={size === 'sm' ? 14 : 16} />
  {/if}
  {#if !iconOnly}<span class="text"><slot /></span>{/if}
</button>

<style>
  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: var(--s-2);
    height: var(--control-h);
    padding: 0 var(--s-4);
    border-radius: var(--r-md);
    border: 1px solid transparent;
    font-size: var(--fs-md);
    font-weight: 550;
    line-height: 1;
    white-space: nowrap;
    transition: background var(--dur) var(--ease), border-color var(--dur) var(--ease), color var(--dur) var(--ease);
  }
  .full {
    width: 100%;
  }
  .sm {
    height: 30px;
    padding: 0 var(--s-3);
    font-size: var(--fs-sm);
  }
  .lg {
    height: 44px;
    padding: 0 var(--s-5);
    font-size: var(--fs-lg);
  }
  .icon-only {
    width: var(--control-h);
    padding: 0;
  }
  .icon-only.sm {
    width: 30px;
  }
  @media (max-width: 720px) {
    .sm {
      height: 40px;
    }
    .icon-only.sm {
      width: 40px;
    }
  }
  .primary {
    background: var(--accent);
    color: var(--text-on-accent);
  }
  .primary:hover:not(:disabled):not([aria-disabled='true']) {
    background: var(--accent-hover);
  }
  .secondary {
    background: var(--surface);
    border-color: var(--border-strong);
    color: var(--text);
  }
  .secondary:hover:not(:disabled):not([aria-disabled='true']) {
    background: var(--surface-2);
  }
  .ghost {
    background: transparent;
    color: var(--text-2);
  }
  .ghost:hover:not(:disabled):not([aria-disabled='true']) {
    background: var(--surface-3);
    color: var(--text);
  }
  .ghost[aria-pressed='true'] {
    background: var(--accent-soft);
    color: var(--accent);
  }
  .danger {
    background: var(--surface);
    border-color: var(--border-strong);
    color: var(--negative);
  }
  .danger:hover:not(:disabled):not([aria-disabled='true']) {
    background: var(--negative-soft);
  }
  .btn:disabled,
  .btn[aria-disabled='true'] {
    opacity: 0.5;
  }
  .btn[aria-busy='true'] {
    opacity: 0.85;
  }
  .spinner {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    border: 2px solid currentColor;
    border-right-color: transparent;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
