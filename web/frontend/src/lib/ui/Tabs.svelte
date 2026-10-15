<script lang="ts">
  export let items: { id: string; label: string; href: string }[];
  export let current: string;
  export let label = 'Sections';
</script>

<nav class="tabs" aria-label={label}>
  {#each items as t (t.id)}
    <a href={t.href} class="tab" aria-current={t.id === current ? 'page' : undefined}>{t.label}</a>
  {/each}
</nav>

<style>
  .tabs {
    display: flex;
    align-items: stretch;
    gap: var(--s-1);
    height: 100%;
    min-width: 0;
    overflow-x: auto;
    scrollbar-width: none;
  }
  .tabs::-webkit-scrollbar {
    display: none;
  }
  .tab {
    position: relative;
    display: inline-flex;
    align-items: center;
    padding: 0 var(--s-3);
    color: var(--text-2);
    font-size: var(--fs-md);
    font-weight: 550;
    text-decoration: none;
    white-space: nowrap;
    border-radius: var(--r-sm);
  }
  .tab:hover {
    color: var(--text);
  }
  .tab[aria-current='page'] {
    color: var(--text);
  }
  .tab[aria-current='page']::after {
    content: '';
    position: absolute;
    left: var(--s-3);
    right: var(--s-3);
    bottom: 0;
    height: 2px;
    border-radius: 2px 2px 0 0;
    background: var(--accent);
  }
  .tab:focus-visible {
    outline-offset: -2px;
  }
</style>
