<script lang="ts">
  export let title: string | null = null;
  export let subtitle: string | null = null;
  export let pad = true;
  export let id: string | undefined = undefined;
</script>

<section class="card" class:pad {id} aria-label={title ?? undefined}>
  {#if title || $$slots.actions}
    <header class="head">
      <div class="titles">
        {#if title}<h2 class="title">{title}</h2>{/if}
        {#if subtitle}<p class="subtitle">{subtitle}</p>{/if}
      </div>
      {#if $$slots.actions}<div class="actions"><slot name="actions" /></div>{/if}
    </header>
  {/if}
  <slot />
</section>

<style>
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    box-shadow: var(--shadow-1);
    min-width: 0;
  }
  .pad {
    padding: var(--s-5);
  }
  .card:not(.pad) {
    overflow: hidden;
  }
  .head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--s-3) var(--s-4);
    flex-wrap: wrap;
    margin-bottom: var(--s-4);
  }
  .card:not(.pad) .head {
    padding: var(--s-5) var(--s-5) 0;
  }
  .titles {
    min-width: 0;
  }
  .title {
    font-size: var(--fs-lg);
    font-weight: 650;
    letter-spacing: -0.01em;
  }
  .subtitle {
    margin-top: 2px;
    color: var(--text-3);
    font-size: var(--fs-sm);
  }
  .actions {
    display: flex;
    align-items: center;
    gap: var(--s-2);
    flex-wrap: wrap;
  }
  @media (max-width: 720px) {
    .pad {
      padding: var(--s-4);
    }
    .card:not(.pad) .head {
      padding: var(--s-4) var(--s-4) 0;
    }
  }
</style>
