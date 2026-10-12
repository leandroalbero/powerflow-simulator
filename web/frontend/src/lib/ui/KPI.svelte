<script lang="ts">
  export let label: string;
  export let value: string;
  export let tone: 'default' | 'positive' | 'negative' = 'default';
  export let hero = false;
</script>

<div class="kpi" class:hero>
  <p class="label">{label}</p>
  <p class="value num {tone}">{value}</p>
  {#if $$slots.default}<div class="sub"><slot /></div>{/if}
</div>

<style>
  .kpi {
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 0;
    padding: var(--s-5);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    box-shadow: var(--shadow-1);
  }
  .label {
    font-size: var(--fs-sm);
    font-weight: 600;
    color: var(--text-2);
  }
  .value {
    font-size: var(--fs-2xl);
    font-weight: 680;
    letter-spacing: -0.02em;
    line-height: 1.1;
    overflow-wrap: anywhere;
  }
  .hero .value {
    font-size: var(--fs-3xl);
  }
  .positive {
    color: var(--positive);
  }
  .negative {
    color: var(--negative);
  }
  /* Pinned to the bottom so the sub-lines line up across cards whose values differ in size. */
  .sub {
    margin-top: auto;
    font-size: var(--fs-sm);
    color: var(--text-3);
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px 8px;
  }
  @media (max-width: 720px) {
    .kpi {
      padding: var(--s-4);
    }
    .value {
      font-size: var(--fs-xl);
    }
    .hero .value {
      font-size: var(--fs-2xl);
    }
  }
</style>
