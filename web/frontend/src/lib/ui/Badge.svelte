<script lang="ts">
  export let tone: 'neutral' | 'accent' | 'warning' | 'positive' | 'negative' = 'neutral';
  export let title: string | undefined = undefined;
</script>

<span class="badge {tone}" {title}><slot /></span>

<style>
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    height: 20px;
    padding: 0 7px;
    border-radius: var(--r-pill);
    font-size: 11.5px;
    font-weight: 600;
    line-height: 1;
    white-space: nowrap;
    vertical-align: middle;
  }
  .neutral {
    background: var(--surface-3);
    color: var(--text-2);
  }
  .accent {
    background: var(--accent-soft);
    color: var(--accent);
  }
  .warning {
    background: var(--warning-soft);
    color: var(--warning);
  }
  .positive {
    background: var(--positive-soft);
    color: var(--positive);
  }
  .negative {
    background: var(--negative-soft);
    color: var(--negative);
  }
</style>
