<script lang="ts">
  import { theme } from '../stores/theme';
  import { strategyColor, strategyHollow } from './colors';

  export let id: string;
  export let size = 10;

  $: color = strategyColor(id, $theme);
  $: hollow = strategyHollow(id);
</script>

<span
  class="dot"
  class:hollow
  class:baseline={id === 'no_battery'}
  style="--c: {color}; --size: {size}px"
  aria-hidden="true"
/>

<style>
  .dot {
    display: inline-block;
    flex: none;
    width: var(--size);
    height: var(--size);
    border-radius: 50%;
    background: var(--c);
  }
  .hollow {
    background: transparent;
    box-shadow: inset 0 0 0 2.5px var(--c);
  }
  .baseline {
    background: transparent;
    box-shadow: inset 0 0 0 2px var(--c);
    border-radius: 2px;
  }
</style>
