<script lang="ts">
  import Button from '../../ui/Button.svelte';
  import { bootState } from '../../stores/app';

  export let onQuickRun: () => void;
  export let disabled = false;
  export let estimate: string;

  // A tiny matrix, darker where the saving is bigger, to hint at what the result looks like.
  const CELLS = [
    [0.15, 0.35, 0.55, 0.62, 0.66],
    [0.25, 0.5, 0.78, 0.86, 0.9],
  ];
</script>

<div class="empty">
  <div class="art" aria-hidden="true">
    {#each CELLS as row}
      <div class="r">
        {#each row as v}
          <span style="--t: {v}" />
        {/each}
      </div>
    {/each}
  </div>
  <div class="text">
    <h1>Which tariff, battery size and strategy saves you the most?</h1>
    <p>
      A comparison runs every combination you pick against your measured solar and house load, and lays the savings out
      as a matrix: tariffs down the side, battery sizes across. Each tariff also runs once with <strong>no battery</strong>,
      so every saving is measured against doing nothing on that tariff, and larger sizes show where extra capacity stops paying off.
    </p>
    <div class="cta">
      <Button variant="secondary" size="lg" icon="play" disabled={disabled || $bootState !== 'ready'} on:click={onQuickRun}>
        Compare 2.0TD vs PVPC for 5–30 kWh, last 12 months
      </Button>
      <span class="hint">{estimate}, with the strategies selected above.</span>
    </div>
  </div>
</div>

<style>
  .empty {
    display: grid;
    grid-template-columns: auto minmax(0, 68ch);
    justify-content: center;
    gap: var(--s-6);
    align-items: center;
    padding: var(--s-7) var(--s-6);
    border: 1px dashed var(--border-strong);
    border-radius: var(--r-lg);
  }
  .art {
    display: grid;
    gap: 5px;
  }
  .r {
    display: flex;
    gap: 5px;
  }
  .r span {
    width: 34px;
    height: 26px;
    border-radius: 5px;
    background: color-mix(in srgb, var(--positive) calc(var(--t) * 100%), var(--surface-3));
  }
  .r:last-child span:nth-child(4) {
    box-shadow: 0 0 0 2px var(--bg), 0 0 0 4px var(--text);
  }
  h1 {
    font-size: var(--fs-xl);
    font-weight: 680;
    letter-spacing: -0.015em;
    margin-bottom: var(--s-2);
  }
  p {
    color: var(--text-2);
    max-width: 66ch;
  }
  .cta {
    margin-top: var(--s-5);
    display: flex;
    align-items: center;
    gap: var(--s-3) var(--s-4);
    flex-wrap: wrap;
  }
  @media (max-width: 720px) {
    .empty {
      grid-template-columns: 1fr;
      padding: var(--s-5) var(--s-4);
      gap: var(--s-4);
    }
    .r span {
      width: 26px;
      height: 20px;
    }
    .cta :global(.btn) {
      white-space: normal;
      height: auto;
      min-height: 44px;
      padding-top: var(--s-2);
      padding-bottom: var(--s-2);
      text-align: left;
      line-height: 1.3;
    }
  }
</style>
