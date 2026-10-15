<script lang="ts">
  import Button from '../../ui/Button.svelte';
  import { bootState } from '../../stores/app';

  export let onQuickRun: () => void;
  export let disabled = false;
</script>

<div class="empty">
  <div class="art" aria-hidden="true">
    <svg viewBox="0 0 220 120" width="220" height="120">
      <path class="grid" d="M10 100H210M10 70H210M10 40H210" />
      <path class="solar" d="M10 100C50 100 60 30 110 30S170 100 210 100" />
      <path class="soc" d="M10 80L40 80L70 60L100 35L130 35L160 55L190 75L210 80" />
    </svg>
  </div>
  <div class="text">
    <h1>Which battery strategy saves you the most?</h1>
    <p>
      Pick a period and a few strategies, then run them against your measured solar and house load, minute by minute.
      Every run includes <strong>no battery</strong> as the baseline, so savings are always compared with doing nothing.
    </p>
    <div class="cta">
      <Button variant="secondary" size="lg" icon="play" disabled={disabled || $bootState !== 'ready'} on:click={onQuickRun}>
        Simulate last 7 days
      </Button>
      <span class="hint">Takes a few seconds for a week.</span>
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
  h1 {
    font-size: var(--fs-xl);
    font-weight: 680;
    letter-spacing: -0.015em;
    margin-bottom: var(--s-2);
  }
  p {
    color: var(--text-2);
    max-width: 64ch;
  }
  .cta {
    margin-top: var(--s-5);
    display: flex;
    align-items: center;
    gap: var(--s-4);
    flex-wrap: wrap;
  }
  svg {
    display: block;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  .grid {
    stroke: var(--border);
    stroke-width: 1;
  }
  .solar {
    stroke: #e0a419;
    stroke-width: 2.5;
  }
  .soc {
    stroke: var(--accent);
    stroke-width: 2.5;
  }
  @media (max-width: 720px) {
    .empty {
      grid-template-columns: 1fr;
      padding: var(--s-5) var(--s-4);
      gap: var(--s-4);
    }
    .art svg {
      width: 160px;
      height: 88px;
    }
  }
</style>
