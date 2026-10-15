<script lang="ts">
  import { presets } from '../../stores/app';
  import { dayCount, fmtRange, num } from '../../ui/format';
  import type { BacktestResult } from '../../types';

  export let used: BacktestResult['hardware_used'];

  $: segs = used.map((u) => {
    const p = $presets?.hardware.find((h) => h.id === u.preset_id);
    const spec = p ? `${num(p.battery.capacity, 2)} kWh · ${num(p.battery.max_charge_rate, 1)}/${num(p.battery.max_discharge_rate, 1)} kW` : null;
    return { ...u, name: p?.name ?? u.preset_id.replace(/_/g, ' '), spec, days: Math.max(1, dayCount(u.from, u.to)) };
  });
  $: total = segs.reduce((a, s) => a + s.days, 0) || 1;
</script>

{#if segs.length}
  <div class="hw" aria-label="Battery hardware simulated">
    <span class="lead">Hardware</span>
    {#if segs.length === 1}
      <span class="one"><strong>{segs[0].name}</strong>{#if segs[0].spec}<span class="muted num">{' · '}{segs[0].spec}</span>{/if}<span class="muted">{' '}for the whole range</span></span>
    {:else}
      <ol class="track">
        {#each segs as s, i (s.preset_id + s.from)}
          <li class="seg" class:alt={i % 2 === 1} style="flex: {s.days} 1 0">
            <span class="name">{s.name}{#if s.spec}<span class="muted num">{' · '}{s.spec}</span>{/if}</span>
            <span class="when num">{fmtRange(s.from, s.to)} · {s.days} d</span>
          </li>
        {/each}
      </ol>
      <span class="sr-only">{segs.map((s) => `${s.name} from ${s.from} to ${s.to}`).join('; ')}. Total {total} days.</span>
    {/if}
  </div>
{/if}

<style>
  .hw {
    display: flex;
    align-items: center;
    gap: var(--s-2) var(--s-3);
    flex-wrap: wrap;
    font-size: var(--fs-sm);
  }
  .lead {
    font-size: var(--fs-xs);
    font-weight: 600;
    color: var(--text-2);
  }
  .one strong {
    font-weight: 600;
  }
  .track {
    display: flex;
    flex: 1 1 420px;
    gap: 2px;
    min-width: 0;
  }
  .seg {
    min-width: 0;
    display: grid;
    gap: 1px;
    padding: 6px 10px;
    background: var(--accent-soft);
    border-top: 3px solid var(--accent);
    border-radius: var(--r-sm);
  }
  .seg.alt {
    background: var(--surface-3);
    border-top-color: var(--text-3);
  }
  .name {
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .when {
    font-size: var(--fs-xs);
    color: var(--text-3);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  @media (max-width: 720px) {
    .track {
      flex-direction: column;
      flex-basis: 100%;
    }
    .seg {
      flex: none !important;
    }
  }
</style>
