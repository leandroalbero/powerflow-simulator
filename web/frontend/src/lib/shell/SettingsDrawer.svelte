<script lang="ts">
  import { settingsOpen } from '../stores/ui';
  import { draft, draftErrors, draftModified, resetDraft } from '../stores/draft';
  import { hasEv } from '../stores/app';
  import Drawer from '../ui/Drawer.svelte';
  import NumberField from '../ui/NumberField.svelte';
  import Switch from '../ui/Switch.svelte';
  import Button from '../ui/Button.svelte';
  import Icon from '../ui/Icon.svelte';
  import type { RateDirection, TariffRate } from '../types';

  $: e = $draftErrors;
  $: errorCount = Object.keys(e).length;

  /**
   * A new period must not overlap the others (the backend rejects that), so it takes the first
   * hours no period covers; when the day is fully covered, it splits the longest period in two.
   */
  function addRate(direction: RateDirection) {
    draft.update((d) => {
      if (!d) return d;
      const rates = d.tariff.rates.map((r) => ({ ...r }));
      const own = rates.filter((r) => r.direction === direction && r.end_hour > r.start_hour);
      const covered = new Array<boolean>(24).fill(false);
      for (const r of own) for (let h = Math.max(0, r.start_hour); h < Math.min(24, r.end_hour); h++) covered[h] = true;
      const gapStart = covered.indexOf(false);
      let rate: TariffRate;
      if (gapStart >= 0) {
        let gapEnd = gapStart;
        while (gapEnd < 24 && !covered[gapEnd]) gapEnd++;
        rate = { start_hour: gapStart, end_hour: gapEnd, price: direction === 'import' ? valleyPrice(rates) : 0.05, direction };
      } else {
        const longest = own.reduce((a, r) => (r.end_hour - r.start_hour > a.end_hour - a.start_hour ? r : a), own[0]);
        if (!longest || longest.end_hour - longest.start_hour < 2) return d;
        const mid = longest.start_hour + Math.floor((longest.end_hour - longest.start_hour) / 2);
        rate = { start_hour: mid, end_hour: longest.end_hour, price: longest.price, direction };
        longest.end_hour = mid;
      }
      return { ...d, tariff: { ...d.tariff, rates: [...rates, rate] } };
    });
  }

  function removeRate(i: number) {
    draft.update((d) => (d ? { ...d, tariff: { ...d.tariff, rates: d.tariff.rates.filter((_, j) => j !== i) } } : d));
  }

  function setWeekend(on: boolean) {
    draft.update((d) =>
      d
        ? {
            ...d,
            tariff: {
              ...d.tariff,
              weekend_rate: on ? { start_hour: 0, end_hour: 24, price: valleyPrice(d.tariff.rates), direction: 'import' } : null,
            },
          }
        : d,
    );
  }

  /** Export-at-PVPC only exists alongside PVPC import prices; leaving it on would send a flag the backend ignores. */
  function onPvpcChange(ev: Event) {
    if (ev.currentTarget instanceof HTMLInputElement && ev.currentTarget.checked) return;
    draft.update((d) => (d && d.tariff.export_follows_pvpc ? { ...d, tariff: { ...d.tariff, use_pvpc: false, export_follows_pvpc: false } } : d));
  }

  function valleyPrice(rates: TariffRate[]): number {
    const imp = rates.filter((r) => r.direction === 'import').map((r) => r.price);
    return imp.length ? Math.min(...imp) : 0.085;
  }

  $: rateRows = $draft ? $draft.tariff.rates.map((r, i) => ({ r, i })) : [];
  $: importRows = rateRows.filter((x) => x.r.direction === 'import').sort((a, b) => a.r.start_hour - b.r.start_hour);
  $: exportRows = rateRows.filter((x) => x.r.direction === 'export');
  // 24 one-hour periods can't be split further; Add would have nothing to add.
  $: canSplit = importRows.length < 24 || importRows.some(({ r }) => r.end_hour - r.start_hour >= 2);
</script>

<Drawer open={$settingsOpen} title="Settings" on:close={() => settingsOpen.set(false)}>
  <svelte:fragment slot="headerExtra">
    {#if $draftModified}<span class="modified"><span class="mdot" />Modified</span>{/if}
  </svelte:fragment>

  {#if $draft}
    <div class="content">
      <p class="intro">These settings are sent with every run. They don't change the server defaults.</p>

      <section aria-labelledby="s-battery">
        <h3 id="s-battery">Battery</h3>
        <div class="grid">
          <NumberField label="Capacity" unit="kWh" min={0} step={0.01} bind:value={$draft.battery.capacity} error={e['battery.capacity']} />
          <NumberField label="Round-trip efficiency" unit="%" scale={100} min={0} max={1} step={1} bind:value={$draft.battery.efficiency} error={e['battery.efficiency']} />
          <NumberField label="Max charge power" unit="kW" min={0} step={0.1} bind:value={$draft.battery.max_charge_rate} error={e['battery.max_charge_rate']} />
          <NumberField label="Max discharge power" unit="kW" min={0} step={0.1} bind:value={$draft.battery.max_discharge_rate} error={e['battery.max_discharge_rate']} />
          <NumberField label="Initial charge" unit="%" scale={100} min={0} max={1} step={1} bind:value={$draft.battery.initial_soc} error={e['battery.initial_soc']} />
          <NumberField
            label="Wear cost"
            unit="€/kWh"
            min={0}
            step={0.001}
            bind:value={$draft.battery.cycle_cost_eur_kwh}
            error={e['battery.cycle_cost_eur_kwh']}
            hint="Penalty per kWh cycled, used by MPC and Oracle only"
          />
          <NumberField label="Taper starts at" unit="%" scale={100} min={0} max={1} step={1} bind:value={$draft.battery.taper_start} error={e['battery.taper_start']} />
          <NumberField
            label="Taper strength"
            unit="%"
            scale={100}
            min={0}
            max={1}
            step={5}
            bind:value={$draft.battery.taper_factor}
            error={e['battery.taper_factor']}
            hint="0% means no taper near full"
          />
        </div>
      </section>

      <section aria-labelledby="s-grid">
        <h3 id="s-grid">Grid connection</h3>
        <div class="grid">
          <NumberField label="Max import" unit="kW" min={0} step={0.1} bind:value={$draft.grid.max_import} error={e['grid.max_import']} hint="Caps grid charging; the house is always served" />
          <NumberField label="Max export" unit="kW" min={0} step={0.1} bind:value={$draft.grid.max_export} error={e['grid.max_export']} />
        </div>
      </section>

      <section aria-labelledby="s-tariff">
        <h3 id="s-tariff">Tariff</h3>
        <div class="stack">
          <Switch
            label="Hourly PVPC import prices"
            description="Replaces the import periods below with hourly regulated prices."
            bind:checked={$draft.tariff.use_pvpc}
            on:change={onPvpcChange}
          />
          <Switch
            label="Export paid at PVPC"
            description="Use the hourly PVPC price for exported energy instead of the flat export rate."
            disabled={!$draft.tariff.use_pvpc}
            bind:checked={$draft.tariff.export_follows_pvpc}
          />
          {#if $draft.tariff.use_pvpc && $draft.tariff.export_follows_pvpc}
            <div class="grid">
              <NumberField
                label="Export discount"
                unit="€/kWh"
                min={0}
                step={0.005}
                bind:value={$draft.tariff.export_pvpc_discount}
                error={e['tariff.export_pvpc_discount']}
                hint="Subtracted from the hourly price, floored at 0"
              />
            </div>
          {/if}
        </div>

        <div class="rates" class:dim={$draft.tariff.use_pvpc}>
          <div class="rates-head">
            <h4>Import periods</h4>
            <Button size="sm" variant="ghost" icon="plus" disabled={!canSplit} on:click={() => addRate('import')}>Add period</Button>
          </div>
          {#if $draft.tariff.use_pvpc}<p class="hint">Not used while PVPC import prices are on.</p>{/if}
          <div class="rate-table" role="table" aria-label="Import periods">
            <div class="rate-row head" role="row">
              <span role="columnheader">From</span><span role="columnheader">To</span><span role="columnheader">Price</span><span role="columnheader"><span class="sr-only">Remove</span></span>
            </div>
            {#each importRows as { i } (i)}
              <div class="rate-row" role="row">
                <div role="cell"><NumberField compact hideLabel label="Period {i + 1} start hour" unit="h" min={0} max={23} step={1} bind:value={$draft.tariff.rates[i].start_hour} error={e[`tariff.rates.${i}.start_hour`]} /></div>
                <div role="cell"><NumberField compact hideLabel label="Period {i + 1} end hour" unit="h" min={1} max={24} step={1} bind:value={$draft.tariff.rates[i].end_hour} error={e[`tariff.rates.${i}.end_hour`]} /></div>
                <div role="cell"><NumberField compact hideLabel label="Period {i + 1} price" unit="€/kWh" min={0} step={0.001} bind:value={$draft.tariff.rates[i].price} error={e[`tariff.rates.${i}.price`]} /></div>
                <div role="cell"><button type="button" class="rm" aria-label="Remove period {i + 1}" title="Remove" on:click={() => removeRate(i)}><Icon name="trash" size={15} /></button></div>
              </div>
            {/each}
          </div>
          {#if e['tariff.rates']}<p class="err">{e['tariff.rates']}</p>{/if}

          <div class="weekend">
            <Switch
              label="Weekends at a flat rate"
              description="Spain 2.0TD: Saturdays, Sundays and national holidays are valley all day."
              checked={$draft.tariff.weekend_rate != null}
              on:change={(ev) => setWeekend(ev.currentTarget instanceof HTMLInputElement && ev.currentTarget.checked)}
            />
            {#if $draft.tariff.weekend_rate}
              <div class="grid">
                <NumberField label="Weekend price" unit="€/kWh" min={0} step={0.001} bind:value={$draft.tariff.weekend_rate.price} error={e['tariff.weekend_rate.price']} />
              </div>
            {/if}
          </div>
        </div>

        <div class="rates">
          <div class="rates-head">
            <h4>Export rate</h4>
            {#if !exportRows.length}<Button size="sm" variant="ghost" icon="plus" on:click={() => addRate('export')}>Add export rate</Button>{/if}
          </div>
          {#if $draft.tariff.use_pvpc && $draft.tariff.export_follows_pvpc}<p class="hint">Not used while export follows PVPC.</p>{/if}
          <div class="rate-table" role="table" aria-label="Export rates">
            {#if exportRows.length}
              <div class="rate-row head" role="row">
                <span role="columnheader">From</span><span role="columnheader">To</span><span role="columnheader">Price</span><span role="columnheader"><span class="sr-only">Remove</span></span>
              </div>
            {/if}
            {#each exportRows as { i } (i)}
              <div class="rate-row" role="row">
                <div role="cell"><NumberField compact hideLabel label="Export {i + 1} start hour" unit="h" min={0} max={23} step={1} bind:value={$draft.tariff.rates[i].start_hour} error={e[`tariff.rates.${i}.start_hour`]} /></div>
                <div role="cell"><NumberField compact hideLabel label="Export {i + 1} end hour" unit="h" min={1} max={24} step={1} bind:value={$draft.tariff.rates[i].end_hour} error={e[`tariff.rates.${i}.end_hour`]} /></div>
                <div role="cell"><NumberField compact hideLabel label="Export {i + 1} price" unit="€/kWh" min={0} step={0.001} bind:value={$draft.tariff.rates[i].price} error={e[`tariff.rates.${i}.price`]} /></div>
                <div role="cell"><button type="button" class="rm" aria-label="Remove export rate {i + 1}" title="Remove" on:click={() => removeRate(i)}><Icon name="trash" size={15} /></button></div>
              </div>
            {/each}
          </div>
          {#if e['tariff.rates.export']}<p class="err">{e['tariff.rates.export']}</p>{/if}
        </div>
      </section>

      <section aria-labelledby="s-strategy">
        <h3 id="s-strategy">Strategy parameters</h3>
        <div class="grid">
          <NumberField label="Minimum charge" unit="%" scale={100} min={0} max={1} step={1} bind:value={$draft.strategy.min_battery_level} error={e['strategy.min_battery_level']} hint="Discharge stops below this" />
          <NumberField label="Valley charge target" unit="%" scale={100} min={0} max={1} step={1} bind:value={$draft.strategy.valley_charge_target} error={e['strategy.valley_charge_target']} />
          <NumberField label="Max grid charge power" unit="kW" min={0} step={0.1} bind:value={$draft.strategy.max_charge_power} error={e['strategy.max_charge_power']} hint="Clipped to the battery's charge power" />
          <NumberField label="MPC horizon" unit="h" min={1} max={48} step={1} bind:value={$draft.tariff.mpc_horizon_hours} error={e['tariff.mpc_horizon_hours']} hint="1–48 hours of lookahead" />
        </div>
      </section>

      {#if $hasEv}
        <section aria-labelledby="s-data">
          <h3 id="s-data">Data</h3>
          <Switch
            label="Exclude EV charging"
            description="Subtract the EV charger's power from the house load."
            bind:checked={$draft.exclude_ev}
          />
        </section>
      {/if}
    </div>
  {:else}
    <p class="content muted">Loading defaults…</p>
  {/if}

  <svelte:fragment slot="footer">
    {#if errorCount}
      <p class="foot-err" role="status">{errorCount === 1 ? '1 value needs fixing' : `${errorCount} values need fixing`}</p>
    {:else}
      <p class="foot-note">Applies to the next run</p>
    {/if}
    <span class="spacer" />
    <Button variant="ghost" disabled={!$draftModified} on:click={resetDraft}>Reset to defaults</Button>
    <Button variant="primary" on:click={() => settingsOpen.set(false)}>Done</Button>
  </svelte:fragment>
</Drawer>

<style>
  .content {
    padding: var(--s-4) var(--s-5) var(--s-6);
  }
  .intro {
    color: var(--text-3);
    font-size: var(--fs-sm);
    margin-bottom: var(--s-2);
  }
  section {
    padding: var(--s-5) 0;
    border-bottom: 1px solid var(--border);
  }
  section:last-child {
    border-bottom: 0;
  }
  h3 {
    font-size: var(--fs-md);
    font-weight: 650;
    margin-bottom: var(--s-4);
  }
  h4 {
    font-size: var(--fs-sm);
    font-weight: 600;
    color: var(--text-2);
  }
  .grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--s-4) var(--s-3);
    align-items: start;
  }
  .stack {
    display: grid;
    gap: var(--s-4);
  }
  .rates {
    margin-top: var(--s-5);
  }
  .rates.dim .rate-table {
    opacity: 0.55;
  }
  .rates-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: var(--s-2);
  }
  .rate-table {
    display: grid;
    gap: var(--s-2);
  }
  .rate-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1.4fr 32px;
    gap: var(--s-2);
    align-items: start;
  }
  .rate-row > [role='cell'] {
    min-width: 0;
  }
  .rate-row.head span {
    font-size: var(--fs-xs);
    color: var(--text-3);
    font-weight: 600;
  }
  .rm {
    width: 32px;
    height: 32px;
    display: grid;
    place-items: center;
    border: 0;
    border-radius: var(--r-sm);
    background: transparent;
    color: var(--text-3);
  }
  .rm:hover {
    background: var(--negative-soft);
    color: var(--negative);
  }
  .weekend {
    margin-top: var(--s-4);
    display: grid;
    gap: var(--s-3);
  }
  .err,
  .foot-err {
    color: var(--negative);
    font-size: var(--fs-xs);
    margin-top: var(--s-2);
  }
  .foot-err,
  .foot-note {
    margin: 0;
    font-size: var(--fs-sm);
  }
  .foot-note {
    color: var(--text-3);
  }
  .spacer {
    flex: 1;
  }
  .modified {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: var(--fs-xs);
    font-weight: 600;
    color: var(--accent);
  }
  .mdot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--accent);
  }
  @media (max-width: 720px) {
    .content {
      padding: var(--s-3) var(--s-4) var(--s-6);
    }
    .rate-row {
      grid-template-columns: 1fr 1fr 1.5fr 40px;
    }
    .rm {
      width: 40px;
      height: 40px;
    }
    .foot-note {
      display: none;
    }
  }
</style>
