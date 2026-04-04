<script lang="ts">
  import { systemConfig } from '../stores/config';
  import { addLog } from '../stores/simulation';
  import { updateConfig } from '../api/client';
  import type { SystemConfig, TariffRate } from '../types/index';

  let collapsed = false;
  let applyPending = false;
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;

  // Local editable copies (initialized from store)
  let batteryCapacity = 0;
  let batteryMaxCharge = 0;
  let batteryMaxDischarge = 0;
  let batteryEfficiency = 0;
  let gridMaxImport = 0;
  let gridMaxExport = 0;
  let tariffRates: TariffRate[] = [];

  // Sync local state from store when config loads
  $: if ($systemConfig) {
    batteryCapacity = $systemConfig.battery.capacity;
    batteryMaxCharge = $systemConfig.battery.max_charge_rate;
    batteryMaxDischarge = $systemConfig.battery.max_discharge_rate;
    batteryEfficiency = $systemConfig.battery.efficiency;
    gridMaxImport = $systemConfig.grid.max_import;
    gridMaxExport = $systemConfig.grid.max_export;
    tariffRates = $systemConfig.tariff.rates ? [...$systemConfig.tariff.rates] : [];
  }

  function debouncedStoreUpdate() {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      const config: SystemConfig = {
        battery: {
          capacity: batteryCapacity,
          max_charge_rate: batteryMaxCharge,
          max_discharge_rate: batteryMaxDischarge,
          efficiency: batteryEfficiency,
        },
        grid: {
          max_import: gridMaxImport,
          max_export: gridMaxExport,
        },
        tariff: {
          rates: tariffRates,
        },
      };
      systemConfig.set(config);
    }, 300);
  }

  async function handleApply() {
    const config: SystemConfig = {
      battery: {
        capacity: batteryCapacity,
        max_charge_rate: batteryMaxCharge,
        max_discharge_rate: batteryMaxDischarge,
        efficiency: batteryEfficiency,
      },
      grid: {
        max_import: gridMaxImport,
        max_export: gridMaxExport,
      },
      tariff: {
        rates: tariffRates,
      },
    };

    applyPending = true;
    try {
      const resp = await updateConfig(config);
      systemConfig.set({
        battery: resp.battery,
        grid: resp.grid,
        tariff: resp.tariff,
      });
      addLog('Config updated successfully');
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      addLog(`Config update failed: ${message}`, 'error');
    } finally {
      applyPending = false;
    }
  }

  function formatHour(h: number): string {
    return String(h).padStart(2, '0') + ':00';
  }
</script>

<div class="panel-header">
  CONFIG
  <button class="collapse-toggle" on:click={() => (collapsed = !collapsed)}>
    {collapsed ? '\u25B6' : '\u25C0'}
  </button>
</div>

{#if !collapsed}
  <div class="config-body">
    <!-- Battery Section -->
    <div class="section">
      <div class="section-title">BATTERY</div>
      <label class="field">
        <span class="field-label">Capacity (kWh)</span>
        <input
          type="number"
          step="0.1"
          min="0"
          bind:value={batteryCapacity}
          on:input={debouncedStoreUpdate}
        />
      </label>
      <label class="field">
        <span class="field-label">Max charge (kW)</span>
        <input
          type="number"
          step="0.1"
          min="0"
          bind:value={batteryMaxCharge}
          on:input={debouncedStoreUpdate}
        />
      </label>
      <label class="field">
        <span class="field-label">Max discharge (kW)</span>
        <input
          type="number"
          step="0.1"
          min="0"
          bind:value={batteryMaxDischarge}
          on:input={debouncedStoreUpdate}
        />
      </label>
      <label class="field">
        <span class="field-label">Efficiency (0-1)</span>
        <input
          type="number"
          step="0.01"
          min="0"
          max="1"
          bind:value={batteryEfficiency}
          on:input={debouncedStoreUpdate}
        />
      </label>
    </div>

    <!-- Grid Section -->
    <div class="section">
      <div class="section-title">GRID</div>
      <label class="field">
        <span class="field-label">Max import (kW)</span>
        <input
          type="number"
          step="0.1"
          min="0"
          bind:value={gridMaxImport}
          on:input={debouncedStoreUpdate}
        />
      </label>
      <label class="field">
        <span class="field-label">Max export (kW)</span>
        <input
          type="number"
          step="0.1"
          min="0"
          bind:value={gridMaxExport}
          on:input={debouncedStoreUpdate}
        />
      </label>
    </div>

    <!-- Tariff Section -->
    <div class="section">
      <div class="section-title">TARIFF</div>
      {#if tariffRates.length > 0}
        <table class="tariff-table">
          <thead>
            <tr>
              <th>Hours</th>
              <th>Price</th>
              <th>Dir</th>
            </tr>
          </thead>
          <tbody>
            {#each tariffRates as rate}
              <tr>
                <td class="mono">{formatHour(rate.start_hour)}-{formatHour(rate.end_hour)}</td>
                <td class="mono numeric">{rate.price.toFixed(2)}</td>
                <td class:import-dir={rate.direction === 'import'} class:export-dir={rate.direction === 'export'}>
                  {rate.direction}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      {:else}
        <div class="no-data">No tariff rates</div>
      {/if}
    </div>

    <!-- Apply button -->
    <div class="apply-section">
      <button class="apply-btn" on:click={handleApply} disabled={applyPending}>
        {applyPending ? 'APPLYING...' : 'APPLY'}
      </button>
    </div>
  </div>
{/if}

<style>
  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 28px;
    padding: 0 var(--spacing-sm);
    font-size: var(--font-size-sm);
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid var(--border);
    background: var(--bg-panel);
    flex-shrink: 0;
  }

  .collapse-toggle {
    all: unset;
    cursor: pointer;
    font-size: 10px;
    color: var(--text-dim);
    padding: 0 var(--spacing-xs);
  }

  .collapse-toggle:hover {
    color: var(--text-secondary);
  }

  .config-body {
    padding: var(--spacing-sm);
    overflow-y: auto;
    flex: 1;
  }

  .section {
    margin-bottom: var(--spacing-md);
  }

  .section-title {
    font-size: var(--font-size-sm);
    font-weight: 500;
    color: var(--text-secondary);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: var(--spacing-xs);
    padding-bottom: var(--spacing-xs);
    border-bottom: 1px solid var(--border);
  }

  .field {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--spacing-sm);
    margin-bottom: var(--spacing-xs);
  }

  .field-label {
    font-size: var(--font-size-sm);
    color: var(--text-dim);
    white-space: nowrap;
  }

  .field input {
    width: 80px;
    height: 24px;
    font-size: var(--font-size-sm);
    font-family: var(--font-mono);
    text-align: right;
    padding: 0 var(--spacing-xs);
  }

  .tariff-table {
    font-size: var(--font-size-sm);
    margin-top: var(--spacing-xs);
  }

  .tariff-table th {
    font-size: 10px;
    padding: 2px var(--spacing-xs);
  }

  .tariff-table td {
    padding: 2px var(--spacing-xs);
    font-size: var(--font-size-sm);
  }

  .tariff-table .mono {
    font-family: var(--font-mono);
  }

  .tariff-table .numeric {
    text-align: right;
  }

  .import-dir {
    color: var(--color-import);
  }

  .export-dir {
    color: var(--color-export);
  }

  .no-data {
    font-size: var(--font-size-sm);
    color: var(--text-dim);
    padding: var(--spacing-xs) 0;
  }

  .apply-section {
    margin-top: var(--spacing-md);
    padding-top: var(--spacing-sm);
    border-top: 1px solid var(--border);
  }

  .apply-btn {
    width: 100%;
    height: 28px;
    font-size: var(--font-size-sm);
    font-weight: 500;
    letter-spacing: 0.05em;
    color: var(--text-primary);
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 2px;
    cursor: pointer;
    transition: border-color 0.1s;
  }

  .apply-btn:hover:not(:disabled) {
    border-color: var(--color-selected);
  }

  .apply-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
</style>
