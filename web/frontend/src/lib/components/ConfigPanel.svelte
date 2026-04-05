<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { systemConfig, dataInfo } from '../stores/config';
  import { addLog } from '../stores/simulation';
  import { updateConfig, uploadData, getDataInfo } from '../api/client';
  import type { SystemConfig, TariffRate } from '../types/index';

  export let collapsed = false;

  const dispatch = createEventDispatcher();

  let applyPending = false;

  // ---- Upload state ----
  let solarFileInput: HTMLInputElement;
  let loadFileInput: HTMLInputElement;
  let solarFile: File | null = null;
  let loadFile: File | null = null;
  let uploading = false;
  let uploadStatus: string | null = null;
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;
  let initialized = false;

  // Local editable copies (initialized from store)
  let batteryCapacity = 0;
  let batteryMaxCharge = 0;
  let batteryMaxDischarge = 0;
  let batteryEfficiency = 0;
  let batteryInitialSoc = 0;
  let batteryTaperStart = 0;
  let batteryTaperFactor = 0;
  let stratMinBatteryLevel = 0;
  let stratMaxChargePower = 0;
  let stratValleyChargeTarget = 0;
  let gridMaxImport = 0;
  let gridMaxExport = 0;
  let tariffRates: TariffRate[] = [];
  let weekendRateEnabled = false;
  let weekendRatePrice = 0.085;
  let weekendRateDirection: 'import' | 'export' = 'import';

  // Sync local state from store only on initial load (not on every store update)
  $: if ($systemConfig && !initialized) {
    batteryCapacity = $systemConfig.battery.capacity;
    batteryMaxCharge = $systemConfig.battery.max_charge_rate;
    batteryMaxDischarge = $systemConfig.battery.max_discharge_rate;
    batteryEfficiency = $systemConfig.battery.efficiency;
    batteryInitialSoc = $systemConfig.battery.initial_soc ?? 0.1;
    batteryTaperStart = $systemConfig.battery.taper_start ?? 0.9;
    batteryTaperFactor = $systemConfig.battery.taper_factor ?? 0.7;
    gridMaxImport = $systemConfig.grid.max_import;
    gridMaxExport = $systemConfig.grid.max_export;
    tariffRates = $systemConfig.tariff.rates ? [...$systemConfig.tariff.rates] : [];
    if ($systemConfig.tariff.weekend_rate) {
      weekendRateEnabled = true;
      weekendRatePrice = $systemConfig.tariff.weekend_rate.price;
      weekendRateDirection = $systemConfig.tariff.weekend_rate.direction as 'import' | 'export';
    } else {
      weekendRateEnabled = false;
    }
    const strat = $systemConfig.strategy ?? { min_battery_level: 0.1, max_charge_power: 2.05, valley_charge_target: 1.0 };
    stratMinBatteryLevel = strat.min_battery_level;
    stratMaxChargePower = strat.max_charge_power;
    stratValleyChargeTarget = strat.valley_charge_target;
    initialized = true;
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
          initial_soc: batteryInitialSoc,
          taper_start: batteryTaperStart,
          taper_factor: batteryTaperFactor,
        },
        grid: {
          max_import: gridMaxImport,
          max_export: gridMaxExport,
        },
        tariff: {
          rates: tariffRates,
          weekend_rate: weekendRateEnabled
            ? { start_hour: 0, end_hour: 24, price: weekendRatePrice, direction: weekendRateDirection }
            : null,
        },
        strategy: {
          min_battery_level: stratMinBatteryLevel,
          max_charge_power: stratMaxChargePower,
          valley_charge_target: stratValleyChargeTarget,
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
        initial_soc: batteryInitialSoc,
        taper_start: batteryTaperStart,
        taper_factor: batteryTaperFactor,
      },
      grid: {
        max_import: gridMaxImport,
        max_export: gridMaxExport,
      },
      tariff: {
        rates: tariffRates,
        weekend_rate: weekendRateEnabled
          ? { start_hour: 0, end_hour: 24, price: weekendRatePrice, direction: weekendRateDirection }
          : null,
      },
      strategy: {
        min_battery_level: stratMinBatteryLevel,
        max_charge_power: stratMaxChargePower,
        valley_charge_target: stratValleyChargeTarget,
      },
    };

    applyPending = true;
    try {
      const resp = await updateConfig(config);
      initialized = false; // allow re-sync from server response
      systemConfig.set({
        battery: resp.battery,
        grid: resp.grid,
        tariff: resp.tariff,
        strategy: resp.strategy ?? { min_battery_level: 0.1, max_charge_power: 2.05, valley_charge_target: 1.0 },
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

  function addRate() {
    tariffRates = [...tariffRates, { start_hour: 0, end_hour: 24, price: 0.1, direction: 'import' }];
    debouncedStoreUpdate();
  }

  function removeRate(index: number) {
    tariffRates = tariffRates.filter((_, i) => i !== index);
    debouncedStoreUpdate();
  }

  // ---- CSV upload ----

  function handleSolarSelect(e: Event) {
    const input = e.target as HTMLInputElement;
    solarFile = input.files?.[0] ?? null;
    uploadStatus = null;
  }

  function handleLoadSelect(e: Event) {
    const input = e.target as HTMLInputElement;
    loadFile = input.files?.[0] ?? null;
    uploadStatus = null;
  }

  async function handleUpload() {
    if (!solarFile && !loadFile) return;
    uploading = true;
    uploadStatus = null;
    try {
      const files: { solarFile?: File; loadFile?: File } = {};
      if (solarFile) files.solarFile = solarFile;
      if (loadFile) files.loadFile = loadFile;
      const info = await uploadData(files);
      dataInfo.set(info);
      uploadStatus = 'success';
      addLog(`Upload complete: ${info.solar_point_count + info.load_point_count} data points`);
      // Reset file inputs
      solarFile = null;
      loadFile = null;
      if (solarFileInput) solarFileInput.value = '';
      if (loadFileInput) loadFileInput.value = '';
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      uploadStatus = 'error';
      addLog(`Upload failed: ${message}`, 'error');
    } finally {
      uploading = false;
    }
  }
</script>

<div class="panel-header">
  CONFIG
  <button class="collapse-toggle" on:click={() => dispatch('toggle')} title="Toggle config panel [">
    &#x25C0;
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
      <label class="field">
        <span class="field-label">Initial SoC (0-1)</span>
        <input type="number" step="0.05" min="0" max="1"
          bind:value={batteryInitialSoc} on:input={debouncedStoreUpdate} />
      </label>
      <label class="field">
        <span class="field-label">Taper start (0-1)</span>
        <input type="number" step="0.05" min="0" max="1"
          bind:value={batteryTaperStart} on:input={debouncedStoreUpdate} />
      </label>
      <label class="field">
        <span class="field-label">Taper factor (0-1)</span>
        <input type="number" step="0.05" min="0" max="1"
          bind:value={batteryTaperFactor} on:input={debouncedStoreUpdate} />
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

    <!-- Strategy Section -->
    <div class="section">
      <div class="section-title">STRATEGY</div>
      <label class="field">
        <span class="field-label">Min battery (0-1)</span>
        <input type="number" step="0.05" min="0" max="1"
          bind:value={stratMinBatteryLevel} on:input={debouncedStoreUpdate} />
      </label>
      <label class="field">
        <span class="field-label">Max charge (kW)</span>
        <input type="number" step="0.05" min="0"
          bind:value={stratMaxChargePower} on:input={debouncedStoreUpdate} />
      </label>
      <label class="field">
        <span class="field-label">Valley target (0-1)</span>
        <input type="number" step="0.05" min="0" max="1"
          bind:value={stratValleyChargeTarget} on:input={debouncedStoreUpdate} />
      </label>
    </div>

    <!-- Tariff Section -->
    <div class="section">
      <div class="section-title">TARIFF</div>
      {#if tariffRates.length > 0}
        <table class="tariff-table">
          <thead>
            <tr>
              <th>From</th>
              <th>To</th>
              <th>Price</th>
              <th>Dir</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {#each tariffRates as rate, i}
              <tr>
                <td>
                  <input type="number" class="tariff-input mono" min="0" max="23" step="1"
                    bind:value={tariffRates[i].start_hour} on:input={debouncedStoreUpdate} />
                </td>
                <td>
                  <input type="number" class="tariff-input mono" min="1" max="24" step="1"
                    bind:value={tariffRates[i].end_hour} on:input={debouncedStoreUpdate} />
                </td>
                <td>
                  <input type="number" class="tariff-input tariff-price mono" min="0" step="0.001"
                    bind:value={tariffRates[i].price} on:input={debouncedStoreUpdate} />
                </td>
                <td>
                  <select class="tariff-select" bind:value={tariffRates[i].direction} on:change={debouncedStoreUpdate}>
                    <option value="import">imp</option>
                    <option value="export">exp</option>
                  </select>
                </td>
                <td>
                  <button class="row-btn remove-btn" on:click={() => removeRate(i)} title="Remove rate">&times;</button>
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      {/if}
      <button class="row-btn add-btn" on:click={addRate}>+ Add rate</button>
      <div class="weekend-section">
        <label class="field">
          <span class="field-label">Weekend override</span>
          <input type="checkbox" class="checkbox-input"
            bind:checked={weekendRateEnabled} on:change={debouncedStoreUpdate} />
        </label>
        {#if weekendRateEnabled}
          <label class="field">
            <span class="field-label">Weekend price</span>
            <input type="number" step="0.001" min="0"
              bind:value={weekendRatePrice} on:input={debouncedStoreUpdate} />
          </label>
          <label class="field">
            <span class="field-label">Direction</span>
            <select class="tariff-select" bind:value={weekendRateDirection} on:change={debouncedStoreUpdate}>
              <option value="import">import</option>
              <option value="export">export</option>
            </select>
          </label>
        {/if}
      </div>
    </div>

    <!-- Apply button -->
    <div class="apply-section">
      <button class="apply-btn" on:click={handleApply} disabled={applyPending}>
        {applyPending ? 'APPLYING...' : 'APPLY'}
      </button>
    </div>

    <!-- CSV Upload Section -->
    <div class="section">
      <div class="section-title">DATA UPLOAD</div>
      <div class="upload-row">
        <span class="field-label">Solar CSV</span>
        <input
          type="file"
          accept=".csv"
          class="file-input"
          bind:this={solarFileInput}
          on:change={handleSolarSelect}
        />
      </div>
      <div class="upload-row">
        <span class="field-label">Load CSV</span>
        <input
          type="file"
          accept=".csv"
          class="file-input"
          bind:this={loadFileInput}
          on:change={handleLoadSelect}
        />
      </div>
      <button
        class="apply-btn upload-btn"
        on:click={handleUpload}
        disabled={uploading || (!solarFile && !loadFile)}
      >
        {#if uploading}
          UPLOADING...
        {:else}
          UPLOAD
        {/if}
      </button>
      {#if uploadStatus === 'success'}
        <div class="upload-status upload-ok">Upload successful</div>
      {:else if uploadStatus === 'error'}
        <div class="upload-status upload-err">Upload failed</div>
      {/if}
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

  /* ---- Upload ---- */

  .upload-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--spacing-sm);
    margin-bottom: var(--spacing-xs);
  }

  .file-input {
    width: 120px;
    font-size: 10px;
    color: var(--text-dim);
    border: none;
    background: none;
    padding: 0;
    height: auto;
  }

  .file-input::file-selector-button {
    font-size: 10px;
    font-family: var(--font-ui);
    color: var(--text-secondary);
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 2px 6px;
    cursor: pointer;
    margin-right: 4px;
  }

  .file-input::file-selector-button:hover {
    border-color: var(--border-active);
  }

  .upload-btn {
    margin-top: var(--spacing-xs);
  }

  .upload-status {
    font-size: var(--font-size-sm);
    margin-top: var(--spacing-xs);
    padding: 2px 0;
  }

  .upload-ok {
    color: var(--color-success);
  }

  .upload-err {
    color: var(--color-error);
  }

  .tariff-input {
    width: 48px;
    height: 22px;
    font-size: var(--font-size-sm);
    font-family: var(--font-mono);
    text-align: right;
    padding: 0 2px;
    background: var(--bg-input);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 2px;
  }

  .tariff-input:focus {
    border-color: var(--border-active);
  }

  .tariff-price {
    width: 56px;
  }

  .tariff-select {
    height: 22px;
    font-size: 10px;
    font-family: var(--font-ui);
    background: var(--bg-input);
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 0 2px;
  }

  .row-btn {
    all: unset;
    cursor: pointer;
    font-size: var(--font-size-sm);
    color: var(--text-dim);
    padding: 0 4px;
  }

  .row-btn:hover {
    color: var(--text-primary);
  }

  .remove-btn {
    font-size: 14px;
    color: var(--color-error);
  }

  .remove-btn:hover {
    opacity: 0.8;
  }

  .add-btn {
    font-size: var(--font-size-sm);
    color: var(--text-dim);
    margin-top: var(--spacing-xs);
    display: block;
  }

  .weekend-section {
    margin-top: var(--spacing-sm);
    padding-top: var(--spacing-xs);
    border-top: 1px solid var(--border);
  }

  .checkbox-input {
    width: 14px;
    height: 14px;
    accent-color: var(--color-selected);
  }
</style>
