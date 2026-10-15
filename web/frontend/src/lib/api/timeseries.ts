import type { Timeseries, TimeseriesResponse } from '../types';

/** Normalise the timeseries response: epoch seconds for x, optional v2 series only when present. */
export function adaptTimeseries(r: TimeseriesResponse): Timeseries {
  const x = r.ts && r.ts.length === r.timestamps.length
    ? r.ts.map((ms) => ms / 1000)
    : r.timestamps.map((t) => Date.parse(t) / 1000);
  const opt = (s: (number | null)[] | undefined) => (s && s.length === x.length ? s : undefined);
  return {
    x,
    solar: r.solar_power,
    load: r.house_consumption,
    gridImport: r.grid_import,
    gridExport: r.grid_export,
    batteryKwh: r.battery_level,
    socPct: opt(r.soc_pct),
    batteryCharge: opt(r.battery_charge),
    batteryDischarge: opt(r.battery_discharge),
    importPrice: opt(r.import_price),
    exportPrice: opt(r.export_price),
    stepCost: opt(r.step_cost),
  };
}

/** SoC in percent: prefer the backend's soc_pct, otherwise convert kWh with the run's capacity. */
export function socPercent(t: Timeseries, capacityKwh: number | null): (number | null)[] | null {
  if (t.socPct) return t.socPct;
  if (!capacityKwh || capacityKwh <= 0) return null;
  return t.batteryKwh.map((v) => (v == null ? null : (v / capacityKwh) * 100));
}
