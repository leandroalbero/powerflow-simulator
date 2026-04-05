# Forecast-Informed Charge Strategy

## Purpose

A new battery strategy that uses day-ahead solar irradiance forecasts to decide how much to charge from the grid at night. On sunny days it skips night charging (leaving capacity for free solar); on cloudy days it charges fully (like charge_night). This should outperform the static charge_night strategy by avoiding unnecessary grid purchases before sunny days.

## Strategy Logic

Extends `ForceChargeAtNightStrategy` with one change: during valley hours, the charge target varies based on tomorrow's forecast instead of always being 1.0.

### Charge target calculation

```
expected_solar_kwh = daily_ghi_forecast_sum * GHI_TO_PV_FACTOR / 1000
charge_target = clamp(1.0 - (expected_solar_kwh / battery_capacity), 0.1, 1.0)
```

Where:
- `daily_ghi_forecast_sum` = sum of hourly `ghi_forecast` values for the upcoming day (W/m² summed over hours → Wh/m²)
- `GHI_TO_PV_FACTOR` = 4.592 (derived from historical GHI vs actual PV regression)
- `battery_capacity` = current battery capacity from config (e.g. 15 kWh)
- Clamped to [0.1, 1.0] — never fully empty (min_battery_level), never above full

### Decision timing

- The strategy checks the forecast for the **current date** during valley hours (00:00-08:00)
- During non-valley hours: identical to ForceChargeAtNightStrategy (use solar, force discharge)
- If no forecast data is available for a date, falls back to charge_target=1.0 (safe default = charge_night behavior)

### What the strategy cannot see (no future peeking)

- Only uses `ghi_forecast` column (the `previous_day1` value — what the weather model predicted yesterday)
- Does not use actual GHI, actual PV generation, or any future data
- The forecast is looked up by date only, not by hour within the day

## Forecast Data Loader

A utility function that reads the monthly irradiance CSVs and returns a lookup dict.

**Input:** `data/irradiance/*.csv` (produced by `scripts/fetch_irradiance.py`)

**Output:** `dict[date, float]` mapping each date to its daily `ghi_forecast` sum

**Location:** `src/domain/strategy/forecast_loader.py`

## Strategy Class

**Location:** `src/domain/strategy/model.py` (alongside existing strategies)

**Class:** `ForecastChargeStrategy(BaseRateAwareStrategy)`

**Constructor:** `__init__(self, battery, grid, tariff, daily_forecasts: dict[date, float])`

**Method:** `calculate_energy_flows(solar_power, load_power, hour, duration) -> EnergyFlow`

## Registration

**simulation_service.py:**
- Add `ForecastChargeStrategy` to `_STRATEGY_CLASSES` as `"forecast_charge"`
- Add to `STRATEGY_REGISTRY` with id=`"forecast_charge"`, name=`"Forecast Charge"`
- Load forecast data in `_run_strategy` when strategy_id is `"forecast_charge"`

## Constants

- `GHI_TO_PV_FACTOR = 4.592` — stored in the strategy module, derived from regression of historical daily GHI sums against actual PV generation (R²=0.56)
- `DEFAULT_CHARGE_TARGET = 1.0` — fallback when no forecast is available

## Out of Scope

- Web UI changes (the strategy will appear automatically via the registry)
- Optimizing the GHI-to-PV factor (can be refined later)
- Multi-day lookahead or intra-day re-optimization
- LP/MPC optimization (future enhancement)
