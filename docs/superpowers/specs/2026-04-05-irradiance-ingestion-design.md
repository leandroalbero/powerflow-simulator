# Irradiance Forecast Ingestion Pipeline

## Purpose

Fetch and store historical solar irradiance forecasts from Open-Meteo's Previous Runs API for Villena, Spain. This data will later be used to train an ML model that decides whether to charge the battery at night or leave capacity for free solar — but this spec covers only the data ingestion step.

## Data Source

**API:** Open-Meteo Previous Runs API  
**Endpoint:** `https://previous-runs-api.open-meteo.com/v1/forecast`  
**Cost:** Free, no API key required  
**Rate limit:** 10,000 calls/day, 5,000/hour  
**Archive depth:** January 2024 onwards  

**Location:** Villena, Spain (lat=38.6361, lon=-0.8654)

## Variables Fetched

For each hourly timestamp, we fetch paired forecast/actual columns:

| Variable | Actual (best estimate) | Day-ahead forecast |
|---|---|---|
| Global Horizontal Irradiance (GHI) | `shortwave_radiation` | `shortwave_radiation_previous_day1` |
| Direct Radiation | `direct_radiation` | `direct_radiation_previous_day1` |
| Direct Normal Irradiance (DNI) | `direct_normal_irradiance` | `direct_normal_irradiance_previous_day1` |

All values in W/m². The "actual" column is the best available near-term model output (stitched from early forecast hours), not satellite-measured ground truth — but it's close enough for training purposes.

## Storage Format

```
data/
  irradiance/
    2024-01.csv
    2024-02.csv
    ...
    2026-04.csv
```

**One CSV per month.** Columns:

```
timestamp,ghi,ghi_forecast,direct_radiation,direct_radiation_forecast,dni,dni_forecast
2024-01-01T00:00,0.0,0.0,0.0,0.0,0.0,0.0
2024-01-01T01:00,0.0,0.0,0.0,0.0,0.0,0.0
...
```

- `timestamp`: ISO 8601, Europe/Madrid timezone (consistent with existing solar/load data)
- Columns use short, readable names (not the raw Open-Meteo parameter names)
- Hourly resolution (one row per hour)

## Idempotency

- If a monthly CSV exists and covers a **complete month** (month is in the past), skip it.
- If a monthly CSV exists for the **current month**, re-fetch it (the month is still accumulating data).
- If a monthly CSV does not exist, fetch and write it.
- The script can be run repeatedly without duplicating data or making unnecessary API calls.

## Script

**Location:** `scripts/fetch_irradiance.py`

**Usage:**
```bash
# Fetch all available months (Jan 2024 to current)
python scripts/fetch_irradiance.py

# Fetch specific range
python scripts/fetch_irradiance.py --start 2024-01 --end 2024-12

# Force re-fetch (ignore existing files)
python scripts/fetch_irradiance.py --force
```

**Behavior:**
1. Determine date range (default: 2024-01 to current month)
2. For each month in range:
   a. Check if CSV exists and month is complete → skip
   b. Otherwise, call Open-Meteo API for that month's date range
   c. Parse JSON response, rename columns, convert timezone
   d. Write CSV to `data/irradiance/YYYY-MM.csv`
3. Print summary: months fetched, months skipped, any errors

**API call pattern:** One HTTP GET per month (~27 calls for the full range). Each call requests one month of hourly data. This is well within rate limits.

## Error Handling

- HTTP errors: retry up to 3 times with backoff, then log error and continue to next month
- Missing data points: keep as NaN in CSV (Open-Meteo may have gaps for recent dates)
- Network failure: partial progress is preserved (already-written months are not re-fetched)

## Dependencies

None beyond what's already installed:
- `pandas` — DataFrame handling and CSV writing
- `requests` or `urllib.request` — HTTP calls (use urllib to avoid adding a dependency; requests is not in requirements.txt)

## Out of Scope

- GHI-to-PV-power conversion (next step)
- ML model training
- Strategy implementation
- Web UI integration for forecast data
