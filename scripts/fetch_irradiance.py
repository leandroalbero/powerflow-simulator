"""
Fetch historical solar irradiance forecasts from Open-Meteo Previous Runs API.

Stores paired actual/day-ahead-forecast data as monthly CSV files in data/irradiance/.
Idempotent: re-running skips already-complete months.

Usage:
    python scripts/fetch_irradiance.py                          # all months
    python scripts/fetch_irradiance.py --start 2024-01 --end 2024-12
    python scripts/fetch_irradiance.py --force                  # re-fetch all
"""

import argparse
import json
import time
import urllib.request
import urllib.error
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

LATITUDE = 38.6361
LONGITUDE = -0.8654
LOCAL_TZ = ZoneInfo("Europe/Madrid")
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "irradiance"

BASE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "shortwave_radiation",
    "shortwave_radiation_previous_day1",
    "direct_radiation",
    "direct_radiation_previous_day1",
    "direct_normal_irradiance",
    "direct_normal_irradiance_previous_day1",
]

COLUMN_RENAME = {
    "shortwave_radiation": "ghi",
    "shortwave_radiation_previous_day1": "ghi_forecast",
    "direct_radiation": "direct_radiation",
    "direct_radiation_previous_day1": "direct_radiation_forecast",
    "direct_normal_irradiance": "dni",
    "direct_normal_irradiance_previous_day1": "dni_forecast",
}


def build_api_url(start_date: str, end_date: str) -> str:
    """Build the Open-Meteo Previous Runs API URL for a date range."""
    params = (
        f"latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&hourly={','.join(HOURLY_VARS)}"
        f"&timezone=Europe/Madrid"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
    )
    return f"{BASE_URL}?{params}"


MAX_RETRIES = 3
RETRY_BACKOFF = [1, 3, 10]  # seconds


def fetch_month_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch irradiance data for a date range from Open-Meteo. Returns a DataFrame."""
    url = build_api_url(start_date, end_date)

    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(url) as resp:
                data = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF[attempt]
                print(f"  HTTP {e.code}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        **{COLUMN_RENAME[var]: hourly[var] for var in HOURLY_VARS},
    })
    df = df.set_index("timestamp")
    return df
