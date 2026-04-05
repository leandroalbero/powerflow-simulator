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
import calendar
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


def generate_months(start: date, end: date) -> list[tuple[date, date]]:
    """Generate (first_day, last_day) pairs for each month in the range."""
    months = []
    cursor = start.replace(day=1)
    end_month = end.replace(day=1)
    while cursor <= end_month:
        last_day = cursor.replace(day=calendar.monthrange(cursor.year, cursor.month)[1])
        months.append((cursor, last_day))
        # Advance to next month
        if cursor.month == 12:
            cursor = cursor.replace(year=cursor.year + 1, month=1)
        else:
            cursor = cursor.replace(month=cursor.month + 1)
    return months


def should_skip_month(
    csv_path: Path,
    month_start: date,
    today: date | None = None,
    force: bool = False,
) -> bool:
    """Return True if this month's CSV already exists and doesn't need re-fetching."""
    if force:
        return False
    if not csv_path.exists():
        return False
    if today is None:
        today = date.today()
    # Current month always re-fetched (still accumulating data)
    is_current_month = (month_start.year == today.year and month_start.month == today.month)
    if is_current_month:
        return False
    return True


ARCHIVE_START = date(2024, 1, 1)


def run_ingestion(
    start: date | None = None,
    end: date | None = None,
    output_dir: Path | None = None,
    force: bool = False,
    today: date | None = None,
) -> None:
    """Fetch irradiance data for each month in range and write CSVs."""
    if today is None:
        today = date.today()
    if start is None:
        start = ARCHIVE_START
    if end is None:
        end = today
    if output_dir is None:
        output_dir = DATA_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    months = generate_months(start, end)

    fetched = 0
    skipped = 0
    errors = 0

    for month_start, month_end in months:
        label = month_start.strftime("%Y-%m")
        csv_path = output_dir / f"{label}.csv"

        if should_skip_month(csv_path, month_start, today=today, force=force):
            print(f"  {label}: skipped (already exists)")
            skipped += 1
            continue

        # Cap end date to today for the current/future month
        effective_end = min(month_end, today)

        print(f"  {label}: fetching...", end=" ", flush=True)
        try:
            df = fetch_month_data(
                month_start.isoformat(),
                effective_end.isoformat(),
            )
            df.to_csv(csv_path)
            print(f"{len(df)} rows written")
            fetched += 1
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1

    print(f"\nDone: {fetched} fetched, {skipped} skipped, {errors} errors")


def parse_month(s: str) -> date:
    """Parse 'YYYY-MM' string to first day of that month."""
    return datetime.strptime(s, "%Y-%m").date()


def main():
    parser = argparse.ArgumentParser(
        description="Fetch solar irradiance forecasts from Open-Meteo"
    )
    parser.add_argument(
        "--start", type=parse_month, default=None,
        help="Start month (YYYY-MM), default: 2024-01",
    )
    parser.add_argument(
        "--end", type=parse_month, default=None,
        help="End month (YYYY-MM), default: current month",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-fetch all months even if CSVs exist",
    )
    args = parser.parse_args()

    print(f"Fetching irradiance data for Villena ({LATITUDE}, {LONGITUDE})")
    print(f"Output: {DATA_DIR}\n")

    run_ingestion(start=args.start, end=args.end, force=args.force)


if __name__ == "__main__":
    main()
