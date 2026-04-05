# Irradiance Forecast Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fetch and store historical solar irradiance forecasts from Open-Meteo's Previous Runs API for Villena, Spain, with idempotent monthly CSV storage.

**Architecture:** Single script that iterates over months, calls the Open-Meteo Previous Runs API once per month, and writes one CSV per month to `data/irradiance/`. Uses urllib (stdlib) for HTTP and pandas for DataFrame/CSV handling. Idempotency: skip complete past months, re-fetch current month.

**Tech Stack:** Python 3.12, urllib.request (stdlib), pandas, argparse (stdlib)

---

### Task 1: Create output directory and write the API client function

**Files:**
- Create: `scripts/fetch_irradiance.py`

- [ ] **Step 1: Write the test for the API URL builder**

Create `tests/scripts/test_fetch_irradiance.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
import json

# We'll test the module's functions directly
import importlib.util
from pathlib import Path

# Load module from scripts/ without it being a package
spec = importlib.util.spec_from_file_location(
    "fetch_irradiance",
    Path(__file__).resolve().parent.parent.parent / "scripts" / "fetch_irradiance.py",
)
mod = importlib.util.module_from_spec(spec)


def _load_module():
    """Load the module (call after the file exists)."""
    spec.loader.exec_module(mod)
    return mod


class TestBuildApiUrl:
    def test_builds_correct_url(self):
        m = _load_module()
        url = m.build_api_url("2024-01-01", "2024-01-31")
        assert "previous-runs-api.open-meteo.com" in url
        assert "latitude=38.6361" in url
        assert "longitude=-0.8654" in url
        assert "start_date=2024-01-01" in url
        assert "end_date=2024-01-31" in url
        assert "shortwave_radiation" in url
        assert "shortwave_radiation_previous_day1" in url
        assert "direct_radiation" in url
        assert "direct_radiation_previous_day1" in url
        assert "direct_normal_irradiance" in url
        assert "direct_normal_irradiance_previous_day1" in url
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/scripts/test_fetch_irradiance.py::TestBuildApiUrl::test_builds_correct_url -v`
Expected: FAIL — module has no `build_api_url`

- [ ] **Step 3: Create the script with constants and URL builder**

Create `scripts/fetch_irradiance.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/scripts/test_fetch_irradiance.py::TestBuildApiUrl::test_builds_correct_url -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_irradiance.py tests/scripts/test_fetch_irradiance.py
git commit -m "feat: add irradiance fetch script with API URL builder"
```

---

### Task 2: Implement the HTTP fetch with retry logic

**Files:**
- Modify: `scripts/fetch_irradiance.py`
- Modify: `tests/scripts/test_fetch_irradiance.py`

- [ ] **Step 1: Write tests for fetch_month_data**

Append to `tests/scripts/test_fetch_irradiance.py`:

```python
class TestFetchMonthData:
    def test_parses_api_response_into_dataframe(self):
        m = _load_module()

        fake_response = {
            "hourly": {
                "time": ["2024-01-01T00:00", "2024-01-01T01:00", "2024-01-01T02:00"],
                "shortwave_radiation": [0.0, 0.0, 10.5],
                "shortwave_radiation_previous_day1": [0.0, 0.0, 8.2],
                "direct_radiation": [0.0, 0.0, 7.3],
                "direct_radiation_previous_day1": [0.0, 0.0, 5.1],
                "direct_normal_irradiance": [0.0, 0.0, 12.0],
                "direct_normal_irradiance_previous_day1": [0.0, 0.0, 9.8],
            }
        }
        response_bytes = json.dumps(fake_response).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = response_bytes
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            df = m.fetch_month_data("2024-01-01", "2024-01-31")

        assert len(df) == 3
        assert list(df.columns) == [
            "ghi", "ghi_forecast",
            "direct_radiation", "direct_radiation_forecast",
            "dni", "dni_forecast",
        ]
        assert df.index.name == "timestamp"
        assert df["ghi"].iloc[2] == 10.5
        assert df["ghi_forecast"].iloc[2] == 8.2

    def test_retries_on_http_error(self):
        m = _load_module()

        fake_response = {
            "hourly": {
                "time": ["2024-01-01T00:00"],
                "shortwave_radiation": [0.0],
                "shortwave_radiation_previous_day1": [0.0],
                "direct_radiation": [0.0],
                "direct_radiation_previous_day1": [0.0],
                "direct_normal_irradiance": [0.0],
                "direct_normal_irradiance_previous_day1": [0.0],
            }
        }
        response_bytes = json.dumps(fake_response).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = response_bytes
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        error = urllib.error.HTTPError(
            url="http://test", code=500, msg="Server Error", hdrs={}, fp=None
        )

        with patch("urllib.request.urlopen", side_effect=[error, mock_resp]) as mock_open:
            with patch("time.sleep"):  # skip actual sleep
                df = m.fetch_month_data("2024-01-01", "2024-01-31")

        assert mock_open.call_count == 2
        assert len(df) == 1

    def test_raises_after_max_retries(self):
        m = _load_module()

        error = urllib.error.HTTPError(
            url="http://test", code=500, msg="Server Error", hdrs={}, fp=None
        )

        with patch("urllib.request.urlopen", side_effect=error):
            with patch("time.sleep"):
                with pytest.raises(urllib.error.HTTPError):
                    m.fetch_month_data("2024-01-01", "2024-01-31")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/scripts/test_fetch_irradiance.py::TestFetchMonthData -v`
Expected: FAIL — `fetch_month_data` not defined

- [ ] **Step 3: Implement fetch_month_data**

Append to `scripts/fetch_irradiance.py` (after `build_api_url`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/test_fetch_irradiance.py::TestFetchMonthData -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_irradiance.py tests/scripts/test_fetch_irradiance.py
git commit -m "feat: add HTTP fetch with retry and DataFrame parsing"
```

---

### Task 3: Implement month iteration and idempotency logic

**Files:**
- Modify: `scripts/fetch_irradiance.py`
- Modify: `tests/scripts/test_fetch_irradiance.py`

- [ ] **Step 1: Write tests for month helpers and idempotency**

Append to `tests/scripts/test_fetch_irradiance.py`:

```python
from datetime import date


class TestMonthHelpers:
    def test_generate_months_full_range(self):
        m = _load_module()
        months = m.generate_months(date(2024, 1, 1), date(2024, 4, 1))
        assert months == [
            (date(2024, 1, 1), date(2024, 1, 31)),
            (date(2024, 2, 1), date(2024, 2, 29)),
            (date(2024, 3, 1), date(2024, 3, 31)),
            (date(2024, 4, 1), date(2024, 4, 30)),
        ]

    def test_month_end_date_december(self):
        m = _load_module()
        months = m.generate_months(date(2024, 12, 1), date(2024, 12, 1))
        assert months == [(date(2024, 12, 1), date(2024, 12, 31))]


class TestShouldSkipMonth:
    def test_skip_complete_past_month(self, tmp_path):
        m = _load_module()
        csv_path = tmp_path / "2024-01.csv"
        csv_path.write_text("timestamp,ghi\n2024-01-01T00:00,0.0\n")
        assert m.should_skip_month(csv_path, date(2024, 1, 1), today=date(2024, 3, 15)) is True

    def test_do_not_skip_current_month(self, tmp_path):
        m = _load_module()
        csv_path = tmp_path / "2024-03.csv"
        csv_path.write_text("timestamp,ghi\n2024-03-01T00:00,0.0\n")
        assert m.should_skip_month(csv_path, date(2024, 3, 1), today=date(2024, 3, 15)) is False

    def test_do_not_skip_missing_file(self, tmp_path):
        m = _load_module()
        csv_path = tmp_path / "2024-01.csv"
        assert m.should_skip_month(csv_path, date(2024, 1, 1), today=date(2024, 3, 15)) is False

    def test_skip_when_forced_is_false(self, tmp_path):
        m = _load_module()
        csv_path = tmp_path / "2024-01.csv"
        csv_path.write_text("timestamp,ghi\n2024-01-01T00:00,0.0\n")
        assert m.should_skip_month(csv_path, date(2024, 1, 1), today=date(2024, 3, 15), force=True) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/scripts/test_fetch_irradiance.py::TestMonthHelpers -v && python -m pytest tests/scripts/test_fetch_irradiance.py::TestShouldSkipMonth -v`
Expected: FAIL — functions not defined

- [ ] **Step 3: Implement month helpers**

Append to `scripts/fetch_irradiance.py` (after `fetch_month_data`):

```python
import calendar


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/test_fetch_irradiance.py::TestMonthHelpers -v && python -m pytest tests/scripts/test_fetch_irradiance.py::TestShouldSkipMonth -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_irradiance.py tests/scripts/test_fetch_irradiance.py
git commit -m "feat: add month iteration and idempotency logic"
```

---

### Task 4: Implement the main function and CLI

**Files:**
- Modify: `scripts/fetch_irradiance.py`
- Modify: `tests/scripts/test_fetch_irradiance.py`

- [ ] **Step 1: Write test for the main orchestration**

Append to `tests/scripts/test_fetch_irradiance.py`:

```python
class TestRunIngestion:
    def test_writes_csv_files(self, tmp_path):
        m = _load_module()

        fake_response = {
            "hourly": {
                "time": [f"2024-01-01T{h:02d}:00" for h in range(24)],
                "shortwave_radiation": [0.0] * 24,
                "shortwave_radiation_previous_day1": [0.0] * 24,
                "direct_radiation": [0.0] * 24,
                "direct_radiation_previous_day1": [0.0] * 24,
                "direct_normal_irradiance": [0.0] * 24,
                "direct_normal_irradiance_previous_day1": [0.0] * 24,
            }
        }
        response_bytes = json.dumps(fake_response).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = response_bytes
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            m.run_ingestion(
                start=date(2024, 1, 1),
                end=date(2024, 1, 31),
                output_dir=tmp_path,
                force=False,
            )

        csv_file = tmp_path / "2024-01.csv"
        assert csv_file.exists()
        df = pd.read_csv(csv_file, index_col="timestamp")
        assert len(df) == 24
        assert "ghi" in df.columns
        assert "ghi_forecast" in df.columns

    def test_skips_existing_complete_month(self, tmp_path):
        m = _load_module()

        csv_file = tmp_path / "2024-01.csv"
        csv_file.write_text("timestamp,ghi\nexisting,data\n")

        with patch("urllib.request.urlopen") as mock_open:
            m.run_ingestion(
                start=date(2024, 1, 1),
                end=date(2024, 1, 31),
                output_dir=tmp_path,
                force=False,
                today=date(2024, 3, 1),
            )

        # Should not have made any API calls
        mock_open.assert_not_called()
        # Original file should be untouched
        assert csv_file.read_text() == "timestamp,ghi\nexisting,data\n"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/scripts/test_fetch_irradiance.py::TestRunIngestion -v`
Expected: FAIL — `run_ingestion` not defined

- [ ] **Step 3: Implement run_ingestion and CLI**

Append to `scripts/fetch_irradiance.py` (after `should_skip_month`):

```python
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

        print(f"  {label}: fetching...", end=" ", flush=True)
        try:
            df = fetch_month_data(
                month_start.isoformat(),
                month_end.isoformat(),
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/scripts/test_fetch_irradiance.py -v`
Expected: PASS (all 10 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_irradiance.py tests/scripts/test_fetch_irradiance.py
git commit -m "feat: add main ingestion loop and CLI with idempotency"
```

---

### Task 5: End-to-end smoke test against real API

**Files:**
- None modified — this is a manual validation step

- [ ] **Step 1: Create the output directory**

Run: `mkdir -p data/irradiance`

- [ ] **Step 2: Fetch a single month to validate**

Run: `python scripts/fetch_irradiance.py --start 2024-06 --end 2024-06`
Expected output:
```
Fetching irradiance data for Villena (38.6361, -0.8654)
Output: .../data/irradiance

  2024-06: fetching... 720 rows written

Done: 1 fetched, 0 skipped, 0 errors
```

- [ ] **Step 3: Verify the CSV content**

Run: `head -5 data/irradiance/2024-06.csv && echo "---" && wc -l data/irradiance/2024-06.csv`
Expected: CSV with header + ~720 rows (30 days × 24 hours), columns: `timestamp,ghi,ghi_forecast,direct_radiation,direct_radiation_forecast,dni,dni_forecast`

- [ ] **Step 4: Verify idempotency — run again, should skip**

Run: `python scripts/fetch_irradiance.py --start 2024-06 --end 2024-06`
Expected:
```
  2024-06: skipped (already exists)

Done: 0 fetched, 1 skipped, 0 errors
```

- [ ] **Step 5: Verify force re-fetch**

Run: `python scripts/fetch_irradiance.py --start 2024-06 --end 2024-06 --force`
Expected:
```
  2024-06: fetching... 720 rows written

Done: 1 fetched, 0 skipped, 0 errors
```

- [ ] **Step 6: Clean up smoke test file and commit**

```bash
rm data/irradiance/2024-06.csv
git add scripts/fetch_irradiance.py tests/scripts/
git commit -m "test: verify irradiance fetch end-to-end against real API"
```

---

### Task 6: Fetch the full dataset

**Files:**
- Creates: `data/irradiance/2024-01.csv` through `data/irradiance/2026-04.csv`

- [ ] **Step 1: Fetch all months**

Run: `python scripts/fetch_irradiance.py`

Expected: ~28 months fetched (Jan 2024 through Apr 2026). This makes ~28 API calls, well within the 10,000/day rate limit. May take 1-2 minutes.

- [ ] **Step 2: Verify the dataset**

Run: `ls -la data/irradiance/ && echo "---" && wc -l data/irradiance/*.csv`

Expected: 28 CSV files, each with ~720-744 rows (depending on month length).

- [ ] **Step 3: Add irradiance directory to .gitignore**

The irradiance data files are fetched from an external API and should not be committed to the repo (same as they can be regenerated). Check if `.gitignore` exists and add the entry:

Append to `.gitignore`:
```
data/irradiance/
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: add data/irradiance/ to gitignore"
```
