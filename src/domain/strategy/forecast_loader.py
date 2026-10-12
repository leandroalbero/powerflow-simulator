"""Load day-ahead GHI forecast sums from irradiance CSVs."""

import csv
from datetime import date, datetime
from pathlib import Path

IRRADIANCE_DIR = Path("data/irradiance")


def load_daily_forecasts(irradiance_dir: Path = IRRADIANCE_DIR) -> dict[date, float]:
    """Read monthly irradiance CSVs and return {date: daily_ghi_forecast_sum}."""
    forecasts: dict[date, float] = {}

    for csv_path in sorted(irradiance_dir.glob("*.csv")):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if "ghi_forecast" not in (reader.fieldnames or []):
                continue
            for row in reader:
                raw = row.get("ghi_forecast", "").strip()
                if not raw:
                    continue
                d = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S").date()
                forecasts[d] = forecasts.get(d, 0.0) + float(raw)

    return forecasts
