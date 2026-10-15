"""Day-ahead GHI forecast sums per local day, from the irradiance CSVs."""

from datetime import date
from pathlib import Path
from typing import Optional

from src.domain.strategy.irradiance import daily_forecast_sums, load_irradiance


def load_daily_forecasts(irradiance_dir: Optional[Path] = None) -> dict[date, float]:
    """{local date: daily ghi_forecast sum}; the directory defaults to the repo's data/irradiance."""
    return daily_forecast_sums(load_irradiance(irradiance_dir))
