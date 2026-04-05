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
