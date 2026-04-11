import pytest
from unittest.mock import patch, MagicMock
import json
import urllib.error
import pandas as pd

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
