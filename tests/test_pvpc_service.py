import json
from datetime import datetime, timedelta, timezone
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
import pytz

from web.backend.services.pvpc_service import PvpcService, _parse_ree_response

_UTC = timezone.utc
_MADRID = pytz.timezone("Europe/Madrid")


def _ree_payload(values):
    """Build a minimal REE response shape with the given PVPC ``values`` list."""
    return {
        "data": {"type": "Precios Mercados Tiempo Real"},
        "included": [
            {"type": "Mercado SPOT Diario", "attributes": {"values": [{"value": 0, "datetime": "2024-01-01T00:00:00.000+01:00"}]}},
            {"type": "PVPC", "attributes": {"values": values}},
        ],
    }


def _urlopen_mock(payload: dict) -> MagicMock:
    """Return a MagicMock suitable for use as urlopen's context-manager return."""
    body = json.dumps(payload).encode("utf-8")
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=body)))
    m.__exit__ = MagicMock(return_value=False)
    return m


def _madrid_hours(day_local_start: datetime, count: int):
    """Yield REE-style PVPC value dicts with a +01:00 offset."""
    # Assume winter (+01:00) for the simple fixture; tests don't cross DST.
    out = []
    for h in range(count):
        ts = day_local_start + timedelta(hours=h)
        # REE format: "YYYY-MM-DDTHH:MM:SS.000+01:00"
        iso = ts.strftime("%Y-%m-%dT%H:%M:%S.000+01:00")
        out.append({"value": 100.0 + h, "percentage": 0.5, "datetime": iso})
    return out


def test_empty_cache_triggers_fetch(tmp_path):
    db = tmp_path / "pvpc.db"
    svc = PvpcService(db_path=db)

    day_local = datetime(2024, 1, 1, 0, 0)  # naive Madrid
    values = _madrid_hours(day_local, 24)
    payload = _ree_payload(values)

    start = _MADRID.localize(datetime(2024, 1, 1, 0, 0))
    end = _MADRID.localize(datetime(2024, 1, 1, 23, 0))

    with patch(
        "web.backend.services.pvpc_service.urllib.request.urlopen",
        return_value=_urlopen_mock(payload),
    ) as mock_urlopen:
        prices = svc.get_prices(start, end)
        assert mock_urlopen.call_count == 1

    assert len(prices) == 24
    # €100/MWh -> €0.10/kWh; hour 0 in Madrid (+01:00) == 23:00 UTC prev day.
    # Check the first value is 100 / 1000 = 0.10.
    assert min(prices.values()) == pytest.approx(0.10)
    assert max(prices.values()) == pytest.approx(0.123)
    for ts in prices:
        assert ts.tzinfo == _UTC


def test_second_call_hits_cache(tmp_path):
    db = tmp_path / "pvpc.db"
    svc = PvpcService(db_path=db)
    day_local = datetime(2024, 1, 1, 0, 0)
    payload = _ree_payload(_madrid_hours(day_local, 24))

    start = _MADRID.localize(datetime(2024, 1, 1, 0, 0))
    end = _MADRID.localize(datetime(2024, 1, 1, 23, 0))

    with patch(
        "web.backend.services.pvpc_service.urllib.request.urlopen",
        return_value=_urlopen_mock(payload),
    ) as mock_urlopen:
        svc.get_prices(start, end)
        assert mock_urlopen.call_count == 1

    # Second call: cache fully covers the range, no HTTP.
    with patch(
        "web.backend.services.pvpc_service.urllib.request.urlopen",
    ) as mock_urlopen:
        prices2 = svc.get_prices(start, end)
        assert mock_urlopen.call_count == 0
    assert len(prices2) == 24


def test_partial_overlap_only_fetches_missing(tmp_path):
    db = tmp_path / "pvpc.db"
    svc = PvpcService(db_path=db)

    # Pre-seed day 1.
    day1_local = datetime(2024, 1, 1, 0, 0)
    payload_day1 = _ree_payload(_madrid_hours(day1_local, 24))

    start_d1 = _MADRID.localize(datetime(2024, 1, 1, 0, 0))
    end_d1 = _MADRID.localize(datetime(2024, 1, 1, 23, 0))
    with patch(
        "web.backend.services.pvpc_service.urllib.request.urlopen",
        return_value=_urlopen_mock(payload_day1),
    ):
        svc.get_prices(start_d1, end_d1)

    # Request day 1 + day 2 — only day 2 must be fetched.
    day2_local = datetime(2024, 1, 2, 0, 0)
    payload_day2 = _ree_payload(_madrid_hours(day2_local, 24))

    start_combined = _MADRID.localize(datetime(2024, 1, 1, 0, 0))
    end_combined = _MADRID.localize(datetime(2024, 1, 2, 23, 0))

    with patch(
        "web.backend.services.pvpc_service.urllib.request.urlopen",
        return_value=_urlopen_mock(payload_day2),
    ) as mock_urlopen:
        prices = svc.get_prices(start_combined, end_combined)
        assert mock_urlopen.call_count == 1

    assert len(prices) == 48


def test_range_over_thirty_days_batches_two_calls(tmp_path):
    db = tmp_path / "pvpc.db"
    svc = PvpcService(db_path=db)

    # 45-day range → need two HTTP calls (first 30 days, then remainder).
    start = _MADRID.localize(datetime(2024, 1, 1, 0, 0))
    end = _MADRID.localize(datetime(2024, 2, 14, 23, 0))

    call_count = {"n": 0}

    def build_side_effect(*args, **kwargs):
        call_count["n"] += 1
        # Emit values covering roughly each window.
        # To simplify, emit 45 days worth of hourly prices every call —
        # INSERT OR REPLACE makes re-inserts idempotent.
        values = []
        t = datetime(2024, 1, 1, 0, 0)
        for h in range(45 * 24):
            iso = (t + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%S.000+01:00")
            values.append({"value": 100.0, "percentage": 0.5, "datetime": iso})
        return _urlopen_mock(_ree_payload(values))

    with patch(
        "web.backend.services.pvpc_service.urllib.request.urlopen",
        side_effect=build_side_effect,
    ):
        prices = svc.get_prices(start, end)

    assert call_count["n"] == 2
    # 45 days * 24 hours - but our simplistic generator only emits 45*24 hours
    # starting from 2024-01-01 Madrid; ensure we got at least the full range.
    assert len(prices) == 45 * 24


def test_fetch_http_error_raises_runtime_error(tmp_path):
    from urllib.error import URLError

    db = tmp_path / "pvpc.db"
    svc = PvpcService(db_path=db)
    start = _MADRID.localize(datetime(2024, 1, 1, 0, 0))
    end = _MADRID.localize(datetime(2024, 1, 1, 23, 0))

    with patch(
        "web.backend.services.pvpc_service.urllib.request.urlopen",
        side_effect=URLError("boom"),
    ):
        with pytest.raises(RuntimeError, match="PVPC fetch failed"):
            svc.get_prices(start, end)


def test_parse_ree_response_handles_valid_payload():
    payload = _ree_payload(
        [
            {"value": 120.0, "percentage": 0.5, "datetime": "2024-01-01T00:00:00.000+01:00"},
            {"value": 85.5, "percentage": 0.3, "datetime": "2024-01-01T01:00:00.000+01:00"},
        ]
    )
    out = _parse_ree_response(payload)
    assert len(out) == 2
    # 2024-01-01T00:00:00 +01:00 == 2023-12-31T23:00:00 UTC
    key = datetime(2023, 12, 31, 23, 0, tzinfo=_UTC)
    assert out[key] == pytest.approx(0.120)


def test_parse_ree_response_missing_pvpc_raises():
    payload = {"included": [{"type": "Mercado SPOT Diario", "attributes": {"values": []}}]}
    with pytest.raises(RuntimeError, match="no PVPC series"):
        _parse_ree_response(payload)


def test_non_strict_skips_unpublished_window(tmp_path):
    from urllib.error import HTTPError

    svc = PvpcService(db_path=tmp_path / "pvpc.db")
    start = _MADRID.localize(datetime(2024, 1, 1, 0, 0))
    with patch(
        "web.backend.services.pvpc_service.urllib.request.urlopen",
        return_value=_urlopen_mock(_ree_payload(_madrid_hours(datetime(2024, 1, 1, 0, 0), 48))),
    ):
        assert len(svc.get_prices(start, _MADRID.localize(datetime(2024, 1, 2, 23, 0)))) == 48

    # "Tomorrow" is 2024-01-03 and every missing hour is in it: REE answers 502 for the whole window.
    bad_gateway = HTTPError("https://apidatos.ree.es", 502, "Bad Gateway", {}, BytesIO(b""))
    end = _MADRID.localize(datetime(2024, 1, 3, 23, 0))
    tomorrow = _MADRID.localize(datetime(2024, 1, 3)).astimezone(_UTC)
    with patch("web.backend.services.pvpc_service._unpublished_from", return_value=tomorrow), \
            patch("web.backend.services.pvpc_service.urllib.request.urlopen", side_effect=bad_gateway):
        prices = svc.get_prices(start, end, strict=False)
        assert len(prices) == 48
        with pytest.raises(RuntimeError, match="PVPC fetch failed"):
            svc.get_prices(start, end)


def test_non_strict_still_raises_for_published_hours(tmp_path):
    from urllib.error import URLError

    svc = PvpcService(db_path=tmp_path / "pvpc.db")
    tomorrow = _MADRID.localize(datetime(2024, 1, 3)).astimezone(_UTC)
    with patch("web.backend.services.pvpc_service._unpublished_from", return_value=tomorrow), \
            patch("web.backend.services.pvpc_service.urllib.request.urlopen", side_effect=URLError("down")):
        with pytest.raises(RuntimeError, match="PVPC fetch failed"):
            svc.get_prices(_MADRID.localize(datetime(2024, 1, 2, 0, 0)),
                           _MADRID.localize(datetime(2024, 1, 3, 23, 0)), strict=False)


def _strict_ree(calls):
    """Fake REE: HTTP 400 when start_date == end_date, else every local hour in [start, end]."""
    from urllib.error import HTTPError
    from urllib.parse import parse_qs, urlparse

    def urlopen(req, timeout=None):
        q = parse_qs(urlparse(req.full_url).query)
        start, end = q["start_date"][0], q["end_date"][0]
        calls.append((start, end))
        if start >= end:
            raise HTTPError(req.full_url, 400, "Bad Request", {}, BytesIO(b""))
        first = _MADRID.localize(datetime.fromisoformat(start), is_dst=True).astimezone(_UTC)
        last = _MADRID.localize(datetime.fromisoformat(end), is_dst=False).astimezone(_UTC)
        values, h = [], first
        while h <= last:
            values.append({"value": 100.0 + h.hour, "datetime": h.astimezone(_MADRID).isoformat()})
            h += timedelta(hours=1)
        return _urlopen_mock(_ree_payload(values))
    return urlopen


@pytest.mark.parametrize("missing_hour", [0, 7, 23])
def test_single_missing_hour_is_fetched_with_a_wider_window(tmp_path, missing_hour):
    svc = PvpcService(db_path=tmp_path / "pvpc.db")
    first = _MADRID.localize(datetime(2026, 10, 5)).astimezone(_UTC)
    hours = [first + timedelta(hours=h) for h in range(24)]
    svc._insert_prices({h: 0.1 for i, h in enumerate(hours) if i != missing_hour})
    calls = []
    with patch("web.backend.services.pvpc_service.urllib.request.urlopen", side_effect=_strict_ree(calls)):
        prices = svc.get_prices(hours[0], hours[-1], strict=False)
    assert sorted(prices) == hours
    assert prices[hours[missing_hour]] == pytest.approx((100.0 + hours[missing_hour].hour) / 1000)
    assert len(calls) == 1
    assert len(svc._query_cached(hours[0] - timedelta(hours=3), hours[-1] + timedelta(hours=3))) == 24


def test_dst_fall_back_second_hour_alone_is_fetchable(tmp_path):
    svc = PvpcService(db_path=tmp_path / "pvpc.db")
    second_two_am = datetime(2026, 10, 25, 1, 0, tzinfo=_UTC)
    calls = []
    with patch("web.backend.services.pvpc_service.urllib.request.urlopen", side_effect=_strict_ree(calls)):
        prices = svc.get_prices(second_two_am, second_two_am)
    assert list(prices) == [second_two_am]
    assert calls[-1][0] < calls[-1][1]


def test_chunk_tail_of_one_hour_is_fetchable(tmp_path):
    svc = PvpcService(db_path=tmp_path / "pvpc.db")
    start = _MADRID.localize(datetime(2026, 1, 1)).astimezone(_UTC)
    end = start + timedelta(days=28)
    calls = []
    with patch("web.backend.services.pvpc_service.urllib.request.urlopen", side_effect=_strict_ree(calls)):
        prices = svc.get_prices(start, end)
    assert len(prices) == 28 * 24 + 1
    assert len(calls) == 2
