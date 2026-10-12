from datetime import date

import numpy as np
import pytest

from custom_components.ml_pv_forecast.core.weather import (
    ARCHIVE_URL,
    FORECAST_URL,
    HISTORICAL_URL,
    HOUR,
    QUARTER,
    OpenMeteoClient,
    OpenMeteoError,
    WeatherData,
    cloud_total,
    parse_block,
)

VARS = [
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "temperature_2m",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "snow_depth",
]


def block(times, value=1.0):
    return {"time": times, **{v: [value] * len(times) for v in VARS}}


class FakeResponse:
    def __init__(self, status, payload):
        self.status, self._payload = status, payload

    async def json(self, content_type=None):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, handler):
        self.calls, self._handler = [], handler

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return FakeResponse(*self._handler(url, params))


def test_parse_block_keys_by_interval_start_and_maps_none_to_nan():
    payload = {"hourly": block([7200, 10800])}
    payload["hourly"]["snow_depth"] = [None, 0.1]
    w = parse_block(payload, "hourly", HOUR, False)
    assert list(w.times) == [3600, 7200]
    assert np.isnan(w.snow_depth[0]) and w.snow_depth[1] == 0.1
    assert w.is_reanalysis.tolist() == [0.0, 0.0]


def test_parse_block_missing_block_raises():
    with pytest.raises(OpenMeteoError):
        parse_block({}, "hourly", HOUR, False)


def test_concat_sorts_and_keeps_last_duplicate():
    a = parse_block({"hourly": block([7200, 10800], 1.0)}, "hourly", HOUR, False)
    b = parse_block({"hourly": block([10800, 14400], 2.0)}, "hourly", HOUR, False)
    w = WeatherData.concat([a, b])
    assert list(w.times) == [3600, 7200, 10800]
    assert w.ghi.tolist() == [1.0, 2.0, 2.0]


def test_take_and_cloud_total():
    w = parse_block({"hourly": block([3600, 7200], 30.0)}, "hourly", HOUR, False)
    assert len(w.take(np.array([1]))) == 1
    assert cloud_total(w).tolist() == [30.0, 30.0]


async def test_forecast_parses_hourly_and_quarter_blocks():
    payload = {"hourly": block([3600]), "minutely_15": block([900, 1800])}
    session = FakeSession(lambda url, params: (200, payload))
    hourly, quarter = await OpenMeteoClient(session, 50.0, 10.0).forecast()
    assert session.calls[0][0] == FORECAST_URL
    assert hourly.interval == HOUR and quarter.interval == QUARTER
    assert list(quarter.times) == [0, 900]


async def test_history_uses_archive_before_2022_and_historical_after(tmp_path):
    def handler(url, params):
        start = date.fromisoformat(params["start_date"])
        t = int((start - date(1970, 1, 1)).days * 86400) + HOUR
        return 200, {"hourly": block([t])}

    session = FakeSession(handler)
    client = OpenMeteoClient(session, 50.0, 10.0, tmp_path)
    w = await client.history(date(2021, 12, 1), date(2022, 1, 10), today=date(2026, 10, 12))
    urls = [c[0] for c in session.calls]
    assert urls == [ARCHIVE_URL, HISTORICAL_URL]
    assert w.is_reanalysis.tolist() == [1.0, 0.0]
    await client.history(date(2021, 12, 1), date(2022, 1, 10), today=date(2026, 10, 12))
    assert len(session.calls) == 2  # second call served from cache


async def test_http_error_raises():
    session = FakeSession(lambda url, params: (400, {"error": True, "reason": "bad"}))
    with pytest.raises(OpenMeteoError, match="bad"):
        await OpenMeteoClient(session, 50.0, 10.0).forecast()
