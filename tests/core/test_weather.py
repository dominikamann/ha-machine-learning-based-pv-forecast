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
    # fixed 90-day boundaries: 2021-09-06..12-04, 2021-12-05..12-31 (ERA5), 2022-01-01.. (archive)
    assert urls == [ARCHIVE_URL, ARCHIVE_URL, HISTORICAL_URL]
    assert w.is_reanalysis.tolist() == [1.0, 1.0, 0.0]
    await client.history(date(2021, 12, 1), date(2022, 1, 10), today=date(2026, 10, 12))
    assert len(session.calls) == 3  # second call served from cache


async def test_http_error_raises():
    session = FakeSession(lambda url, params: (400, {"error": True, "reason": "bad"}))
    with pytest.raises(OpenMeteoError, match="bad"):
        await OpenMeteoClient(session, 50.0, 10.0).forecast()


from custom_components.ml_pv_forecast.core.weather import (  # noqa: E402
    ENSEMBLE_MODELS,
    PREVIOUS_RUNS_URL,
)

ENS_VARS = ["shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation", "cloud_cover"]


def multi_block(times, suffix="", base=1.0, ens=2.0):
    data = {"time": times}
    for v in VARS:
        data[f"{v}{suffix}_best_match"] = [base] * len(times)
    for m in ENSEMBLE_MODELS:
        for v in ENS_VARS:
            data[f"{v}{suffix}_{m}"] = [ens] * len(times)
    return data


def test_parse_multi_model_block_with_lead():
    payload = {"hourly": multi_block([7200, 10800], suffix="_previous_day1")}
    w = parse_block(payload, "hourly", HOUR, False, suffix="_previous_day1", lead=1)
    assert w.ghi.tolist() == [1.0, 1.0]
    assert w.ens_ghi.shape == (2, len(ENSEMBLE_MODELS)) and (w.ens_ghi == 2.0).all()
    assert w.lead_days.tolist() == [1.0, 1.0]


def test_weather_without_ensemble_gets_nan_columns_that_survive_take_and_concat():
    w = parse_block({"hourly": block([3600, 7200])}, "hourly", HOUR, False)
    assert w.ens_ghi.shape == (2, len(ENSEMBLE_MODELS)) and np.isnan(w.ens_ghi).all()
    assert w.lead_days.tolist() == [0.0, 0.0]
    both = WeatherData.concat([w, w.take(np.array([1]))])
    assert both.ens_cloud.shape == (2, len(ENSEMBLE_MODELS))


async def test_history_leads_fetch_previous_runs_from_2024(tmp_path):
    def handler(url, params):
        start = date.fromisoformat(params["start_date"])
        t = int((start - date(1970, 1, 1)).days * 86400) + HOUR
        hourly = params["hourly"].split(",")
        suffix = (
            "_previous_day" + hourly[0].split("_previous_day")[1]
            if "_previous_day" in hourly[0]
            else ""
        )
        return 200, {"hourly": multi_block([t], suffix=suffix)}

    session = FakeSession(handler)
    client = OpenMeteoClient(session, 50.0, 10.0, tmp_path)
    leads = await client.history_leads(
        date(2023, 12, 1), date(2024, 1, 10), today=date(2026, 10, 12)
    )
    assert [w.lead_days[0] for w in leads] == [0.0, 1.0, 2.0]
    previous = [c for c in session.calls if c[0] == PREVIOUS_RUNS_URL]
    assert {c[1]["start_date"] for c in previous} == {"2024-01-01"}
    assert all("models" in c[1] for c in session.calls)
    assert leads[1].ens_ghi[0, 0] == 2.0


async def test_rate_limit_waits_and_retries(monkeypatch):
    from custom_components.ml_pv_forecast.core import weather as weather_module

    waits = []

    async def fake_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(weather_module.asyncio, "sleep", fake_sleep)
    answers = iter([(429, {"error": True, "reason": "Minutely API request limit exceeded"})])

    def handler(url, params):
        return next(answers, (200, {"hourly": block([3600]), "minutely_15": block([900])}))

    hourly, _ = await OpenMeteoClient(FakeSession(handler), 50.0, 10.0).forecast()
    assert len(hourly) == 1 and waits == [61]


async def test_rate_limit_gives_up_after_retries(monkeypatch):
    from custom_components.ml_pv_forecast.core import weather as weather_module

    async def fake_sleep(seconds):
        return None

    monkeypatch.setattr(weather_module.asyncio, "sleep", fake_sleep)
    session = FakeSession(lambda url, params: (429, {"error": True, "reason": "limit"}))
    with pytest.raises(OpenMeteoError, match="limit"):
        await OpenMeteoClient(session, 50.0, 10.0).forecast()


async def test_cache_is_shared_between_different_start_dates(tmp_path):
    def handler(url, params):
        start = date.fromisoformat(params["start_date"])
        t = int((start - date(1970, 1, 1)).days * 86400) + HOUR
        return 200, {"hourly": multi_block([t])}

    session = FakeSession(handler)
    client = OpenMeteoClient(session, 50.0, 10.0, tmp_path)
    today = date(2026, 10, 12)
    await client.history(date(2023, 1, 5), date(2024, 6, 30), today)
    first = len(session.calls)
    await client.history(date(2023, 2, 20), date(2024, 6, 30), today)
    assert len(session.calls) == first  # a plant starting later reuses every cached chunk


def test_weather_code_is_parsed_when_present():
    payload = {"hourly": {**multi_block([7200]), "weather_code_best_match": [61]}}
    w = parse_block(payload, "hourly", HOUR, False, suffix="")
    assert w.weather_code.tolist() == [61.0]
    plain = parse_block({"hourly": block([7200])}, "hourly", HOUR, False)
    assert np.isnan(plain.weather_code).all()
