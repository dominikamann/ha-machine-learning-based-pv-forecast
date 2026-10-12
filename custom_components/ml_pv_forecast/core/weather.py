"""Open-Meteo client and the weather data container."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from dataclasses import dataclass, fields, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np

HOUR = 3600
QUARTER = 900

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
HISTORICAL_START = date(2022, 1, 1)
PREVIOUS_RUNS_START = date(2024, 1, 1)
CHUNK_DAYS = 90
LEADS = (0, 1, 2)
CACHE_VERSION = 2
RATE_LIMIT_RETRIES = 5
RATE_LIMIT_WAIT = 61  # seconds; Open-Meteo's limit window is one minute

VARIABLES = {
    "shortwave_radiation": "ghi",
    "direct_normal_irradiance": "dni",
    "diffuse_radiation": "dhi",
    "temperature_2m": "temperature",
    "cloud_cover_low": "cloud_low",
    "cloud_cover_mid": "cloud_mid",
    "cloud_cover_high": "cloud_high",
    "snow_depth": "snow_depth",
}
QUARTER_VARIABLES = ("shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation")

# Several weather models give the model a second opinion and a measure of agreement.
ENSEMBLE_MODELS = ("icon_seamless", "meteofrance_seamless", "ecmwf_ifs025", "gfs_seamless")
ENSEMBLE_VARIABLES = {
    "shortwave_radiation": "ens_ghi",
    "direct_normal_irradiance": "ens_dni",
    "diffuse_radiation": "ens_dhi",
    "cloud_cover": "ens_cloud",
}
REQUEST_MODELS = ("best_match", *ENSEMBLE_MODELS)
HOURLY_REQUEST = [*VARIABLES, *(v for v in ENSEMBLE_VARIABLES if v not in VARIABLES)]


class OpenMeteoError(Exception):
    """Open-Meteo could not deliver data."""


class _RateLimited(Exception):
    """HTTP 429 from Open-Meteo; retried after a pause."""


@dataclass(frozen=True)
class WeatherData:
    """Weather series keyed by interval start (unix seconds, UTC)."""

    times: np.ndarray
    interval: int
    ghi: np.ndarray
    dni: np.ndarray
    dhi: np.ndarray
    temperature: np.ndarray
    cloud_low: np.ndarray
    cloud_mid: np.ndarray
    cloud_high: np.ndarray
    snow_depth: np.ndarray
    is_reanalysis: np.ndarray
    ens_ghi: np.ndarray | None = None  # (n, len(ENSEMBLE_MODELS)), NaN where a model is missing
    ens_dni: np.ndarray | None = None
    ens_dhi: np.ndarray | None = None
    ens_cloud: np.ndarray | None = None
    lead_days: np.ndarray | None = None  # 0 = same-day forecast, 1 = issued a day ahead, ...

    def __post_init__(self) -> None:
        n = len(self.times)
        for name in ENSEMBLE_VARIABLES.values():
            if getattr(self, name) is None:
                object.__setattr__(self, name, np.full((n, len(ENSEMBLE_MODELS)), np.nan))
        if self.lead_days is None:
            object.__setattr__(self, "lead_days", np.zeros(n))

    def __len__(self) -> int:
        return len(self.times)

    def columns(self) -> list[str]:
        return [f.name for f in fields(self) if f.name not in ("times", "interval")]

    def take(self, index: np.ndarray) -> WeatherData:
        """Rows selected by a boolean mask or an index array."""
        return replace(
            self, times=self.times[index], **{c: getattr(self, c)[index] for c in self.columns()}
        )

    @staticmethod
    def concat(parts: list[WeatherData]) -> WeatherData:
        """Concatenate, sort by time, keep the last value for duplicate times."""
        parts = [p for p in parts if len(p)]
        if not parts:
            raise OpenMeteoError("no weather data")
        times = np.concatenate([p.times for p in parts])
        reverse_first = np.unique(times[::-1], return_index=True)[1]
        keep = len(times) - 1 - reverse_first
        columns = {
            c: np.concatenate([getattr(p, c) for p in parts])[keep] for c in parts[0].columns()
        }
        return WeatherData(times=times[keep], interval=parts[0].interval, **columns)


def cloud_total(weather: WeatherData) -> np.ndarray:
    """Mean of low, mid and high cloud cover in percent."""
    return (
        np.nan_to_num(weather.cloud_low)
        + np.nan_to_num(weather.cloud_mid)
        + np.nan_to_num(weather.cloud_high)
    ) / 3


def _values(raw: list | None, n: int) -> np.ndarray:
    if raw is None:
        return np.full(n, np.nan)
    return np.array([np.nan if v is None else v for v in raw], dtype=float)


def parse_block(
    payload: dict[str, Any],
    block: str,
    interval: int,
    is_reanalysis: bool,
    suffix: str | None = None,
    lead: int = 0,
) -> WeatherData:
    """Parse one Open-Meteo time block.

    With ``suffix`` set the response comes from a multi-model request (``models=REQUEST_MODELS``):
    keys look like ``shortwave_radiation{suffix}_icon_seamless``; base values use ``best_match``.
    """
    data = payload.get(block)
    if not data or "time" not in data:
        raise OpenMeteoError(f"response has no {block} data")
    # Open-Meteo labels radiation with the end of the interval it averages; we key by start.
    times = np.asarray(data["time"], dtype=np.int64) - interval
    n = len(times)
    multi = suffix is not None
    columns = {
        name: _values(data.get(f"{api}{suffix}_best_match" if multi else api), n)
        for api, name in VARIABLES.items()
    }
    if multi:
        for api, name in ENSEMBLE_VARIABLES.items():
            columns[name] = np.column_stack(
                [_values(data.get(f"{api}{suffix}_{model}"), n) for model in ENSEMBLE_MODELS]
            )
    return WeatherData(
        times=times,
        interval=interval,
        is_reanalysis=np.full(n, float(is_reanalysis)),
        lead_days=np.full(n, float(lead)),
        **columns,
    )


class OpenMeteoClient:
    """Fetches forecasts and history from Open-Meteo (free, no API key)."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        latitude: float,
        longitude: float,
        cache_dir: Path | None = None,
    ) -> None:
        self._session = session
        self._lat = latitude
        self._lon = longitude
        self._cache_dir = cache_dir

    def _params(self, **extra: str) -> dict[str, str]:
        return {
            "latitude": f"{self._lat:.4f}",
            "longitude": f"{self._lon:.4f}",
            "timezone": "GMT",
            "timeformat": "unixtime",
            **extra,
        }

    async def _get(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        """GET with retries when Open-Meteo's per-minute limit is hit (large history downloads)."""
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                return await self._get_once(url, params)
            except _RateLimited as err:
                if attempt == RATE_LIMIT_RETRIES:
                    raise OpenMeteoError(str(err)) from err
                await asyncio.sleep(RATE_LIMIT_WAIT)
        raise AssertionError("unreachable")

    async def _get_once(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        try:
            async with self._session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                payload = await resp.json(content_type=None)
                if resp.status != 200 or payload.get("error"):
                    reason = payload.get("reason", "") if isinstance(payload, dict) else ""
                    message = f"{url}: HTTP {resp.status} {reason}".strip()
                    if resp.status == 429:
                        raise _RateLimited(message)
                    raise OpenMeteoError(message)
                return payload
        except (aiohttp.ClientError, TimeoutError, ValueError) as err:
            raise OpenMeteoError(f"{url}: {err}") from err

    async def forecast(self) -> tuple[WeatherData, WeatherData]:
        """Hourly (all models) and 15-minute forecast from yesterday to three days ahead."""
        hourly = await self._get(
            FORECAST_URL,
            self._params(
                hourly=",".join(HOURLY_REQUEST),
                models=",".join(REQUEST_MODELS),
                past_days="1",
                forecast_days="4",
            ),
        )
        quarter = await self._get(
            FORECAST_URL,
            self._params(minutely_15=",".join(QUARTER_VARIABLES), past_days="1", forecast_days="4"),
        )
        return (
            parse_block(hourly, "hourly", HOUR, False, suffix=""),
            parse_block(quarter, "minutely_15", QUARTER, False),
        )

    async def history_leads(self, start: date, end: date, today: date) -> list[WeatherData]:
        """Training weather per forecast lead: same-day (from 2022), one and two days ahead."""
        leads = [await self.history(start, end, today)]
        first = max(start, PREVIOUS_RUNS_START)
        for lead in LEADS[1:]:
            if first > end:
                break
            parts = [
                await self._previous_runs_chunk(chunk_start, chunk_end, lead, today)
                for chunk_start, chunk_end, _ in _chunks(first, end)
            ]
            leads.append(WeatherData.concat(parts))
        return leads

    async def _previous_runs_chunk(
        self, start: date, end: date, lead: int, today: date
    ) -> WeatherData:
        suffix = f"_previous_day{lead}"
        payload = await self._cached(
            f"prev{lead}",
            start,
            end,
            today,
            PREVIOUS_RUNS_URL,
            hourly=",".join(f"{v}{suffix}" for v in HOURLY_REQUEST),
            models=",".join(REQUEST_MODELS),
        )
        return parse_block(payload, "hourly", HOUR, False, suffix=suffix, lead=lead)

    async def _cached(
        self, kind: str, start: date, end: date, today: date, url: str, **params: str
    ) -> dict[str, Any]:
        path = None
        if self._cache_dir is not None and end < today - timedelta(days=7):
            name = f"v{CACHE_VERSION}_{kind}_{self._lat:.2f}_{self._lon:.2f}_{start}_{end}.json"
            path = self._cache_dir / name
        payload = await asyncio.to_thread(_read_json, path) if path else None
        if payload is None:
            payload = await self._get(
                url, self._params(start_date=start.isoformat(), end_date=end.isoformat(), **params)
            )
            if path:
                await asyncio.to_thread(_write_json, path, payload)
        return payload

    async def history(self, start: date, end: date, today: date) -> WeatherData:
        """Hourly archived forecasts (ERA5 reanalysis before 2022) for [start, end]."""
        parts = [
            await self._history_chunk(chunk_start, chunk_end, reanalysis, today)
            for chunk_start, chunk_end, reanalysis in _chunks(start, end)
        ]
        return WeatherData.concat(parts)

    async def _history_chunk(
        self, start: date, end: date, reanalysis: bool, today: date
    ) -> WeatherData:
        if reanalysis:  # ERA5 is a single reanalysis, no model choice
            payload = await self._cached(
                "era5", start, end, today, ARCHIVE_URL, hourly=",".join(VARIABLES)
            )
            return parse_block(payload, "hourly", HOUR, True)
        payload = await self._cached(
            "hist",
            start,
            end,
            today,
            HISTORICAL_URL,
            hourly=",".join(HOURLY_REQUEST),
            models=",".join(REQUEST_MODELS),
        )
        return parse_block(payload, "hourly", HOUR, False, suffix="")


def _chunks(start: date, end: date) -> Iterator[tuple[date, date, bool]]:
    """Split [start, end] into pieces of at most CHUNK_DAYS not crossing HISTORICAL_START."""
    current = start
    while current <= end:
        reanalysis = current < HISTORICAL_START
        last = min(end, current + timedelta(days=CHUNK_DAYS - 1))
        if reanalysis:
            last = min(last, HISTORICAL_START - timedelta(days=1))
        yield current, last, reanalysis
        current = last + timedelta(days=1)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
