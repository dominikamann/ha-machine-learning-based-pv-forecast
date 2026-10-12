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
HISTORICAL_START = date(2022, 1, 1)
CHUNK_DAYS = 90

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


class OpenMeteoError(Exception):
    """Open-Meteo could not deliver data."""


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


def parse_block(
    payload: dict[str, Any], block: str, interval: int, is_reanalysis: bool
) -> WeatherData:
    """Parse one Open-Meteo time block."""
    data = payload.get(block)
    if not data or "time" not in data:
        raise OpenMeteoError(f"response has no {block} data")
    # Open-Meteo labels radiation with the end of the interval it averages; we key by start.
    times = np.asarray(data["time"], dtype=np.int64) - interval
    n = len(times)
    columns = {}
    for api_name, name in VARIABLES.items():
        raw = data.get(api_name)
        columns[name] = (
            np.full(n, np.nan)
            if raw is None
            else np.array([np.nan if v is None else v for v in raw], dtype=float)
        )
    return WeatherData(
        times=times, interval=interval, is_reanalysis=np.full(n, float(is_reanalysis)), **columns
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
        try:
            async with self._session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                payload = await resp.json(content_type=None)
                if resp.status != 200 or payload.get("error"):
                    reason = payload.get("reason", "") if isinstance(payload, dict) else ""
                    raise OpenMeteoError(f"{url}: HTTP {resp.status} {reason}".strip())
                return payload
        except (aiohttp.ClientError, TimeoutError, ValueError) as err:
            raise OpenMeteoError(f"{url}: {err}") from err

    async def forecast(self) -> tuple[WeatherData, WeatherData]:
        """Hourly and 15-minute forecast from yesterday to three days ahead."""
        payload = await self._get(
            FORECAST_URL,
            self._params(
                hourly=",".join(VARIABLES),
                minutely_15=",".join(QUARTER_VARIABLES),
                past_days="1",
                forecast_days="4",
            ),
        )
        return (
            parse_block(payload, "hourly", HOUR, False),
            parse_block(payload, "minutely_15", QUARTER, False),
        )

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
        path = None
        if self._cache_dir is not None and end < today - timedelta(days=7):
            kind = "era5" if reanalysis else "hist"
            path = self._cache_dir / f"{kind}_{self._lat:.2f}_{self._lon:.2f}_{start}_{end}.json"
        payload = await asyncio.to_thread(_read_json, path) if path else None
        if payload is None:
            payload = await self._get(
                ARCHIVE_URL if reanalysis else HISTORICAL_URL,
                self._params(
                    hourly=",".join(VARIABLES),
                    start_date=start.isoformat(),
                    end_date=end.isoformat(),
                ),
            )
            if path:
                await asyncio.to_thread(_write_json, path, payload)
        return parse_block(payload, "hourly", HOUR, reanalysis)


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
