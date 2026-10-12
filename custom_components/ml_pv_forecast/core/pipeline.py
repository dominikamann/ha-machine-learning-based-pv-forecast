"""Glue between weather, features, models and the forecast container."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

import numpy as np

from .downscale import to_quarters
from .evaluation import Correction
from .features import build_features
from .model import PlantModel, TrainResult, train
from .nowcast import DEFAULT_TAU, adjust
from .weather import WeatherData, cloud_total

HOUR = 3600
QUARTER = 900
DAY = 86400


def _integrate(times: np.ndarray, values: np.ndarray, step: int, start: int, end: int) -> np.ndarray:
    overlap = np.clip(np.minimum(times + step, end) - np.maximum(times, start), 0, None) / step
    return (values * overlap[:, None]).sum(axis=0)


@dataclass
class PlantForecast:
    """Forecast for one plant (or the total). Values are Wh per interval, P10/P50/P90."""

    plant_id: str
    issued_at: int
    hour_times: np.ndarray
    hourly: np.ndarray
    quarter_times: np.ndarray
    quarter: np.ndarray
    quarter_raw: np.ndarray
    ratio: float | None = None

    def energy(self, start: int, end: int) -> np.ndarray:
        if len(self.quarter_times):
            return _integrate(self.quarter_times, self.quarter, QUARTER, start, end)
        return _integrate(self.hour_times, self.hourly, HOUR, start, end)

    def raw_energy(self, start: int, end: int) -> np.ndarray:
        return _integrate(self.quarter_times, self.quarter_raw, QUARTER, start, end)

    def peak(self, start: int, end: int) -> tuple[float, int] | None:
        mask = (self.quarter_times >= start) & (self.quarter_times < end)
        if not mask.any():
            return None
        watts = self.quarter[mask, 1] * (HOUR / QUARTER)
        i = int(np.argmax(watts))
        return float(watts[i]), int(self.quarter_times[mask][i])


def _aligned_sum(times: np.ndarray, parts: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    return sum(values[np.searchsorted(t, times)] for t, values in parts)


def total_forecast(forecasts: list[PlantForecast], issued_at: int) -> PlantForecast:
    hour_times = reduce(np.intersect1d, [f.hour_times for f in forecasts])
    quarter_times = reduce(np.intersect1d, [f.quarter_times for f in forecasts])
    return PlantForecast(
        "total",
        issued_at,
        hour_times,
        _aligned_sum(hour_times, [(f.hour_times, f.hourly) for f in forecasts]),
        quarter_times,
        _aligned_sum(quarter_times, [(f.quarter_times, f.quarter) for f in forecasts]),
        _aligned_sum(quarter_times, [(f.quarter_times, f.quarter_raw) for f in forecasts]),
    )


def window_cloud(weather: WeatherData, start: int, end: int) -> float:
    mask = (weather.times + weather.interval > start) & (weather.times < end)
    return float(np.mean(cloud_total(weather)[mask])) if mask.any() else 0.0


def train_plant(
    prod_times: np.ndarray,
    prod_wh: np.ndarray,
    weather: WeatherData,
    latitude: float,
    longitude: float,
    now: int,
    active: PlantModel | None,
    limit_w: float | None,
    config_hash: str,
) -> TrainResult:
    common, prod_index, weather_index = np.intersect1d(
        prod_times, weather.times, return_indices=True
    )
    X = build_features(weather.take(weather_index), latitude, longitude)
    result = train(common, X, prod_wh[prod_index], now, active, limit_w)
    result.model.config_hash = config_hash
    return result


def forecast_plant(
    plant_id: str,
    model: PlantModel,
    weather_hourly: WeatherData,
    weather_quarter: WeatherData,
    latitude: float,
    longitude: float,
    now: int,
    correction: Correction | None = None,
    ratio: float | None = None,
    tau: float = DEFAULT_TAU,
) -> PlantForecast:
    times = weather_hourly.times
    hourly = model.predict(build_features(weather_hourly, latitude, longitude))
    if correction is not None:
        factors = correction.factors(
            (times - now) / HOUR,
            (times % DAY) / HOUR,
            (times / DAY) % 365.2425,
            cloud_total(weather_hourly),
            hourly[:, 1] / model.capacity,
        )
        hourly = hourly * factors[:, None]
    quarter_raw = to_quarters(times, hourly, weather_quarter.times, weather_quarter.ghi)
    valid = np.isfinite(quarter_raw).all(axis=1)
    quarter_times = weather_quarter.times[valid]
    quarter_raw = quarter_raw[valid]
    quarter = (
        adjust(quarter_raw, (quarter_times - now) / 60, ratio, tau)
        if ratio is not None
        else quarter_raw.copy()
    )
    return PlantForecast(plant_id, now, times, hourly, quarter_times, quarter, quarter_raw, ratio)
