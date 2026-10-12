"""Synthetic weather and PV plants with known properties."""

from __future__ import annotations

import numpy as np

from custom_components.ml_pv_forecast.core.solar import (
    clear_sky_ghi,
    plane_irradiance,
    sun_position,
)
from custom_components.ml_pv_forecast.core.weather import WeatherData

LAT, LON = 50.0, 10.0
EAST_WEST = ((30.0, 90.0, 0.5), (30.0, 270.0, 0.5))
START_2024 = 1704067200  # 2024-01-01 00:00 UTC


def synthetic_weather(start: int, count: int, interval: int = 3600, seed: int = 0) -> WeatherData:
    rng = np.random.default_rng(seed)
    times = start + np.arange(count, dtype=np.int64) * interval
    elevation, _ = sun_position(times + interval / 2, LAT, LON)
    clearsky = clear_sky_ghi(elevation)
    day = (times - times[0]) // 86400
    daily_cloud = rng.uniform(0.0, 1.0, int(day.max()) + 1)[day]
    cloud = np.clip(daily_cloud + rng.normal(0.0, 0.1, count), 0.0, 1.0)
    ghi = clearsky * (1.0 - 0.75 * cloud**2)
    dhi = ghi * np.clip(0.15 + 0.8 * cloud, 0.0, 1.0)
    sin_el = np.sin(np.radians(np.maximum(elevation, 1.0)))
    dni = np.where(elevation > 1.0, (ghi - dhi) / sin_el, 0.0)
    doy = (times / 86400) % 365.2425
    temperature = 10.0 - 10.0 * np.cos(2 * np.pi * (doy - 15) / 365.2425)
    temperature = temperature + rng.normal(0.0, 2.0, count)
    pct = cloud * 100.0
    zeros = np.zeros(count)
    return WeatherData(
        times=times,
        interval=interval,
        ghi=ghi,
        dni=dni,
        dhi=dhi,
        temperature=temperature,
        cloud_low=pct,
        cloud_mid=pct * 0.5,
        cloud_high=pct * 0.3,
        snow_depth=zeros,
        is_reanalysis=zeros.copy(),
    )


def synthetic_production(
    weather: WeatherData,
    planes=EAST_WEST,
    kwp: float = 8.0,
    shade_after_hour_utc: float | None = None,
    clip_w: float | None = None,
    seed: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mid = weather.times + weather.interval / 2
    elevation, azimuth = sun_position(mid, LAT, LON)
    poa = sum(
        share
        * plane_irradiance(weather.ghi, weather.dni, weather.dhi, elevation, azimuth, tilt, az)
        for tilt, az, share in planes
    )
    watts = poa / 1000.0 * kwp * 1000.0 * 0.85
    watts = watts * (1.0 - 0.004 * (weather.temperature + poa / 40.0 - 25.0))
    if shade_after_hour_utc is not None:
        hour = (mid % 86400) / 3600
        watts = np.where(hour >= shade_after_hour_utc, watts * 0.3, watts)
    watts = np.maximum(watts * rng.normal(1.0, 0.03, len(watts)), 0.0)
    if clip_w is not None:
        watts = np.minimum(watts, clip_w)
    return watts * weather.interval / 3600.0
