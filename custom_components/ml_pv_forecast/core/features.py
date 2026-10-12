"""Feature matrix for the PV models."""

from __future__ import annotations

import numpy as np

from .solar import VIRTUAL_PLANES, clear_sky_ghi, plane_irradiance, sun_position
from .weather import ENSEMBLE_MODELS, WeatherData

# Bump when the feature set changes: stored models with another version are retrained.
FEATURE_VERSION = 2

PLANE_FEATURES = [f"plane_{int(tilt)}_{int(azimuth)}" for tilt, azimuth in VIRTUAL_PLANES]
FEATURE_NAMES = [
    *PLANE_FEATURES,
    "ghi",
    "dni",
    "dhi",
    "clearsky_ghi",
    "clearsky_index",
    "sun_elevation",
    "sun_azimuth",
    "temperature",
    "cloud_low",
    "cloud_mid",
    "cloud_high",
    "snow_depth",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "is_reanalysis",
    *(f"ens_ghi_{model}" for model in ENSEMBLE_MODELS),
    "ens_ghi_mean",
    "ens_ghi_spread",
    "ens_cloud_mean",
    "ens_count",
    "lead_days",
]
_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
PLANE_COLUMNS = [_INDEX[name] for name in PLANE_FEATURES]


def col(name: str) -> int:
    """Column index of a feature."""
    return _INDEX[name]


def build_features(weather: WeatherData, latitude: float, longitude: float) -> np.ndarray:
    """One row per weather interval, columns as in FEATURE_NAMES."""
    mid = weather.times.astype(float) + weather.interval / 2
    elevation, azimuth = sun_position(mid, latitude, longitude)
    ghi = np.nan_to_num(weather.ghi)
    dni = np.nan_to_num(weather.dni)
    dhi = np.nan_to_num(weather.dhi)
    clearsky = clear_sky_ghi(elevation)
    with np.errstate(divide="ignore", invalid="ignore"):
        clearsky_index = np.where(clearsky > 20.0, np.clip(ghi / clearsky, 0.0, 1.5), np.nan)
    planes = [
        plane_irradiance(ghi, dni, dhi, elevation, azimuth, tilt, plane_azimuth)
        for tilt, plane_azimuth in VIRTUAL_PLANES
    ]
    hour = (mid % 86400) / 3600
    doy = (mid / 86400) % 365.2425
    columns = [
        *planes,
        ghi,
        dni,
        dhi,
        clearsky,
        clearsky_index,
        elevation,
        azimuth,
        weather.temperature,
        weather.cloud_low,
        weather.cloud_mid,
        weather.cloud_high,
        weather.snow_depth,
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * doy / 365.2425),
        np.cos(2 * np.pi * doy / 365.2425),
        weather.is_reanalysis,
        *_ensemble_columns(weather),
        weather.lead_days,
    ]
    return np.column_stack(columns).astype(float)


def _nan_stat(values: np.ndarray, count: np.ndarray, fn) -> np.ndarray:
    """Row statistic over the available models; NaN where no model is available."""
    filled = np.where(np.isfinite(values), values, 0.0)
    out = fn(filled, values, np.maximum(count, 1))
    return np.where(count > 0, out, np.nan)


def _ensemble_columns(weather: WeatherData) -> list[np.ndarray]:
    ghi, cloud = weather.ens_ghi, weather.ens_cloud
    count = np.isfinite(ghi).sum(axis=1).astype(float)
    mean = _nan_stat(ghi, count, lambda f, v, c: f.sum(axis=1) / c)
    spread = _nan_stat(
        ghi,
        count,
        lambda f, v, c: np.sqrt(
            np.where(np.isfinite(v), (f - (f.sum(axis=1) / c)[:, None]) ** 2, 0.0).sum(axis=1) / c
        ),
    )
    cloud_count = np.isfinite(cloud).sum(axis=1).astype(float)
    cloud_mean = _nan_stat(cloud, cloud_count, lambda f, v, c: f.sum(axis=1) / c)
    return [*ghi.T, mean, spread, cloud_mean, count]
