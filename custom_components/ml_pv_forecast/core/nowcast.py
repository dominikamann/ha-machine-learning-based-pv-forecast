"""Short-horizon correction from live production."""

from __future__ import annotations

import numpy as np

DEFAULT_TAU = 90.0
TAU_GRID = np.arange(30.0, 361.0, 15.0)
MIN_ROWS = 50


def live_ratio(
    actual_wh: float,
    forecast_wh: float,
    elevation: float,
    capacity_wh: float,
    window_hours: float = 0.75,
) -> float | None:
    """Actual/forecast over the recent window, or None when it is not meaningful."""
    if not (np.isfinite(actual_wh) and np.isfinite(forecast_wh)):
        return None
    if elevation <= 5.0 or forecast_wh < 0.05 * capacity_wh * window_hours:
        return None
    return float(np.clip(actual_wh / forecast_wh, 0.0, 2.0))


def adjust(values: np.ndarray, lead_minutes: np.ndarray, ratio: float, tau: float) -> np.ndarray:
    """Blend the live ratio into quarters that are not finished yet, fading with lead time."""
    factor = 1.0 + (ratio - 1.0) * np.exp(-np.maximum(lead_minutes, 0.0) / tau)
    factor = np.where(lead_minutes > -15.0, factor, 1.0)
    return values * factor[:, None]


def fit_tau(
    raw_p50: np.ndarray, ratio: np.ndarray, lead_minutes: np.ndarray, actual: np.ndarray
) -> float:
    """Decay constant (minutes) that minimises the adjusted error."""
    if len(actual) < MIN_ROWS:
        return DEFAULT_TAU
    errors = [
        np.mean(np.abs(raw_p50 * (1 + (ratio - 1) * np.exp(-lead_minutes / tau)) - actual))
        for tau in TAU_GRID
    ]
    return float(TAU_GRID[int(np.argmin(errors))])
