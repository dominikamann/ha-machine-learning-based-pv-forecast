"""Marks rows that must not be used for training."""

from __future__ import annotations

import numpy as np

from .features import col

YEAR = 365 * 86400


def effective_capacity(times: np.ndarray, production: np.ndarray, now: int) -> float:
    """99th percentile of hourly production over the last year (all data if less)."""
    finite = np.isfinite(production)
    recent = finite & (times >= now - YEAR)
    values = production[recent] if recent.sum() >= 24 else production[finite]
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, 99))


def usable_mask(production: np.ndarray, X: np.ndarray, capacity: float) -> np.ndarray:
    """True for hourly rows that are safe to train on.

    Hours capped by an export or inverter limit are kept: the capped value is what the meter will
    show, and dropping them would leave only the weaker hours at high irradiance and bias the
    forecast low.
    """
    elevation = X[:, col("sun_elevation")]
    clearsky_index = np.nan_to_num(X[:, col("clearsky_index")])
    with np.errstate(invalid="ignore"):
        ok = (
            np.isfinite(production)
            & (elevation >= 2.0)
            & (production >= 0)
            & (production <= 1.5 * capacity)
        )
        # Sun but no production is an outage – unless snow covers the panels: keep those hours so
        # the model learns what snow does.
        snow = np.nan_to_num(X[:, col("snow_depth")]) > 0
        ok &= ~((production < 0.02 * capacity) & (clearsky_index > 0.5) & ~snow)
    return ok
