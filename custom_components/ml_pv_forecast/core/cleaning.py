"""Marks rows that must not be used for training."""

from __future__ import annotations

import numpy as np

from .features import PLANE_COLUMNS, col

YEAR = 365 * 86400
HOUR = 3600


def effective_capacity(times: np.ndarray, production: np.ndarray, now: int) -> float:
    """99th percentile of hourly production over the last year (all data if less)."""
    finite = np.isfinite(production)
    recent = finite & (times >= now - YEAR)
    values = production[recent] if recent.sum() >= 24 else production[finite]
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, 99))


def plateau_mask(times: np.ndarray, production: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Hours in a flat 3-hour plateau at the monthly maximum while irradiance changes."""
    n = len(times)
    mask = np.zeros(n, dtype=bool)
    if n < 3:
        return mask
    month = np.floor((times / 86400) % 365.2425 / 30.44).astype(int)
    month_max = np.zeros(13)
    np.maximum.at(month_max, month, np.nan_to_num(production))
    irradiance = X[:, PLANE_COLUMNS].max(axis=1)
    a, b, c = slice(0, n - 2), slice(1, n - 1), slice(2, n)
    contiguous = (times[b] - times[a] == HOUR) & (times[c] - times[b] == HOUR)
    p = np.stack([production[a], production[b], production[c]])
    g = np.stack([irradiance[a], irradiance[b], irradiance[c]])
    with np.errstate(invalid="ignore"):
        p_max, p_min = p.max(axis=0), p.min(axis=0)
        g_max, g_min = g.max(axis=0), g.min(axis=0)
        flat = (
            np.isfinite(p).all(axis=0)
            & (p_max > 0)
            & (p_max - p_min <= 0.02 * p_max)
            & (p_max >= 0.95 * month_max[month[b]])
            & (g_max - g_min > 0.1 * g_max)
        )
    hits = np.flatnonzero(contiguous & flat)
    for offset in (0, 1, 2):
        mask[hits + offset] = True
    return mask


def usable_mask(
    times: np.ndarray,
    production: np.ndarray,
    X: np.ndarray,
    capacity: float,
    limit_wh: float | None = None,
) -> np.ndarray:
    """True for hourly rows that are safe to train on."""
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
        if limit_wh:
            ok &= production < 0.97 * limit_wh
    ok &= ~plateau_mask(times, production, X)
    return ok
