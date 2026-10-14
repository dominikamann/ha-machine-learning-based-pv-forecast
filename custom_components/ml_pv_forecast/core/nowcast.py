"""Short-horizon correction from live production and from the day so far."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Defaults from backtests on two plants over a year (2-3 h ahead about 0.2 points better than
# the live ratio alone); each plant then learns its own values from the archive.
DEFAULT_TAU = 60.0
# Capped at three hours: a forecast that was too low for days rewards an ever longer decay, which
# then carries a weak morning hour into the whole afternoon.
TAU_GRID = np.arange(30.0, 181.0, 15.0)
MIN_ROWS = 50
DEFAULT_DAY_WEIGHT = 0.6
DEFAULT_DAY_TAU = 240.0
DAY_WEIGHT_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DAY_TAU_GRID = (60.0, 120.0, 240.0, 480.0)
MIN_DAY_ROWS = 500  # about two weeks of short-horizon rows


@dataclass(frozen=True)
class DayFactor:
    """How today has gone so far: actual/forecast since midnight and the share of the day done."""

    ratio: float
    share: float
    weight: float = DEFAULT_DAY_WEIGHT
    tau: float = DEFAULT_DAY_TAU


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


def day_ratio(actual_wh: float, forecast_wh: float, capacity_wh: float) -> float | None:
    """Actual/forecast since midnight, or None while too little was forecast to judge."""
    if not (np.isfinite(actual_wh) and np.isfinite(forecast_wh)):
        return None
    if forecast_wh < 0.01 * capacity_wh:
        return None
    return float(np.clip(actual_wh / forecast_wh, 0.0, 3.0))


def _factor(
    lead: np.ndarray,
    ratio: np.ndarray | float | None,
    tau: float,
    day_ratio_: np.ndarray | float | None = None,
    share: np.ndarray | float = 0.0,
    weight: float = 0.0,
    day_tau: float = DEFAULT_DAY_TAU,
) -> np.ndarray:
    """The live ratio rules the next minutes; the day so far takes over as it fades."""
    lead = np.maximum(lead, 0.0)
    live = np.exp(-lead / tau)
    factor = np.ones_like(lead) if ratio is None else 1.0 + (np.asarray(ratio) - 1.0) * live
    if day_ratio_ is not None and weight:
        day = weight * np.clip(share, 0.0, 1.0) * (np.asarray(day_ratio_) - 1.0)
        factor = factor + day * (1.0 - live) * np.exp(-lead / day_tau)
    return factor


def adjust(
    values: np.ndarray,
    lead_minutes: np.ndarray,
    ratio: float | None,
    tau: float,
    day: DayFactor | None = None,
) -> np.ndarray:
    """Blend the live ratio and the day so far into quarters that are not finished yet."""
    factor = _factor(
        lead_minutes,
        ratio,
        tau,
        None if day is None else day.ratio,
        0.0 if day is None else day.share,
        0.0 if day is None else day.weight,
        DEFAULT_DAY_TAU if day is None else day.tau,
    )
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


def fit_day(
    raw_p50: np.ndarray,
    ratio: np.ndarray,
    lead_minutes: np.ndarray,
    tau: float,
    day_ratio_: np.ndarray,
    share: np.ndarray,
    actual: np.ndarray,
) -> tuple[float, float]:
    """Weight and decay (minutes) of the day so far that minimise the adjusted error."""
    if len(actual) < MIN_DAY_ROWS:
        return DEFAULT_DAY_WEIGHT, DEFAULT_DAY_TAU
    best, best_error = (DEFAULT_DAY_WEIGHT, DEFAULT_DAY_TAU), np.inf
    for weight in DAY_WEIGHT_GRID:
        for day_tau in DAY_TAU_GRID if weight else DAY_TAU_GRID[:1]:
            factor = _factor(lead_minutes, ratio, tau, day_ratio_, share, weight, day_tau)
            error = float(np.mean(np.abs(raw_p50 * factor - actual)))
            if error < best_error - 1e-9:
                best, best_error = (weight, day_tau), error
    return best
