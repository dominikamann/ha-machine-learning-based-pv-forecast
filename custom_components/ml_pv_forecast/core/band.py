"""Calibration of the P10–P90 band from how this plant's morning forecasts actually turned out."""

from __future__ import annotations

import numpy as np

QUARTER = 900
HOUR = 3600
NEUTRAL = (1.0, 1.0)  # (lower, upper) scale of the band around P50
TARGET = 0.1  # a tenth of the hours below P10 and a tenth above P90
GRID = np.round(np.arange(0.5, 3.0001, 0.05), 2)
# Backtests on two plants over a year: half the measured gap keeps a well calibrated plant as sharp
# as before and helps plants that drift (outages, shading, ageing) a lot.
SHRINK = 0.5
WINDOW_DAYS = 60
MIN_HOURS = 70  # about a week of daylight hours
# Days with measured hours; fewer often come from an older model version and mislead the band.
MIN_DAYS = 14


def widen(values: np.ndarray, scale: tuple[float, float]) -> np.ndarray:
    """Scale the lower and upper half of the band (columns P10, P50, P90) around P50."""
    out = values.copy()
    out[:, 0] = np.maximum(values[:, 1] - scale[0] * (values[:, 1] - values[:, 0]), 0.0)
    out[:, 2] = values[:, 1] + scale[1] * (values[:, 2] - values[:, 1])
    return out


def hourly_rows(
    curve: list,
    applied: tuple[float, float],
    actual_times: np.ndarray,
    actual_wh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hourly P10/P50/P90 of a morning curve (with ``applied`` undone) and the actual hours.

    Only hours whose four quarters were all measured are returned.
    """
    rows = np.asarray(curve, dtype=float).reshape(-1, 4)
    hours = (rows[:, 0] // HOUR).astype(np.int64)
    measured = dict(zip(actual_times.astype(np.int64).tolist(), actual_wh.tolist(), strict=True))
    out = []
    for hour in np.unique(hours):
        part = rows[hours == hour]
        values = [measured[t] for t in part[:, 0].astype(np.int64).tolist() if t in measured]
        if len(part) != HOUR // QUARTER or len(values) != len(part):
            continue
        p10, p50, p90 = part[:, 1:].sum(axis=0)
        out.append(
            (p50 - (p50 - p10) / applied[0], p50, p50 + (p90 - p50) / applied[1], sum(values))
        )
    if not out:
        empty = np.array([])
        return empty, empty, empty, empty
    p10, p50, p90, actual = (np.array(c) for c in zip(*out, strict=True))
    return p10, p50, p90, actual


def fit_band(
    p10: np.ndarray, p50: np.ndarray, p90: np.ndarray, actual: np.ndarray, capacity_wh: float
) -> tuple[float, float]:
    """Band scale that would have put a tenth of the hours below P10 and a tenth above P90."""
    producing = (p90 > 0.02 * capacity_wh) | (actual > 0.02 * capacity_wh)
    if producing.sum() < MIN_HOURS:
        return NEUTRAL
    p10, p50, p90, actual = p10[producing], p50[producing], p90[producing], actual[producing]
    below, above = p50 - actual, actual - p50
    low_gap, high_gap = np.maximum(p50 - p10, 1e-6), np.maximum(p90 - p50, 1e-6)

    def needed(miss: np.ndarray, gap: np.ndarray) -> float:
        k = next((k for k in GRID if np.mean(miss > k * gap) <= TARGET), GRID[-1])
        return round(float(1 + SHRINK * (k - 1)), 3)

    return needed(below, low_gap), needed(above, high_gap)


def fit_days(
    days: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], capacity_wh: float
) -> tuple[float, float]:
    """``fit_band`` over the hourly rows of several days, once enough days were measured."""
    days = [d for d in days if len(d[0])]
    if len(days) < MIN_DAYS:
        return NEUTRAL
    p10, p50, p90, actual = (np.concatenate([d[i] for d in days]) for i in range(4))
    return fit_band(p10, p50, p90, actual, capacity_wh)
