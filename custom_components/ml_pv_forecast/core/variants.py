"""Model variants a plant can run, and the rule that picks one at each training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import col

# Weather inputs only the "weather_plus" variant uses (fog, haze, rain and wind).
EXTRA_WEATHER = ("humidity", "dew_point_spread", "precipitation", "wind_speed", "visibility")
MIN_GAIN = 0.02  # a challenger must be at least 2 % better than the active variant ...
BOOTSTRAP = 500  # ... and consistently so over the holdout days


@dataclass(frozen=True)
class Variant:
    key: str
    extra_weather: bool
    half_life_days: float  # how fast older training days lose weight


STANDARD = Variant("standard", False, 365.0)
VARIANTS = (
    STANDARD,
    Variant("weather_plus", True, 365.0),
    Variant("short_memory", False, 90.0),
    Variant("weather_plus_short_memory", True, 90.0),
)
BY_KEY = {v.key: v for v in VARIANTS}
_EXTRA_COLUMNS = [col(name) for name in EXTRA_WEATHER]


def inputs(variant: str, X: np.ndarray) -> np.ndarray:
    """The feature matrix as this variant sees it (extra weather hidden unless it uses it)."""
    if BY_KEY.get(variant, STANDARD).extra_weather:
        return X
    X = X.copy()
    X[:, _EXTRA_COLUMNS] = np.nan
    return X


def choose(day_errors: dict[str, np.ndarray], active: str, seed: int = 0) -> str:
    """The variant to run: the active one unless another is clearly and consistently better.

    ``day_errors`` holds each variant's absolute error per holdout day (same days for all).
    """
    if active not in day_errors:
        active = STANDARD.key if STANDARD.key in day_errors else next(iter(day_errors))
    best = min(day_errors, key=lambda k: day_errors[k].sum())
    ours, theirs = day_errors[active], day_errors[best]
    if best == active or theirs.sum() > (1 - MIN_GAIN) * ours.sum():
        return active
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(ours), (BOOTSTRAP, len(ours)))
    differences = theirs[samples].sum(axis=1) - ours[samples].sum(axis=1)
    return best if np.percentile(differences, 97.5) < 0 else active
