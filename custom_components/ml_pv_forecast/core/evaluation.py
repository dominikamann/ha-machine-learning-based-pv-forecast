"""Accuracy metrics, the correction layer and the nowcast decay estimate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .archive import EvalRow
from .gbm import GradientBoosting
from .nowcast import DEFAULT_TAU, fit_tau

DAY = 86400
CORRECTION_MIN_DAYS = 30
VALIDATION_DAYS = 14
SHORT_HORIZONS = ("30m", "1h", "6h")
CORRECTION_HORIZON = "1h"


@dataclass(frozen=True)
class Metrics:
    mae: float  # Wh
    nmae: float | None  # percent of mean actual
    bias: float  # Wh, positive = forecast too high
    skill: float | None  # 1 - MAE / persistence MAE
    count: int


def compute_metrics(rows: list[EvalRow]) -> Metrics | None:
    rows = [r for r in rows if r.actual > 0 or r.forecast.p50 > 0]
    if not rows:
        return None
    forecast = np.array([r.forecast.p50 for r in rows])
    actual = np.array([r.actual for r in rows])
    error = forecast - actual
    mae = float(np.mean(np.abs(error)))
    mean_actual = float(actual.mean())
    skill = None
    with_persistence = [r for r in rows if r.persistence is not None]
    if with_persistence:
        p_mae = np.mean([abs(r.persistence - r.actual) for r in with_persistence])
        f_mae = np.mean([abs(r.forecast.p50 - r.actual) for r in with_persistence])
        skill = float(1 - f_mae / p_mae) if p_mae > 0 else None
    return Metrics(
        mae=mae,
        nmae=mae / mean_actual * 100 if mean_actual > 0 else None,
        bias=float(error.mean()),
        skill=skill,
        count=len(rows),
    )


def correction_features(
    hour: np.ndarray,
    doy: np.ndarray,
    cloud: np.ndarray,
    p50_norm: np.ndarray,
) -> np.ndarray:
    return np.column_stack([hour, doy, cloud, p50_norm])


def _row_features(rows: list[EvalRow]) -> np.ndarray:
    f = [r.forecast for r in rows]
    start = np.array([x.start for x in f], dtype=float)
    hours = np.array([(x.end - x.start) / 3600 for x in f])
    return correction_features(
        (start % DAY) / 3600,
        (start / DAY) % 365.2425,
        np.array([x.cloud for x in f]),
        np.array([x.model_p50 for x in f]) / (np.array([x.capacity for x in f]) * hours),
    )


def _fit(X: np.ndarray, y: np.ndarray) -> GradientBoosting:
    return GradientBoosting(
        loss="squared_error", max_iter=100, learning_rate=0.05, min_samples_leaf=30
    ).fit(X, y)


class Correction:
    """Learns the ratio actual / model P50 from archived hourly windows.

    Trained on the pure model output (before correction and nowcast) and on 1-hour windows, so it
    is fitted on exactly what it is later applied to: hourly model predictions.
    """

    def __init__(self, estimator: GradientBoosting) -> None:
        self._estimator = estimator

    def factors(
        self,
        hour: np.ndarray,
        doy: np.ndarray,
        cloud: np.ndarray,
        p50_norm: np.ndarray,
    ) -> np.ndarray:
        X = correction_features(hour, doy, cloud, p50_norm)
        return np.clip(self._estimator.predict(X), 0.5, 1.5)

    @staticmethod
    def from_rows(rows: list[EvalRow], now: int) -> Correction | None:
        """A correction that beats the uncorrected model on the last 14 days, else None."""
        rows = [
            r
            for r in rows
            if r.forecast.horizon == CORRECTION_HORIZON
            and r.forecast.model_p50 > 0.02 * r.forecast.capacity
        ]
        if not rows:
            return None
        start = np.array([r.forecast.start for r in rows])
        if start.max() - start.min() < CORRECTION_MIN_DAYS * DAY:
            return None
        X = _row_features(rows)
        p50 = np.array([r.forecast.model_p50 for r in rows])
        actual = np.array([r.actual for r in rows])
        target = np.clip(actual / p50, 0.0, 3.0)
        validation = start >= now - VALIDATION_DAYS * DAY
        if validation.sum() < 20 or (~validation).sum() < 100:
            return None
        trial = _fit(X[~validation], target[~validation])
        corrected = p50[validation] * np.clip(trial.predict(X[validation]), 0.5, 1.5)
        if np.mean(np.abs(corrected - actual[validation])) >= np.mean(
            np.abs(p50[validation] - actual[validation])
        ):
            return None
        return Correction(_fit(X, target))


def tau_from_rows(rows: list[EvalRow]) -> float:
    rows = [
        r for r in rows if r.forecast.horizon in SHORT_HORIZONS and r.forecast.ratio is not None
    ]
    if not rows:
        return DEFAULT_TAU
    f = [r.forecast for r in rows]
    lead = np.array([((x.start + x.end) / 2 - x.issued_at) / 60 for x in f])
    return fit_tau(
        np.array([x.raw_p50 for x in f]),
        np.array([x.ratio for x in f]),
        lead,
        np.array([r.actual for r in rows]),
    )
