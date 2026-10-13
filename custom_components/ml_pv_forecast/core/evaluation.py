"""Accuracy metrics, the correction layer and the nowcast decay estimate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .archive import EvalRow
from .gbm import GradientBoosting
from .model import quantile_score
from .nowcast import DEFAULT_TAU, fit_day, fit_tau
from .variants import choose

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
    # How well the P10–P90 band fits: share of windows below P10 / above P90 (10 % each is right),
    # and the quantile score of the whole band in percent of mean actual (lower is better).
    below_p10: float | None = None
    above_p90: float | None = None
    score: float | None = None


def compute_metrics(rows: list[EvalRow]) -> Metrics | None:
    rows = [r for r in rows if r.actual > 0 or r.forecast.p50 > 0]
    if not rows:
        return None
    forecast = np.array([r.forecast.p50 for r in rows])
    actual = np.array([r.actual for r in rows])
    band = np.array([(r.forecast.p10, r.forecast.p50, r.forecast.p90) for r in rows])
    error = forecast - actual
    mae = float(np.mean(np.abs(error)))
    mean_actual = float(actual.mean())
    skill = None
    with_persistence = [(r, r.persistence) for r in rows if r.persistence is not None]
    if with_persistence:
        p_mae = np.mean([abs(p - r.actual) for r, p in with_persistence])
        f_mae = np.mean([abs(r.forecast.p50 - r.actual) for r, _ in with_persistence])
        skill = float(1 - f_mae / p_mae) if p_mae > 0 else None
    return Metrics(
        mae=mae,
        nmae=mae / mean_actual * 100 if mean_actual > 0 else None,
        bias=float(error.mean()),
        skill=skill,
        count=len(rows),
        below_p10=float(np.mean(actual < band[:, 0]) * 100),
        above_p90=float(np.mean(actual > band[:, 2]) * 100),
        score=float(quantile_score(band, actual).mean() / mean_actual * 100)
        if mean_actual > 0
        else None,
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


def _fit(X: np.ndarray, y: np.ndarray, weight: np.ndarray) -> GradientBoosting:
    return GradientBoosting(
        loss="squared_error", max_iter=100, learning_rate=0.05, min_samples_leaf=30
    ).fit(X, y, sample_weight=weight)


class Correction:
    """Learns the ratio actual / model P50 from archived hourly windows.

    Trained on the pure model output (before correction and nowcast) and on 1-hour windows, so it
    is fitted on exactly what it is later applied to: hourly model predictions. Hours are weighted
    by the model's energy: the ratio of a dim hour is noisy and matters little in Wh.

    The archive holds the output of whichever model was active, so the correction also learns
    from models that were replaced since. Keeping only rows of the current model is no option:
    models are refitted weekly and the correction needs a month of rows. The validation on the
    last days, all by recent models, guards against a correction that no longer fits.
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
        """A correction that beats the uncorrected model on the last 14 days, else None.

        Like a model variant it must be clearly and consistently better, day by day
        (``variants.choose``): a small gain on two weeks of data is often just noise.
        """
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
        trial = _fit(X[~validation], target[~validation], p50[~validation])
        corrected = p50[validation] * np.clip(trial.predict(X[validation]), 0.5, 1.5)
        _, day = np.unique(start[validation] // DAY, return_inverse=True)
        day_errors = {
            "raw": np.bincount(day, np.abs(p50[validation] - actual[validation])),
            "corrected": np.bincount(day, np.abs(corrected - actual[validation])),
        }
        if choose(day_errors, "raw") != "corrected":
            return None
        return Correction(_fit(X, target, p50))


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


def day_from_rows(rows: list[EvalRow], tau: float) -> tuple[float, float]:
    """Weight and decay of the day so far, learned from short-horizon rows that recorded it."""
    rows = [
        r
        for r in rows
        if r.forecast.horizon in SHORT_HORIZONS
        and r.forecast.day_ratio is not None
        and r.forecast.day_share is not None
    ]
    f = [r.forecast for r in rows]
    return fit_day(
        np.array([x.raw_p50 for x in f]),
        np.array([1.0 if x.ratio is None else x.ratio for x in f]),
        np.array([((x.start + x.end) / 2 - x.issued_at) / 60 for x in f]),
        tau,
        np.array([x.day_ratio for x in f]),
        np.array([x.day_share for x in f]),
        np.array([r.actual for r in rows]),
    )


def day_review(
    morning_curve: list | None,
    actual_times: np.ndarray,
    actual_wh: np.ndarray,
    day_start: int,
    day_end: int,
) -> dict:
    """How well the morning forecast matched one day.

    ``morning_curve`` holds quarter rows ``[start, p10, p50, p90]`` in Wh, actuals are quarter
    Wh. Hit rate: 100 % minus the deviation of the daily forecast from the actual energy.
    Deviation: the same with its sign, positive when the forecast was too optimistic.
    In band: share of producing hours whose actual energy lay between P10 and P90.
    """
    hours = np.arange(day_start, day_end, 3600)
    actual_h = np.zeros(len(hours))
    mask = (actual_times >= day_start) & (actual_times < day_end)
    np.add.at(actual_h, ((actual_times[mask] - day_start) // 3600).astype(int), actual_wh[mask])
    complete = int(mask.sum()) >= 0.9 * (day_end - day_start) / 900
    review = {
        "date": int(day_start),
        "actual_kwh": round(float(actual_h.sum()) / 1000, 2),
        "forecast_kwh": None,
        "hit_rate": None,
        "deviation_pct": None,
        "in_band": None,
        "complete": complete,
        "hourly": [
            {"time": int(t), "w": round(float(a), 1)} for t, a in zip(hours, actual_h, strict=True)
        ],
    }
    if not morning_curve:
        return review
    curve = np.asarray(morning_curve, dtype=float)
    forecast_h = np.zeros((len(hours), 3))
    inside = (curve[:, 0] >= day_start) & (curve[:, 0] < day_end)
    np.add.at(forecast_h, ((curve[inside, 0] - day_start) // 3600).astype(int), curve[inside, 1:])
    forecast = float(forecast_h[:, 1].sum())
    actual = float(actual_h.sum())
    if actual > 0:
        hit = max(0.0, 1 - abs(forecast - actual) / actual) * 100
    else:
        hit = 100.0 if forecast <= 0 else 0.0
    producing = (forecast_h[:, 2] > 0) | (actual_h > 0)
    within = (actual_h >= forecast_h[:, 0]) & (actual_h <= forecast_h[:, 2])
    review.update(
        forecast_kwh=round(forecast / 1000, 2),
        hit_rate=round(hit, 1),
        deviation_pct=round((forecast - actual) / actual * 100, 1) if actual > 0 else None,
        in_band=round(float(within[producing].mean()) * 100, 1) if producing.any() else None,
    )
    for row, f in zip(review["hourly"], forecast_h, strict=True):
        row.update(p10=round(float(f[0]), 1), p50=round(float(f[1]), 1), p90=round(float(f[2]), 1))
    return review


def today_so_far(
    morning_curve: list | None,
    actual_times: np.ndarray,
    actual_wh: np.ndarray,
    day_start: int,
    day_end: int,
) -> dict:
    """Today's production so far and how well the morning forecast matched the finished hours.

    The hit rate and the hours in range use the same rules as ``day_review``, limited to the
    hours whose production is complete (``until``); the hour still running is left out.
    """
    mask = (actual_times >= day_start) & (actual_times < day_end)
    today = {
        "actual_kwh": round(float(actual_wh[mask].sum()) / 1000, 2),
        "forecast_kwh": None,
        "hit_rate": None,
        "deviation_pct": None,
        "in_band": None,
        "until": None,
    }
    if morning_curve:
        curve = np.asarray(morning_curve, dtype=float)
        inside = (curve[:, 0] >= day_start) & (curve[:, 0] < day_end)
        today["forecast_kwh"] = round(float(curve[inside, 2].sum()) / 1000, 2)
    if not mask.any():
        return today
    last_end = int(actual_times[mask].max()) + 900
    until = day_start + (last_end - day_start) // 3600 * 3600
    if until <= day_start:
        return today
    today["until"] = int(until)
    if morning_curve:
        scored = day_review(morning_curve, actual_times, actual_wh, day_start, until)
        if scored["in_band"] is not None:  # at least one producing hour
            today.update(
                hit_rate=scored["hit_rate"],
                deviation_pct=scored["deviation_pct"],
                in_band=scored["in_band"],
            )
    return today


def tendency(days: list[dict]) -> float | None:
    """Signed deviation of the morning forecasts over several days, in % of the actual energy.

    Positive: the forecasts were too optimistic, negative: too pessimistic. Days are weighted by
    their energy, so a dull winter day does not outweigh a sunny one.
    """
    scored = [d for d in days if d.get("forecast_kwh") is not None and d.get("actual_kwh")]
    actual = sum(d["actual_kwh"] for d in scored)
    if not actual:
        return None
    return round((sum(d["forecast_kwh"] for d in scored) - actual) / actual * 100, 1)
