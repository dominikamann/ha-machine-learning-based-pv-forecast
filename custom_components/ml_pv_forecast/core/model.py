"""Per-plant quantile models, the NNLS fallback, training, promotion and persistence."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls

from .cleaning import effective_capacity, usable_mask
from .features import FEATURE_VERSION, PLANE_COLUMNS, col
from .gbm import GradientBoosting

QUANTILES = (0.1, 0.5, 0.9)
DAY = 86400
HOLDOUT_DAYS = 60
MIN_GBM_DAYS = 30
MIN_FALLBACK_ROWS = 24
MIN_UNSEEN_ROWS = 24
HALF_LIFE_DAYS = 365
REANALYSIS_WEIGHT = 0.3
ACTIVE_TOLERANCE = 1.02


class NotEnoughData(Exception):
    """Too little usable history to train anything."""


@dataclass
class PlantModel:
    """A trained model for one plant. Predictions are Wh per hour."""

    kind: str  # "fallback" or "gbm"
    capacity: float
    trained_at: int
    config_hash: str = ""
    coef: np.ndarray | None = None
    spread: tuple[float, float] = (0.6, 1.3)
    estimators: dict[float, Any] = field(default_factory=dict)
    feature_version: int = FEATURE_VERSION
    rows: int = 0  # usable training hours
    history_start: int = 0  # first and last training hour (unix seconds)
    history_end: int = 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """P10, P50, P90 per row."""
        if self.kind == "fallback":
            p50 = X[:, PLANE_COLUMNS] @ self.coef
            out = np.column_stack([p50 * self.spread[0], p50, p50 * self.spread[1]])
        else:
            out = (
                np.column_stack([self.estimators[q].predict(X) for q in QUANTILES]) * self.capacity
            )
        out = np.clip(out, 0.0, 1.1 * self.capacity)
        out.sort(axis=1)
        out[X[:, col("sun_elevation")] < 0] = 0.0
        return out


@dataclass
class TrainResult:
    model: PlantModel
    promoted: bool
    rows: int
    holdout_mae: float | None
    baseline_mae: float | None
    status: str  # "learning" or "ready"


def sample_weights(times: np.ndarray, now: int, is_reanalysis: np.ndarray) -> np.ndarray:
    age_days = np.maximum(now - times, 0) / DAY
    return 0.5 ** (age_days / HALF_LIFE_DAYS) * np.where(is_reanalysis > 0, REANALYSIS_WEIGHT, 1.0)


def fit_fallback(X: np.ndarray, y: np.ndarray, capacity: float, now: int) -> PlantModel:
    """Non-negative least squares of production on the virtual-plane irradiances."""
    A = X[:, PLANE_COLUMNS]
    coef, _ = nnls(A, y)
    p50 = A @ coef
    big = p50 > 0.05 * capacity
    spread = (0.6, 1.3)
    if big.sum() >= 10:
        ratio = y[big] / p50[big]
        spread = (float(np.quantile(ratio, 0.1)), float(np.quantile(ratio, 0.9)))
    return PlantModel("fallback", capacity, now, coef=coef, spread=spread)


def fit_gbm(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray, capacity: float, now: int
) -> PlantModel:
    target = y / capacity
    estimators = {}
    for q in QUANTILES:
        estimator = GradientBoosting(loss="quantile", quantile=q)
        estimator.fit(X, target, sample_weight=weights)
        estimators[q] = estimator
    return PlantModel("gbm", capacity, now, estimators=estimators)


def _mae(model: PlantModel, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(model.predict(X)[:, 1] - y)))


def train(
    times: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    now: int,
    active: PlantModel | None = None,
    limit_w: float | None = None,
) -> TrainResult:
    """Train and record which history a newly fitted model has seen."""
    result = _train(times, X, y, now, active, limit_w)
    if result.model is not active:
        result.model.rows = result.rows
        result.model.history_start = int(times.min())
        result.model.history_end = int(times.max())
    return result


def _train(
    times: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    now: int,
    active: PlantModel | None = None,
    limit_w: float | None = None,
) -> TrainResult:
    """Train a candidate and decide whether it replaces the active model (spec §5.7, §5.8)."""
    capacity = effective_capacity(times, y, now)
    if capacity <= 0:
        raise NotEnoughData("no production recorded")
    mask = usable_mask(times, y, X, capacity, limit_w)  # hourly rows: W limit == Wh per hour
    t, Xu, yu = times[mask], X[mask], y[mask]
    if len(t) < MIN_FALLBACK_ROWS:
        raise NotEnoughData(f"only {len(t)} usable hours")
    weights = sample_weights(t, now, Xu[:, col("is_reanalysis")])
    active_gbm = active if active is not None and active.kind == "gbm" else None

    if len(np.unique(t // DAY)) < MIN_GBM_DAYS:
        if active_gbm is not None:
            return TrainResult(active_gbm, False, len(t), None, None, "ready")
        return TrainResult(
            fit_fallback(Xu, yu, capacity, now), True, len(t), None, None, "learning"
        )

    holdout = t >= now - HOLDOUT_DAYS * DAY
    if holdout.sum() < MIN_UNSEEN_ROWS or (~holdout).sum() < MIN_UNSEEN_ROWS:
        holdout = t >= np.quantile(t, 0.8)
    fit_part = ~holdout
    candidate = fit_gbm(Xu[fit_part], yu[fit_part], weights[fit_part], capacity, now)
    baseline = fit_fallback(Xu[fit_part], yu[fit_part], capacity, now)
    candidate_mae = _mae(candidate, Xu[holdout], yu[holdout])
    baseline_mae = _mae(baseline, Xu[holdout], yu[holdout])

    promoted = candidate_mae < baseline_mae
    if promoted and active_gbm is not None:
        # Compare at the same data cutoff: a rival trained on exactly the data the active model
        # had must not do worse on the hours neither has seen. This catches regressions (bad data,
        # changed cleaning) without penalising the candidate for being trained on older data.
        seen = t <= active_gbm.trained_at
        unseen = ~seen
        if unseen.sum() >= MIN_UNSEEN_ROWS and seen.sum() >= MIN_UNSEEN_ROWS:
            rival = fit_gbm(Xu[seen], yu[seen], weights[seen], capacity, now)
            promoted = _mae(rival, Xu[unseen], yu[unseen]) <= ACTIVE_TOLERANCE * _mae(
                active_gbm, Xu[unseen], yu[unseen]
            )

    if promoted:
        final = fit_gbm(Xu, yu, weights, capacity, now)
    elif active_gbm is not None:
        final = active_gbm
    else:
        final, promoted = fit_fallback(Xu, yu, capacity, now), True
    return TrainResult(final, promoted, len(t), candidate_mae, baseline_mae, "ready")


def is_current(model: PlantModel) -> bool:
    """False for models trained on an older feature set (pickles from before have no field)."""
    # vars(): an unpickled old model would otherwise see the class default (the current version).
    return vars(model).get("feature_version", 1) == FEATURE_VERSION


def save_model(model: PlantModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(model))
    tmp.replace(path)


def load_model(path: Path) -> PlantModel | None:
    """The stored model, or None if missing or unreadable (caller retrains)."""
    if not path.exists():
        return None
    try:
        model = pickle.loads(path.read_bytes())
    except Exception:  # noqa: BLE001 - a corrupt pickle can raise almost anything
        return None
    return model if isinstance(model, PlantModel) else None
