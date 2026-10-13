"""Per-plant quantile models, the NNLS fallback, training, promotion and persistence."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls

from .cleaning import effective_capacity, usable_mask
from .features import FEATURE_VERSION, PLANE_COLUMNS, col
from .gbm import GradientBoosting
from .variants import BY_KEY, STANDARD, VARIANTS, Variant, choose, inputs

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
    variant: str = STANDARD.key  # which model variant this is (see variants.py)
    variant_since: int = 0  # when this plant switched to the variant
    variant_scores: dict[str, float] = field(default_factory=dict)  # holdout score per variant
    limit: float | None = None  # export or inverter limit in W (= Wh per hour), caps the forecast

    def predict(self, X: np.ndarray) -> np.ndarray:
        """P10, P50, P90 per row."""
        if self.kind == "fallback":
            p50 = X[:, PLANE_COLUMNS] @ self.coef
            out = np.column_stack([p50 * self.spread[0], p50, p50 * self.spread[1]])
        else:
            seen = inputs(vars(self).get("variant", STANDARD.key), X)
            out = (
                np.column_stack([self.estimators[q].predict(seen) for q in QUANTILES])
                * self.capacity
            )
        out = np.clip(out, 0.0, 1.1 * self.capacity)
        limit = vars(self).get("limit")
        if limit:
            out = np.minimum(out, limit)
        out.sort(axis=1)
        out[X[:, col("sun_elevation")] < 0] = 0.0
        return out


@dataclass
class TrainResult:
    model: PlantModel
    promoted: bool
    rows: int
    holdout_score: float | None  # quantile score (see quantile_score), Wh per hour
    baseline_score: float | None
    status: str  # "learning" or "ready"
    # quantile score of each variant on the holdout days, in % of the actual energy
    variant_scores: dict[str, float] = field(default_factory=dict)


def sample_weights(
    times: np.ndarray,
    now: int,
    is_reanalysis: np.ndarray,
    half_life_days: float = HALF_LIFE_DAYS,
) -> np.ndarray:
    age_days = np.maximum(now - times, 0) / DAY
    return 0.5 ** (age_days / half_life_days) * np.where(is_reanalysis > 0, REANALYSIS_WEIGHT, 1.0)


def quantile_score(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Per row: pinball loss of P10, P50 and P90, summed and scaled by 2/3.

    This is the usual estimate of the CRPS from three quantiles. It rewards a sharp band that
    still holds the actual value, and equals the absolute error of P50 when the band is zero
    wide, so it reads like a MAE in Wh.
    """
    q = np.asarray(QUANTILES)
    miss = actual[:, None] - predicted
    return 2 / len(q) * np.maximum(q * miss, (q - 1) * miss).sum(axis=1)


def fit_fallback(
    X: np.ndarray, y: np.ndarray, capacity: float, now: int, limit: float | None = None
) -> PlantModel:
    """Non-negative least squares of production on the virtual-plane irradiances."""
    A = X[:, PLANE_COLUMNS]
    coef, _ = nnls(A, y)
    p50 = A @ coef
    big = p50 > 0.05 * capacity
    spread = (0.6, 1.3)
    if big.sum() >= 10:
        ratio = y[big] / p50[big]
        spread = (float(np.quantile(ratio, 0.1)), float(np.quantile(ratio, 0.9)))
    return PlantModel("fallback", capacity, now, coef=coef, spread=spread, limit=limit)


def fit_gbm(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    capacity: float,
    now: int,
    variant: str = STANDARD.key,
    limit: float | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> PlantModel:
    target = y / capacity
    seen = inputs(variant, X)
    estimators = {}
    for q in QUANTILES:
        estimator = GradientBoosting(loss="quantile", quantile=q)
        estimator.fit(seen, target, sample_weight=weights, should_stop=should_stop)
        estimators[q] = estimator
    return PlantModel("gbm", capacity, now, estimators=estimators, variant=variant, limit=limit)


def _score(model: PlantModel, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(quantile_score(model.predict(X), y)))


def train(
    times: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    now: int,
    active: PlantModel | None = None,
    limit_w: float | None = None,
    variants: tuple[Variant, ...] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> TrainResult:
    """Train and record which history a newly fitted model has seen (all variants by default)."""
    variants = VARIANTS if variants is None else variants
    result = _train(times, X, y, now, active, limit_w, variants, should_stop)
    if result.model is not active:
        previous = vars(active) if active is not None else {}
        same = previous.get("variant", STANDARD.key) == result.model.variant
        result.model.variant_since = previous.get("variant_since", 0) if same else now
        result.model.variant_since = result.model.variant_since or now
        result.model.variant_scores = result.variant_scores
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
    variants: tuple[Variant, ...] = (STANDARD,),
    should_stop: Callable[[], bool] | None = None,
) -> TrainResult:
    """Train a candidate and decide whether it replaces the active model (spec §5.7, §5.8).

    Every variant is fitted on the same rows and scored on the same holdout days; the active
    variant is kept unless another is clearly and consistently better (``variants.choose``).
    All comparisons use ``quantile_score``, so the P10–P90 band counts as well as P50.
    Hours at the export limit stay in the training data: they are what the meter will show.
    """
    capacity = effective_capacity(times, y, now)
    if capacity <= 0:
        raise NotEnoughData("no production recorded")
    mask = usable_mask(y, X, capacity)
    t, Xu, yu = times[mask], X[mask], y[mask]
    if len(t) < MIN_FALLBACK_ROWS:
        raise NotEnoughData(f"only {len(t)} usable hours")
    reanalysis = Xu[:, col("is_reanalysis")]
    active_gbm = active if active is not None and active.kind == "gbm" else None

    if len(np.unique(t // DAY)) < MIN_GBM_DAYS:
        if active_gbm is not None:
            return TrainResult(active_gbm, False, len(t), None, None, "ready")
        return TrainResult(
            fit_fallback(Xu, yu, capacity, now, limit_w), True, len(t), None, None, "learning"
        )

    holdout = t >= now - HOLDOUT_DAYS * DAY
    if holdout.sum() < MIN_UNSEEN_ROWS or (~holdout).sum() < MIN_UNSEEN_ROWS:
        holdout = t >= np.quantile(t, 0.8)
    fit_part = ~holdout
    days = (t[holdout] // DAY).astype(np.int64)
    _, day_index = np.unique(days, return_inverse=True)
    candidates, day_errors, scores = {}, {}, {}
    for v in variants:
        w = sample_weights(t, now, reanalysis, v.half_life_days)
        model = fit_gbm(
            Xu[fit_part], yu[fit_part], w[fit_part], capacity, now, v.key, limit_w, should_stop
        )
        error = quantile_score(model.predict(Xu[holdout]), yu[holdout])
        candidates[v.key] = model
        day_errors[v.key] = np.bincount(day_index, error)
        scores[v.key] = round(float(error.sum() / max(yu[holdout].sum(), 1e-9) * 100), 2)
    active_variant = vars(active_gbm).get("variant", STANDARD.key) if active_gbm else STANDARD.key
    variant = BY_KEY[choose(day_errors, active_variant)]
    weights = sample_weights(t, now, reanalysis, variant.half_life_days)
    candidate = candidates[variant.key]
    baseline = fit_fallback(Xu[fit_part], yu[fit_part], capacity, now, limit_w)
    candidate_score = _score(candidate, Xu[holdout], yu[holdout])
    baseline_score = _score(baseline, Xu[holdout], yu[holdout])

    promoted = candidate_score < baseline_score
    if promoted and active_gbm is not None:
        # Compare at the same data cutoff: a rival trained on exactly the data the active model
        # had must not do worse on the hours neither has seen. This catches regressions (bad data,
        # changed cleaning) without penalising the candidate for being trained on older data.
        seen = t <= active_gbm.trained_at
        unseen = ~seen
        if unseen.sum() >= MIN_UNSEEN_ROWS and seen.sum() >= MIN_UNSEEN_ROWS:
            rival = fit_gbm(
                Xu[seen], yu[seen], weights[seen], capacity, now, variant.key, limit_w, should_stop
            )
            promoted = _score(rival, Xu[unseen], yu[unseen]) <= ACTIVE_TOLERANCE * _score(
                active_gbm, Xu[unseen], yu[unseen]
            )

    if promoted:
        final = fit_gbm(Xu, yu, weights, capacity, now, variant.key, limit_w, should_stop)
    elif active_gbm is not None:
        final = active_gbm
    else:
        final, promoted = fit_fallback(Xu, yu, capacity, now, limit_w), True
    return TrainResult(final, promoted, len(t), candidate_score, baseline_score, "ready", scores)


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
