import pickle

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from custom_components.ml_pv_forecast.core.gbm import GradientBoosting


def make_data(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 6))
    signal = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + np.where(X[:, 2] > 0.3, 1.0, 0.0)
    noise = rng.normal(0, 0.2 + 0.3 * (X[:, 3] > 0), n)
    return X, signal + noise


def test_median_accuracy_close_to_scikit_learn():
    X, y = make_data()
    Xt, yt = make_data(seed=1)
    ours = GradientBoosting(loss="quantile", quantile=0.5).fit(X, y)
    ref = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.5, learning_rate=0.05, max_iter=300, random_state=0
    ).fit(X, y)
    ours_mae = np.mean(np.abs(ours.predict(Xt) - yt))
    ref_mae = np.mean(np.abs(ref.predict(Xt) - yt))
    assert ours_mae <= 1.1 * ref_mae


@pytest.mark.parametrize("q", [0.1, 0.9])
def test_quantile_coverage(q):
    X, y = make_data()
    Xt, yt = make_data(seed=2)
    model = GradientBoosting(loss="quantile", quantile=q).fit(X, y)
    ref = HistGradientBoostingRegressor(
        loss="quantile", quantile=q, learning_rate=0.05, max_iter=300, random_state=0
    ).fit(X, y)
    # Out-of-sample quantiles shrink inwards for any boosted model; match scikit-learn's behaviour.
    ours = np.mean(yt <= model.predict(Xt))
    assert ours == pytest.approx(np.mean(yt <= ref.predict(Xt)), abs=0.02)
    assert ours == pytest.approx(q, abs=0.07)


def test_squared_error_learns_mean():
    X, y = make_data()
    Xt, yt = make_data(seed=3)
    model = GradientBoosting(loss="squared_error").fit(X, y)
    assert np.mean((model.predict(Xt) - yt) ** 2) < 0.6 * np.var(yt)


def test_zero_weight_rows_are_ignored():
    X, y = make_data(3000)
    corrupt = y.copy()
    corrupt[:1500] = 100.0
    w = np.r_[np.zeros(1500), np.ones(1500)]
    a = GradientBoosting(loss="quantile", quantile=0.5).fit(X, corrupt, sample_weight=w)
    b = GradientBoosting(loss="quantile", quantile=0.5).fit(X[1500:], y[1500:])
    np.testing.assert_allclose(a.predict(X[:200]), b.predict(X[:200]), atol=0.05)


def test_nan_features_and_pickle_roundtrip():
    X, y = make_data(3000)
    X[::7, 1] = np.nan
    model = GradientBoosting(loss="quantile", quantile=0.5).fit(X, y)
    p = model.predict(X)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(pickle.loads(pickle.dumps(model)).predict(X), p)
