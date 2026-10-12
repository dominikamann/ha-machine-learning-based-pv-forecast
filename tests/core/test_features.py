import numpy as np

from custom_components.ml_pv_forecast.core.features import (
    FEATURE_NAMES,
    PLANE_COLUMNS,
    build_features,
    col,
)
from tests.synthetic import LAT, LON, START_2024, synthetic_weather


def test_shape_and_names():
    w = synthetic_weather(START_2024, 48)
    X = build_features(w, LAT, LON)
    assert X.shape == (48, len(FEATURE_NAMES))
    assert len(PLANE_COLUMNS) == 7


def test_horizontal_plane_matches_ghi_when_sun_is_up():
    w = synthetic_weather(START_2024 + 180 * 86400, 48)
    X = build_features(w, LAT, LON)
    up = X[:, col("sun_elevation")] > 1.0
    assert up.any()
    np.testing.assert_allclose(X[up, col("plane_0_180")], w.ghi[up], rtol=1e-6)


def test_planes_are_zero_at_night_and_finite():
    w = synthetic_weather(START_2024, 48)
    X = build_features(w, LAT, LON)
    night = X[:, col("sun_elevation")] < -5
    assert np.all(X[night][:, PLANE_COLUMNS] == 0)
    assert np.isfinite(X[:, PLANE_COLUMNS]).all()


def test_ensemble_and_lead_features():
    from dataclasses import replace

    from custom_components.ml_pv_forecast.core.features import FEATURE_VERSION
    from custom_components.ml_pv_forecast.core.weather import ENSEMBLE_MODELS

    w = synthetic_weather(START_2024 + 180 * 86400, 4)
    ens = np.column_stack([w.ghi * f for f in (0.8, 1.0, 1.2, np.nan)])
    w = replace(w, ens_ghi=ens, lead_days=np.full(4, 1.0))
    X = build_features(w, LAT, LON)
    assert FEATURE_VERSION >= 2
    assert X[:, col("ens_count")].tolist() == [3.0] * 4
    np.testing.assert_allclose(X[:, col("ens_ghi_mean")], w.ghi, rtol=1e-9)
    assert (X[:, col("ens_ghi_spread")] >= 0).all()
    assert X[:, col("lead_days")].tolist() == [1.0] * 4
    assert np.isnan(X[:, col(f"ens_ghi_{ENSEMBLE_MODELS[3]}")]).all()


def test_no_ensemble_gives_nan_not_crash():
    X = build_features(synthetic_weather(START_2024, 3), LAT, LON)
    assert X[:, col("ens_count")].tolist() == [0.0] * 3
    assert np.isnan(X[:, col("ens_ghi_mean")]).all()
