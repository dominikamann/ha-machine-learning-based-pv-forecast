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
