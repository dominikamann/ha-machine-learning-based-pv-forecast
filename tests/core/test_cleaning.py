import numpy as np

from custom_components.ml_pv_forecast.core.cleaning import effective_capacity, usable_mask
from custom_components.ml_pv_forecast.core.features import build_features, col
from tests.synthetic import LAT, LON, START_2024, synthetic_production, synthetic_weather

DAYS = 120


def setup(clip_w=None):
    w = synthetic_weather(START_2024 + 90 * 86400, DAYS * 24)
    X = build_features(w, LAT, LON)
    y = synthetic_production(w, clip_w=clip_w)
    return w.times, X, y


def test_capacity_is_p99_of_recent_year():
    times, _, y = setup()
    cap = effective_capacity(times, y, int(times[-1]))
    assert 0.8 * y.max() < cap <= y.max()


def test_night_is_excluded_and_clean_day_rows_kept():
    times, X, y = setup()
    cap = effective_capacity(times, y, int(times[-1]))
    ok = usable_mask(y, X, cap)
    assert not ok[X[:, col("sun_elevation")] < 0].any()
    day = X[:, col("sun_elevation")] >= 2
    assert ok[day].mean() > 0.95


def test_outage_on_sunny_hours_is_excluded():
    times, X, y = setup()
    cap = effective_capacity(times, y, int(times[-1]))
    sunny = (np.nan_to_num(X[:, col("clearsky_index")]) > 0.7) & (X[:, col("sun_elevation")] > 10)
    broken = y.copy()
    broken[sunny] = 0.0
    ok = usable_mask(broken, X, cap)
    assert not ok[sunny].any()


def test_hours_capped_by_a_limit_are_kept():
    # The capped value is what the meter shows; dropping these hours biases the forecast low.
    times, X, y = setup(clip_w=3000.0)
    cap = effective_capacity(times, y, int(times[-1]))
    ok = usable_mask(y, X, cap)
    clipped = y >= 2999.0
    assert clipped.sum() > 50
    assert ok[clipped].all()


def test_no_production_gives_zero_capacity():
    assert effective_capacity(np.arange(5), np.full(5, np.nan), 10) == 0.0


def test_snow_covered_sunny_hours_are_kept_for_training():
    from dataclasses import replace

    from custom_components.ml_pv_forecast.core.features import build_features
    from tests.synthetic import LAT, LON

    w = synthetic_weather(START_2024 + 20 * 86400, 10 * 24)
    w = replace(w, snow_depth=np.full(len(w), 0.2))
    X = build_features(w, LAT, LON)
    y = synthetic_production(w)
    cap = effective_capacity(w.times, y, int(w.times[-1]))
    sunny = (np.nan_to_num(X[:, col("clearsky_index")]) > 0.7) & (X[:, col("sun_elevation")] > 10)
    covered = y.copy()
    covered[sunny] = 0.0  # panels under snow
    ok = usable_mask(covered, X, cap)
    assert ok[sunny].all()
