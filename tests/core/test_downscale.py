import numpy as np

from custom_components.ml_pv_forecast.core.downscale import to_quarters


def test_quarters_sum_to_hour_and_follow_ghi():
    hourly = np.array([[40.0, 100.0, 160.0]])
    q = to_quarters(np.array([0]), hourly, np.array([0, 900, 1800, 2700]), np.array([1, 1, 1, 7.0]))
    np.testing.assert_allclose(q.sum(axis=0), hourly[0])
    assert q[3, 1] == 70.0


def test_zero_ghi_splits_evenly_and_missing_hour_is_nan():
    hourly = np.array([[0.0, 8.0, 8.0]])
    q = to_quarters(np.array([0]), hourly, np.array([0, 900, 1800, 2700, 3600]), np.zeros(5))
    np.testing.assert_allclose(q[:4, 1], 2.0)
    assert np.isnan(q[4]).all()
