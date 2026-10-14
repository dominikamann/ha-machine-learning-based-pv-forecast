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


def test_a_partial_hour_at_the_data_edge_keeps_its_quarter_size():
    # The 15-minute data ends after two quarters of the last hour: those two quarters get their
    # share of the hour (a quarter each at even sun), not the whole hour between them.
    hour_times = np.array([0, 3600])
    hourly = np.array([[400.0, 400.0, 400.0], [400.0, 400.0, 400.0]])
    quarter_times = np.arange(6) * 900
    out = to_quarters(hour_times, hourly, quarter_times, np.full(6, 100.0))
    np.testing.assert_allclose(out[:, 1], [100.0] * 6)
