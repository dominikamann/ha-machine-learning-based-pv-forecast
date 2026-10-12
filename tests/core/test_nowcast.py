import numpy as np
import pytest

from custom_components.ml_pv_forecast.core.nowcast import adjust, fit_tau, live_ratio


def test_live_ratio_rules():
    assert live_ratio(500, 1000, 30, 4000) == 0.5
    assert live_ratio(5000, 1000, 30, 4000) == 2.0
    assert live_ratio(10, 0.0, 30, 4000) is None  # nothing forecast
    assert live_ratio(10, 100, 30, 4000) is None  # below 5 % of capacity
    assert live_ratio(500, 1000, 3, 4000) is None  # sun too low / night


def test_adjust_decays_with_lead_and_leaves_finished_quarters():
    values = np.ones((4, 3))
    out = adjust(values, np.array([-30.0, 0.0, 90.0, 10000.0]), 0.5, 90.0)
    assert out[0, 1] == 1.0
    assert out[1, 1] == pytest.approx(0.5)
    assert out[2, 1] == pytest.approx(1 - 0.5 * np.exp(-1))
    assert out[3, 1] == pytest.approx(1.0, abs=1e-6)


def test_fit_tau_recovers_true_decay():
    rng = np.random.default_rng(0)
    n = 400
    raw = rng.uniform(100, 1000, n)
    ratio = rng.uniform(0.3, 1.7, n)
    lead = rng.uniform(0, 360, n)
    actual = raw * (1 + (ratio - 1) * np.exp(-lead / 120.0))
    assert fit_tau(raw, ratio, lead, actual) == 120.0


def test_fit_tau_default_with_little_data():
    assert fit_tau(np.ones(3), np.ones(3), np.ones(3), np.ones(3)) == 90.0
