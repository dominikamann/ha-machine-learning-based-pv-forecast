import numpy as np
import pytest

from custom_components.ml_pv_forecast.core.nowcast import (
    DEFAULT_DAY_TAU,
    DEFAULT_DAY_WEIGHT,
    DayFactor,
    adjust,
    day_ratio,
    fit_day,
    fit_tau,
    live_ratio,
)


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
    assert fit_tau(np.ones(3), np.ones(3), np.ones(3), np.ones(3)) == 60.0


def test_live_ratio_ignores_missing_actuals():
    assert live_ratio(float("nan"), 1000, 30, 4000) is None


def test_day_ratio_rules():
    assert day_ratio(1150.0, 1000.0, 4000.0) == pytest.approx(1.15)
    assert day_ratio(10.0, 20.0, 4000.0) is None  # too little forecast so far to judge
    assert day_ratio(9000.0, 1000.0, 4000.0) == 3.0  # clipped
    assert day_ratio(float("nan"), 1000.0, 4000.0) is None


def test_day_factor_grows_with_the_share_of_the_day_and_fades_with_lead():
    values = np.ones((4, 3))
    lead = np.array([-30.0, 0.0, 120.0, 100000.0])
    day = DayFactor(ratio=1.5, share=0.5, weight=0.6, tau=240.0)
    out = adjust(values, lead, None, 60.0, day)
    assert out[0, 1] == 1.0  # finished quarters stay
    assert out[1, 1] == pytest.approx(1.0)  # right now the live ratio rules (none here)
    g = np.exp(-2.0)
    assert out[2, 1] == pytest.approx(1 + 0.6 * 0.5 * 0.5 * (1 - g) * np.exp(-0.5))
    assert out[3, 1] == pytest.approx(1.0, abs=1e-6)
    both = adjust(values, lead, 0.8, 60.0, day)
    assert both[2, 1] == pytest.approx(out[2, 1] - 0.2 * g)


def test_fit_day_recovers_weight_and_decay():
    rng = np.random.default_rng(1)
    n = 3000
    raw = rng.uniform(100, 1000, n)
    ratio = rng.uniform(0.5, 1.5, n)
    lead = rng.uniform(0, 360, n)
    day = rng.uniform(0.5, 1.5, n)
    share = rng.uniform(0, 1, n)
    g = np.exp(-lead / 60.0)
    actual = raw * (1 + (ratio - 1) * g + 0.4 * share * (day - 1) * (1 - g) * np.exp(-lead / 120.0))
    assert fit_day(raw, ratio, lead, 60.0, day, share, actual) == (0.4, 120.0)


def test_fit_day_keeps_defaults_with_little_data():
    ones = np.ones(10)
    assert fit_day(ones, ones, ones, 60.0, ones, ones, ones) == (
        DEFAULT_DAY_WEIGHT,
        DEFAULT_DAY_TAU,
    )
