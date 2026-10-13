import numpy as np
import pytest

from custom_components.ml_pv_forecast.core.band import NEUTRAL, fit_band, hourly_rows, widen


def test_widen_scales_each_half_of_the_band_around_p50():
    values = np.array([[80.0, 100.0, 110.0], [0.0, 5.0, 10.0]])
    out = widen(values, (1.5, 2.0))
    assert out[0].tolist() == [70.0, 100.0, 120.0]
    assert out[1, 0] == 0.0  # never below zero
    assert (
        widen(values, NEUTRAL) is not values and widen(values, NEUTRAL).tolist() == values.tolist()
    )


def test_fit_band_widens_a_band_that_is_too_narrow_by_half_the_gap():
    rng = np.random.default_rng(0)
    n = 2000
    p50 = rng.uniform(200, 1000, n)
    actual = p50 * (1 + rng.normal(0, 0.2, n))
    # the forecast band covers only +-10 %, the errors need about +-25.6 % for 80 %
    p10, p90 = p50 * 0.9, p50 * 1.1
    lo, hi = fit_band(p10, p50, p90, actual, 1000.0)
    assert lo == pytest.approx(1 + 0.5 * (2.55 - 1), abs=0.1)
    assert hi == pytest.approx(1 + 0.5 * (2.55 - 1), abs=0.1)


def test_fit_band_keeps_a_good_band_and_needs_enough_hours():
    rng = np.random.default_rng(1)
    n = 2000
    p50 = rng.uniform(200, 1000, n)
    actual = p50 * (1 + rng.normal(0, 0.2, n))
    p10, p90 = p50 * (1 - 0.256), p50 * (1 + 0.256)
    lo, hi = fit_band(p10, p50, p90, actual, 1000.0)
    assert lo == pytest.approx(1.0, abs=0.05) and hi == pytest.approx(1.0, abs=0.05)
    assert fit_band(p10[:20], p50[:20], p90[:20], actual[:20], 1000.0) == NEUTRAL


def test_hourly_rows_undo_the_band_scale_that_was_applied():
    day = 86400 * 100
    quarters = day + np.arange(96) * 900
    # a morning curve that was already widened by (2, 1.5): p10 50, p50 100, p90 150 per quarter
    curve = [[int(t), 50.0, 100.0, 150.0] for t in quarters]
    actual_times = quarters[:40]  # actuals until 10:00
    p10, p50, p90, actual = hourly_rows(curve, (2.0, 1.5), actual_times, np.full(40, 90.0))
    assert len(p50) == 10  # only hours with all four quarters measured
    assert p50[0] == 400.0 and actual[0] == 360.0
    assert p10[0] == pytest.approx(400 - 200 / 2.0)  # back to the uncalibrated band
    assert p90[0] == pytest.approx(400 + 200 / 1.5)
