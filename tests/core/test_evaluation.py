import numpy as np
import pytest

from custom_components.ml_pv_forecast.core.archive import ArchivedForecast, EvalRow
from custom_components.ml_pv_forecast.core.evaluation import (
    Correction,
    compute_metrics,
    tau_from_rows,
)

DAY = 86400


def row(
    p50,
    actual,
    persistence=None,
    start=0,
    horizon="day_ahead",
    issued=None,
    ratio=None,
    model_p50=None,
):
    end = start + (DAY if horizon == "day_ahead" else 3600)
    f = ArchivedForecast(
        "a",
        horizon,
        start - 3600 if issued is None else issued,
        start,
        end,
        p50 * 0.8,
        p50,
        p50 * 1.2,
        p50,
        p50 if model_p50 is None else model_p50,
        ratio,
        50.0,
        1000.0,
    )
    return EvalRow(f, actual, persistence)


def test_metrics_values_and_zero_rows_ignored():
    rows = [row(110, 100, 150), row(90, 100, 50), row(0, 0, 0)]
    m = compute_metrics(rows)
    assert m.count == 2
    assert m.mae == 10 and m.bias == 0
    assert m.nmae == pytest.approx(10.0)
    assert m.skill == pytest.approx(0.8)


def test_metrics_none_when_empty():
    assert compute_metrics([row(0, 0)]) is None


def test_correction_learns_systematic_overestimate():
    rng = np.random.default_rng(0)
    now = 120 * DAY
    rows = [
        row(p, 0.8 * p, start=d * DAY + h * 3600, horizon="1h")
        for d in range(60, 120)
        for h in range(6, 18)
        for p in [rng.uniform(200, 900)]
    ]
    corr = Correction.from_rows(rows, now)
    assert corr is not None
    f = corr.factors(np.array([12.0]), np.array([100.0]), np.array([50.0]), np.array([0.5]))
    assert f[0] == pytest.approx(0.8, abs=0.05)


def test_correction_learns_from_uncorrected_model_output():
    # Archived p50 was already corrected (equals actual); the model itself is 25 % too high.
    rng = np.random.default_rng(1)
    rows = [
        row(0.8 * m, 0.8 * m, start=d * DAY + h * 3600, horizon="1h", model_p50=m)
        for d in range(60, 120)
        for h in range(6, 18)
        for m in [rng.uniform(200, 900)]
    ]
    corr = Correction.from_rows(rows, 120 * DAY)
    assert corr is not None
    f = corr.factors(np.array([12.0]), np.array([100.0]), np.array([50.0]), np.array([0.5]))
    assert f[0] == pytest.approx(0.8, abs=0.05)


def test_correction_ignores_day_windows():
    rows = [row(500, 400, start=d * DAY) for d in range(60, 120)]
    assert Correction.from_rows(rows, 120 * DAY) is None


def test_correction_needs_thirty_days():
    rows = [row(500, 400, start=d * DAY + 12 * 3600, horizon="1h") for d in range(10)]
    assert Correction.from_rows(rows, 10 * DAY) is None


def test_tau_from_rows_uses_short_rows_with_ratio():
    rows = [
        row(500, 500, start=1800 * i, horizon="1h", issued=1800 * i, ratio=1.0) for i in range(60)
    ]
    assert tau_from_rows(rows) == 30.0  # ratio 1 everywhere: every tau ties, first grid value wins
    assert tau_from_rows([row(500, 500, horizon="1h")]) == 90.0  # no ratio -> default
    assert tau_from_rows([]) == 90.0


def test_day_review_hit_rate_band_and_hourly():
    from custom_components.ml_pv_forecast.core.evaluation import day_review

    day = 86400 * 100
    quarters = day + np.arange(96) * 900
    sun = (np.arange(96) >= 32) & (np.arange(96) < 64)  # 08:00–16:00
    curve = [[int(t), 80.0 * s, 100.0 * s, 120.0 * s] for t, s in zip(quarters, sun, strict=True)]
    actual = np.where(sun, 90.0, 0.0)  # 10 % below the forecast, inside the band
    review = day_review(curve, quarters, actual, day, day + 86400)
    assert review["forecast_kwh"] == pytest.approx(3.2)
    assert review["actual_kwh"] == pytest.approx(2.88)
    assert review["hit_rate"] == pytest.approx(88.9, abs=0.1)
    assert review["in_band"] == 100.0
    assert review["complete"] is True
    assert len(review["hourly"]) == 24
    ten = review["hourly"][10]
    assert ten["p50"] == pytest.approx(400.0) and ten["w"] == 360.0


def test_day_review_without_morning_forecast_or_with_gaps():
    from custom_components.ml_pv_forecast.core.evaluation import day_review

    day = 86400 * 100
    quarters = day + np.arange(40) * 900
    review = day_review(None, quarters, np.full(40, 50.0), day, day + 86400)
    assert review["forecast_kwh"] is None and review["hit_rate"] is None
    assert review["actual_kwh"] == pytest.approx(2.0)
    assert review["complete"] is False


def test_today_so_far_scores_only_finished_hours():
    from custom_components.ml_pv_forecast.core.evaluation import today_so_far

    day = 86400 * 100
    quarters = day + np.arange(96) * 900
    sun = (np.arange(96) >= 32) & (np.arange(96) < 64)  # 08:00–16:00
    curve = [[int(t), 80.0 * s, 100.0 * s, 120.0 * s] for t, s in zip(quarters, sun, strict=True)]
    # Actuals up to 10:45 (the 10 o'clock hour is still running): only 08:00–10:00 count.
    done = quarters < day + 10 * 3600 + 45 * 60
    actual = np.where(sun, 90.0, 0.0)[done]
    today = today_so_far(curve, quarters[done], actual, day, day + 86400)
    assert today["actual_kwh"] == pytest.approx(0.99)  # all 11 producing quarters so far
    assert today["forecast_kwh"] == pytest.approx(3.2)  # the whole morning forecast
    assert today["hit_rate"] == pytest.approx(88.9, abs=0.1)  # 720 Wh against 800 Wh, 08–10
    assert today["in_band"] == 100.0
    assert today["until"] == day + 10 * 3600


def test_today_so_far_before_sunrise_or_without_morning_forecast():
    from custom_components.ml_pv_forecast.core.evaluation import today_so_far

    day = 86400 * 100
    quarters = day + np.arange(24) * 900  # until 06:00, nothing produced
    curve = [[int(t), 0.0, 0.0, 0.0] for t in day + np.arange(96) * 900]
    today = today_so_far(curve, quarters, np.zeros(24), day, day + 86400)
    assert today["hit_rate"] is None and today["in_band"] is None
    nothing = today_so_far(None, np.array([]), np.array([]), day, day + 86400)
    assert nothing["actual_kwh"] == 0.0 and nothing["forecast_kwh"] is None
    assert nothing["hit_rate"] is None and nothing["until"] is None
