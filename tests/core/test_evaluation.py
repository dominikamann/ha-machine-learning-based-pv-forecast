from dataclasses import replace

import numpy as np
import pytest

from custom_components.ml_pv_forecast.core.archive import ArchivedForecast, EvalRow
from custom_components.ml_pv_forecast.core.evaluation import (
    Correction,
    compute_metrics,
    day_from_rows,
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


def test_metrics_check_the_band():
    # row(): P10 = 0.8 * P50, P90 = 1.2 * P50
    rows = [row(100, 70), row(100, 100), row(100, 100), row(100, 130)]
    m = compute_metrics(rows)
    assert m.below_p10 == 25.0 and m.above_p90 == 25.0
    # quantile score: 2/3 of the summed pinball losses, mean actual 100 Wh
    # actual 70 or 130: 9 + 15 + 5 = 29; actual 100: 2 + 0 + 2 = 4
    assert m.score == pytest.approx((29 + 4 + 4 + 29) / 4 * 2 / 3)


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


@pytest.mark.parametrize("seed", [1, 5])
def test_correction_is_dropped_when_it_only_learned_the_weather_of_some_days(seed):
    # The model is right on average; each day's weather made it too high or too low as a whole.
    # A correction can fit those days (it knows the day of the year) and look better on the
    # validation days by chance; on fresh weather it made the forecast about 15 % worse.
    rng = np.random.default_rng(seed)
    days = range(60, 120)
    weather = rng.uniform(0.7, 1.3, len(days))
    rows = [
        row(p, p * weather[i] * rng.uniform(0.9, 1.1), start=d * DAY + h * 3600, horizon="1h")
        for i, d in enumerate(days)
        for h in range(6, 18)
        for p in [rng.uniform(200, 900)]
    ]
    assert Correction.from_rows(rows, 120 * DAY) is None


def test_correction_ignores_day_windows():
    rows = [row(500, 400, start=d * DAY) for d in range(60, 120)]
    assert Correction.from_rows(rows, 120 * DAY) is None


def test_correction_needs_thirty_days():
    rows = [row(500, 400, start=d * DAY + 12 * 3600, horizon="1h") for d in range(10)]
    assert Correction.from_rows(rows, 10 * DAY) is None


def test_tau_from_rows_uses_short_rows_with_ratio():
    rows = [
        row(500, 500, start=DAY * i, horizon="1h", issued=DAY * i, ratio=1.0) for i in range(60)
    ]
    assert tau_from_rows(rows) == 30.0  # ratio 1 everywhere: every tau ties, first grid value wins
    assert tau_from_rows([row(500, 500, horizon="1h")]) == 60.0  # no ratio -> default
    assert tau_from_rows([]) == 60.0


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
    assert review["deviation_pct"] == pytest.approx(11.1, abs=0.1)  # forecast above: optimistic
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
    assert today["deviation_pct"] == pytest.approx(11.1, abs=0.1)
    assert today["until"] == day + 10 * 3600
    assert today["too_early"] is False


def test_today_so_far_waits_until_enough_of_the_day_is_done():
    from custom_components.ml_pv_forecast.core.evaluation import today_so_far

    # 2026-10-14: after the first producing hour (0.27 kWh against 0.7 kWh in the morning haze)
    # the card showed a hit rate of 0 % and "166 % too optimistic" for a 20 kWh day.
    day = 86400 * 100
    quarters = day + np.arange(96) * 900
    sun = (np.arange(96) >= 32) & (np.arange(96) < 72)  # 08:00–18:00, 100 Wh per quarter
    curve = [[int(t), 80.0 * s, 100.0 * s, 120.0 * s] for t, s in zip(quarters, sun, strict=True)]

    def at(hour, wh):
        done = quarters < day + hour * 3600
        return today_so_far(curve, quarters[done], np.where(sun, wh, 0.0)[done], day, day + 86400)

    early = at(9, 40.0)  # one finished hour, 10 % of the day forecast
    assert early["too_early"] is True
    assert early["hit_rate"] is None and early["deviation_pct"] is None
    assert early["in_band"] is None
    assert early["actual_kwh"] == pytest.approx(0.16) and early["until"] == day + 9 * 3600
    later = at(10, 80.0)  # two finished hours and 20 % of the day forecast
    assert later["too_early"] is False and later["hit_rate"] == 75.0
    # a forecast that was far too low: two hours that already brought much more than expected
    surprise = [[t, lo / 20, p50 / 20, hi / 20] for t, lo, p50, hi in curve]
    done = quarters < day + 10 * 3600
    today = today_so_far(
        surprise, quarters[done], np.where(sun, 100.0, 0.0)[done], day, day + 86400
    )
    assert today["too_early"] is False and today["hit_rate"] == pytest.approx(5.0)


def test_today_so_far_before_sunrise_or_without_morning_forecast():
    from custom_components.ml_pv_forecast.core.evaluation import today_so_far

    day = 86400 * 100
    quarters = day + np.arange(24) * 900  # until 06:00, nothing produced
    curve = [[int(t), 0.0, 0.0, 0.0] for t in day + np.arange(96) * 900]
    today = today_so_far(curve, quarters, np.zeros(24), day, day + 86400)
    assert today["hit_rate"] is None and today["in_band"] is None
    assert today["deviation_pct"] is None
    nothing = today_so_far(None, np.array([]), np.array([]), day, day + 86400)
    assert nothing["actual_kwh"] == 0.0 and nothing["forecast_kwh"] is None
    assert nothing["hit_rate"] is None and nothing["until"] is None
    assert nothing["too_early"] is False and today["too_early"] is True


def test_tendency_weights_days_by_energy_and_skips_days_without_forecast():
    from custom_components.ml_pv_forecast.core.evaluation import tendency

    days = [
        {"forecast_kwh": 9.0, "actual_kwh": 10.0},  # 1 kWh too low
        {"forecast_kwh": 32.0, "actual_kwh": 30.0},  # 2 kWh too high
        {"forecast_kwh": None, "actual_kwh": 20.0},
    ]
    assert tendency(days) == pytest.approx(2.5)  # +1 kWh over 40 kWh: slightly optimistic
    assert tendency([{"forecast_kwh": None, "actual_kwh": 5.0}]) is None
    assert tendency([]) is None


def test_day_from_rows_learns_from_short_rows_with_the_day_so_far():
    from custom_components.ml_pv_forecast.core.nowcast import DEFAULT_DAY_TAU, DEFAULT_DAY_WEIGHT

    rng = np.random.default_rng(2)
    rows = []
    for _ in range(800):
        day, share, lead_h = rng.uniform(0.5, 1.5), rng.uniform(0, 1), int(rng.integers(1, 6))
        g = np.exp(-lead_h)  # live decay 60 min, lead measured to the window's middle
        actual = 500 * (1 + 0.4 * share * (day - 1) * (1 - g) * np.exp(-lead_h * 60 / 120.0))
        issued = int(rng.integers(0, 30)) * DAY
        r = row(
            500, actual, start=issued + lead_h * 3600 - 1800, horizon="1h", issued=issued, ratio=1.0
        )
        rows.append(replace(r, forecast=replace(r.forecast, day_ratio=day, day_share=share)))
    assert day_from_rows(rows, 60.0) == (pytest.approx(0.4), 120.0)
    # rows without the day so far (archived before 0.7.0) keep the defaults
    plain = [row(500, 500, start=1800, horizon="1h", issued=0, ratio=1.0)] * 600
    assert day_from_rows(plain, 60.0) == (DEFAULT_DAY_WEIGHT, DEFAULT_DAY_TAU)
    # many rows from only a few days are not enough to learn from
    few = [r for r in rows if r.forecast.issued_at < 5 * DAY]
    assert day_from_rows(few * 4, 60.0) == (DEFAULT_DAY_WEIGHT, DEFAULT_DAY_TAU)


def test_tau_is_not_learned_from_a_few_days_of_a_biased_forecast():
    # 2026-10-14: three days in which the old model was about 25 % too low everywhere. A long
    # decay always wins then, and the learned 360 min dragged a weak first morning hour into
    # the whole afternoon.
    rng = np.random.default_rng(3)

    def biased(days):
        rows = []
        for d in range(days):
            for q in range(40):
                issued = d * DAY + 7 * 3600 + q * 900
                ratio = rng.uniform(1.1, 1.5)
                for horizon, length in (("30m", 1800), ("1h", 3600), ("6h", 21600)):
                    start = issued + 900
                    r = row(400, 500, start=start, horizon="1h", issued=issued, ratio=ratio)
                    rows.append(
                        EvalRow(replace(r.forecast, horizon=horizon, end=start + length), 500, None)
                    )
        return rows

    assert tau_from_rows(biased(3)) == 60.0  # 360 rows, but only three days: keep the default
    assert tau_from_rows(biased(20)) <= 180.0  # learned, but never longer than three hours
