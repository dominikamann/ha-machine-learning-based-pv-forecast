import numpy as np
import pytest

from custom_components.ml_pv_forecast.core.archive import ArchivedForecast, EvalRow
from custom_components.ml_pv_forecast.core.evaluation import (
    Correction,
    compute_metrics,
    tau_from_rows,
)

DAY = 86400


def row(p50, actual, persistence=None, start=0, horizon="day_ahead", issued=None, ratio=None):
    end = start + (DAY if horizon == "day_ahead" else 3600)
    f = ArchivedForecast("a", horizon, start - 3600 if issued is None else issued, start, end,
                         p50 * 0.8, p50, p50 * 1.2, p50, ratio, 50.0, 1000.0)
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
    f = corr.factors(np.array([1.0]), np.array([12.0]), np.array([100.0]), np.array([50.0]),
                     np.array([0.5]))
    assert f[0] == pytest.approx(0.8, abs=0.05)


def test_correction_needs_thirty_days():
    rows = [row(500, 400, start=d * DAY + 12 * 3600, horizon="1h") for d in range(10)]
    assert Correction.from_rows(rows, 10 * DAY) is None


def test_tau_from_rows_uses_short_rows_with_ratio():
    rows = [row(500, 500, start=1800 * i, horizon="1h", issued=1800 * i, ratio=1.0) for i in range(60)]
    assert tau_from_rows(rows) == 30.0  # ratio 1 everywhere: every tau ties, first grid value wins
    assert tau_from_rows([row(500, 500, horizon="1h")]) == 90.0  # no ratio -> default
    assert tau_from_rows([]) == 90.0
