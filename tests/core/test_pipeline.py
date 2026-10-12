import numpy as np
import pytest

from custom_components.ml_pv_forecast.core.pipeline import (
    PlantForecast,
    forecast_plant,
    total_forecast,
    train_plant,
    window_cloud,
)
from tests.synthetic import LAT, LON, START_2024, synthetic_production, synthetic_weather

DAY = 86400


def simple_forecast(pid="a", value=100.0):
    qt = np.arange(0, 7200, 900)
    q = np.full((8, 3), value)
    return PlantForecast(pid, 0, np.array([0, 3600]), np.full((2, 3), value * 4), qt, q, q.copy())


def test_energy_is_pro_rata_and_peak_in_watts():
    f = simple_forecast()
    assert f.energy(0, 1800)[1] == 200.0
    assert f.energy(450, 1350)[1] == pytest.approx(100.0)
    watts, at = f.peak(0, 7200)
    assert watts == 400.0 and at == 0


def test_total_sums_plants():
    t = total_forecast([simple_forecast("a", 100), simple_forecast("b", 50)], 0)
    assert t.plant_id == "total" and t.energy(0, 900)[1] == 150.0


def test_window_cloud_is_mean_over_overlapping_hours():
    w = synthetic_weather(START_2024, 4)
    assert window_cloud(w, int(w.times[0]), int(w.times[2])) == pytest.approx(
        np.mean(((w.cloud_low + w.cloud_mid + w.cloud_high) / 3)[:2])
    )


def test_end_to_end_accuracy_on_synthetic_plant():
    days = 400
    w = synthetic_weather(START_2024, days * 24)
    y = synthetic_production(w, shade_after_hour_utc=14.0)
    now = int(w.times[-1]) + 3600
    result = train_plant(w.times, y, w, LAT, LON, now, None, None, "h1")
    assert result.model.kind == "gbm" and result.model.config_hash == "h1"
    day = y[w.times >= now - 60 * DAY]
    assert result.holdout_mae / day[day > 0].mean() < 0.15

    future_h = synthetic_weather(now, 48, seed=5)
    future_q = synthetic_weather(now, 192, interval=900, seed=5)
    fc = forecast_plant("a", result.model, future_h, future_q, LAT, LON, now)
    np.testing.assert_allclose(fc.energy(now, now + 86400)[1], fc.hourly[:24, 1].sum(), rtol=1e-6)

    boosted = forecast_plant("a", result.model, future_h, future_q, LAT, LON, now, ratio=1.5)
    first = boosted.quarter[0, 1]
    assert first == pytest.approx(1.5 * fc.quarter[0, 1])
    np.testing.assert_allclose(boosted.quarter_raw, fc.quarter_raw)
