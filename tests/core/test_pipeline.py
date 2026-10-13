from dataclasses import replace

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
    assert result.holdout_score / day[day > 0].mean() < 0.15

    future_h = synthetic_weather(now, 48, seed=5)
    future_q = synthetic_weather(now, 192, interval=900, seed=5)
    fc = forecast_plant("a", result.model, future_h, future_q, LAT, LON, now)
    np.testing.assert_allclose(fc.energy(now, now + 86400)[1], fc.hourly[:24, 1].sum(), rtol=1e-6)

    limited = replace(result.model, limit=0.5 * float(fc.hourly[:, 2].max()))
    capped = forecast_plant(
        "a", limited, future_h, future_q, LAT, LON, now, ratio=2.0, band=(2.0, 2.0)
    )
    assert capped.hourly.max() <= limited.limit
    assert capped.quarter.max() <= limited.limit / 4
    assert capped.quarter_raw.max() <= limited.limit / 4

    boosted = forecast_plant("a", result.model, future_h, future_q, LAT, LON, now, ratio=1.5)
    first = boosted.quarter[0, 1]
    assert first == pytest.approx(1.5 * fc.quarter[0, 1])
    np.testing.assert_allclose(boosted.quarter_raw, fc.quarter_raw)


class HalfCorrection:
    def factors(self, hour, doy, cloud, p50_norm):
        return np.full(len(hour), 0.5)


def test_model_energy_is_before_correction_and_nowcast():
    w = synthetic_weather(START_2024, 60 * 24)
    y = synthetic_production(w)
    now = int(w.times[-1]) + 3600
    model = train_plant(w.times, y, w, LAT, LON, now, None, None, "h").model
    hourly = synthetic_weather(now, 48, seed=5)
    quarter = synthetic_weather(now, 192, interval=900, seed=5)
    fc = forecast_plant(
        "a", model, hourly, quarter, LAT, LON, now, correction=HalfCorrection(), ratio=1.5
    )
    end = now + 86400
    assert fc.model_energy(now, end)[1] > 0
    assert fc.raw_energy(now, end)[1] == pytest.approx(0.5 * fc.model_energy(now, end)[1])
    total = total_forecast([fc, fc], now)
    assert total.model_energy(now, end)[1] == pytest.approx(2 * fc.model_energy(now, end)[1])


def test_with_lead_assigns_days_ahead():
    from custom_components.ml_pv_forecast.core.pipeline import with_lead

    w = synthetic_weather(START_2024, 120)
    now = int(w.times[10])
    leads = with_lead(w, now).lead_days
    assert leads[0] == 0 and leads[10] == 0
    assert leads[10 + 30] == 1 and leads[10 + 60] == 2 and leads[10 + 100] == 2


def test_training_stacks_leads_and_skips_rows_without_weather():
    from dataclasses import replace

    from custom_components.ml_pv_forecast.core.features import FEATURE_VERSION
    from custom_components.ml_pv_forecast.core.model import is_current

    w0 = synthetic_weather(START_2024, 20 * 24)
    w1 = replace(
        w0, lead_days=np.ones(len(w0)), ghi=np.where(np.arange(len(w0)) < 48, np.nan, w0.ghi)
    )
    y = synthetic_production(w0)
    now = int(w0.times[-1]) + 3600
    single = train_plant(w0.times, y, w0, LAT, LON, now, None, None, "h")
    stacked = train_plant(w0.times, y, [w0, w1], LAT, LON, now, None, None, "h")
    assert stacked.rows > single.rows
    assert stacked.rows < 2 * single.rows  # rows without lead-1 weather were skipped
    assert stacked.model.feature_version == FEATURE_VERSION and is_current(stacked.model)


def test_day_so_far_lifts_the_later_hours_and_is_recorded():
    from custom_components.ml_pv_forecast.core.nowcast import DayFactor

    w = synthetic_weather(START_2024, 60 * 24)
    y = synthetic_production(w)
    now = int(w.times[-1]) + 3600
    model = train_plant(w.times, y, w, LAT, LON, now, None, None, "h").model
    hourly = synthetic_weather(now, 48, seed=5)
    quarter = synthetic_weather(now, 192, interval=900, seed=5)
    plain = forecast_plant("a", model, hourly, quarter, LAT, LON, now)
    day = DayFactor(ratio=1.4, share=0.5)
    lifted = forecast_plant("a", model, hourly, quarter, LAT, LON, now, day=day)
    later = (lifted.quarter_times >= now + 7200) & (plain.quarter[:, 1] > 0)
    assert (lifted.quarter[later, 1] > plain.quarter[later, 1]).all()
    np.testing.assert_allclose(lifted.quarter_raw, plain.quarter_raw)
    assert (lifted.day_ratio, lifted.day_share) == (1.4, 0.5)
    assert (plain.day_ratio, plain.day_share) == (None, None)


def test_band_scale_widens_hourly_and_quarter_bands_but_not_p50():
    w = synthetic_weather(START_2024, 60 * 24)
    y = synthetic_production(w)
    now = int(w.times[-1]) + 3600
    model = train_plant(w.times, y, w, LAT, LON, now, None, None, "h").model
    hourly = synthetic_weather(now, 48, seed=5)
    quarter = synthetic_weather(now, 192, interval=900, seed=5)
    plain = forecast_plant("a", model, hourly, quarter, LAT, LON, now)
    wide = forecast_plant("a", model, hourly, quarter, LAT, LON, now, band=(1.5, 2.0))
    np.testing.assert_allclose(wide.quarter[:, 1], plain.quarter[:, 1])
    np.testing.assert_allclose(wide.hourly[:, 1], plain.hourly[:, 1])
    np.testing.assert_allclose(wide.quarter_raw, plain.quarter_raw)
    up = plain.quarter[:, 2] - plain.quarter[:, 1]
    np.testing.assert_allclose(wide.quarter[:, 2] - wide.quarter[:, 1], 2.0 * up, atol=1e-6)
    assert (wide.hourly[:, 2] >= plain.hourly[:, 2]).all()
