import numpy as np
import pytest

from custom_components.ml_pv_forecast.core import model as model_module
from custom_components.ml_pv_forecast.core.features import build_features
from custom_components.ml_pv_forecast.core.model import (
    NotEnoughData,
    PlantModel,
    fit_fallback,
    load_model,
    quantile_score,
    sample_weights,
    save_model,
    train,
)
from tests.synthetic import LAT, LON, START_2024, synthetic_production, synthetic_weather


def data(days, shade=None, start=START_2024):
    w = synthetic_weather(start, days * 24)
    return w.times, build_features(w, LAT, LON), synthetic_production(w, shade_after_hour_utc=shade)


def test_sample_weights_halve_after_a_year_and_downweight_reanalysis():
    w = sample_weights(np.array([0, 365 * 86400]), 365 * 86400, np.array([0.0, 1.0]))
    assert w[0] == pytest.approx(0.5)
    assert w[1] == pytest.approx(0.3)
    short = sample_weights(np.array([0]), 90 * 86400, np.array([0.0]), half_life_days=90)
    assert short[0] == pytest.approx(0.5)


def test_quantile_score_is_the_absolute_error_for_a_zero_wide_band():
    actual = np.array([100.0, 100.0])
    sharp = np.array([[80.0, 80.0, 80.0], [130.0, 130.0, 130.0]])
    np.testing.assert_allclose(quantile_score(sharp, actual), [20.0, 30.0])
    # A band that holds the actual value scores better than the same P50 without a band ...
    banded = np.array([[60.0, 80.0, 110.0]])
    assert quantile_score(banded, actual[:1])[0] < 20.0
    # ... and a needlessly wide band is penalised.
    wide = np.array([[0.0, 80.0, 400.0]])
    assert quantile_score(wide, actual[:1])[0] > quantile_score(banded, actual[:1])[0]


def test_limit_caps_every_quantile():
    times, X, y = data(20)
    m = fit_fallback(X, y, float(y.max()), int(times[-1]), limit=1000.0)
    p = m.predict(X)
    assert p.max() == 1000.0 and (p[:, 0] <= p[:, 1]).all() and (p[:, 1] <= p[:, 2]).all()


def test_capped_plant_is_forecast_without_bias():
    # A south roof whose production is capped at 60 % of its peak (balcony plant, export limit),
    # forecast for the summer after a year of history.
    w = synthetic_weather(START_2024 + 180 * 86400, 400 * 24)
    X = build_features(w, LAT, LON)
    free = synthetic_production(w, planes=((30.0, 180.0, 1.0),))
    limit = 0.6 * float(free.max())
    y = np.minimum(free, limit)
    now = int(w.times[-1]) + 3600
    train_part = w.times < now - 60 * 86400
    result = train(w.times[train_part], X[train_part], y[train_part], now, limit_w=limit)
    assert result.model.kind == "gbm"
    test = ~train_part & (y > 0)
    p = result.model.predict(X[test])
    assert p.max() <= limit
    assert abs(p[:, 1].sum() / y[test].sum() - 1) < 0.02
    capped = free[test] >= limit
    assert capped.sum() > 50
    assert np.mean(p[capped, 1] / y[test][capped]) > 0.98
    # the band holds the actual value (1 Wh of slack at the limit, which P90 approaches from
    # below) and moves with the weather instead of sticking to the limit
    assert np.mean(y[test] > p[:, 2] + 1.0) < 0.2 and p[~capped, 2].min() < 0.5 * limit


def test_fallback_tracks_production():
    times, X, y = data(20)
    m = fit_fallback(X, y, float(y.max()), int(times[-1]))
    p = m.predict(X)
    assert np.corrcoef(p[:, 1], y)[0, 1] > 0.95
    assert (p[:, 0] <= p[:, 1]).all() and (p[:, 1] <= p[:, 2]).all()


def test_short_history_trains_fallback_and_reports_learning():
    times, X, y = data(10)
    result = train(times, X, y, int(times[-1]) + 3600)
    assert result.model.kind == "fallback"
    assert result.status == "learning"


def test_long_history_trains_gbm_that_beats_baseline_with_shading():
    times, X, y = data(200, shade=14.0)
    result = train(times, X, y, int(times[-1]) + 3600)
    assert result.model.kind == "gbm" and result.promoted and result.status == "ready"
    assert result.holdout_score < 0.8 * result.baseline_score


def test_worse_candidate_keeps_active_model(monkeypatch):
    times, X, y = data(200)
    now = int(times[-1]) + 3600
    good = train(times, X, y, now - 10 * 86400).model

    def zero_gbm(X, y, weights, capacity, now, variant="standard", limit=None, should_stop=None):
        m = PlantModel("gbm", capacity, now, coef=None)
        m.predict = lambda X: np.zeros((len(X), 3))  # type: ignore[method-assign]
        return m

    monkeypatch.setattr(model_module, "fit_gbm", zero_gbm)
    result = train(times, X, y, now, active=good)
    assert result.model is good and not result.promoted


def test_no_production_raises():
    times, X, y = data(5)
    with pytest.raises(NotEnoughData):
        train(times, X, np.zeros_like(y), int(times[-1]))


def test_save_load_roundtrip_and_corrupt_file(tmp_path):
    times, X, y = data(10)
    m = fit_fallback(X, y, float(y.max()), 0, limit=900.0)
    path = tmp_path / "m.npz"
    save_model(m, path)
    loaded = load_model(path)
    assert {k: v for k, v in vars(loaded).items() if k != "coef"} == {
        k: v for k, v in vars(m).items() if k != "coef"
    }
    np.testing.assert_array_equal(loaded.coef, m.coef)
    np.testing.assert_array_equal(loaded.predict(X), m.predict(X))
    path.write_bytes(b"garbage")
    assert load_model(path) is None
    assert load_model(tmp_path / "missing.npz") is None


def test_gbm_survives_saving_bit_for_bit(tmp_path):
    times, X, y = data(200, shade=14.0)  # shading: the GBM clearly beats the physics model
    result = train(times, X, y, int(times[-1]) + 3600, limit_w=4000.0)
    assert result.model.kind == "gbm"
    save_model(result.model, tmp_path / "m.npz")
    loaded = load_model(tmp_path / "m.npz")
    np.testing.assert_array_equal(loaded.predict(X), result.model.predict(X))
    assert (loaded.variant, loaded.limit, loaded.rows) == ("standard", 4000.0, result.model.rows)


def test_files_of_another_format_or_pickles_are_not_loaded(tmp_path):
    import pickle

    from custom_components.ml_pv_forecast.core import model as model_module

    times, X, y = data(10)
    m = fit_fallback(X, y, float(y.max()), 0)
    (tmp_path / "old.npz").write_bytes(pickle.dumps(m))  # a pickle is never unpickled
    assert load_model(tmp_path / "old.npz") is None
    save_model(m, tmp_path / "m.npz")
    model_module.FORMAT_VERSION += 1
    try:
        assert load_model(tmp_path / "m.npz") is None
    finally:
        model_module.FORMAT_VERSION -= 1


def test_saving_removes_the_pickle_of_older_versions(tmp_path):
    times, X, y = data(10)
    (tmp_path / "m.pkl").write_bytes(b"old")
    save_model(fit_fallback(X, y, float(y.max()), 0), tmp_path / "m.npz")
    assert not (tmp_path / "m.pkl").exists()


@pytest.mark.slow
def test_fresh_refit_replaces_older_model_after_a_change():
    # New shading started 50 days ago. The active model (trained a week ago) already knows it;
    # a weekly refit must not be rejected just because its holdout candidate predates the change.
    times, X, y = data(200)
    now = int(times[-1]) + 3600
    hour = (times % 86400) / 3600
    y = np.where((times >= now - 50 * 86400) & (hour >= 13), y * 0.4, y)
    before = times < now - 7 * 86400
    active = train(times[before], X[before], y[before], now - 7 * 86400).model
    result = train(times, X, y, now, active=active)
    assert result.promoted and result.model is not active


def test_models_from_older_feature_sets_are_not_current():
    from custom_components.ml_pv_forecast.core.model import is_current

    times, X, y = data(10)
    m = fit_fallback(X, y, float(y.max()), 0)
    assert is_current(m)
    m.feature_version = 1
    assert not is_current(m)


def test_model_remembers_its_training_history():
    times, X, y = data(40)
    result = train(times, X, y, int(times[-1]) + 3600)
    m = result.model
    assert m.rows == result.rows > 0
    assert times[0] <= m.history_start < m.history_end <= times[-1]


@pytest.mark.slow
@pytest.mark.all_variants
def test_every_variant_is_scored_and_the_choice_is_recorded():
    from custom_components.ml_pv_forecast.core.variants import STANDARD, VARIANTS

    times, X, y = data(200)
    result = train(times, X, y, int(times[-1]) + 3600)
    assert set(result.variant_scores) == {v.key for v in VARIANTS}
    assert all(0 < s < 100 for s in result.variant_scores.values())
    assert result.model.variant in result.variant_scores
    only = train(times, X, y, int(times[-1]) + 3600, variants=(STANDARD,))
    assert set(only.variant_scores) == {"standard"} and only.model.variant == "standard"


@pytest.mark.slow
@pytest.mark.all_variants
def test_a_variant_that_sees_a_helpful_input_wins():
    from custom_components.ml_pv_forecast.core.features import col

    times, X, y = data(200)
    rng = np.random.default_rng(3)
    fog = rng.uniform(0, 1, len(y)) ** 3  # mostly clear, sometimes thick fog
    X = X.copy()
    X[:, col("visibility")] = 50000 * (1 - fog)
    y = y * (1 - 0.8 * fog)
    result = train(times, X, y, int(times[-1]) + 3600)
    assert result.model.variant in ("weather_plus", "weather_plus_short_memory")
    assert result.variant_scores[result.model.variant] < 0.9 * result.variant_scores["standard"]
    # the model keeps seeing visibility when it forecasts
    clear, foggy = X[:1].copy(), X[:1].copy()
    foggy[:, col("visibility")] = 0.0
    assert result.model.predict(foggy)[0, 1] <= result.model.predict(clear)[0, 1]
