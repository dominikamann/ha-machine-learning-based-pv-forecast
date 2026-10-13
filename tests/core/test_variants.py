import numpy as np

from custom_components.ml_pv_forecast.core.features import col
from custom_components.ml_pv_forecast.core.variants import (
    EXTRA_WEATHER,
    STANDARD,
    VARIANTS,
    choose,
    inputs,
)


def test_standard_hides_the_extra_weather_inputs_and_weather_plus_keeps_them():
    X = np.ones((3, max(col(n) for n in EXTRA_WEATHER) + 1))
    plain = inputs("standard", X)
    assert np.isnan(plain[:, [col(n) for n in EXTRA_WEATHER]]).all()
    assert np.isfinite(np.delete(plain, [col(n) for n in EXTRA_WEATHER], axis=1)).all()
    assert np.isfinite(inputs("weather_plus", X)).all()
    assert np.isfinite(X).all()  # the input is not changed
    assert [v.key for v in VARIANTS] == [
        "standard",
        "weather_plus",
        "short_memory",
        "weather_plus_short_memory",
    ]
    assert np.isfinite(inputs("weather_plus_short_memory", X)).all()
    assert STANDARD.key == "standard"


def test_choose_switches_only_for_a_clear_and_consistent_gain():
    rng = np.random.default_rng(0)
    actual = rng.uniform(5000, 30000, 60)
    active = actual * 0.15 * rng.uniform(0.8, 1.2, 60)
    errors = {
        "standard": active,
        "weather_plus": active * 0.9,  # 10 % better on every day
        "short_memory": active * 1.1,
    }
    assert choose(errors, "standard") == "weather_plus"


def test_choose_keeps_the_active_variant_for_small_or_noisy_gains():
    rng = np.random.default_rng(1)
    actual = rng.uniform(5000, 30000, 60)
    active = actual * 0.15
    small = {"standard": active, "weather_plus": active * 0.99, "short_memory": active}
    assert choose(small, "standard") == "standard"  # below the 2 % threshold
    noise = active * rng.uniform(0.3, 1.6, 60)
    noisy = {"standard": active, "weather_plus": noise * (active.sum() * 0.97 / noise.sum())}
    assert choose(noisy, "standard") == "standard"  # 3 % on average but not consistent
    assert choose({"weather_plus": active, "standard": active * 1.2}, "weather_plus") == (
        "weather_plus"
    )
    assert choose({"standard": active}, "short_memory") == "standard"  # active not evaluated
