from datetime import UTC, datetime

import numpy as np
import pytest

from custom_components.ml_pv_forecast.core.solar import (
    VIRTUAL_PLANES,
    clear_sky_ghi,
    plane_irradiance,
    sun_position,
)


def ts(*args):
    return np.array([datetime(*args, tzinfo=UTC).timestamp()])


def test_summer_solstice_noon_berlin():
    el, az = sun_position(ts(2024, 6, 21, 11, 7), 52.52, 13.40)
    assert el[0] == pytest.approx(60.9, abs=0.6)
    assert az[0] == pytest.approx(180.0, abs=3.0)


def test_morning_sun_is_east_and_night_is_below_horizon():
    el, az = sun_position(ts(2024, 6, 21, 5, 0), 52.52, 13.40)
    assert 0 < el[0] < 30 and 45 < az[0] < 110
    el, _ = sun_position(ts(2024, 6, 21, 23, 0), 52.52, 13.40)
    assert el[0] < 0


def test_clear_sky_zero_at_night_and_positive_by_day():
    out = clear_sky_ghi(np.array([-5.0, 0.0, 30.0, 60.0]))
    assert out[0] == 0 and out[1] == 0
    assert 300 < out[2] < out[3] < 1100


def test_horizontal_plane_equals_ghi_for_consistent_components():
    el, az = np.array([40.0]), np.array([180.0])
    dni, dhi = np.array([600.0]), np.array([100.0])
    ghi = dni * np.sin(np.radians(el)) + dhi
    out = plane_irradiance(ghi, dni, dhi, el, az, 0.0, 180.0)
    assert out[0] == pytest.approx(ghi[0])


def test_south_plane_beats_north_plane_at_noon():
    el, az = np.array([20.0]), np.array([180.0])
    ghi, dni, dhi = np.array([400.0]), np.array([700.0]), np.array([160.0])
    south = plane_irradiance(ghi, dni, dhi, el, az, 30.0, 180.0)
    north = plane_irradiance(ghi, dni, dhi, el, az, 30.0, 0.0)
    assert south[0] > north[0]


def test_seven_virtual_planes():
    assert len(VIRTUAL_PLANES) == 7
