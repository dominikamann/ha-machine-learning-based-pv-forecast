"""Sun position, clear-sky irradiance and plane-of-array irradiance."""

from __future__ import annotations

import numpy as np

# (tilt, azimuth) of the fixed virtual planes used as model inputs. They are the same for every
# installation and have nothing to do with the user's real roofs.
VIRTUAL_PLANES: tuple[tuple[float, float], ...] = (
    (0.0, 180.0),
    (30.0, 90.0),
    (30.0, 135.0),
    (30.0, 180.0),
    (30.0, 225.0),
    (30.0, 270.0),
    (60.0, 180.0),
)

_J2000_DAYS = 10957.5  # days from 1970-01-01 00:00 UTC to 2000-01-01 12:00 UTC


def sun_position(
    times: np.ndarray, latitude: float, longitude: float
) -> tuple[np.ndarray, np.ndarray]:
    """Elevation and azimuth (degrees, azimuth clockwise from north) at unix times."""
    days = np.asarray(times, dtype=float) / 86400.0 - _J2000_DAYS
    mean_anomaly = np.radians((357.529 + 0.98560028 * days) % 360)
    mean_longitude = (280.459 + 0.98564736 * days) % 360
    ecliptic_longitude = np.radians(
        mean_longitude + 1.915 * np.sin(mean_anomaly) + 0.020 * np.sin(2 * mean_anomaly)
    )
    obliquity = np.radians(23.439 - 0.00000036 * days)
    right_ascension = np.arctan2(
        np.cos(obliquity) * np.sin(ecliptic_longitude), np.cos(ecliptic_longitude)
    )
    declination = np.arcsin(np.sin(obliquity) * np.sin(ecliptic_longitude))
    sidereal_hours = (18.697374558 + 24.06570982441908 * days) % 24
    hour_angle = np.radians(sidereal_hours * 15 + longitude) - right_ascension
    phi = np.radians(latitude)
    sin_elevation = np.sin(phi) * np.sin(declination) + np.cos(phi) * np.cos(
        declination
    ) * np.cos(hour_angle)
    elevation = np.arcsin(np.clip(sin_elevation, -1.0, 1.0))
    azimuth = np.arctan2(
        -np.sin(hour_angle),
        np.tan(declination) * np.cos(phi) - np.sin(phi) * np.cos(hour_angle),
    )
    return np.degrees(elevation), np.degrees(azimuth) % 360


def clear_sky_ghi(elevation: np.ndarray) -> np.ndarray:
    """Haurwitz clear-sky global horizontal irradiance in W/m²."""
    sin_el = np.sin(np.radians(np.asarray(elevation, dtype=float)))
    safe = np.where(sin_el > 0, sin_el, 1.0)
    return np.where(sin_el > 0, 1098.0 * sin_el * np.exp(-0.057 / safe), 0.0)


def plane_irradiance(
    ghi: np.ndarray,
    dni: np.ndarray,
    dhi: np.ndarray,
    elevation: np.ndarray,
    azimuth: np.ndarray,
    tilt: float,
    plane_azimuth: float,
    albedo: float = 0.2,
) -> np.ndarray:
    """Irradiance on a tilted plane (isotropic sky model) in W/m²."""
    beta = np.radians(tilt)
    zenith = np.radians(90.0 - np.asarray(elevation, dtype=float))
    cos_aoi = np.cos(zenith) * np.cos(beta) + np.sin(zenith) * np.sin(beta) * np.cos(
        np.radians(np.asarray(azimuth, dtype=float) - plane_azimuth)
    )
    beam = np.where(np.asarray(elevation) > 0, dni * np.clip(cos_aoi, 0.0, None), 0.0)
    sky = dhi * (1 + np.cos(beta)) / 2
    ground = ghi * albedo * (1 - np.cos(beta)) / 2
    return np.maximum(beam + sky + ground, 0.0)
