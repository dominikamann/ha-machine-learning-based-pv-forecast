from datetime import timedelta

import numpy as np
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.components.recorder.common import (
    async_wait_recording_done,
)

from custom_components.ml_pv_forecast.history import (
    EntityProblem,
    async_production,
    entity_kind,
    quarter_sums,
    validate_entities,
)

from .conftest import import_statistics


def set_power(hass, entity_id, unit="W", state_class="measurement"):
    hass.states.async_set(entity_id, "0", {"unit_of_measurement": unit, "state_class": state_class})


async def test_entity_validation(recorder_mock, hass):
    set_power(hass, "sensor.p")
    hass.states.async_set(
        "sensor.e", "1", {"unit_of_measurement": "kWh", "state_class": "total_increasing"}
    )
    hass.states.async_set(
        "sensor.t", "1", {"unit_of_measurement": "°C", "state_class": "measurement"}
    )
    set_power(hass, "sensor.nostats", state_class=None)
    assert entity_kind(hass, "sensor.p") == "power"
    assert entity_kind(hass, "sensor.e") == "energy"
    assert validate_entities(hass, ["sensor.p", "sensor.e"]) is None
    assert validate_entities(hass, []) == "no_entities"
    assert validate_entities(hass, ["sensor.missing"]) == "missing"
    assert validate_entities(hass, ["sensor.t"]) == "unsupported_unit"
    assert validate_entities(hass, ["sensor.nostats"]) == "no_statistics"
    try:
        entity_kind(hass, "sensor.missing")
    except EntityProblem as err:
        assert err.reason == "missing"


async def test_power_in_kw_is_converted_and_late_entity_counts_as_zero(recorder_mock, hass):
    start = dt_util.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=10)
    set_power(hass, "sensor.a", unit="kW")
    set_power(hass, "sensor.b")
    import_statistics(hass, "sensor.a", "power", "kW", start, [1.0] * 6)
    import_statistics(
        hass, "sensor.b", "power", "W", start + timedelta(hours=3), [500.0, -20.0, 500.0]
    )
    await async_wait_recording_done(hass)
    times, wh = await async_production(
        hass, ["sensor.a", "sensor.b"], start, start + timedelta(hours=6), "hour"
    )
    assert len(times) == 6
    assert wh.tolist() == [1000.0, 1000.0, 1000.0, 1500.0, 1000.0, 1500.0]


async def test_energy_counter_reset_is_never_negative(recorder_mock, hass):
    start = dt_util.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=10)
    hass.states.async_set(
        "sensor.e", "1", {"unit_of_measurement": "kWh", "state_class": "total_increasing"}
    )
    import_statistics(hass, "sensor.e", "energy", "kWh", start, [1.0, 2.0, -5.0, 1.0])
    await async_wait_recording_done(hass)
    times, wh = await async_production(
        hass, ["sensor.e"], start, start + timedelta(hours=4), "hour"
    )
    assert (wh >= 0).all()
    assert 2000.0 in wh.tolist()


def test_quarter_sums_keep_only_complete_quarters():
    times = np.array([0, 300, 600, 900, 1200])
    t, wh = quarter_sums(times, np.ones(5))
    assert t.tolist() == [0] and wh.tolist() == [3.0]


def test_quarter_sums_drop_quarters_with_gaps():
    t, wh = quarter_sums(np.array([0, 300, 600, 900, 1200, 1500]), np.array([1, np.nan, 1, 1, 1, 1.0]))
    assert t.tolist() == [900] and wh.tolist() == [3.0]
