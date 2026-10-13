from unittest.mock import patch

from homeassistant.config_entries import SOURCE_USER
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.ml_pv_forecast.const import (
    CONF_ENTITIES,
    CONF_EVCC,
    CONF_LIMIT_W,
    CONF_VARIANTS,
    DOMAIN,
)

SETUP = "custom_components.ml_pv_forecast.async_setup_entry"


async def test_user_flow_creates_single_entry(hass):
    with patch(SETUP, return_value=True, create=True):
        result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_EVCC: True}
        )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["options"] == {CONF_EVCC: True}
    again = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
    assert again["type"] is FlowResultType.ABORT


async def _start_plant(hass, entry, name="Roof", measurement="power"):
    result = await hass.config_entries.subentries.async_init(
        (entry.entry_id, "plant"), context={"source": SOURCE_USER}
    )
    assert result["step_id"] == "user"
    return await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"name": name, "measurement": measurement}
    )


async def test_plant_flow_asks_measurement_then_matching_sensors(hass):
    entry = MockConfigEntry(domain=DOMAIN, options={CONF_EVCC: True})
    entry.add_to_hass(hass)
    hass.states.async_set(
        "sensor.pv", "0", {"unit_of_measurement": "W", "state_class": "measurement"}
    )
    hass.states.async_set(
        "sensor.meter", "0", {"unit_of_measurement": "kWh", "state_class": "total_increasing"}
    )
    hass.states.async_set(
        "sensor.temp", "0", {"unit_of_measurement": "°C", "state_class": "measurement"}
    )

    result = await _start_plant(hass, entry)
    assert result["step_id"] == "sensors"
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_ENTITIES: ["sensor.temp"]}
    )
    assert result["errors"] == {CONF_ENTITIES: "unsupported_unit"}
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_ENTITIES: ["sensor.meter"]}
    )
    assert result["errors"] == {CONF_ENTITIES: "wrong_kind"}
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_ENTITIES: ["sensor.pv"], CONF_LIMIT_W: 7000}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    sub = next(iter(entry.subentries.values()))
    assert sub.title == "Roof" and sub.data[CONF_ENTITIES] == ["sensor.pv"]
    assert sub.data[CONF_LIMIT_W] == 7000


async def test_negative_power_history_asks_for_confirmation(hass):
    from datetime import timedelta

    from homeassistant.util import dt as dt_util
    from pytest_homeassistant_custom_component.components.recorder.common import (
        async_wait_recording_done,
    )

    from .conftest import import_statistics

    entry = MockConfigEntry(domain=DOMAIN, options={CONF_EVCC: True})
    entry.add_to_hass(hass)
    hass.states.async_set(
        "sensor.ac", "0", {"unit_of_measurement": "W", "state_class": "measurement"}
    )
    start = dt_util.utcnow().replace(minute=0, second=0, microsecond=0) - timedelta(hours=5)
    import_statistics(hass, "sensor.ac", "power", "W", start, [3000.0, -800.0, 2000.0])
    await async_wait_recording_done(hass)

    result = await _start_plant(hass, entry)
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {CONF_ENTITIES: ["sensor.ac"]}
    )
    assert result["step_id"] == "confirm_negative"
    assert result["description_placeholders"]["entity_id"] == "sensor.ac"
    result = await hass.config_entries.subentries.async_configure(result["flow_id"], {})
    assert result["type"] is FlowResultType.CREATE_ENTRY


async def test_options_flow_toggles_evcc(hass):
    entry = MockConfigEntry(domain=DOMAIN, options={CONF_EVCC: True})
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_EVCC: False}
    )
    assert entry.options == {CONF_EVCC: False, CONF_VARIANTS: True}
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_EVCC: False, CONF_VARIANTS: False}
    )
    assert entry.options == {CONF_EVCC: False, CONF_VARIANTS: False}
