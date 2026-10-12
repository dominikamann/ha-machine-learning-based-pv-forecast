from unittest.mock import patch

from homeassistant.config_entries import SOURCE_USER
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.ml_pv_forecast.const import CONF_ENTITIES, CONF_EVCC, CONF_LIMIT_W, DOMAIN

SETUP = "custom_components.ml_pv_forecast.async_setup_entry"


async def test_user_flow_creates_single_entry(hass):
    with patch(SETUP, return_value=True, create=True):
        result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
        result = await hass.config_entries.flow.async_configure(result["flow_id"], {CONF_EVCC: True})
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["options"] == {CONF_EVCC: True}
    again = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
    assert again["type"] is FlowResultType.ABORT


async def test_plant_subentry_validates_entities(hass):
    entry = MockConfigEntry(domain=DOMAIN, options={CONF_EVCC: True})
    entry.add_to_hass(hass)
    hass.states.async_set("sensor.pv", "0", {"unit_of_measurement": "W", "state_class": "measurement"})
    hass.states.async_set("sensor.temp", "0", {"unit_of_measurement": "°C", "state_class": "measurement"})

    result = await hass.config_entries.subentries.async_init((entry.entry_id, "plant"), context={"source": SOURCE_USER})
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"name": "Roof", CONF_ENTITIES: ["sensor.temp"]}
    )
    assert result["errors"] == {CONF_ENTITIES: "unsupported_unit"}
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"name": "Roof", CONF_ENTITIES: ["sensor.pv"], CONF_LIMIT_W: 7000}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    sub = next(iter(entry.subentries.values()))
    assert sub.title == "Roof" and sub.data[CONF_ENTITIES] == ["sensor.pv"]


async def test_options_flow_toggles_evcc(hass):
    entry = MockConfigEntry(domain=DOMAIN, options={CONF_EVCC: True})
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(result["flow_id"], {CONF_EVCC: False})
    assert entry.options == {CONF_EVCC: False}
