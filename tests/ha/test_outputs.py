from http import HTTPStatus

from custom_components.ml_pv_forecast.const import CONF_EVCC
from custom_components.ml_pv_forecast.diagnostics import async_get_config_entry_diagnostics
from custom_components.ml_pv_forecast.energy import async_get_solar_forecast
from homeassistant.setup import async_setup_component

from .conftest import setup_plant


async def test_energy_platform_returns_hourly_wh(recorder_mock, hass, open_meteo):
    entry = await setup_plant(hass, open_meteo)
    result = await async_get_solar_forecast(hass, entry.entry_id)
    assert result and len(result["wh_hours"]) > 24
    assert all(v >= 0 for v in result["wh_hours"].values())


async def test_evcc_endpoint(recorder_mock, hass, hass_client, open_meteo):
    await async_setup_component(hass, "http", {})
    entry = await setup_plant(hass, open_meteo)
    client = await hass_client()
    resp = await client.get("/api/ml_pv_forecast/evcc")
    assert resp.status == HTTPStatus.OK
    rates = await resp.json()
    assert {"start", "end", "value"} <= set(rates[0])
    assert (await client.get("/api/ml_pv_forecast/evcc?quantile=p99")).status == HTTPStatus.BAD_REQUEST
    assert (await client.get("/api/ml_pv_forecast/evcc?plant=nope")).status == HTTPStatus.NOT_FOUND
    hass.config_entries.async_update_entry(entry, options={CONF_EVCC: False})
    await hass.async_block_till_done()
    assert (await client.get("/api/ml_pv_forecast/evcc")).status == HTTPStatus.NOT_FOUND


async def test_diagnostics_redact_location(recorder_mock, hass, open_meteo):
    entry = await setup_plant(hass, open_meteo)
    diag = await async_get_config_entry_diagnostics(hass, entry)
    assert diag["latitude"] == "**REDACTED**" and diag["plants"]
