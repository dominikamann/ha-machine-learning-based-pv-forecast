from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from homeassistant.helpers import entity_registry as er

from custom_components.ml_pv_forecast.sensor import Windows

from .conftest import setup_plant


async def test_windows_follow_dst(hass, freezer):
    await hass.config.async_set_time_zone("Europe/Berlin")
    freezer.move_to(datetime(2026, 10, 25, 10, 0, tzinfo=ZoneInfo("Europe/Berlin")))
    w = Windows.current()
    assert w.tomorrow - w.today == 25 * 3600
    assert w.day_after - w.tomorrow == 24 * 3600


async def test_sensors_created_with_values(recorder_mock, hass, open_meteo):
    entry = await setup_plant(hass, open_meteo)
    registry = er.async_get(hass)
    entities = er.async_entries_for_config_entry(registry, entry.entry_id)
    keys = {e.unique_id.rsplit("_", 1)[-1] for e in entities}
    assert {"today", "tomorrow", "status", "curve"} <= keys
    today = next(e for e in entities if e.unique_id.endswith("total_today"))
    state = hass.states.get(today.entity_id)
    assert float(state.state) >= 0
    assert "p10" in state.attributes and "p90" in state.attributes
    curve = next(e for e in entities if e.unique_id.endswith("total_curve"))
    assert len(hass.states.get(curve.entity_id).attributes["hourly"]) > 24


async def test_accuracy_without_data_is_unknown_not_unavailable(recorder_mock, hass, open_meteo):
    entry = await setup_plant(hass, open_meteo)
    registry = er.async_get(hass)
    accuracy = next(
        e
        for e in er.async_entries_for_config_entry(registry, entry.entry_id)
        if e.unique_id.endswith("total_accuracy_day_ahead")
    )
    state = hass.states.get(accuracy.entity_id)
    assert state.state == "unknown"
    assert state.attributes["status"] == "collecting"


async def test_curve_has_actual_today_daily_totals_and_hourly_from_midnight(
    recorder_mock, hass, open_meteo
):
    from homeassistant.util import dt as dt_util

    entry = await setup_plant(hass, open_meteo)
    coordinator = entry.runtime_data
    midnight = int(dt_util.start_of_local_day().timestamp())
    times = midnight + np.arange(4) * 900
    await hass.async_add_executor_job(
        coordinator.archive.add_actuals, "total", times, np.array([100.0, 200.0, 300.0, 400.0])
    )
    await coordinator.async_refresh()
    await hass.async_block_till_done()
    registry = er.async_get(hass)
    curve = next(
        e
        for e in er.async_entries_for_config_entry(registry, entry.entry_id)
        if e.unique_id.endswith("total_curve")
    )
    attrs = hass.states.get(curve.entity_id).attributes
    assert [round(a["w"]) for a in attrs["actual"][:4]] == [400, 800, 1200, 1600]
    assert set(attrs["daily"]) == {"today", "tomorrow", "day_after_tomorrow"}
    first = dt_util.parse_datetime(attrs["hourly"][0]["time"])
    assert first.timestamp() <= midnight + 3600


async def test_card_script_is_served(recorder_mock, hass, hass_client, open_meteo):
    from homeassistant.setup import async_setup_component

    await async_setup_component(hass, "http", {})
    await setup_plant(hass, open_meteo)
    client = await hass_client()
    resp = await client.get("/ml_pv_forecast/ml-pv-forecast-card.js")
    assert resp.status == 200
    assert "ml-pv-forecast-card" in await resp.text()


async def test_status_shows_history_coverage(recorder_mock, hass, open_meteo):
    entry = await setup_plant(hass, open_meteo)
    registry = er.async_get(hass)
    status = next(
        e
        for e in er.async_entries_for_config_entry(registry, entry.entry_id)
        if e.unique_id.endswith("_status") and "total" not in e.unique_id
    )
    attrs = hass.states.get(status.entity_id).attributes
    assert attrs["history_days"] >= 50 and attrs["history_from"] and attrs["rows"] > 0
    assert attrs["full_year"] is False


def _entity(hass, entry, suffix):
    registry = er.async_get(hass)
    return next(
        e.entity_id
        for e in er.async_entries_for_config_entry(registry, entry.entry_id)
        if e.unique_id.endswith(suffix)
    )


async def test_actual_power_follows_source_sensors_live(recorder_mock, hass, open_meteo):
    entry = await setup_plant(hass, open_meteo)
    plant_id = next(iter(entry.runtime_data.plants))
    attrs = dict(hass.states.get("sensor.pv").attributes)
    hass.states.async_set("sensor.pv", "1500", attrs)
    await hass.async_block_till_done()
    assert float(hass.states.get(_entity(hass, entry, f"{plant_id}_actual_power")).state) == 1500
    hass.states.async_set("sensor.pv", "2.5", {**attrs, "unit_of_measurement": "kW"})
    await hass.async_block_till_done()
    assert float(hass.states.get(_entity(hass, entry, f"{plant_id}_actual_power")).state) == 2500
    assert float(hass.states.get(_entity(hass, entry, "total_actual_power")).state) == 2500


async def test_actual_power_from_energy_meter(hass, freezer):
    from custom_components.ml_pv_forecast.sensor import PowerFromEnergy

    meter = PowerFromEnergy()
    t0 = 1_000_000.0
    assert meter.update(10_000.0, t0) is None  # needs two readings
    assert meter.update(10_050.0, t0 + 300) == pytest.approx(600.0)  # 50 Wh in 5 min
    assert meter.update(10_050.0, t0 + 300 + 1200) == 0.0  # no increase for 20 min
    assert meter.update(5.0, t0 + 1600) is None  # counter reset starts over


async def test_actual_energy_today_from_archive(recorder_mock, hass, open_meteo):
    from homeassistant.util import dt as dt_util

    entry = await setup_plant(hass, open_meteo)
    coordinator = entry.runtime_data
    midnight = int(dt_util.start_of_local_day().timestamp())
    await hass.async_add_executor_job(
        coordinator.archive.add_actuals, "total", midnight + np.arange(4) * 900, np.full(4, 250.0)
    )
    await coordinator.async_refresh()
    await hass.async_block_till_done()
    state = hass.states.get(_entity(hass, entry, "total_actual_energy_today"))
    assert float(state.state) == 1.0
    assert state.attributes["state_class"] == "total_increasing"


async def test_status_history_unknown_for_models_without_history(recorder_mock, hass, open_meteo):
    entry = await setup_plant(hass, open_meteo)
    plant = next(iter(entry.runtime_data.plants.values()))
    for key in ("history_start", "history_end", "rows"):
        vars(plant.model).pop(key, None)  # a model saved by 0.2.x
    await entry.runtime_data.async_refresh()
    await hass.async_block_till_done()
    attrs = hass.states.get(_entity(hass, entry, f"{plant.plant_id}_status")).attributes
    assert attrs["history_days"] is None and attrs["full_year"] is None


async def test_energy_meter_power_available_right_after_start(
    recorder_mock, hass, freezer, open_meteo
):
    from datetime import timedelta

    from homeassistant.config_entries import ConfigSubentryData
    from homeassistant.util import dt as dt_util
    from pytest_homeassistant_custom_component.common import MockConfigEntry
    from pytest_homeassistant_custom_component.components.recorder.common import (
        async_wait_recording_done,
    )

    from custom_components.ml_pv_forecast.const import CONF_ENTITIES, CONF_EVCC, DOMAIN

    meter = {"unit_of_measurement": "kWh", "state_class": "total_increasing"}
    start = dt_util.utcnow()
    hass.states.async_set("sensor.meter", "10.000", meter)
    await async_wait_recording_done(hass)
    freezer.move_to(start + timedelta(minutes=5))
    hass.states.async_set("sensor.meter", "10.050", meter)  # 50 Wh in 5 min = 600 W
    await async_wait_recording_done(hass)

    entry = MockConfigEntry(
        domain=DOMAIN,
        options={CONF_EVCC: False},
        subentries_data=[
            ConfigSubentryData(
                data={"name": "Roof", CONF_ENTITIES: ["sensor.meter"]},
                subentry_type="plant",
                title="Roof",
                unique_id=None,
            )
        ],
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)  # like a restart: sensor starts fresh
    await hass.async_block_till_done()
    state = hass.states.get(_entity(hass, entry, "total_actual_power"))
    assert state.state != "unavailable"
    assert float(state.state) == pytest.approx(600.0, rel=0.01)
