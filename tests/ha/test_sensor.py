from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
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
