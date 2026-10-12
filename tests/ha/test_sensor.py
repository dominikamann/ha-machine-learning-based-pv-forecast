from datetime import datetime
from zoneinfo import ZoneInfo

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
