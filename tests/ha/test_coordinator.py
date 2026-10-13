from datetime import timedelta
from unittest.mock import patch

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.helpers import issue_registry as ir

from custom_components.ml_pv_forecast.const import DOMAIN
from custom_components.ml_pv_forecast.core.weather import OpenMeteoError

from .conftest import setup_plant


async def fail():
    raise OpenMeteoError("down")


async def test_setup_trains_and_forecasts(recorder_mock, hass, open_meteo):
    entry = await setup_plant(hass, open_meteo)
    assert entry.state is ConfigEntryState.LOADED
    coordinator = entry.runtime_data
    plant = next(iter(coordinator.plants.values()))
    assert plant.model is not None and plant.status in ("learning", "ready")
    assert plant.plant_id in coordinator.data and "total" in coordinator.data
    assert coordinator.data["total"].energy(0, 2**40)[1] > 0


async def test_open_meteo_down_at_startup_retries(recorder_mock, hass, open_meteo):
    open_meteo.forecast.side_effect = fail
    entry = await setup_plant(hass, open_meteo)
    assert entry.state is ConfigEntryState.SETUP_RETRY


async def test_open_meteo_down_later_keeps_forecast_and_raises_issue(
    recorder_mock, hass, open_meteo
):
    entry = await setup_plant(hass, open_meteo)
    coordinator = entry.runtime_data
    before = coordinator.data
    coordinator.weather_fetched_at -= timedelta(hours=2)
    open_meteo.forecast.side_effect = fail
    await coordinator.async_refresh()
    assert coordinator.last_update_success and coordinator.data.keys() == before.keys()
    coordinator._weather_failing_since -= timedelta(hours=4)
    await coordinator.async_refresh()
    assert ir.async_get(hass).async_get_issue(DOMAIN, "open_meteo_unreachable") is not None


async def test_no_training_starts_while_setup_is_retrying(recorder_mock, hass, open_meteo):
    open_meteo.forecast.side_effect = fail
    with patch("custom_components.ml_pv_forecast.coordinator.MlPvCoordinator.async_train") as train:
        entry = await setup_plant(hass, open_meteo)
    assert entry.state is ConfigEntryState.SETUP_RETRY
    train.assert_not_called()


async def test_plant_added_during_setup_is_picked_up(recorder_mock, hass, open_meteo):
    import asyncio

    from homeassistant.config_entries import ConfigSubentry

    release = asyncio.Event()
    weather = open_meteo.forecast.side_effect

    async def slow_forecast():
        await release.wait()
        return await weather()

    open_meteo.forecast.side_effect = slow_forecast
    hass.states.async_set(
        "sensor.balcony", "0", {"unit_of_measurement": "W", "state_class": "measurement"}
    )
    setup = hass.async_create_task(setup_plant(hass, open_meteo))
    try:
        async with asyncio.timeout(30):  # wait until setup blocks on the first weather fetch
            while not open_meteo.forecast.called:
                await asyncio.sleep(0.05)
        entry = hass.config_entries.async_entries(DOMAIN)[0]
        assert entry.state is ConfigEntryState.SETUP_IN_PROGRESS
        hass.config_entries.async_add_subentry(
            entry,
            ConfigSubentry(
                data={"name": "Balcony", "entities": ["sensor.balcony"]},
                subentry_type="plant",
                title="Balcony",
                unique_id=None,
            ),
        )
    finally:
        release.set()
    entry = await setup
    await hass.async_block_till_done(wait_background_tasks=True)
    names = {p.name for p in entry.runtime_data.plants.values()}
    assert names == {"Roof", "Balcony"}


async def test_briefly_missing_sensor_raises_no_repair(recorder_mock, hass, open_meteo):
    entry = await setup_plant(hass, open_meteo)
    coordinator = entry.runtime_data
    plant = next(iter(coordinator.plants.values()))
    issue_id = f"entity_{plant.plant_id}"
    attributes = dict(hass.states.get("sensor.pv").attributes)
    hass.states.async_remove("sensor.pv")  # e.g. its integration is still starting
    await coordinator.async_refresh()
    assert ir.async_get(hass).async_get_issue(DOMAIN, issue_id) is None
    hass.states.async_set("sensor.pv", "0", attributes)
    await coordinator.async_refresh()
    assert plant.status in ("ready", "learning") and plant.plant_id in coordinator.data


async def test_sensor_missing_for_long_raises_repair(recorder_mock, hass, open_meteo):
    entry = await setup_plant(hass, open_meteo)
    coordinator = entry.runtime_data
    plant = next(iter(coordinator.plants.values()))
    hass.states.async_remove("sensor.pv")
    await coordinator.async_refresh()
    plant.problem_since -= timedelta(minutes=31)
    await coordinator.async_refresh()
    assert ir.async_get(hass).async_get_issue(DOMAIN, f"entity_{plant.plant_id}") is not None


async def test_plant_without_model_trains_once_its_sensor_is_back(recorder_mock, hass, open_meteo):
    entry = await setup_plant(hass, open_meteo)
    coordinator = entry.runtime_data
    plant = next(iter(coordinator.plants.values()))
    # State after a start-up where the sensor did not exist yet and training failed.
    plant.model, plant.status, plant.error = None, "learning", "sensor.pv: missing"
    plant.retry_at = None
    await coordinator.async_refresh()
    await hass.async_block_till_done(wait_background_tasks=True)
    assert plant.model is not None and plant.error is None


async def test_retrain_service_trains_named_or_all_plants(recorder_mock, hass, open_meteo):
    entry = await setup_plant(hass, open_meteo)
    plant = next(iter(entry.runtime_data.plants.values()))
    before = plant.model
    await hass.services.async_call(DOMAIN, "retrain", {"plant": "Roof"}, blocking=True)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert plant.model is not before
    with patch("custom_components.ml_pv_forecast.coordinator.MlPvCoordinator.async_train") as train:
        await hass.services.async_call(DOMAIN, "retrain", {}, blocking=True)
        await hass.async_block_till_done(wait_background_tasks=True)
    assert train.call_count == 1


async def test_retrain_service_rejects_unknown_plant(recorder_mock, hass, open_meteo):
    import pytest
    from homeassistant.exceptions import ServiceValidationError

    await setup_plant(hass, open_meteo)
    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(DOMAIN, "retrain", {"plant": "Nope"}, blocking=True)


async def test_recent_actuals_are_backfilled_from_short_term_statistics(
    recorder_mock, hass, open_meteo
):
    import numpy as np
    from homeassistant.util import dt as dt_util

    midnight = int(dt_util.start_of_local_day().timestamp())
    five_min = midnight + 6 * 3600 + np.arange(12) * 300  # 06:00–07:00 local
    real = __import__("custom_components.ml_pv_forecast.coordinator", fromlist=["x"])
    original = real.async_production

    async def fake_production(hass_, entity_ids, start, end, period):
        if period == "5minute" and end - start > timedelta(hours=3):
            return five_min, np.full(12, 25.0)
        return await original(hass_, entity_ids, start, end, period)

    with patch.object(real, "async_production", fake_production):
        entry = await setup_plant(hass, open_meteo)
        await hass.async_block_till_done(wait_background_tasks=True)
    coordinator = entry.runtime_data
    plant_id = next(iter(coordinator.plants))
    for key in (plant_id, "total"):
        times, wh = await hass.async_add_executor_job(
            coordinator.archive.actuals, key, midnight, midnight + 86400
        )
        assert len(times) == 4 and wh.tolist() == [75.0] * 4


async def test_day_factor_compares_today_so_far_with_the_forecast(recorder_mock, hass, open_meteo):
    import numpy as np
    from homeassistant.util import dt as dt_util

    from custom_components.ml_pv_forecast.core.pipeline import PlantForecast

    entry = await setup_plant(hass, open_meteo)
    coordinator = entry.runtime_data
    plant = next(iter(coordinator.plants.values()))
    midnight = int(dt_util.start_of_local_day().timestamp())
    quarters = midnight + np.arange(96) * 900
    raw = np.full((96, 3), 100.0)
    coordinator.data[plant.plant_id] = PlantForecast(
        plant.plant_id, midnight, quarters[::4], raw[::4] * 4, quarters, raw, raw.copy()
    )
    coordinator.actual_today[plant.plant_id] = (quarters[:24], np.full(24, 150.0))
    day = coordinator._day_factor(plant, midnight)
    assert day.ratio == pytest.approx(1.5)
    assert day.share == pytest.approx(0.25)
    assert (day.weight, day.tau) == (plant.day_weight, plant.day_tau)
    coordinator.actual_today[plant.plant_id] = (quarters[:0], np.zeros(0))
    assert coordinator._day_factor(plant, midnight) is None  # nothing measured yet today


def test_morning_forecast_only_frozen_in_the_morning():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from custom_components.ml_pv_forecast.coordinator import MORNING_UNTIL_HOUR, should_freeze

    tz = ZoneInfo("Europe/Berlin")
    assert should_freeze(datetime(2026, 10, 12, 6, 0, tzinfo=tz), None)
    assert not should_freeze(datetime(2026, 10, 12, 4, 0, tzinfo=tz), None)
    assert not should_freeze(datetime(2026, 10, 12, MORNING_UNTIL_HOUR, 0, tzinfo=tz), None)
    assert not should_freeze(datetime(2026, 10, 12, 6, 0, tzinfo=tz), "2026-10-12")
