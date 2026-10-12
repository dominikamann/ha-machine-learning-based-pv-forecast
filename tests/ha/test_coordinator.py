from datetime import timedelta
from unittest.mock import patch

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
