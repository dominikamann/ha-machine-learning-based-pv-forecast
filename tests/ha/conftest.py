"""Home Assistant test helpers."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pytest
from homeassistant.components.recorder.models import StatisticMetaData
from homeassistant.components.recorder.statistics import async_import_statistics
from homeassistant.config_entries import ConfigEntryState, ConfigSubentryData
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.components.recorder.common import (
    async_wait_recording_done,
)

from custom_components.ml_pv_forecast.const import CONF_ENTITIES, CONF_EVCC, CONF_VARIANTS, DOMAIN
from tests.synthetic import synthetic_production, synthetic_weather


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(recorder_mock, enable_custom_integrations, hass, tmp_path):
    """Recorder must start before hass; every test gets its own config dir for models/archive."""
    hass.config.config_dir = str(tmp_path)
    yield


def import_statistics(hass, entity_id: str, kind: str, unit: str, start: datetime, values):
    """Import hourly statistics: means for power, cumulative sums for energy."""
    metadata = {
        "has_mean": kind == "power",
        "has_sum": kind == "energy",
        "name": None,
        "source": "recorder",
        "statistic_id": entity_id,
        "unit_of_measurement": unit,
    }
    annotations = getattr(StatisticMetaData, "__annotations__", {})
    if "mean_type" in annotations:
        from homeassistant.components.recorder.models import StatisticMeanType

        metadata["mean_type"] = (
            StatisticMeanType.ARITHMETIC if kind == "power" else StatisticMeanType.NONE
        )
    if "unit_class" in annotations:
        metadata["unit_class"] = kind
    rows, total = [], 0.0
    for i, value in enumerate(values):
        hour = start + timedelta(hours=i)
        if kind == "power":
            rows.append({"start": hour, "mean": value, "min": value, "max": value})
        else:
            total += value
            rows.append({"start": hour, "state": total, "sum": total})
    async_import_statistics(hass, metadata, rows)


CLIENT = "custom_components.ml_pv_forecast.coordinator.OpenMeteoClient"
DAYS = 60


async def _value(value):
    return value


@pytest.fixture
async def open_meteo(hass):
    """Patched Open-Meteo client for the whole test (also across entry reloads)."""
    hass.config.latitude, hass.config.longitude = 50.0, 10.0
    now = dt_util.utcnow().replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=DAYS)
    history = synthetic_weather(int(start.timestamp()), DAYS * 24)
    future_start = int(now.timestamp()) - 86400
    hourly = synthetic_weather(future_start, 5 * 24, seed=3)
    quarter = synthetic_weather(future_start, 5 * 96, interval=900, seed=3)
    with patch(CLIENT) as client_cls:
        client = client_cls.return_value
        client.forecast.side_effect = lambda **_: _value((hourly, quarter))
        client.history.side_effect = lambda *args, **kwargs: _value(history)
        leads = [history, replace(history, lead_days=np.ones(len(history)))]
        client.history_leads.side_effect = lambda *args, **kwargs: _value(leads)
        client.history_start = start
        client.history_weather = history
        yield client
        for entry in hass.config_entries.async_entries(DOMAIN):
            await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()


async def setup_plant(hass, open_meteo, variants=False):
    """One plant ("Roof", sensor.pv) with 60 days of statistics, set up and refreshed.

    Only the standard variant is trained unless ``variants`` (four variants are slow to train).
    """
    hass.states.async_set(
        "sensor.pv", "0", {"unit_of_measurement": "W", "state_class": "measurement"}
    )
    production = synthetic_production(open_meteo.history_weather).tolist()
    import_statistics(hass, "sensor.pv", "power", "W", open_meteo.history_start, production)
    await async_wait_recording_done(hass)
    entry = MockConfigEntry(
        domain=DOMAIN,
        options={CONF_EVCC: True, CONF_VARIANTS: variants},
        subentries_data=[
            ConfigSubentryData(
                data={"name": "Roof", CONF_ENTITIES: ["sensor.pv"]},
                subentry_type="plant",
                title="Roof",
                unique_id=None,
            )
        ],
    )
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    if entry.state is ConfigEntryState.LOADED:
        await entry.runtime_data.async_refresh()
        await hass.async_block_till_done()
    return entry
