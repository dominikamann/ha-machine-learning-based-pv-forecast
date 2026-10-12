"""Home Assistant test helpers."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from homeassistant.components.recorder.models import StatisticMetaData
from homeassistant.components.recorder.statistics import async_import_statistics


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(recorder_mock, enable_custom_integrations):
    """Recorder must start before hass; every test here needs both."""
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
