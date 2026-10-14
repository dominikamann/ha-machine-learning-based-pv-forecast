"""Solar forecast for the Energy dashboard."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import TOTAL


async def async_get_solar_forecast(
    hass: HomeAssistant, config_entry_id: str
) -> dict[str, dict[str, float | int]] | None:
    entry = hass.config_entries.async_get_entry(config_entry_id)
    if entry is None or entry.state is not ConfigEntryState.LOADED:
        return None
    forecast = (entry.runtime_data.data or {}).get(TOTAL)
    if forecast is None:
        return None
    return {
        "wh_hours": {
            dt_util.utc_from_timestamp(int(t)).isoformat(): round(float(v), 1)
            # the same live-corrected values as the sensors and evcc
            for t, v in zip(forecast.hour_times, forecast.hourly_p50(), strict=True)
        }
    }
