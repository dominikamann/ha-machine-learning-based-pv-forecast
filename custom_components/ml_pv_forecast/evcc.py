"""Forecast endpoint in the format evcc expects for custom solar forecasts."""

from __future__ import annotations

from http import HTTPStatus

from aiohttp import web
from homeassistant.components.http import HomeAssistantView
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.http import KEY_HASS
from homeassistant.util import dt as dt_util

from .const import CONF_EVCC, DOMAIN, TOTAL
from .core.pipeline import PlantForecast

QUANTILE_INDEX = {"p10": 0, "p50": 1, "p90": 2}
_REGISTERED = f"{DOMAIN}_evcc_view"


def evcc_rates(forecast: PlantForecast, index: int) -> list[dict]:
    """15-minute slots with average power in W."""
    return [
        {
            "start": dt_util.utc_from_timestamp(int(t)).isoformat(),
            "end": dt_util.utc_from_timestamp(int(t) + 900).isoformat(),
            "value": round(float(v[index]) * 4, 1),
        }
        for t, v in zip(forecast.quarter_times, forecast.quarter, strict=True)
    ]


class EvccForecastView(HomeAssistantView):
    url = "/api/ml_pv_forecast/evcc"
    name = "api:ml_pv_forecast:evcc"
    requires_auth = True

    async def get(self, request: web.Request) -> web.Response:
        hass = request.app[KEY_HASS]
        entry = next(
            (
                e
                for e in hass.config_entries.async_entries(DOMAIN)
                if e.state is ConfigEntryState.LOADED
            ),
            None,
        )
        if entry is None or not entry.options.get(CONF_EVCC, True):
            return self.json_message("evcc endpoint is disabled", HTTPStatus.NOT_FOUND)
        quantile = request.query.get("quantile", "p50")
        if quantile not in QUANTILE_INDEX:
            return self.json_message("quantile must be p10, p50 or p90", HTTPStatus.BAD_REQUEST)
        forecast = (entry.runtime_data.data or {}).get(request.query.get("plant", TOTAL))
        if forecast is None:
            return self.json_message("unknown plant or no forecast yet", HTTPStatus.NOT_FOUND)
        return self.json(evcc_rates(forecast, QUANTILE_INDEX[quantile]))


@callback
def async_register_view(hass: HomeAssistant) -> None:
    if hass.data.get(_REGISTERED):
        return
    hass.http.register_view(EvccForecastView())
    hass.data[_REGISTERED] = True
