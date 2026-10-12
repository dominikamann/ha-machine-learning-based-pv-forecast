"""Serves the bundled dashboard card and adds it to the frontend."""

from __future__ import annotations

import json
from pathlib import Path

from homeassistant.components.http import StaticPathConfig
from homeassistant.core import HomeAssistant

from .const import DOMAIN

CARD_FILE = Path(__file__).parent / "cards" / "ml-pv-forecast-card.js"
CARD_URL = f"/{DOMAIN}/ml-pv-forecast-card.js"
_REGISTERED = f"{DOMAIN}_card_registered"


async def async_register_card(hass: HomeAssistant) -> None:
    """Register once per Home Assistant run; the version in the URL busts browser caches."""
    if hass.data.get(_REGISTERED):
        return
    hass.data[_REGISTERED] = True
    await hass.http.async_register_static_paths([StaticPathConfig(CARD_URL, str(CARD_FILE), False)])
    if "frontend" in hass.config.components:
        from homeassistant.components.frontend import add_extra_js_url

        manifest = await hass.async_add_executor_job(
            (Path(__file__).parent / "manifest.json").read_text
        )
        add_extra_js_url(hass, f"{CARD_URL}?v={json.loads(manifest)['version']}")
