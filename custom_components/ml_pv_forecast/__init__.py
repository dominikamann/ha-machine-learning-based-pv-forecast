"""Machine-Learning-Based PV Forecast."""

from __future__ import annotations

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.typing import ConfigType

from .const import DOMAIN
from .coordinator import MlPvCoordinator
from .evcc import async_register_view
from .frontend import async_register_card

PLATFORMS = [Platform.SENSOR]

type MlPvConfigEntry = ConfigEntry[MlPvCoordinator]

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)
RETRAIN_SCHEMA = vol.Schema({vol.Optional("plant"): cv.string})


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    async def retrain(call: ServiceCall) -> None:
        """Start training now for the named plant (by name) or for all plants."""
        name = call.data.get("plant")
        started = 0
        for entry in hass.config_entries.async_entries(DOMAIN):
            if entry.state is ConfigEntryState.LOADED:
                started += entry.runtime_data.async_retrain(name)
        if started == 0:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="unknown_plant",
                translation_placeholders={"plant": name or ""},
            )

    hass.services.async_register(DOMAIN, "retrain", retrain, schema=RETRAIN_SCHEMA)
    # Here, not in the entry setup: dashboards need the card even while that setup is retrying.
    await async_register_card(hass)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: MlPvConfigEntry) -> bool:
    # Listen before reading the plants: a plant added while setup runs then triggers a reload.
    entry.async_on_unload(entry.add_update_listener(_async_reload))
    coordinator = MlPvCoordinator(hass, entry)
    try:
        await coordinator.async_config_entry_first_refresh()
    except Exception:
        await coordinator.async_unload()  # the archive was opened in _async_setup
        raise
    entry.runtime_data = coordinator
    coordinator.async_start()
    async_register_view(hass)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def _async_reload(hass: HomeAssistant, entry: MlPvConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: MlPvConfigEntry) -> bool:
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        await entry.runtime_data.async_unload()
    return unloaded
