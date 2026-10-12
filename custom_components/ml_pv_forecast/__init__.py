"""Machine-Learning-Based PV Forecast."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .coordinator import MlPvCoordinator
from .evcc import async_register_view

PLATFORMS = [Platform.SENSOR]

type MlPvConfigEntry = ConfigEntry[MlPvCoordinator]


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
