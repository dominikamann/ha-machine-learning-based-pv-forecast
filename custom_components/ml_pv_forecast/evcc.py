"""evcc forecast endpoint."""

from homeassistant.core import HomeAssistant, callback


@callback
def async_register_view(hass: HomeAssistant) -> None:
    """Registered in Task 14."""
