"""Config flow: one entry per installation, one subentry per PV plant."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    OptionsFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_NAME
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
)

from .const import CONF_ENTITIES, CONF_EVCC, CONF_LIMIT_W, DOMAIN, SUBENTRY_PLANT
from .history import validate_entities


def _evcc_schema(default: bool) -> vol.Schema:
    return vol.Schema({vol.Required(CONF_EVCC, default=default): bool})


def _plant_schema(values: dict[str, Any]) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(CONF_NAME, description={"suggested_value": values.get(CONF_NAME)}): str,
            vol.Required(CONF_ENTITIES, default=values.get(CONF_ENTITIES, [])): EntitySelector(
                EntitySelectorConfig(
                    domain="sensor",
                    device_class=[SensorDeviceClass.POWER, SensorDeviceClass.ENERGY],
                    multiple=True,
                )
            ),
            vol.Optional(
                CONF_LIMIT_W, description={"suggested_value": values.get(CONF_LIMIT_W)}
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, step=1, unit_of_measurement="W", mode=NumberSelectorMode.BOX
                )
            ),
        }
    )


class MlPvForecastConfigFlow(ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()
        if user_input is not None:
            return self.async_create_entry(
                title="Machine-Learning-Based PV Forecast", data={}, options=user_input
            )
        return self.async_show_form(step_id="user", data_schema=_evcc_schema(True))

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        return MlPvForecastOptionsFlow()

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        return {SUBENTRY_PLANT: PlantSubentryFlow}


class MlPvForecastOptionsFlow(OptionsFlow):
    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        if user_input is not None:
            return self.async_create_entry(data=user_input)
        return self.async_show_form(
            step_id="init", data_schema=_evcc_schema(self.config_entry.options.get(CONF_EVCC, True))
        )


class PlantSubentryFlow(ConfigSubentryFlow):
    """Add or change a PV plant."""

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            error = validate_entities(self.hass, user_input[CONF_ENTITIES])
            if error is None:
                return self.async_create_entry(title=user_input[CONF_NAME], data=user_input)
            errors[CONF_ENTITIES] = error
        return self.async_show_form(
            step_id="user", data_schema=_plant_schema(user_input or {}), errors=errors
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        subentry = self._get_reconfigure_subentry()
        errors: dict[str, str] = {}
        if user_input is not None:
            error = validate_entities(self.hass, user_input[CONF_ENTITIES])
            if error is None:
                return self.async_update_and_abort(
                    self._get_entry(), subentry, title=user_input[CONF_NAME], data=user_input
                )
            errors[CONF_ENTITIES] = error
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=_plant_schema(user_input or dict(subentry.data)),
            errors=errors,
        )
