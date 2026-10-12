"""Config flow: one entry per installation, one subentry per PV plant."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.config_entries import (
    SOURCE_RECONFIGURE,
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
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from .const import (
    CONF_ENTITIES,
    CONF_EVCC,
    CONF_LIMIT_W,
    CONF_MEASUREMENT,
    DOMAIN,
    MEASUREMENT_ENERGY,
    MEASUREMENT_POWER,
    SUBENTRY_PLANT,
)
from .history import EntityProblem, async_lowest_hourly_mean, entity_kind, validate_entities

# A power sensor this far below zero is usually an inverter's AC output with a battery.
NEGATIVE_POWER_W = -50.0


def _evcc_schema(default: bool) -> vol.Schema:
    return vol.Schema({vol.Required(CONF_EVCC, default=default): bool})


def _plant_schema(values: dict[str, Any]) -> vol.Schema:
    return vol.Schema(
        {
            vol.Required(CONF_NAME, description={"suggested_value": values.get(CONF_NAME)}): str,
            vol.Required(
                CONF_MEASUREMENT, default=values.get(CONF_MEASUREMENT, MEASUREMENT_POWER)
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[MEASUREMENT_POWER, MEASUREMENT_ENERGY],
                    translation_key=CONF_MEASUREMENT,
                    mode=SelectSelectorMode.LIST,
                )
            ),
        }
    )


def _sensors_schema(values: dict[str, Any], measurement: str) -> vol.Schema:
    device_class = (
        SensorDeviceClass.POWER if measurement == MEASUREMENT_POWER else SensorDeviceClass.ENERGY
    )
    return vol.Schema(
        {
            vol.Required(CONF_ENTITIES, default=values.get(CONF_ENTITIES, [])): EntitySelector(
                EntitySelectorConfig(domain="sensor", device_class=device_class, multiple=True)
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
    """Add or change a PV plant: name and kind of measurement, then the matching sensors."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> SubentryFlowResult:
        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_sensors()
        return self.async_show_form(step_id="user", data_schema=_plant_schema(self._data))

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        if not self._data:
            self._data = dict(self._get_reconfigure_subentry().data)
            self._data.setdefault(CONF_MEASUREMENT, _measurement_of(self.hass, self._data))
        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_sensors()
        return self.async_show_form(step_id="reconfigure", data_schema=_plant_schema(self._data))

    async def async_step_sensors(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        errors: dict[str, str] = {}
        measurement = self._data[CONF_MEASUREMENT]
        if user_input is not None:
            error = validate_entities(self.hass, user_input[CONF_ENTITIES])
            if error is None and any(
                entity_kind(self.hass, e) != measurement for e in user_input[CONF_ENTITIES]
            ):
                error = "wrong_kind"
            if error is None:
                self._data[CONF_ENTITIES] = user_input[CONF_ENTITIES]
                self._data.pop(CONF_LIMIT_W, None)
                if user_input.get(CONF_LIMIT_W):
                    self._data[CONF_LIMIT_W] = user_input[CONF_LIMIT_W]
                if measurement == MEASUREMENT_POWER:
                    for entity_id in user_input[CONF_ENTITIES]:
                        lowest = await async_lowest_hourly_mean(self.hass, entity_id)
                        if lowest is not None and lowest < NEGATIVE_POWER_W:
                            self._suspicious = (entity_id, lowest)
                            return await self.async_step_confirm_negative()
                return self._finish()
            errors[CONF_ENTITIES] = error
        return self.async_show_form(
            step_id="sensors",
            data_schema=_sensors_schema(user_input or self._data, measurement),
            errors=errors,
        )

    async def async_step_confirm_negative(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """A power sensor that goes clearly negative is usually an inverter's AC output."""
        if user_input is not None:
            return self._finish()
        entity_id, lowest = self._suspicious
        return self.async_show_form(
            step_id="confirm_negative",
            data_schema=vol.Schema({}),
            description_placeholders={"entity_id": entity_id, "lowest": f"{lowest:.0f}"},
        )

    def _finish(self) -> SubentryFlowResult:
        title = self._data[CONF_NAME]
        if self.source == SOURCE_RECONFIGURE:
            return self.async_update_and_abort(
                self._get_entry(), self._get_reconfigure_subentry(), title=title, data=self._data
            )
        return self.async_create_entry(title=title, data=self._data)


def _measurement_of(hass, data: dict[str, Any]) -> str:
    """Measurement of a plant configured before the measurement step existed."""
    try:
        return entity_kind(hass, data[CONF_ENTITIES][0])
    except (EntityProblem, IndexError, KeyError):
        return MEASUREMENT_POWER
