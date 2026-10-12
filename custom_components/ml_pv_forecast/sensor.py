"""Sensor entities per plant and for the total."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.const import EntityCategory, UnitOfEnergy, UnitOfPower
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util

from .const import DOMAIN, TOTAL
from .coordinator import MlPvCoordinator


@dataclass(frozen=True)
class Windows:
    now: int
    today: int
    tomorrow: int
    day_after: int
    day_after_end: int

    @classmethod
    def current(cls) -> Windows:
        day = dt_util.now().date()
        starts = [
            int(dt_util.start_of_local_day(day + timedelta(days=i)).timestamp()) for i in range(4)
        ]
        return cls(int(dt_util.utcnow().timestamp()), *starts)


ENERGY_WINDOWS: dict[str, Callable[[Windows], tuple[int, int]]] = {
    "next_30m": lambda w: (w.now, w.now + 1800),
    "next_1h": lambda w: (w.now, w.now + 3600),
    "next_6h": lambda w: (w.now, w.now + 21600),
    "today": lambda w: (w.today, w.tomorrow),
    "today_remaining": lambda w: (w.now, w.tomorrow),
    "tomorrow": lambda w: (w.tomorrow, w.day_after),
    "day_after_tomorrow": lambda w: (w.day_after, w.day_after_end),
}
PEAK_WINDOWS = {
    "peak_today": lambda w: (w.today, w.tomorrow),
    "peak_tomorrow": lambda w: (w.tomorrow, w.day_after),
}
ACCURACY_HORIZONS = ("30m", "1h", "6h", "today_morning", "day_ahead")


def _iso(ts: int) -> str:
    return dt_util.utc_from_timestamp(int(ts)).isoformat()


async def async_setup_entry(
    hass: HomeAssistant, entry, async_add_entities: AddConfigEntryEntitiesCallback
) -> None:
    coordinator: MlPvCoordinator = entry.runtime_data
    for plant_id in coordinator.plants:
        async_add_entities(_entities(coordinator, plant_id), config_subentry_id=plant_id)
    if coordinator.plants:
        async_add_entities(_entities(coordinator, TOTAL))


def _entities(coordinator: MlPvCoordinator, plant_id: str) -> list[SensorEntity]:
    return [
        *(EnergySensor(coordinator, plant_id, key, fn) for key, fn in ENERGY_WINDOWS.items()),
        *(PeakSensor(coordinator, plant_id, key, fn) for key, fn in PEAK_WINDOWS.items()),
        CurveSensor(coordinator, plant_id),
        *(AccuracySensor(coordinator, plant_id, h) for h in ACCURACY_HORIZONS),
        StatusSensor(coordinator, plant_id),
    ]


class ForecastEntity(CoordinatorEntity[MlPvCoordinator], SensorEntity):
    _attr_has_entity_name = True

    def __init__(self, coordinator: MlPvCoordinator, plant_id: str, key: str) -> None:
        super().__init__(coordinator)
        self._plant_id = plant_id
        self._attr_translation_key = key
        entry_id = coordinator.config_entry.entry_id
        self._attr_unique_id = f"{entry_id}_{plant_id}_{key}"
        name = "Total" if plant_id == TOTAL else coordinator.plants[plant_id].name
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, f"{entry_id}_{plant_id}")},
            name=f"PV Forecast {name}",
            manufacturer="Machine-Learning-Based PV Forecast",
            entry_type=DeviceEntryType.SERVICE,
        )

    @property
    def _forecast(self):
        return (self.coordinator.data or {}).get(self._plant_id)


class EnergySensor(ForecastEntity):
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_suggested_display_precision = 2

    def __init__(self, coordinator, plant_id, key, window) -> None:
        super().__init__(coordinator, plant_id, key)
        self._window = window

    @property
    def available(self) -> bool:
        return super().available and self._forecast is not None

    def _values(self):
        return self._forecast.energy(*self._window(Windows.current())) / 1000

    @property
    def native_value(self) -> float | None:
        return round(float(self._values()[1]), 3) if self._forecast else None

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        if not self._forecast:
            return None
        p10, _, p90 = self._values()
        return {"p10": round(float(p10), 3), "p90": round(float(p90), 3)}


class PeakSensor(ForecastEntity):
    _attr_device_class = SensorDeviceClass.POWER
    _attr_native_unit_of_measurement = UnitOfPower.WATT

    def __init__(self, coordinator, plant_id, key, window) -> None:
        super().__init__(coordinator, plant_id, key)
        self._window = window

    def _peak(self):
        return self._forecast.peak(*self._window(Windows.current())) if self._forecast else None

    @property
    def available(self) -> bool:
        return super().available and self._peak() is not None

    @property
    def native_value(self) -> float | None:
        peak = self._peak()
        return round(peak[0]) if peak else None

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        peak = self._peak()
        return {"time": _iso(peak[1])} if peak else None


class CurveSensor(ForecastEntity):
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _unrecorded_attributes = frozenset({"hourly", "quarters", "morning"})

    def __init__(self, coordinator, plant_id) -> None:
        super().__init__(coordinator, plant_id, "curve")

    @property
    def available(self) -> bool:
        return super().available and self._forecast is not None

    @property
    def native_value(self) -> float | None:
        if not self._forecast:
            return None
        w = Windows.current()
        return round(float(self._forecast.energy(w.today, w.tomorrow)[1]) / 1000, 3)

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        fc = self._forecast
        if not fc:
            return None
        w = Windows.current()
        hour_start = w.now - w.now % 3600

        def rows(times, values, start, end, scale):
            return [
                {
                    "time": _iso(t),
                    "p10": round(v[0] * scale, 1),
                    "p50": round(v[1] * scale, 1),
                    "p90": round(v[2] * scale, 1),
                }
                for t, v in zip(times, values, strict=True)
                if start <= t < end
            ]

        morning = self.coordinator.morning.get(self._plant_id, [])
        return {
            "hourly": rows(fc.hour_times, fc.hourly, hour_start, hour_start + 72 * 3600, 1.0),
            "quarters": rows(fc.quarter_times, fc.quarter, w.today, w.now + 86400, 4.0),
            "morning": [
                {
                    "time": _iso(t),
                    "p10": round(a * 4, 1),
                    "p50": round(b * 4, 1),
                    "p90": round(c * 4, 1),
                }
                for t, a, b, c in morning
            ],
        }


class AccuracySensor(ForecastEntity):
    _attr_native_unit_of_measurement = "%"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator, plant_id, horizon) -> None:
        super().__init__(coordinator, plant_id, f"accuracy_{horizon}")
        self._horizon = horizon

    def _metrics(self):
        return self.coordinator.metrics.get(self._plant_id, {}).get(self._horizon)

    @property
    def native_value(self) -> float | None:
        # Unknown (not unavailable) until the nightly evaluation has archived results to compare.
        m = self._metrics()
        return round(m.nmae, 1) if m and m.nmae is not None else None

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        m = self._metrics()
        if not m:
            return {"status": "collecting"}
        return {
            "status": "ready",
            "mae_kwh": round(m.mae / 1000, 3),
            "bias_kwh": round(m.bias / 1000, 3),
            "skill": None if m.skill is None else round(m.skill, 3),
            "count": m.count,
        }


class StatusSensor(ForecastEntity):
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = ["learning", "ready", "error"]
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator, plant_id) -> None:
        super().__init__(coordinator, plant_id, "status")

    @property
    def available(self) -> bool:
        return True

    @property
    def native_value(self) -> str:
        if self._plant_id == TOTAL:
            return self.coordinator.total_status()
        return self.coordinator.plants[self._plant_id].status

    @property
    def extra_state_attributes(self) -> dict[str, Any] | None:
        if self._plant_id == TOTAL:
            return None
        plant = self.coordinator.plants[self._plant_id]
        return {
            "model": None if plant.model is None else plant.model.kind,
            "last_trained": None if plant.last_trained is None else _iso(plant.last_trained),
            "rows": plant.rows,
            "error": plant.error,
        }
