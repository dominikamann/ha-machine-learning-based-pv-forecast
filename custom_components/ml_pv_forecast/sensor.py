"""Sensor entities per plant and for the total."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from typing import Any

import numpy as np
from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.history import state_changes_during_period
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity, SensorStateClass
from homeassistant.const import EntityCategory, UnitOfEnergy, UnitOfPower
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import EnergyConverter, PowerConverter

from .const import DOMAIN, TOTAL
from .coordinator import MlPvCoordinator
from .core.evaluation import tendency, today_so_far
from .core.variants import VARIANTS
from .history import EntityProblem, entity_kind

_LOGGER = logging.getLogger(__name__)
SEED_LOOKBACK = timedelta(minutes=30)


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
        ActualPowerSensor(coordinator, plant_id),
        ActualEnergyTodaySensor(coordinator, plant_id),
        HitRateSensor(coordinator, plant_id, "hit_rate_yesterday"),
        HitRateSensor(coordinator, plant_id, "hit_rate_week"),
        *([] if plant_id == TOTAL else [VariantSensor(coordinator, plant_id)]),
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
    _unrecorded_attributes = frozenset(
        {"hourly", "quarters", "morning", "actual", "daily", "review", "today", "past"}
    )

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

        def rows(times, values, start, end, scale):
            return [
                {
                    "time": _iso(t),
                    # float(): numpy numbers break JSON-based consumers such as templates
                    "p10": round(float(v[0]) * scale, 1),
                    "p50": round(float(v[1]) * scale, 1),
                    "p90": round(float(v[2]) * scale, 1),
                }
                for t, v in zip(times, values, strict=True)
                if start <= t < end
            ]

        morning = self.coordinator.morning.get(self._plant_id, [])
        actual_times, actual_wh = self.coordinator.actual_today.get(
            self._plant_id, (np.array([]), np.array([]))
        )

        today = today_so_far(morning, actual_times, actual_wh, w.today, w.tomorrow)

        def kwh(start: int, end: int) -> float:
            return round(float(fc.energy(start, end)[1]) / 1000, 2)

        return {
            # lets the dashboard card find every plant's curve
            "plant": None
            if self._plant_id == TOTAL
            else self.coordinator.plants[self._plant_id].name,
            "daily": {
                "today": kwh(w.today, w.tomorrow),
                "tomorrow": kwh(w.tomorrow, w.day_after),
                "day_after_tomorrow": kwh(w.day_after, w.day_after_end),
            },
            "actual": [
                {"time": _iso(t), "w": round(float(v) * 4, 1)}
                for t, v in zip(actual_times, actual_wh, strict=True)
                if w.today <= t < w.tomorrow
            ],
            "today": {
                **today,
                "until": _iso(today["until"]) if today["until"] is not None else None,
            },
            "review": [
                {
                    **day,
                    "date": _iso(day["date"]),
                    "hourly": [{**h, "time": _iso(h["time"])} for h in day["hourly"]],
                }
                for day in self.coordinator.review.get(self._plant_id, [])
            ],
            "hourly": [
                {**row, **self.coordinator.hour_weather(row["time"])}
                for row in rows(fc.hour_times, fc.hourly, w.today, w.day_after_end, 1.0)
            ],
            "quarters": rows(fc.quarter_times, fc.quarter, w.today, w.now + 86400, 4.0),
            # today's past hours as the live forecast saw them just before (W)
            "past": [
                {"time": _iso(t), "p10": round(a, 1), "p50": round(b, 1), "p90": round(c, 1)}
                for t, a, b, c in self.coordinator.past_hours.get(self._plant_id, [])
            ],
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
            # band check: about 10 % each is right, more means the band is too narrow
            "below_p10_pct": None if m.below_p10 is None else round(m.below_p10, 1),
            "above_p90_pct": None if m.above_p90 is None else round(m.above_p90, 1),
            "score_pct": None if m.score is None else round(m.score, 1),
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
        model = plant.model
        start, end = (model.history_start, model.history_end) if model else (0, 0)
        days = round((end - start) / 86400) if start and end else None
        return {
            "model": None if model is None else model.kind,
            "last_trained": None if plant.last_trained is None else _iso(plant.last_trained),
            "rows": model.rows if model else plant.rows,
            "history_from": _iso(start) if start else None,
            "history_days": days,
            "full_year": None if days is None else days >= 365,
            "error": plant.error,
        }


class VariantSensor(ForecastEntity):
    """Which model variant this plant runs, and how each variant scored at the last training."""

    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = [*(v.key for v in VARIANTS), "fallback"]
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:source-branch"

    def __init__(self, coordinator, plant_id) -> None:
        super().__init__(coordinator, plant_id, "variant")

    @property
    def available(self) -> bool:
        return True

    @property
    def native_value(self) -> str | None:
        model = self.coordinator.plants[self._plant_id].model
        if model is None:
            return None
        if model.kind != "gbm":
            return "fallback"  # the simple model: first weeks, or it beat every variant
        return model.variant

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        plant = self.coordinator.plants[self._plant_id]
        model = plant.model
        since = (model.variant_since or model.trained_at) if model else None
        return {
            # holdout quantile score of every variant at the last training, % of the actual energy
            "scores": plant.variant_scores or (model.variant_scores if model else {}),
            "since": _iso(since) if since else None,
            "testing": len(self.coordinator.variants) > 1,
        }


class PowerFromEnergy:
    """Average power (W) from an energy meter's rise over the last few minutes."""

    WINDOW = 300  # seconds

    def __init__(self) -> None:
        self._samples: deque[tuple[float, float]] = deque()

    def update(self, wh: float, now: float) -> float | None:
        if self._samples and wh < self._samples[-1][1]:
            self._samples.clear()  # meter reset
        self._samples.append((now, wh))
        while len(self._samples) > 2 and self._samples[1][0] <= now - self.WINDOW:
            self._samples.popleft()
        start, start_wh = self._samples[0]
        if len(self._samples) < 2 or now <= start:
            return None
        return max((wh - start_wh) / (now - start) * 3600, 0.0)


class ActualPowerSensor(ForecastEntity):
    """Current production, summed live from the plant's (or all plants') source sensors."""

    _attr_device_class = SensorDeviceClass.POWER
    _attr_native_unit_of_measurement = UnitOfPower.WATT
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 0

    def __init__(self, coordinator, plant_id) -> None:
        super().__init__(coordinator, plant_id, "actual_power")
        plants = (
            coordinator.plants.values() if plant_id == TOTAL else [coordinator.plants[plant_id]]
        )
        self._sources = [e for p in plants for e in p.entity_ids]
        self._meters = {e: PowerFromEnergy() for e in self._sources}
        self._value: float | None = None

    @property
    def available(self) -> bool:
        return self._value is not None

    @property
    def native_value(self) -> float | None:
        return None if self._value is None else round(self._value, 1)

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self.async_on_remove(
            async_track_state_change_event(self.hass, self._sources, self._async_source_changed)
        )
        # Energy meters do not report while production is zero; re-evaluate regularly.
        self.async_on_remove(
            async_track_time_interval(self.hass, self._async_tick, timedelta(minutes=1))
        )
        await self._async_seed_energy_meters()
        self._recompute()

    async def _async_seed_energy_meters(self) -> None:
        """Feed recent meter readings from the recorder, so power is known right after a restart."""
        start = dt_util.utcnow() - SEED_LOOKBACK
        for entity_id in self._sources:
            current = self.hass.states.get(entity_id)
            try:
                if entity_kind(self.hass, entity_id) != "energy":
                    continue
            except EntityProblem:
                continue
            unit = current.attributes.get("unit_of_measurement")
            try:
                changes = await get_instance(self.hass).async_add_executor_job(
                    partial(
                        state_changes_during_period,
                        self.hass,
                        start,
                        entity_id=entity_id,
                        no_attributes=True,
                        descending=True,
                        limit=3,
                    )
                )
            except Exception:  # noqa: BLE001 - seeding is best effort, live updates follow
                _LOGGER.debug("Could not read recent states of %s", entity_id, exc_info=True)
                continue
            for state in reversed(changes.get(entity_id, [])):
                try:
                    wh = EnergyConverter.convert(float(state.state), unit, UnitOfEnergy.WATT_HOUR)
                except ValueError:
                    continue
                self._meters[entity_id].update(wh, state.last_changed.timestamp())

    @callback
    def _async_source_changed(self, event) -> None:
        self._recompute()
        self.async_write_ha_state()

    @callback
    def _async_tick(self, now) -> None:
        self._recompute()
        self.async_write_ha_state()

    @callback
    def _handle_coordinator_update(self) -> None:
        self.async_write_ha_state()

    def _recompute(self) -> None:
        now = dt_util.utcnow().timestamp()
        total = 0.0
        for entity_id in self._sources:
            watts = self._watts(entity_id, now)
            if watts is None:
                self._value = None
                return
            total += watts
        self._value = total

    def _watts(self, entity_id: str, now: float) -> float | None:
        state = self.hass.states.get(entity_id)
        try:
            kind = entity_kind(self.hass, entity_id)
            value = float(state.state)
        except (EntityProblem, AttributeError, ValueError):
            return None
        unit = state.attributes.get("unit_of_measurement")
        if kind == "power":
            return max(PowerConverter.convert(value, unit, UnitOfPower.WATT), 0.0)
        wh = EnergyConverter.convert(value, unit, UnitOfEnergy.WATT_HOUR)
        return self._meters[entity_id].update(wh, now)


class ActualEnergyTodaySensor(ForecastEntity):
    """Today's production from the 15-minute actuals the forecast is evaluated against."""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_suggested_display_precision = 2

    def __init__(self, coordinator, plant_id) -> None:
        super().__init__(coordinator, plant_id, "actual_energy_today")

    @property
    def native_value(self) -> float | None:
        times, wh = self.coordinator.actual_today.get(self._plant_id, (np.array([]), np.array([])))
        today = Windows.current()
        mask = (times >= today.today) & (times < today.tomorrow)
        return round(float(np.sum(wh[mask])) / 1000, 3)


class HitRateSensor(ForecastEntity):
    """How closely this morning's forecast matched the actual production (yesterday, 7 days)."""

    _attr_native_unit_of_measurement = "%"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 0
    _attr_icon = "mdi:bullseye-arrow"

    def __init__(self, coordinator, plant_id, key) -> None:
        super().__init__(coordinator, plant_id, key)
        self._week = key == "hit_rate_week"

    def _days(self) -> list[dict]:
        days = self.coordinator.review.get(self._plant_id, [])
        days = days if self._week else days[:1]
        return [d for d in days if d["hit_rate"] is not None and d["complete"]]

    @property
    def native_value(self) -> float | None:
        days = self._days()
        if not days:
            return None
        return round(sum(d["hit_rate"] for d in days) / len(days), 1)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        days = self._days()
        bands = [d["in_band"] for d in days if d["in_band"] is not None]
        return {
            "days": len(days),
            "forecast_kwh": round(sum(d["forecast_kwh"] for d in days), 2) if days else None,
            "actual_kwh": round(sum(d["actual_kwh"] for d in days), 2) if days else None,
            "in_band": round(sum(bands) / len(bands), 1) if bands else None,
            # signed: positive when the morning forecasts were too optimistic
            "tendency_pct": tendency(days),
        }
