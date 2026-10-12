"""Schedules forecasting, training and evaluation for all plants."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import async_track_time_change
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .const import CONF_ENTITIES, CONF_LIMIT_W, DOMAIN, SUBENTRY_PLANT, TOTAL
from .core.archive import Archive, ArchivedForecast
from .core.evaluation import Correction, Metrics, compute_metrics, tau_from_rows
from .core.model import NotEnoughData, PlantModel, load_model, save_model
from .core.nowcast import DEFAULT_TAU, live_ratio
from .core.pipeline import PlantForecast, forecast_plant, total_forecast, train_plant, window_cloud
from .core.solar import sun_position
from .core.weather import OpenMeteoClient, OpenMeteoError, WeatherData
from .history import EntityProblem, async_production, quarter_sums

_LOGGER = logging.getLogger(__name__)

SHORT_HORIZONS = {"30m": 1800, "1h": 3600, "6h": 21600}
HISTORY_DAYS = 6 * 365
WEATHER_MAX_AGE = timedelta(minutes=55)
WEATHER_REPAIR_AFTER = timedelta(hours=3)
EVALUATION_DAYS = 30
CORRECTION_DAYS = 365
RETENTION = timedelta(days=730)
MORNING_HOUR = 5
LIVE_WINDOW = 2700


def config_hash(entity_ids: list[str], limit_w: float | None) -> str:
    return hashlib.sha256(json.dumps([sorted(entity_ids), limit_w]).encode()).hexdigest()[:16]


@dataclass
class PlantRuntime:
    plant_id: str
    name: str
    entity_ids: list[str]
    limit_w: float | None
    config_hash: str
    model: PlantModel | None = None
    status: str = "learning"
    error: str | None = None
    rows: int = 0
    last_trained: int | None = None
    correction: Correction | None = None
    tau: float = DEFAULT_TAU
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class MlPvCoordinator(DataUpdateCoordinator[dict[str, PlantForecast]]):
    """Owns models, weather, archive and the latest forecasts."""

    config_entry: ConfigEntry

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(
            hass, _LOGGER, config_entry=entry, name=DOMAIN, update_interval=timedelta(minutes=15)
        )
        self.latitude = hass.config.latitude
        self.longitude = hass.config.longitude
        self.base_dir = Path(hass.config.path(".storage", DOMAIN))
        self.client = OpenMeteoClient(
            async_get_clientsession(hass), self.latitude, self.longitude, self.base_dir / "weather"
        )
        self.archive: Archive | None = None
        self.plants: dict[str, PlantRuntime] = {}
        for subentry in entry.subentries.values():
            if subentry.subentry_type != SUBENTRY_PLANT:
                continue
            ids = list(subentry.data[CONF_ENTITIES])
            limit = subentry.data.get(CONF_LIMIT_W)
            self.plants[subentry.subentry_id] = PlantRuntime(
                subentry.subentry_id, subentry.title, ids, limit, config_hash(ids, limit)
            )
        self.metrics: dict[str, dict[str, Metrics]] = {}
        self.morning: dict[str, list] = {}
        self.weather_fetched_at: datetime | None = None
        self._weather: tuple[WeatherData, WeatherData] | None = None
        self._weather_failing_since: datetime | None = None
        self._morning_day: str | None = None
        self._unsubscribe: list = []

    # ---- lifecycle -------------------------------------------------------------------------

    async def _async_setup(self) -> None:
        self.archive = await self.hass.async_add_executor_job(Archive, self.base_dir / "archive.db")
        if self.archive.recovered:
            _LOGGER.warning("Forecast archive was unreadable and has been reset")
        today = dt_util.now().date().isoformat()
        for plant in self.plants.values():
            model = await self.hass.async_add_executor_job(
                load_model, self._model_path(plant.plant_id)
            )
            if model is not None and model.config_hash == plant.config_hash:
                plant.model = model
                plant.status = "ready" if model.kind == "gbm" else "learning"
                plant.last_trained = model.trained_at
        for key in [*self.plants, TOTAL]:
            curve = await self.hass.async_add_executor_job(self.archive.load_curve, key, today)
            if curve is not None:
                self.morning[key] = curve
                self._morning_day = today
        self._unsubscribe = [
            async_track_time_change(self.hass, self._async_nightly, hour=2, minute=30, second=0),
            async_track_time_change(self.hass, self._async_weekly, hour=3, minute=0, second=0),
        ]
        for plant in self.plants.values():
            if plant.model is None:
                self.config_entry.async_create_background_task(
                    self.hass, self.async_train(plant.plant_id), f"{DOMAIN}_train_{plant.plant_id}"
                )
        self.config_entry.async_create_background_task(
            self.hass, self.async_evaluate(dt_util.utcnow()), f"{DOMAIN}_evaluate"
        )

    async def async_unload(self) -> None:
        for unsubscribe in self._unsubscribe:
            unsubscribe()
        if self.archive is not None:
            await self.hass.async_add_executor_job(self.archive.close)

    def _model_path(self, plant_id: str) -> Path:
        return self.base_dir / "models" / f"{plant_id}.joblib"

    def capacity(self, plant_id: str) -> float:
        if plant_id == TOTAL:
            return sum(p.model.capacity for p in self.plants.values() if p.model is not None)
        model = self.plants[plant_id].model
        return model.capacity if model is not None else 0.0

    def total_status(self) -> str:
        statuses = {p.status for p in self.plants.values()}
        if "error" in statuses:
            return "error"
        return "ready" if statuses == {"ready"} else "learning"

    # ---- issues ----------------------------------------------------------------------------

    def _issue(self, issue_id: str, key: str, **placeholders: str) -> None:
        ir.async_create_issue(
            self.hass,
            DOMAIN,
            issue_id,
            is_fixable=False,
            severity=ir.IssueSeverity.WARNING,
            translation_key=key,
            translation_placeholders=placeholders,
        )

    def _clear_issue(self, issue_id: str) -> None:
        ir.async_delete_issue(self.hass, DOMAIN, issue_id)

    def _entity_problem(self, plant: PlantRuntime, err: EntityProblem) -> None:
        plant.status, plant.error = "error", str(err)
        self._issue(
            f"entity_{plant.plant_id}",
            "entity_problem",
            plant=plant.name,
            entity_id=err.entity_id,
            reason=err.reason,
        )

    # ---- forecasting -----------------------------------------------------------------------

    async def _async_refresh_weather(self, now: datetime) -> None:
        if (
            self._weather is not None
            and self.weather_fetched_at is not None
            and now - self.weather_fetched_at < WEATHER_MAX_AGE
        ):
            return
        try:
            self._weather = await self.client.forecast()
        except OpenMeteoError as err:
            self._weather_failing_since = self._weather_failing_since or now
            if now - self._weather_failing_since >= WEATHER_REPAIR_AFTER:
                self._issue("open_meteo_unreachable", "open_meteo_unreachable", error=str(err))
            if self._weather is None:
                raise UpdateFailed(f"Open-Meteo: {err}") from err
            _LOGGER.warning("Open-Meteo failed, keeping previous forecast: %s", err)
            return
        self.weather_fetched_at = now
        self._weather_failing_since = None
        self._clear_issue("open_meteo_unreachable")

    async def _async_update_data(self) -> dict[str, PlantForecast]:
        now = dt_util.utcnow()
        now_ts = int(now.timestamp())
        await self._async_refresh_weather(now)
        hourly, quarter = self._weather
        if hourly.times[-1] + hourly.interval <= now_ts:
            raise UpdateFailed("weather forecast expired")

        forecasts: dict[str, PlantForecast] = {}
        actuals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for plant in self.plants.values():
            if plant.model is None:
                continue
            try:
                times, wh = await async_production(
                    self.hass, plant.entity_ids, now - timedelta(hours=2), now, "5minute"
                )
            except EntityProblem as err:
                self._entity_problem(plant, err)
                continue
            if plant.status == "error" and plant.error and "training" not in plant.error:
                plant.status = "ready" if plant.model.kind == "gbm" else "learning"
                plant.error = None
            self._clear_issue(f"entity_{plant.plant_id}")
            actuals[plant.plant_id] = quarter_sums(times, wh)
            ratio = self._live_ratio(plant, *actuals[plant.plant_id])
            forecasts[plant.plant_id] = await self.hass.async_add_executor_job(
                forecast_plant,
                plant.plant_id,
                plant.model,
                hourly,
                quarter,
                self.latitude,
                self.longitude,
                now_ts,
                plant.correction,
                ratio,
                plant.tau,
            )
        if not forecasts:
            return {}
        forecasts[TOTAL] = total_forecast(list(forecasts.values()), now_ts)
        actuals[TOTAL] = _sum_actuals(list(actuals.values()))
        daily = self._daily_windows(now)
        await self.hass.async_add_executor_job(
            self._archive_cycle, forecasts, actuals, now_ts, daily
        )
        if daily is not None:
            self._morning_day = daily[3]
        return forecasts

    def _live_ratio(self, plant: PlantRuntime, times: np.ndarray, wh: np.ndarray) -> float | None:
        previous = (self.data or {}).get(plant.plant_id)
        if previous is None or len(times) < 3:
            return None
        end = int(times[-1]) + 900
        start = end - LIVE_WINDOW
        if int(times[-3]) != start:
            return None
        elevation, _ = sun_position(
            np.array([start + LIVE_WINDOW / 2]), self.latitude, self.longitude
        )
        return live_ratio(
            float(wh[-3:].sum()),
            float(previous.raw_energy(start, end)[1]),
            float(elevation[0]),
            plant.model.capacity,
        )

    def _daily_windows(self, now: datetime) -> tuple[int, int, int, str] | None:
        local = dt_util.as_local(now)
        day = local.date().isoformat()
        if local.hour < MORNING_HOUR or self._morning_day == day:
            return None
        today = dt_util.start_of_local_day(local.date())
        tomorrow = dt_util.start_of_local_day(local.date() + timedelta(days=1))
        after = dt_util.start_of_local_day(local.date() + timedelta(days=2))
        return int(today.timestamp()), int(tomorrow.timestamp()), int(after.timestamp()), day

    def _archive_cycle(
        self,
        forecasts: dict[str, PlantForecast],
        actuals: dict[str, tuple[np.ndarray, np.ndarray]],
        now: int,
        daily: tuple[int, int, int, str] | None,
    ) -> None:
        hourly = self._weather[0]
        start = now - now % 900 + (900 if now % 900 else 0)
        rows = []
        for key, fc in forecasts.items():
            capacity = self.capacity(key)
            windows = [(h, start, start + length) for h, length in SHORT_HORIZONS.items()]
            if daily is not None:
                today, tomorrow, after, day = daily
                windows += [("today_morning", today, tomorrow), ("day_ahead", tomorrow, after)]
                mask = (fc.quarter_times >= today) & (fc.quarter_times < tomorrow)
                curve = [
                    [int(t), *map(float, v)]
                    for t, v in zip(fc.quarter_times[mask], fc.quarter[mask], strict=True)
                ]
                self.archive.save_curve(key, day, curve)
                self.morning[key] = curve
            for horizon, w_start, w_end in windows:
                p = fc.energy(w_start, w_end)
                rows.append(
                    ArchivedForecast(
                        key,
                        horizon,
                        now,
                        w_start,
                        w_end,
                        float(p[0]),
                        float(p[1]),
                        float(p[2]),
                        float(fc.raw_energy(w_start, w_end)[1]),
                        fc.ratio,
                        window_cloud(hourly, w_start, w_end),
                        capacity,
                    )
                )
        self.archive.add_forecasts(rows)
        for key, (times, wh) in actuals.items():
            self.archive.add_actuals(key, times, wh)

    # ---- training and evaluation -----------------------------------------------------------

    async def async_train(self, plant_id: str) -> None:
        plant = self.plants[plant_id]
        async with plant.lock:
            now = dt_util.utcnow()
            now_ts = int(now.timestamp())
            try:
                times, wh = await async_production(
                    self.hass, plant.entity_ids, now - timedelta(days=HISTORY_DAYS), now, "hour"
                )
                if len(times) == 0:
                    raise NotEnoughData("no statistics yet")
                first = dt_util.utc_from_timestamp(int(times[0])).date()
                last = (now - timedelta(days=1)).date()
                weather = await self.client.history(first, max(first, last), now.date())
                result = await self.hass.async_add_executor_job(
                    train_plant,
                    times,
                    wh,
                    weather,
                    self.latitude,
                    self.longitude,
                    now_ts,
                    plant.model,
                    plant.limit_w,
                    plant.config_hash,
                )
            except NotEnoughData as err:
                plant.status, plant.error = "learning", str(err)
                return
            except EntityProblem as err:
                self._entity_problem(plant, err)
                return
            except Exception as err:  # noqa: BLE001 - any failure keeps the active model
                _LOGGER.exception("Training plant %s failed", plant.name)
                if plant.model is None:
                    plant.status = "error"
                plant.error = f"training: {err}"
                self._issue(
                    f"training_{plant_id}", "training_failed", plant=plant.name, error=str(err)
                )
                return
            if result.promoted:
                await self.hass.async_add_executor_job(
                    save_model, result.model, self._model_path(plant_id)
                )
            plant.model, plant.status, plant.rows = result.model, result.status, result.rows
            plant.error, plant.last_trained = None, now_ts
            self._clear_issue(f"training_{plant_id}")
        await self.async_request_refresh()

    async def async_evaluate(self, now: datetime) -> None:
        if self.archive is None:
            return
        now_ts = int(now.timestamp())
        for key in [*self.plants, TOTAL]:
            rows = await self.hass.async_add_executor_job(
                self.archive.evaluated, key, None, now_ts - CORRECTION_DAYS * 86400, now_ts
            )
            recent = [r for r in rows if r.forecast.start >= now_ts - EVALUATION_DAYS * 86400]
            horizons = {r.forecast.horizon for r in recent}
            self.metrics[key] = {
                h: m
                for h in horizons
                if (m := compute_metrics([r for r in recent if r.forecast.horizon == h]))
            }
            if key in self.plants:
                plant = self.plants[key]
                plant.correction = await self.hass.async_add_executor_job(
                    Correction.from_rows, rows, now_ts
                )
                plant.tau = await self.hass.async_add_executor_job(tau_from_rows, rows)
        cutoff = now - RETENTION
        await self.hass.async_add_executor_job(
            self.archive.prune, int(cutoff.timestamp()), dt_util.as_local(cutoff).date().isoformat()
        )
        self.async_update_listeners()

    async def _async_nightly(self, now: datetime) -> None:
        await self.async_evaluate(dt_util.utcnow())
        for plant in list(self.plants.values()):
            if plant.model is None or plant.model.kind == "fallback":
                await self.async_train(plant.plant_id)

    async def _async_weekly(self, now: datetime) -> None:
        if dt_util.as_local(now).weekday() != 6:
            return
        for plant_id in list(self.plants):
            await self.async_train(plant_id)


def _sum_actuals(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Sum quarter actuals over plants; only quarters every plant has."""
    if not parts:
        return np.array([], dtype=np.int64), np.array([])
    times = parts[0][0]
    for t, _ in parts[1:]:
        times = np.intersect1d(times, t)
    total = sum(wh[np.searchsorted(t, times)] for t, wh in parts)
    return times, np.asarray(total, dtype=float)
