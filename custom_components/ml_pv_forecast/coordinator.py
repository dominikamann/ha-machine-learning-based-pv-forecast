"""Schedules forecasting, training and evaluation for all plants."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import async_track_time_change
from homeassistant.helpers.start import async_at_started
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .const import CONF_ENTITIES, CONF_LIMIT_W, CONF_VARIANTS, DOMAIN, SUBENTRY_PLANT, TOTAL
from .core.archive import Archive, ArchivedForecast
from .core.band import NEUTRAL, fit_band, hourly_rows
from .core.band import WINDOW_DAYS as BAND_WINDOW_DAYS
from .core.evaluation import (
    Correction,
    Metrics,
    compute_metrics,
    day_from_rows,
    day_review,
    tau_from_rows,
)
from .core.gbm import Cancelled
from .core.model import NotEnoughData, PlantModel, is_current, load_model, save_model
from .core.nowcast import (
    DEFAULT_DAY_TAU,
    DEFAULT_DAY_WEIGHT,
    DEFAULT_TAU,
    DayFactor,
    day_ratio,
    live_ratio,
)
from .core.pipeline import PlantForecast, forecast_plant, total_forecast, train_plant, window_cloud
from .core.solar import sun_position
from .core.variants import STANDARD, VARIANTS, Variant
from .core.weather import OpenMeteoClient, OpenMeteoError, WeatherData
from .history import EntityProblem, async_production, entity_kind, quarter_sums

_LOGGER = logging.getLogger(__name__)

SHORT_HORIZONS = {"30m": 1800, "1h": 3600, "6h": 21600}
HISTORY_DAYS = 6 * 365
WEATHER_MAX_AGE = timedelta(minutes=55)
WEATHER_REPAIR_AFTER = timedelta(hours=3)
ENTITY_REPAIR_AFTER = timedelta(minutes=30)
TRAIN_RETRY_AFTER = timedelta(hours=1)
EVALUATION_DAYS = 30
CORRECTION_DAYS = 365
RETENTION = timedelta(days=730)
SHORT_RETENTION_DAYS = CORRECTION_DAYS + 35  # short-horizon rows are the bulk of the archive
MORNING_HOUR = 5
MORNING_UNTIL_HOUR = 10  # a "morning forecast" issued later than this would be misleading
REVIEW_DAYS = 7
BACKFILL = timedelta(days=10)  # Home Assistant keeps 5-minute statistics for 10 days by default


def should_freeze(local_now: datetime, frozen_day: str | None) -> bool:
    """Whether to freeze today's morning forecast now (once a day, only in the morning)."""
    return (
        MORNING_HOUR <= local_now.hour < MORNING_UNTIL_HOUR
        and frozen_day != local_now.date().isoformat()
    )


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
    day_weight: float = DEFAULT_DAY_WEIGHT
    day_tau: float = DEFAULT_DAY_TAU
    band: tuple[float, float] = NEUTRAL
    variant_scores: dict[str, float] = field(default_factory=dict)  # from the last training
    problem_since: datetime | None = None
    retry_at: datetime | None = None
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
        self.actual_today: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.review: dict[str, list[dict]] = {}  # last REVIEW_DAYS days, newest first
        self.weather_fetched_at: datetime | None = None
        self._weather: tuple[WeatherData, WeatherData] | None = None
        self._weather_failing_since: datetime | None = None
        self._morning_day: str | None = None
        self._freeze_morning = False
        self._unsubscribe: list = []
        # Training runs for minutes in an executor thread; unloading sets this to stop it.
        self._stop_training = threading.Event()

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
            if model is not None and model.config_hash == plant.config_hash and is_current(model):
                plant.model = model
                plant.status = "ready" if model.kind == "gbm" else "learning"
                plant.last_trained = model.trained_at
        midnight = int(dt_util.start_of_local_day().timestamp())
        for key in [*self.plants, TOTAL]:
            self.actual_today[key] = await self.hass.async_add_executor_job(
                self.archive.actuals, key, midnight, midnight + 2 * 86400
            )
            curve = await self.hass.async_add_executor_job(self.archive.load_curve, key, today)
            if curve is not None:
                self.morning[key] = curve
                self._morning_day = today

    @callback
    def async_start(self) -> None:
        """Start training, evaluation and the schedules once the first refresh succeeded."""
        self._unsubscribe = [
            async_track_time_change(self.hass, self._nightly, hour=2, minute=30, second=0),
            async_track_time_change(self.hass, self._weekly, hour=3, minute=0, second=0),
        ]
        self._background(self.async_evaluate(dt_util.utcnow()), "evaluate")
        # Train only once Home Assistant has started: sensors of integrations that load after us
        # (ESPHome, Shelly, ...) do not exist before. Runs right away if HA is already running.
        self._unsubscribe.append(async_at_started(self.hass, self._async_started))

    async def _async_started(self, hass: HomeAssistant) -> None:
        self._background(self._async_backfill_actuals(), "backfill")
        for plant in self.plants.values():
            if plant.model is None:
                self._start_training(plant)
        await self.async_request_refresh()

    @callback
    def async_retrain(self, name: str | None = None) -> int:
        """Start training for the plant with this name (any case) or all; returns how many."""
        plants = [
            p for p in self.plants.values() if name is None or p.name.casefold() == name.casefold()
        ]
        for plant in plants:
            self._start_training(plant)
        return len(plants)

    async def async_update_review(self) -> None:
        """Morning forecast against actual production for each of the last days."""
        today = dt_util.now().date()
        days = []
        for back in range(1, REVIEW_DAYS + 1):
            day = today - timedelta(days=back)
            start = int(dt_util.start_of_local_day(day).timestamp())
            end = int(dt_util.start_of_local_day(day + timedelta(days=1)).timestamp())
            days.append((day.isoformat(), start, end))
        for key in [*self.plants, TOTAL]:
            self.review[key] = await self.hass.async_add_executor_job(self._review, key, days)
        self.async_update_listeners()

    def _review(self, key: str, days: list[tuple[str, int, int]]) -> list[dict]:
        out = []
        for day, start, end in days:
            times, wh = self.archive.actuals(key, start, end)
            out.append(day_review(self.archive.load_curve(key, day), times, wh, start, end))
        return out

    async def _async_backfill_actuals(self) -> None:
        """Take the last days of actual production from Home Assistant's 5-minute statistics.

        The archive only records while the integration runs; without this, charts and evaluation
        would start empty after installing (or miss what happened during a long downtime).
        """
        now = dt_util.utcnow()
        parts = []
        for plant in self.plants.values():
            try:
                times, wh = await async_production(
                    self.hass, plant.entity_ids, now - BACKFILL, now, "5minute"
                )
            except EntityProblem:
                continue
            quarters = quarter_sums(times, wh)
            parts.append(quarters)
            await self.hass.async_add_executor_job(
                self.archive.add_actuals, plant.plant_id, *quarters
            )
        if len(parts) == len(self.plants) and parts:
            await self.hass.async_add_executor_job(
                self.archive.add_actuals, TOTAL, *_sum_actuals(parts)
            )
        midnight = int(dt_util.start_of_local_day().timestamp())
        for key in [*self.plants, TOTAL]:
            self.actual_today[key] = await self.hass.async_add_executor_job(
                self.archive.actuals, key, midnight, midnight + 2 * 86400
            )
        await self.async_update_review()

    def _start_training(self, plant: PlantRuntime) -> None:
        plant.retry_at = dt_util.utcnow() + TRAIN_RETRY_AFTER
        self._background(self.async_train(plant.plant_id), f"train_{plant.plant_id}")

    def _background(self, coro, name: str) -> None:
        # Tracked by the entry, so HA cancels it on unload before the archive is closed.
        self.config_entry.async_create_background_task(self.hass, coro, f"{DOMAIN}_{name}")

    @callback
    def _nightly(self, now: datetime) -> None:
        self._background(self._async_nightly(now), "nightly")

    @callback
    def _weekly(self, now: datetime) -> None:
        self._background(self._async_weekly(now), "weekly")

    async def async_unload(self) -> None:
        self._stop_training.set()
        for unsubscribe in self._unsubscribe:
            unsubscribe()
        if self.archive is not None:
            await self.hass.async_add_executor_job(self.archive.close)

    def _model_path(self, plant_id: str) -> Path:
        return self.base_dir / "models" / f"{plant_id}.pkl"

    def hour_weather(self, iso_time: str) -> dict:
        """Weather code (WMO) and temperature for one forecast hour, for display."""
        if self._weather is None:
            return {"wc": None, "t": None}
        hourly = self._weather[0]
        start = int(dt_util.parse_datetime(iso_time).timestamp())
        i = int(np.searchsorted(hourly.times, start))
        if i >= len(hourly.times) or hourly.times[i] != start:
            return {"wc": None, "t": None}
        code, temp = hourly.weather_code[i], hourly.temperature[i]
        return {
            "wc": int(code) if np.isfinite(code) else None,
            "t": round(float(temp), 1) if np.isfinite(temp) else None,
        }

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
        # Source sensors may appear only after their integration started; report persistent ones.
        now = dt_util.utcnow()
        plant.problem_since = plant.problem_since or now
        plant.error = str(err)
        if now - plant.problem_since < ENTITY_REPAIR_AFTER:
            return
        plant.status = "error"
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
        midnight = int(dt_util.start_of_local_day().timestamp())

        forecasts: dict[str, PlantForecast] = {}
        actuals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for plant in self.plants.values():
            if plant.model is None:
                # No model yet (first start, or training failed): retry once the sensor is usable.
                retry_due = plant.retry_at is None or now >= plant.retry_at
                if retry_due and not plant.lock.locked() and self._entities_usable(plant):
                    self._start_training(plant)
                continue
            try:
                times, wh = await async_production(
                    self.hass, plant.entity_ids, now - timedelta(hours=2), now, "5minute"
                )
            except EntityProblem as err:
                self._entity_problem(plant, err)
                continue
            plant.problem_since = None
            if plant.error and "training" not in plant.error:
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
                self._day_factor(plant, midnight),
                plant.band,
            )
        if not forecasts:
            return {}
        forecasts[TOTAL] = total_forecast(list(forecasts.values()), now_ts)
        actuals[TOTAL] = _sum_actuals(list(actuals.values()))
        daily = self._daily_windows(now)
        self._freeze_morning = should_freeze(dt_util.as_local(now), self._morning_day)
        await self.hass.async_add_executor_job(
            self._archive_cycle, forecasts, actuals, now_ts, daily, midnight
        )
        if daily is not None:
            self._morning_day = daily[3]
        return forecasts

    def _entities_usable(self, plant: PlantRuntime) -> bool:
        try:
            for entity_id in plant.entity_ids:
                entity_kind(self.hass, entity_id)
        except EntityProblem:
            return False
        return True

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

    def _day_factor(self, plant: PlantRuntime, midnight: int) -> DayFactor | None:
        """Today's production so far against what the last forecast expected for those quarters."""
        previous = (self.data or {}).get(plant.plant_id)
        times, wh = self.actual_today.get(plant.plant_id, (np.array([]), np.array([])))
        if previous is None or plant.model is None:
            return None
        known = (times >= midnight) & np.isin(times, previous.quarter_times)
        if not known.any():
            return None
        done = np.isin(previous.quarter_times, times[known])
        forecast_done = float(previous.quarter_raw[done, 1].sum())
        tomorrow = int(
            dt_util.start_of_local_day(
                dt_util.as_local(dt_util.utc_from_timestamp(midnight)).date() + timedelta(days=1)
            ).timestamp()
        )
        day_total = float(previous.raw_energy(midnight, tomorrow)[1])
        ratio = day_ratio(float(wh[known].sum()), forecast_done, plant.model.capacity)
        if ratio is None or day_total <= 0:
            return None
        return DayFactor(
            ratio, min(1.0, forecast_done / day_total), plant.day_weight, plant.day_tau
        )

    @property
    def variants(self) -> tuple[Variant, ...]:
        """Variants trained each week: all of them, or only the standard when switched off."""
        return VARIANTS if self.config_entry.options.get(CONF_VARIANTS, True) else (STANDARD,)

    def _fit_band(self, key: str, capacity: float) -> tuple[float, float]:
        """Band scale from the finished days' morning curves against what was produced."""
        today = dt_util.now().date()
        first = (today - timedelta(days=BAND_WINDOW_DAYS)).isoformat()
        parts = []
        for day, curve, applied in self.archive.curves_since(key, first):
            if day >= today.isoformat() or not curve:
                continue
            times, wh = self.archive.actuals(key, int(curve[0][0]), int(curve[-1][0]) + 900)
            parts.append(hourly_rows(curve, applied, times, wh))
        if not parts:
            return NEUTRAL
        return fit_band(*(np.concatenate([p[i] for p in parts]) for i in range(4)), capacity)

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
        midnight: int,
    ) -> None:
        hourly = self._weather[0]
        start = now - now % 900 + (900 if now % 900 else 0)
        rows = []
        for key, fc in forecasts.items():
            capacity = self.capacity(key)
            windows = [(h, start, start + length) for h, length in SHORT_HORIZONS.items()]
            if daily is not None:
                today, tomorrow, after, day = daily
                windows.append(("day_ahead", tomorrow, after))
                if self._freeze_morning:
                    windows.append(("today_morning", today, tomorrow))
                    mask = (fc.quarter_times >= today) & (fc.quarter_times < tomorrow)
                    curve = [
                        [int(t), *map(float, v)]
                        for t, v in zip(fc.quarter_times[mask], fc.quarter[mask], strict=True)
                    ]
                    band = self.plants[key].band if key in self.plants else NEUTRAL
                    self.archive.save_curve(key, day, curve, band)
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
                        float(fc.model_energy(w_start, w_end)[1]),
                        fc.ratio,
                        window_cloud(hourly, w_start, w_end),
                        capacity,
                        fc.day_ratio,
                        fc.day_share,
                    )
                )
        self.archive.add_forecasts(rows)
        for key, (times, wh) in actuals.items():
            self.archive.add_actuals(key, times, wh)
        for key in forecasts:
            self.actual_today[key] = self.archive.actuals(key, midnight, now + 900)

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
                _LOGGER.info(
                    "Training %s: loading weather %s to %s (first run takes several minutes)",
                    plant.name,
                    first,
                    last,
                )
                weather = await self.client.history_leads(first, max(first, last), now.date())
                _LOGGER.info("Training %s: fitting models", plant.name)
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
                    self.variants,
                    self._stop_training.is_set,
                )
            except Cancelled:
                return
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
            plant.variant_scores = result.variant_scores
            plant.error, plant.last_trained, plant.retry_at = None, now_ts, None
            _LOGGER.info(
                "Training %s done in %.0f s: %s model, %d hours, %s",
                plant.name,
                (dt_util.utcnow() - now).total_seconds(),
                result.model.kind,
                result.rows,
                "now active" if result.promoted else "previous model kept",
            )
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
                plant.day_weight, plant.day_tau = await self.hass.async_add_executor_job(
                    day_from_rows, rows, plant.tau
                )
                if plant.model is not None:
                    plant.band = await self.hass.async_add_executor_job(
                        self._fit_band, key, plant.model.capacity
                    )
        await self.async_update_review()
        cutoff = now - RETENTION
        await self.hass.async_add_executor_job(
            self.archive.prune,
            int(cutoff.timestamp()),
            dt_util.as_local(cutoff).date().isoformat(),
            now_ts - SHORT_RETENTION_DAYS * 86400,
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
