"""Reads plant production from recorder statistics."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.statistics import statistics_during_period
from homeassistant.const import ATTR_UNIT_OF_MEASUREMENT, UnitOfEnergy, UnitOfPower
from homeassistant.core import HomeAssistant

POWER_UNITS = {unit.value for unit in UnitOfPower}
ENERGY_UNITS = {unit.value for unit in UnitOfEnergy}
CHUNK = timedelta(days=90)
STEP = {"hour": 3600, "5minute": 300}


class EntityProblem(Exception):
    """A configured production entity cannot be used."""

    def __init__(self, entity_id: str, reason: str) -> None:
        super().__init__(f"{entity_id}: {reason}")
        self.entity_id = entity_id
        self.reason = reason


def entity_kind(hass: HomeAssistant, entity_id: str) -> str:
    state = hass.states.get(entity_id)
    if state is None:
        raise EntityProblem(entity_id, "missing")
    unit = state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)
    state_class = state.attributes.get("state_class")
    if unit in POWER_UNITS:
        if state_class != "measurement":
            raise EntityProblem(entity_id, "no_statistics")
        return "power"
    if unit in ENERGY_UNITS:
        if state_class not in ("total", "total_increasing"):
            raise EntityProblem(entity_id, "no_statistics")
        return "energy"
    raise EntityProblem(entity_id, "unsupported_unit")


def validate_entities(hass: HomeAssistant, entity_ids: list[str]) -> str | None:
    if not entity_ids:
        return "no_entities"
    for entity_id in entity_ids:
        try:
            entity_kind(hass, entity_id)
        except EntityProblem as err:
            return err.reason
    return None


async def async_production(
    hass: HomeAssistant, entity_ids: list[str], start: datetime, end: datetime, period: str
) -> tuple[np.ndarray, np.ndarray]:
    """Wh per period summed over the entities, keyed by period start (unix seconds)."""
    step = STEP[period]
    per_entity = []
    for entity_id in entity_ids:
        kind = entity_kind(hass, entity_id)
        rows = await _fetch(hass, entity_id, kind, start, end, period)
        per_entity.append(_to_wh(rows, kind, step))
    return _combine(per_entity)


async def _fetch(
    hass: HomeAssistant, entity_id: str, kind: str, start: datetime, end: datetime, period: str
) -> list:
    units = {"power": UnitOfPower.WATT} if kind == "power" else {"energy": UnitOfEnergy.WATT_HOUR}
    types = {"mean"} if kind == "power" else {"change"}
    rows: list = []
    current = start
    while current < end:
        upper = min(current + CHUNK, end)
        result = await get_instance(hass).async_add_executor_job(
            statistics_during_period, hass, current, upper, {entity_id}, period, units, types
        )
        rows.extend(result.get(entity_id, []))
        current = upper
    return rows


def _to_wh(rows: list, kind: str, step: int) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in rows:
        value = row.get("mean") if kind == "power" else row.get("change")
        if value is None:
            continue
        wh = value * step / 3600 if kind == "power" else value
        out[int(row["start"])] = max(float(wh), 0.0)  # counter resets / standby draw
    return out


def _combine(per_entity: list[dict[int, float]]) -> tuple[np.ndarray, np.ndarray]:
    all_times = sorted(set().union(*per_entity)) if per_entity else []
    times = np.array(all_times, dtype=np.int64)
    total = np.zeros(len(times))
    for series in per_entity:
        if not series:
            continue
        values = np.array([series.get(t, np.nan) for t in all_times])
        values[times < min(series)] = 0.0  # entity added later: no production before it
        total += values
    return times, total


def quarter_sums(times: np.ndarray, wh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum 5-minute values into complete quarter hours."""
    if len(times) == 0:
        return times, wh
    quarter = times - times % 900
    unique, inverse, counts = np.unique(quarter, return_inverse=True, return_counts=True)
    sums = np.zeros(len(unique))
    np.add.at(sums, inverse, wh)
    keep = counts == 3
    return unique[keep], sums[keep]
