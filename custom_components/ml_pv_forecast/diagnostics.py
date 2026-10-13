"""Diagnostics download."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.core import HomeAssistant

from .const import TOTAL

TO_REDACT = {"latitude", "longitude"}


async def async_get_config_entry_diagnostics(hass: HomeAssistant, entry) -> dict[str, Any]:
    c = entry.runtime_data
    plants = {
        pid: {
            "name": p.name,
            "entities": p.entity_ids,
            "limit_w": p.limit_w,
            "status": p.status,
            "error": p.error,
            "rows": p.rows,
            "model": None
            if p.model is None
            else {
                "kind": p.model.kind,
                "capacity_wh": p.model.capacity,
                "trained_at": p.model.trained_at,
            },
            "tau": p.tau,
            "day_weight": p.day_weight,
            "day_tau": p.day_tau,
            "correction": p.correction is not None,
            "metrics": {h: asdict(m) for h, m in c.metrics.get(pid, {}).items()},
        }
        for pid, p in c.plants.items()
    }
    return async_redact_data(
        {
            "latitude": c.latitude,
            "longitude": c.longitude,
            "options": dict(entry.options),
            "weather_fetched_at": None
            if c.weather_fetched_at is None
            else c.weather_fetched_at.isoformat(),
            "plants": plants,
            "total_metrics": {h: asdict(m) for h, m in c.metrics.get(TOTAL, {}).items()},
        },
        TO_REDACT,
    )
