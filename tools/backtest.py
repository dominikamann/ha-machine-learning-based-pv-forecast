"""Backtest the forecast model on a running Home Assistant.

Trains once on everything before the last 12 months (all model variants, as the integration does)
and scores the model over those 12 months for each forecast lead (same day, one and two days
ahead) against two references: the physics baseline (the integration's fallback model) and
persistence (the last complete day happens again). Days are local days of the Home Assistant.

    uv run python tools/backtest.py --url http://homeassistant.local:8123 \
        --token "$HA_TOKEN" --entity sensor.solaredge_ac_power [--markdown]

Only the model is scored: the live correction, the short-term adjustment and the band calibration
of the running integration come on top.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout

from custom_components.ml_pv_forecast.core.features import build_features  # noqa: E402
from custom_components.ml_pv_forecast.core.model import (  # noqa: E402
    PlantModel,
    fit_fallback,
    quantile_score,
)
from custom_components.ml_pv_forecast.core.pipeline import train_plant  # noqa: E402
from custom_components.ml_pv_forecast.core.weather import OpenMeteoClient  # noqa: E402

DAY = 86400
LEAD_NAMES = ("same day", "1 day ahead", "2 days ahead")
OFF_SHARE = 0.1  # a day below this share of the physics forecast counts as switched off


def local_days(times: np.ndarray, time_zone: str) -> np.ndarray:
    """Local calendar day (days since 1970-01-01) of each unix time."""
    zone = ZoneInfo(time_zone)
    return (
        np.array(
            [datetime.fromtimestamp(int(t), zone).date().toordinal() for t in times], dtype=np.int64
        )
        - date(1970, 1, 1).toordinal()
    )


def evaluate(
    days: np.ndarray,
    actual: np.ndarray,
    model: np.ndarray,
    physics: np.ndarray,
    all_days: np.ndarray,
    all_actual: np.ndarray,
    lead: int,
) -> dict[str, float]:
    """Scores of one lead over the test hours.

    ``model`` and ``physics`` hold P10/P50/P90 per hour (Wh). ``all_days``/``all_actual`` cover
    the whole history, so persistence can look back before the first test day: the forecast for
    a day issued in its morning knows the actual energy up to the day before, i.e. for lead ``n``
    the day ``n + 1`` days before the target day.

    Days on which the plant gave less than a tenth of what the physics baseline expected (switched
    off, broken, under snow) are left out: they say nothing about the forecast. So are days whose
    persistence day is not known.
    """
    unique, inverse = np.unique(days, return_inverse=True)

    def daily(values: np.ndarray) -> np.ndarray:
        out = np.zeros(len(unique))
        np.add.at(out, inverse, values)
        return out

    known = np.zeros(int(all_days.max() - all_days.min()) + 1)
    np.add.at(known, all_days - all_days.min(), all_actual)
    back = unique - (lead + 1) - all_days.min()
    physics_all = daily(physics[:, 1])
    off = daily(actual) < OFF_SHARE * physics_all
    usable = (back >= 0) & ~off
    actual_d = daily(actual)[usable]
    model_d, physics_d = daily(model[:, 1])[usable], daily(physics[:, 1])[usable]
    persistence_d = known[back[usable]]
    mae = float(np.mean(np.abs(model_d - actual_d)))
    daylight = ((actual > 0) | (model[:, 2] > 0)) & usable[inverse]
    a, m = actual[daylight], model[daylight]
    return {
        "days": int(usable.sum()),
        "days_off": int(off.sum()),
        "nmae_day": mae / float(actual_d.mean()) * 100,
        "skill_physics": 1 - mae / float(np.mean(np.abs(physics_d - actual_d))),
        "skill_persistence": 1 - mae / float(np.mean(np.abs(persistence_d - actual_d))),
        "bias_day": float((model_d.sum() - actual_d.sum()) / actual_d.sum() * 100),
        "score_hour": float(quantile_score(m, a).mean() / a.mean() * 100),
        "score_hour_physics": float(quantile_score(physics[daylight], a).mean() / a.mean() * 100),
        "below_p10": float(np.mean(a < m[:, 0]) * 100),
        "below_p50": float(np.mean(a < m[:, 1]) * 100),
        "below_p90": float(np.mean(a < m[:, 2]) * 100),
    }


def markdown(results: dict[str, dict[str, float]]) -> str:
    """The scores as the table used in the README."""
    rows = [
        ("Days scored", "days", "{:.0f}"),
        ("Days left out (plant off)", "days_off", "{:.0f}"),
        ("Daily energy error (nMAE)", "nmae_day", "{:.1f} %"),
        ("Daily energy bias", "bias_day", "{:+.1f} %"),
        ("Skill vs. physics baseline", "skill_physics", "{:+.2f}"),
        ("Skill vs. persistence", "skill_persistence", "{:+.2f}"),
        ("Hourly quantile score (model)", "score_hour", "{:.1f} %"),
        ("Hourly quantile score (physics)", "score_hour_physics", "{:.1f} %"),
        ("Hours below P10 (ideal 10 %)", "below_p10", "{:.1f} %"),
        ("Hours below P50 (ideal 50 %)", "below_p50", "{:.1f} %"),
        ("Hours below P90 (ideal 90 %)", "below_p90", "{:.1f} %"),
    ]
    head = "| | " + " | ".join(results) + " |\n|---|" + "---|" * len(results) + "\n"
    body = "".join(
        f"| {label} | " + " | ".join(fmt.format(r[key]) for r in results.values()) + " |\n"
        for label, key, fmt in rows
    )
    return head + body


async def _ws_statistics(url: str, token: str, entities: list[str], start: datetime, end: datetime):
    async with (
        aiohttp.ClientSession() as session,
        session.ws_connect(url.rstrip("/") + "/api/websocket") as ws,
    ):
        await ws.receive_json()
        await ws.send_json({"type": "auth", "access_token": token})
        if (await ws.receive_json()).get("type") != "auth_ok":
            raise SystemExit("authentication failed")
        await ws.send_json({"id": 1, "type": "get_config"})
        config = (await ws.receive_json())["result"]
        series: dict[int, float] = {}
        msg_id = 2
        current = start
        while current < end:
            upper = min(current + timedelta(days=90), end)
            await ws.send_json(
                {
                    "id": msg_id,
                    "type": "recorder/statistics_during_period",
                    "start_time": current.isoformat(),
                    "end_time": upper.isoformat(),
                    "statistic_ids": entities,
                    "period": "hour",
                    "units": {"power": "W", "energy": "Wh"},
                    "types": ["mean", "change"],
                }
            )
            result = (await ws.receive_json())["result"]
            for rows in result.values():
                for row in rows:
                    value = row.get("mean") if row.get("mean") is not None else row.get("change")
                    if value is not None:
                        t = int(row["start"] / 1000)  # websocket API uses milliseconds
                        series[t] = series.get(t, 0.0) + max(float(value), 0.0)
            msg_id += 1
            current = upper
        times = np.array(sorted(series), dtype=np.int64)
        return config, times, np.array([series[t] for t in times])


def score(
    times: np.ndarray,
    wh: np.ndarray,
    leads: list,
    latitude: float,
    longitude: float,
    cutoff: int,
    limit: float | None,
    time_zone: str,
) -> tuple[PlantModel, int, dict[str, dict[str, float]]]:
    """Train on the hours before ``cutoff``, score every lead on the hours after it."""
    train_mask = times < cutoff
    result = train_plant(
        times[train_mask], wh[train_mask], leads, latitude, longitude, cutoff, None, limit, ""
    )
    all_days = local_days(times, time_zone)
    results = {}
    for lead, (name, weather) in enumerate(zip(LEAD_NAMES, leads, strict=False)):
        common, pi, wi = np.intersect1d(times, weather.times, return_indices=True)
        rows = weather.take(wi)
        has_weather = np.isfinite(rows.ghi)
        X = build_features(rows.take(has_weather), latitude, longitude)
        t, y = common[has_weather], wh[pi][has_weather]
        test = t >= cutoff
        if test.sum() < 24 * 30:
            continue  # this lead's archive is too short for a year of testing
        baseline = fit_fallback(X[~test], y[~test], result.model.capacity, cutoff, limit)
        results[name] = evaluate(
            local_days(t[test], time_zone),
            y[test],
            result.model.predict(X[test]),
            baseline.predict(X[test]),
            all_days,
            wh,
            lead,
        )
    return result.model, result.rows, results


async def load(
    url: str, token: str, entities: list[str], cache: Path | None
) -> tuple[dict, np.ndarray, np.ndarray, list, int]:
    """Hourly production, the training weather per lead and the start of the test year."""
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    config, times, wh = await _ws_statistics(
        url, token, entities, end - timedelta(days=6 * 365), end
    )
    if len(times) == 0:
        raise SystemExit("no statistics found")
    async with aiohttp.ClientSession() as session:
        client = OpenMeteoClient(session, config["latitude"], config["longitude"], cache)
        leads = await client.history_leads(
            datetime.fromtimestamp(int(times[0]), UTC).date(),
            (end - timedelta(days=1)).date(),
            date.today(),
        )
    return config, times, wh, leads, int((end - timedelta(days=365)).timestamp())


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--entity", action="append", required=True)
    parser.add_argument("--limit", type=float, help="export or inverter limit of the plant in W")
    parser.add_argument("--markdown", action="store_true", help="print a table for the README")
    parser.add_argument("--cache", type=Path, help="folder to keep downloaded weather between runs")
    args = parser.parse_args()

    config, times, wh, leads, cutoff = await load(args.url, args.token, args.entity, args.cache)
    model, rows, results = score(
        times,
        wh,
        leads,
        config["latitude"],
        config["longitude"],
        cutoff,
        args.limit,
        config["time_zone"],
    )
    print(f"model: {model.kind}, variant {model.variant}, {rows} hours")
    if args.markdown:
        print(markdown(results))
        return
    for name, r in results.items():
        print(f"{name}: " + ", ".join(f"{k} {v:.3g}" for k, v in r.items()))


if __name__ == "__main__":
    asyncio.run(main())
