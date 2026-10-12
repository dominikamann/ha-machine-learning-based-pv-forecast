"""Backtest the forecast model on a running Home Assistant.

Trains on everything before the last 12 months and reports the daily-energy error of the model
and of the physics baseline over those 12 months.

    uv run python tools/backtest.py --url http://homeassistant.local:8123 \
        --token "$HA_TOKEN" --entity sensor.solaredge_ac_power
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, date, datetime, timedelta

import aiohttp
import numpy as np

from custom_components.ml_pv_forecast.core.features import build_features
from custom_components.ml_pv_forecast.core.model import fit_fallback
from custom_components.ml_pv_forecast.core.pipeline import train_plant
from custom_components.ml_pv_forecast.core.weather import OpenMeteoClient

DAY = 86400


def daily_errors(
    times: np.ndarray, actual_wh: np.ndarray, predicted_wh: np.ndarray, tz_offset_s: int = 0
) -> tuple[float, float]:
    """Mean absolute daily error in kWh and as percent of mean daily production."""
    days = (times + tz_offset_s) // DAY
    unique, inverse = np.unique(days, return_inverse=True)
    actual = np.zeros(len(unique))
    predicted = np.zeros(len(unique))
    np.add.at(actual, inverse, actual_wh)
    np.add.at(predicted, inverse, predicted_wh)
    mae = float(np.mean(np.abs(predicted - actual)))
    return round(mae / 1000, 3), round(mae / float(actual.mean()) * 100, 1)


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


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--entity", action="append", required=True)
    args = parser.parse_args()

    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    config, times, wh = await _ws_statistics(
        args.url, args.token, args.entity, end - timedelta(days=6 * 365), end
    )
    if len(times) == 0:
        raise SystemExit("no statistics found")
    lat, lon = config["latitude"], config["longitude"]
    async with aiohttp.ClientSession() as session:
        weather = await OpenMeteoClient(session, lat, lon).history(
            datetime.fromtimestamp(int(times[0]), UTC).date(),
            (end - timedelta(days=1)).date(),
            date.today(),
        )
    cutoff = int((end - timedelta(days=365)).timestamp())
    train_mask = times < cutoff
    result = train_plant(
        times[train_mask], wh[train_mask], weather, lat, lon, cutoff, None, None, ""
    )

    common, pi, wi = np.intersect1d(times, weather.times, return_indices=True)
    X = build_features(weather.take(wi), lat, lon)
    test = common >= cutoff
    baseline = fit_fallback(X[~test], wh[pi][~test], result.model.capacity, cutoff)
    model_err = daily_errors(common[test], wh[pi][test], result.model.predict(X[test])[:, 1])
    base_err = daily_errors(common[test], wh[pi][test], baseline.predict(X[test])[:, 1])
    print(f"model ({result.model.kind}): MAE {model_err[0]} kWh/day, nMAE {model_err[1]} %")
    print(f"physics baseline:   MAE {base_err[0]} kWh/day, nMAE {base_err[1]} %")


if __name__ == "__main__":
    asyncio.run(main())
