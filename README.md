# Machine-Learning-Based PV Forecast

[![CI](https://github.com/dominikamann/ha-machine-learning-based-pv-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/dominikamann/ha-machine-learning-based-pv-forecast/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**A PV forecast for Home Assistant that learns from your own history: no tilt, no azimuth, no kWp,
no API key.**

You pick the sensors your PV plants already have. The integration reads their history from the Home
Assistant recorder, combines it with weather data from [Open-Meteo](https://open-meteo.com) and
trains one model per plant. Panels in several directions on one inverter, shading by trees or the
neighbour's roof, a balcony plant: the model learns what your plant actually does instead of what a
datasheet says it should do.

## Features

- Forecasts for the **next 30 minutes, next hour, next 6 hours, today, today remaining, tomorrow and
  the day after tomorrow**, plus a **72-hour hourly curve**
- Energy forecasts as **P10 / P50 / P90** (pessimistic, expected, optimistic)
- **Per plant and in total**
- **Short-term correction from live production**: if a cloud passes right now, the next minutes
  follow it
- **Learns from its own mistakes**: every issued forecast is archived and compared with what really
  happened; a correction layer and the weekly retraining use those errors, and a new model only
  replaces the old one when it is measurably better
- Forecast in the **Energy dashboard**
- Endpoint for **[evcc](https://evcc.io)**
- Accuracy sensors per horizon, so you can watch the model improve

## How it works

- **History**: hourly long-term statistics from the recorder (years of data) plus 5-minute
  statistics for the last hours.
- **Weather**: Open-Meteo's archive of past forecasts (from 2022; ERA5 reanalysis before that,
  weighted lower) for training, the current forecast for predictions. Training on past *forecasts*
  means the model learns from the same kind of input it gets later.
- **No geometry**: internally the irradiance on seven fixed virtual planes (flat, 30° east to west,
  60° south) is computed. They are model inputs, identical for every installation; the model learns
  which mix fits your plant.
- **Model**: gradient boosting (scikit-learn) per plant with quantile outputs; recent data counts
  more than old data. Curtailment, outages and night hours are removed before training.
- **Every night**: accuracy per horizon, correction layer, short-term decay; **every Sunday**:
  retraining with a promotion gate.

Open-Meteo's free API is for non-commercial use, which covers a private home. No API key is needed.

## Installation

1. HACS → ⋮ → *Custom repositories* → add
   `https://github.com/dominikamann/ha-machine-learning-based-pv-forecast`, category *Integration*.
2. Install **Machine-Learning-Based PV Forecast** and restart Home Assistant.
3. *Settings → Devices & services → Add integration →* **Machine-Learning-Based PV Forecast**.
4. On the integration, choose **Add plant**, give it a name and select its production sensors.
   Repeat for every plant.

The location is taken from Home Assistant. The first model is trained right after adding a plant.

## Requirements

- Production sensors with long-term statistics:
  - power sensors (W, kW) with `state_class: measurement`, or
  - energy sensors (Wh, kWh) with `state_class: total` or `total_increasing`.
- The more history, the better. With less than 30 days of usable data a simple fallback model is
  used and the status sensor shows *learning*.
- Optional per plant: the export or inverter limit in W. Without it, curtailment is detected
  automatically.

## Sensors

Per plant and for the total:

| Sensor | Unit | Attributes |
|---|---|---|
| Next 30 minutes, next hour, next 6 hours | kWh | `p10`, `p90` |
| Today, today remaining, tomorrow, day after tomorrow | kWh | `p10`, `p90` |
| Peak power today, peak power tomorrow | W | `time` |
| Forecast curve (state: today in kWh) | kWh | `hourly` (72 h, Wh per hour), `quarters` (W), `morning` (this morning's forecast for today, W) |
| Accuracy 30 minutes / 1 hour / 6 hours / today (morning forecast) / day ahead | % | `mae_kwh`, `bias_kwh`, `skill`, `count` |
| Model status | learning / ready / error | `model`, `last_trained`, `rows`, `error` |

Accuracy is the mean absolute error over the last 30 days as a percentage of actual production
(lower is better). `skill` compares with simply repeating the previous period (above 0 is better
than that).

## evcc

The integration serves the forecast at `/api/ml_pv_forecast/evcc` in evcc's format (15-minute
slots, power in W). Create a long-lived access token in Home Assistant (your profile → *Security*)
and add to `evcc.yaml`:

```yaml
tariffs:
  solar:
    - type: custom
      tariff: solar
      forecast:
        source: http
        uri: http://homeassistant.local:8123/api/ml_pv_forecast/evcc
        headers:
          - Authorization: Bearer YOUR_LONG_LIVED_ACCESS_TOKEN
```

Optional query parameters: `?plant=<plant id>` for a single plant and `?quantile=p10|p50|p90`
(default `p50`; `p10` plans conservatively). The endpoint can be switched off in the integration
options.

## Dashboards

[`docs/dashboards/`](docs/dashboards) contains templates for
[apexcharts-card](https://github.com/RomRider/apexcharts-card) (install it from HACS):

- `today.yaml` – actual production against this morning's forecast and the live forecast
- `72h.yaml` – the next 72 hours per plant, stacked
- `horizons.yaml` – the short-term forecasts
- `accuracy.yaml` – forecast error over the weeks

Entity ids depend on your plant names and your Home Assistant language; adjust them after pasting.

## Backtest

To see how well the model works on your data before relying on it:

```bash
uv sync
uv run python tools/backtest.py --url http://homeassistant.local:8123 \
  --token "$HA_TOKEN" --entity sensor.your_pv_power
```

It trains on everything except the last 12 months and reports the daily error of the model and of
a physics baseline over those 12 months.

## Troubleshooting

Problems appear under *Settings → Repairs*:

- **Open-Meteo unreachable**: no new weather data for more than 3 hours. The last forecast is used
  while it lasts.
- **Sensor problem**: a production sensor is missing or has the wrong unit. Only that plant stops.
- **Training failed**: the previous model stays active.

*Download diagnostics* on the integration gives model status and accuracy figures (location
redacted) for bug reports.

## License

MIT, see [LICENSE](LICENSE).
