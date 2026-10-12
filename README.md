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
- **Several weather models**: besides Open-Meteo's best match, the forecasts of DWD ICON,
  Météo-France, ECMWF and NOAA GFS are model inputs, together with their mean and spread. When the
  models disagree, the P10–P90 band widens.
- **Trained per lead time**: every past hour is learned three times – with the forecast from the
  same day, from the day before and from two days before (Open-Meteo's archive of previous model
  runs, from 2024). Tomorrow's forecast is therefore trained on what forecasts for tomorrow really
  looked like, not on same-day forecasts.
- **No geometry**: internally the irradiance on seven fixed virtual planes (flat, 30° east to west,
  60° south) is computed. They are model inputs, identical for every installation; the model learns
  which mix fits your plant.
- **Model**: histogram gradient boosting per plant with quantile outputs (P10/P50/P90); recent
  data counts more than old data. Curtailment, outages and night hours are removed before training.
  The boosting is a small numpy implementation of the algorithm behind scikit-learn's
  `HistGradientBoostingRegressor` (tested against it), because scikit-learn cannot be installed on
  Home Assistant OS. The only extra dependency is scipy.
- **Every night**: accuracy per horizon, correction layer, short-term decay; **every Sunday**:
  retraining with a promotion gate.

Open-Meteo's free API is for non-commercial use, which covers a private home. No API key is needed.
The first training downloads several years of weather for five models; if Open-Meteo's per-minute
limit is reached, the integration waits a minute and continues (a few minutes in total, cached
afterwards).

## Installation

1. HACS → ⋮ → *Custom repositories* → add
   `https://github.com/dominikamann/ha-machine-learning-based-pv-forecast`, category *Integration*.
2. Install **Machine-Learning-Based PV Forecast** and restart Home Assistant.
3. *Settings → Devices & services → Add integration →* **Machine-Learning-Based PV Forecast**.
4. On the integration, choose **Add plant**: give it a name, choose how production is measured
   (power in W/kW or an energy meter in Wh/kWh), then pick the matching sensors. A power sensor
   that goes clearly negative (typical for a hybrid inverter's AC output) asks for confirmation.
   Repeat for every plant.
5. Add the **PV Forecast** card to a dashboard (see [Dashboard card](#dashboard-card)).

The location is taken from Home Assistant. The first model is trained as soon as Home Assistant
has started; the first run downloads several years of weather and takes a few minutes (progress
in the log). Weather is cached, later trainings only fetch the last days.

## Requirements

- Production sensors with long-term statistics:
  - power sensors (W, kW) with `state_class: measurement`, or
  - energy sensors (Wh, kWh) with `state_class: total` or `total_increasing`.
- History: all available long-term statistics are used (up to 6 years). **At least one year is
  recommended** so the model has seen every season; the status sensor shows the covered days and
  `full_year`. With less than 30 days a simple fallback model is used (status *learning*); with
  less than a year the forecast is weaker for seasons not seen yet and improves with the weekly
  retraining.
- Optional per plant: the export or inverter limit in W. Without it, curtailment is detected
  automatically.
- Home Assistant 2025.3 or newer (2026.3 or newer to show the integration's own icon).

### Choosing the right sensor

Use a sensor that measures what the **panels** produce:

- Hybrid inverters with a battery (e.g. SolarEdge StorEdge): the inverter's AC power includes
  battery discharge and can go negative while charging from the grid. Use the panel production
  instead (DC power of inverter and battery combined, or an energy sensor integrated from it).
- Smart plugs whose energy counter resets after a restart: prefer the power sensor.

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

Accuracy sensors stay *unknown* (attribute `status: collecting`) until the nightly evaluation at
02:30 has archived forecasts to compare: the short horizons after the first night, the day horizons
after one to two days. Accuracy is the mean absolute error over the last 30 days as a percentage of actual production
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

## Dashboard card

The integration ships its own card. It needs
[apexcharts-card](https://github.com/RomRider/apexcharts-card) (install it from HACS).

```yaml
type: custom:ml-pv-forecast-card
# optional:
# entity: sensor.pv_forecast_total_forecast_curve   # default: the total, found automatically
# days: 3                                            # 1–3
# title: PV
```

It shows hourly bars of today's actual production, this morning's forecast, the live forecast
with its P10–P90 range for today and the next days, a *now* marker and the daily totals.
It is also in the card picker as **PV Forecast**.

### More templates

[`docs/dashboards/`](docs/dashboards) contains further `apexcharts-card` templates:

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
a physics baseline over those 12 months. The model is trained once and not updated during those
12 months, so live results with weekly retraining and the correction layer tend to be better.

On two real plants the daily error over 12 months (days on which a plant was switched off
excluded) was:

| | Today | Tomorrow |
|---|---|---|
| Plant 1 | 9.3 % | 11.7 % |
| Plant 2 | 11.4 % | 13.6 % |

Before version 0.2.0 (one weather model, trained on same-day forecasts only) tomorrow's error was
20.8 % and 24.8 %.

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

---

<p align="center">Proudly provided by <a href="https://amannlabs.eu">amannlabs.eu</a></p>
