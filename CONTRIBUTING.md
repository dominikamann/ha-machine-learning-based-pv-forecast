# Contributing

## Setup

```bash
uv sync
```

## Checks

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest -q
```

## Layout

- `custom_components/ml_pv_forecast/core/` – the forecasting core (weather, features, models,
  archive, evaluation). It must not import Home Assistant, so it can be tested on synthetic data.
- `custom_components/ml_pv_forecast/` – the Home Assistant layer (recorder history, coordinator,
  config flow, sensors, Energy dashboard, evcc endpoint).
- `tests/core/` – tests of the core; `tests/ha/` – Home Assistant tests; `tests/synthetic.py` –
  synthetic weather and plants with known properties.

Heavy work (training, prediction, sqlite, file access) always runs in the executor.
