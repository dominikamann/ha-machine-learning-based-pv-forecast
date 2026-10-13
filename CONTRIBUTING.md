# Contributing

## Setup

```bash
uv sync
```

## Checks

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright
uv run pytest -q -m "not slow"   # about 6 minutes; the slow tests train models on months of data
uv run pytest -q                 # everything, about 14 minutes (CI runs both halves in parallel)
```

Changes to the model must not make it worse: compare with the backtest (`tools/backtest.py`, see
the README) before and after, on real plants where possible.

## Layout

- `custom_components/ml_pv_forecast/core/` – the forecasting core (weather, features, models,
  archive, evaluation). It must not import Home Assistant, so it can be tested on synthetic data.
- `core/gbm.py` – the numpy gradient boosting. Runtime dependencies must have wheels for Home
  Assistant OS (musl, aarch64 and amd64); scikit-learn and joblib do not, so they are dev-only
  (scikit-learn serves as the reference in `tests/core/test_gbm.py`).
- `custom_components/ml_pv_forecast/` – the Home Assistant layer (recorder history, coordinator,
  config flow, sensors, Energy dashboard, evcc endpoint).
- `tests/core/` – tests of the core; `tests/ha/` – Home Assistant tests; `tests/synthetic.py` –
  synthetic weather and plants with known properties.

Heavy work (training, prediction, sqlite, file access) always runs in the executor.

Trained models are stored as plain arrays in `.npz` files (`core/model.py`, `save_model`), not as
pickles: they load without running code and survive refactoring. Bump `FORMAT_VERSION` when the
layout changes and `FEATURE_VERSION` (`core/features.py`) when the model inputs change; stored
models of another version are retrained.
