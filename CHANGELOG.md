# Changelog

## 0.2.1 – 2026-10-12

- Fix: models saved by 0.1.x were treated as current after the update and kept in use instead of
  being retrained with the new weather inputs.

## 0.2.0 – 2026-10-12

- Forecasts use five weather models (best match, DWD ICON, Météo-France, ECMWF, NOAA GFS) with
  their mean and spread as inputs.
- Training per lead time with Open-Meteo's previous model runs: tomorrow's daily error on two real
  plants dropped from 20.8 % / 24.8 % to 11.7 % / 13.6 %.
- Models from older versions are retrained automatically after the update.
- Waits and retries when Open-Meteo's per-minute request limit is reached.
- No repair issue for sensors that appear shortly after Home Assistant started; forecasts refresh
  as soon as Home Assistant has started.
- Backtest reports same-day and day-ahead errors.

## 0.1.2 – 2026-10-12

- Fix: a plant added while the integration was still starting up was ignored until the next
  restart.
- Accuracy sensors show *unknown* with `status: collecting` until the first nightly evaluation
  instead of *unavailable*.
- All eight brand images (icon and logo, light and dark, 1x and 2x).
- German name of the day-ahead accuracy sensor clarified.

## 0.1.1 – 2026-10-12

- Runs on Home Assistant OS: replaced scikit-learn (no wheels for musl/aarch64) with a built-in
  numpy gradient boosting; the only extra dependency is scipy. Saved models move from `.joblib`
  to `.pkl` and are retrained once.
- Integration icon (shown from Home Assistant 2026.3).
- Backtest script runs from a checkout as documented.
- README: how to choose the right production sensor.

## 0.1.0 – 2026-10-12

- First version: geometry-free per-plant PV forecast learned from recorder history and Open-Meteo,
  nowcasting, self-evaluation with correction layer, Energy dashboard, evcc endpoint.
