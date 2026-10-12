# Changelog

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
