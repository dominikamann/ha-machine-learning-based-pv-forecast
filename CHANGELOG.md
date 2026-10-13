# Changelog

## 0.6.0 – 2026-10-13

- Card: new **Today** view – today's chart with the production so far, the expected total (so far
  plus the live forecast for the rest of the day), the morning forecast and the hit rate and hours
  in range for the hours finished so far. Cards set to `view: today` show it again.
- Card: `views:` chooses which tabs are shown and in which order, e.g. `views: [today, review]`.
  With a single view the tab bar is hidden.
- Forecast curve sensor: new `today` attribute (actual so far, morning forecast, hit rate, hours
  in range, scored until the last finished hour).

## 0.5.4 – 2026-10-12

- Card: removed the "Rest of today" view; cards set to `view: today` show the overview.
- Card: weather symbols only for hours with sun. A clear night sky has the same weather code as a
  sunny hour and showed up as a sun in the evening.

## 0.5.3 – 2026-10-12

- Card, Plants view: today's actual production per plant as light lines over the forecast areas,
  and the actual energy so far next to today's forecast in the table.
- Card: weather symbols sit above the top grid line instead of on it.

## 0.5.2 – 2026-10-12

- Card on narrow screens: weather symbols spaced by the available width, weekday names instead of
  "Tomorrow" / "Day after tomorrow", tab names fit.
- README: reload the dashboard after an update.

## 0.5.1 – 2026-10-12

- Card: the Plants view now really stacks the plants (each band sits on the plants before it;
  the tooltip shows each plant's own power).
- Card: weather symbols one row below the day labels in continuous charts, so they no longer
  overlap.
- README screenshots with weather symbols and the Plants view.

## 0.5.0 – 2026-10-12

- Card: the weather in a subtle way – prevailing weather and highest temperature next to each day,
  small monochrome weather symbols (sun, clouds, rain, snow, fog, thunder) along the top of the
  charts. The forecast curve's hourly rows carry the WMO weather code (`wc`) and temperature (`t`).
- Snow: sunny hours with snow on the ground are no longer discarded as outages, so the model learns
  how snow on the panels lowers production.
- README images load in HACS too (absolute URLs).
- Card: new **Plants** view – the next days per plant, stacked, with a table of daily energy per
  plant and in total. The forecast curve names its plant (`plant` attribute).
- README: evcc example with `interval: 15m`, so evcc picks up the 15-minute updates.
- Card: charts always fit the card width, also when a dashboard column becomes narrower after
  loading (phones).

## 0.4.2 – 2026-10-12

- Fix: the card sometimes showed "Custom element doesn't exist: ml-pv-forecast-card". It loaded
  before Home Assistant had set up its element registry and was lost; it now registers once Home
  Assistant is ready.

## 0.4.1 – 2026-10-12

- Card: days of the overview stacked vertically (readable on phones), a full-width tab bar with
  short names, `layout: timeline` for all days in one continuous chart, compact day headings,
  sun hours detected relative to the day's peak.
- README with screenshots.

## 0.4.0 – 2026-10-12

- Card with four views: **Overview**, **Day by day** (rotates through today, tomorrow and the day
  after every `interval` seconds), **Rest of today** (what is still to come, in 15-minute
  steps, with peak hour and end of production) and **Review** (step back through the last 7 days:
  morning forecast against actual production, hit rate, hours in range). `view:` and `tabs:` to
  use single views as separate cards. No title unless `title:` is set.
- New sensors per plant and in total: **Hit rate yesterday** and **Hit rate 7 days**.
- Overview shows one chart per day covering only the hours with sun (nights are left out), all
  on one kW scale, with the day's totals above each chart; no nested card frame; own legend.
- Fix: forecast curve attributes contained numpy numbers, which broke templates.

## 0.3.3 – 2026-10-12

- Card: blue forecast, green actual, violet morning forecast; P10/P90 as one legend entry;
  day separators and the *now* line without misplaced label boxes.
- Actual production of the last 10 days is taken from Home Assistant's 5-minute statistics at
  start-up, so charts and *Production today* are complete right after installing.
- The morning forecast is only frozen before 10:00; after a later start that day has none instead
  of a misleading afternoon forecast.

## 0.3.2 – 2026-10-12

- *Production now* is available right after a restart for plants measured by an energy meter
  (recent readings are taken from the recorder) instead of 1–2 minutes later.
- New action `ml_pv_forecast.retrain` to train one or all plants right away.

## 0.3.1 – 2026-10-12

- New sensors per plant and in total: **Production now** (W, live sum of the source sensors;
  derived from the meter for energy sensors) and **Production today** (kWh). Both have long-term
  statistics, so actual and forecast can be charted side by side without template sensors.
- Status sensor shows the history range as unknown (instead of 0 days) for models trained before
  0.3.0, until their next training.

## 0.3.0 – 2026-10-12

- Dashboard card `custom:ml-pv-forecast-card` (based on apexcharts-card): today's actual
  production, this morning's and the live forecast, P10–P90 range and daily totals for up to
  three days. The forecast curve sensor now carries today's actual production and the daily totals.
- Clearer plant setup: choose power or energy first, only matching sensors are offered, hints for
  hybrid inverters and smart plugs, confirmation for power sensors that go negative.
- First training starts once Home Assistant has started; plants whose sensor was missing are
  retrained as soon as it appears (instead of the next night).
- Progress of downloads and training in the log.
- Weather cache on fixed 90-day boundaries: shared by all plants and every retraining (one more
  download after the update).
- Status sensor shows the covered history (`history_from`, `history_days`, `full_year`).

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
