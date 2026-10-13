# TODO

Open points from the code review of 2026-10-13. The medium points need real data before anything
changes: a change only goes in when the numbers show it helps.

## Check with real data – about four weeks after installing the update

The update retrains every plant (export-limit fix, quantile score), so judge the weeks after it.
Look at the attributes of the *Forecast error* sensors per horizon (or the diagnostics download):
`below_p10_pct`, `above_p90_pct` (about 10 % each is right) and `score_pct`.

Baseline before the update (archive of 2026-10-11 to 2026-10-13, only 2–3 days, old version):
30 min to 6 h ahead the forecast was 13–32 % too low, and the actual value lay above P90 in
43–63 % of the windows (below P10: 2–6 %). Too short to conclude anything, but check first whether
this low bias is gone after the update (the export-limit fix removes one known cause of it).

- [ ] **Low bias of the short horizons.** Compare the `bias_kwh` and `above_p90_pct` of the
      30 min / 1 h / 6 h sensors with the baseline above. Still clearly too low after a few weeks
      → look into the live ratio and the correction layer first, then the band.

- [ ] **Band for the short horizons (review point 7).** The band scale is learned from the
      morning forecasts but also applied to the 30 min / 1 h / 6 h quarters, which the live
      production already pulls close to the actual value.
      *Look at:* 30 min and 1 h. Clearly below 10 % on both sides means the band is too wide
      there → calibrate the band per horizon from the archived short-horizon rows (they hold
      P10/P90 and the actual value).
- [ ] **Lead time (review point 4).** Training sees "same-day" weather from the first hours of
      each weather-model run, while the forecast treats everything up to 24 h ahead as same-day.
      *Look at:* *today (morning forecast)* against *day ahead*. If the day ahead misses its band
      much more often (e.g. above 15 % on one side) while the morning forecast is fine, test
      another mapping of hours ahead to `lead_days` (e.g. rounding instead of floor) in a
      backtest before changing it.
- [ ] **Visibility in Weather Plus (review point 6).** Visibility is missing (NaN) in the
      training weather for one and two days ahead but present when forecasting.
      *Look at:* only matters for plants running a Weather Plus variant (*Model variant*
      sensor). If such a plant's day-ahead error or band is clearly worse than its same-day
      forecast, hide visibility for lead ≥ 1 when forecasting, as in training.
- [ ] **Band of the total (review point 7).** The total's P10/P90 is the sum of the plants'
      P10/P90, which assumes all plants miss together (same weather). *Look at:* the total's
      `below_p10_pct` / `above_p90_pct` against the plants'. Only relevant with several plants
      that differ a lot (e.g. east and west roofs).

## Smaller points (no data needed)

- [ ] CI: `uv sync --locked`; pin `home-assistant/actions/hassfest` and `hacs/action` to
      versions instead of `@master` / `@main`; add coverage and a type checker (pyright).
- [ ] Test suite takes about 12 minutes: mark the slow training tests (`@pytest.mark.slow`) and
      run them in a separate CI job.
- [ ] GBM: missing values always go right (into the highest bin); scikit-learn learns the
      direction. Affects `clearsky_index` at low sun and the visibility point above.
- [ ] GBM: `_Binner` stores bins as `uint8`; guard against `max_bins > 255`.
- [ ] Quarter-hour split uses GHI; plants facing east or west would be better served by the
      plant's own mix of the virtual planes (the NNLS weights).
- [ ] Temperature and cloud cover are instantaneous values at the end of the hour in Open-Meteo
      but are shifted to the start like the radiation averages.
- [ ] Models are stored with pickle; a versioned format (npz/JSON) would survive refactorings.
- [ ] Training speed: the histogram-subtraction trick saves about 40 % but changes P10/P90
      slightly (rounding); not used, because models must stay bit-identical. Revisit only
      together with a backtest showing no loss.
- [ ] README: a reproducible evaluation report (skill against the physics model and
      persistence, quantile score, band coverage per horizon) from `tools/backtest.py`.
