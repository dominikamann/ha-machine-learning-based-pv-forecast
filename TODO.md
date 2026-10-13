# TODO

Open points from the code review of 2026-10-13. A change to the model only goes in when the
backtest (`tools/backtest.py`) or the live sensors show that it helps.

## Next: the P90 of the model is too low

In the backtest over a year (README, *Evaluation*) the actual value lay below the model's P90 in
only about 65 % of the hours instead of 90 %; P10 is about right. The live band calibration
(0.8.0) widens the band from the morning forecasts but only by half of what was off, and the
archive of the first days showed the actual value above P90 in about half of the short-horizon
windows.

- [ ] Calibrate the quantiles at training time from the holdout (conformalised quantile
      regression: widen P10/P90 by the holdout's coverage error), then check in the backtest that
      the hours below P90 move towards 90 % without the quantile score getting worse. Afterwards
      decide whether the live band calibration still needs its 50 % shrink.

## Check with real data – about four weeks after installing 0.9.0

The update retrains every plant, so judge the weeks after it. Look at the attributes of the
*Forecast error* sensors per horizon (or the diagnostics download): `below_p10_pct`,
`above_p90_pct` (about 10 % each is right), `score_pct` and `bias_kwh`.

Baseline before the update (archive of 2026-10-11 to 2026-10-13, only 2–3 days, old version):
30 min to 6 h ahead the forecast was 13–32 % too low, and the actual value lay above P90 in
43–63 % of the windows (below P10: 2–6 %).

- [ ] **Low bias of the short horizons.** Compare `bias_kwh` and `above_p90_pct` of the 30 min /
      1 h / 6 h sensors with the baseline above. Still clearly too low → look into the live ratio
      and the correction layer first, then the band.
- [ ] **Band for the short horizons.** The band scale is learned from the morning forecasts but
      also applied to the 30 min / 1 h / 6 h quarters, which the live production already pulls
      close to the actual value. Clearly below 10 % on both sides at 30 min and 1 h means the band
      is too wide there → calibrate it per horizon from the archived short-horizon rows.
- [ ] **Lead time.** Training sees "same-day" weather from the first hours of each weather-model
      run, while the forecast treats everything up to 24 h ahead as same-day. If *day ahead*
      misses its band much more often than *today (morning forecast)*, try another mapping of
      hours ahead to `lead_days` (e.g. rounding instead of floor) in the backtest.
- [ ] **Visibility in Weather Plus.** Visibility is missing (NaN) in the training weather for one
      and two days ahead but present when forecasting, and the GBM always sends missing values to
      the highest bin (scikit-learn learns the direction). Only matters for plants running a
      Weather Plus variant: if their day-ahead error is clearly worse than same-day, hide
      visibility for lead ≥ 1 when forecasting, or let the GBM learn the missing direction.
- [ ] **Band of the total.** The total's P10/P90 is the sum of the plants' P10/P90 (assumes all
      plants miss together). Compare the total's `below_p10_pct` / `above_p90_pct` with the
      plants'.

## Later

- [ ] Training speed: subtracting histograms saves about 40 % of the training time but changes
      P10/P90 slightly through rounding; not used, because models must stay bit-identical.
      Revisit only together with a backtest that shows no loss.
- [ ] Quarter-hour split: by GHI. The plant's own plane mix was slightly worse on two plants in
      October (see `core/downscale.py`); test again with a summer of archived quarter actuals.
