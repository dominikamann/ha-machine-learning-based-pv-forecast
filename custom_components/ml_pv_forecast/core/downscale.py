"""Distributes hourly forecasts over quarter hours."""

from __future__ import annotations

import numpy as np

HOUR = 3600


def to_quarters(
    hour_times: np.ndarray, hourly: np.ndarray, quarter_times: np.ndarray, quarter_ghi: np.ndarray
) -> np.ndarray:
    """Split each hourly value over its quarters in proportion to the quarter-hour GHI."""
    hour_start = quarter_times - quarter_times % HOUR
    pos = np.searchsorted(hour_times, hour_start)
    pos_c = np.minimum(pos, len(hour_times) - 1)
    valid = (pos < len(hour_times)) & (hour_times[pos_c] == hour_start)
    weight = np.where(valid, np.maximum(np.nan_to_num(quarter_ghi), 0.0), 0.0)
    sums = np.zeros(len(hour_times))
    counts = np.zeros(len(hour_times))
    np.add.at(sums, pos_c[valid], weight[valid])
    np.add.at(counts, pos_c[valid], 1)
    hour_sum = sums[pos_c]
    share = np.where(
        hour_sum > 0,
        weight / np.where(hour_sum > 0, hour_sum, 1.0),
        1.0 / np.maximum(counts[pos_c], 1),
    )
    out = hourly[pos_c] * share[:, None]
    out[~valid] = np.nan
    return out
