import numpy as np
import pytest

from tools.backtest import evaluate, local_days, markdown

HOUR = 3600


def test_local_days_follow_the_time_zone():
    # 23:30 UTC on 2024-06-30 is already July 1st in Berlin (UTC+2)
    t = np.array([1719790200])
    assert local_days(t, "UTC")[0] + 1 == local_days(t, "Europe/Berlin")[0]


def test_evaluate_scores_against_physics_and_persistence():
    # Ten days of 10 h at 100 Wh: 1000 Wh a day; day 4 is dull (500 Wh).
    days = np.repeat(np.arange(10), 10)
    actual = np.full(100, 100.0)
    actual[days == 4] = 50.0
    model = np.column_stack([actual * 0.8, actual * 1.1, actual * 1.2])  # 10 % too high
    physics = np.column_stack([actual, actual * 1.3, actual * 1.6])  # 30 % too high
    r = evaluate(days[20:], actual[20:], model[20:], physics[20:], days, actual, lead=0)
    assert r["days"] == 8
    actual_d = np.array([1000, 1000, 500, 1000, 1000, 1000, 1000, 1000])
    assert r["nmae_day"] == pytest.approx(10.0)
    assert r["bias_day"] == pytest.approx(10.0)
    assert r["skill_physics"] == pytest.approx(1 - 0.1 / 0.3)
    # persistence for day d is day d - 1: wrong by 500 Wh on day 4 and on day 5
    persistence_mae = 1000 / 8
    assert r["skill_persistence"] == pytest.approx(1 - np.mean(0.1 * actual_d) / persistence_mae)
    # P10 is always below, P50 and P90 always above the actual value
    assert (r["below_p10"], r["below_p50"], r["below_p90"]) == (0.0, 100.0, 100.0)
    assert r["score_hour"] < r["score_hour_physics"]
    assert r["days_off"] == 0
    table = markdown({"same day": r})
    assert "| Skill vs. physics baseline | +0.67 |" in table


def test_days_the_plant_was_off_are_left_out():
    days = np.repeat(np.arange(6), 10)
    actual = np.repeat([100.0, 80.0, 120.0, 0.0, 90.0, 110.0], 10)  # day 3: switched off
    model = np.column_stack([actual * 0.9, actual * 1.1, actual * 1.2])
    physics = np.column_stack([np.full(60, 80.0), np.full(60, 100.0), np.full(60, 120.0)])
    r = evaluate(days[10:], actual[10:], model[10:], physics[10:], days, actual, lead=0)
    assert r["days_off"] == 1 and r["days"] == 4
    assert r["nmae_day"] == pytest.approx(10.0)  # day 3 does not count, nor its zero hours


def test_script_runs_as_documented():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, "tools/backtest.py", "--help"], cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
