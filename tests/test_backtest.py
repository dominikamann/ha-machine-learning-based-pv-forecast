import numpy as np

from tools.backtest import daily_errors


def test_daily_errors():
    times = np.arange(48) * 3600
    actual = np.full(48, 100.0)
    predicted = np.full(48, 110.0)
    mae, nmae = daily_errors(times, actual, predicted)
    assert mae == 0.24 and nmae == 10.0


def test_script_runs_as_documented():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, "tools/backtest.py", "--help"], cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
