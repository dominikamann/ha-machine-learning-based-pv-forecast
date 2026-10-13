import numpy as np

from custom_components.ml_pv_forecast.core.archive import Archive, ArchivedForecast


def fc(start, end, p50, horizon="1h", issued=0):
    return ArchivedForecast(
        "a", horizon, issued, start, end, p50 * 0.8, p50, p50 * 1.2, p50, p50, None, 20.0, 1000.0
    )


def test_evaluated_needs_complete_actuals_and_gives_persistence(tmp_path):
    ar = Archive(tmp_path / "a.db")
    ar.add_forecasts([fc(3600, 7200, 500.0, issued=3600), fc(7200, 10800, 400.0, issued=7200)])
    ar.add_actuals("a", np.arange(0, 7200, 900), np.full(8, 100.0))
    rows = ar.evaluated("a", "1h", 0, 20000)
    assert len(rows) == 1  # second window has no actuals
    assert rows[0].actual == 400.0 and rows[0].persistence == 400.0
    ar.close()


def test_curves_and_prune(tmp_path):
    ar = Archive(tmp_path / "a.db")
    ar.save_curve("a", "2026-10-12", [[0, 1, 2, 3]])
    assert ar.load_curve("a", "2026-10-12") == [[0, 1, 2, 3]]
    ar.add_forecasts([fc(0, 3600, 1.0)])
    ar.prune(10**9, "2026-10-13")
    assert ar.evaluated("a", None, 0, 10**9) == []
    assert ar.load_curve("a", "2026-10-12") is None
    ar.close()


def test_corrupt_database_is_replaced(tmp_path):
    path = tmp_path / "a.db"
    path.write_bytes(b"not a database" * 100)
    ar = Archive(path)
    assert ar.recovered
    assert list(tmp_path.glob("a.db.corrupt-*"))
    ar.close()


def test_non_finite_forecasts_are_skipped(tmp_path):
    ar = Archive(tmp_path / "a.db")
    ar.add_forecasts([fc(0, 3600, float("nan")), fc(3600, 7200, 5.0, issued=1)])
    ar.add_actuals("a", np.arange(0, 7200, 900), np.ones(8))
    assert len(ar.evaluated("a", None, 0, 10**9)) == 1
    ar.close()


def test_outdated_schema_is_replaced(tmp_path):
    import sqlite3

    path = tmp_path / "a.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE forecasts (plant TEXT, horizon TEXT)")
    old.commit()
    old.close()
    ar = Archive(path)
    assert ar.recovered
    ar.add_forecasts([fc(0, 3600, 1.0)])
    ar.close()


def test_closed_archive_ignores_late_calls(tmp_path):
    ar = Archive(tmp_path / "a.db")
    ar.close()
    ar.add_forecasts([fc(0, 3600, 1.0)])
    ar.add_actuals("a", np.arange(0, 3600, 900), np.ones(4))
    ar.prune(0, "2000-01-01")
    assert ar.evaluated("a", None, 0, 10**9) == []
    assert ar.load_curve("a", "2026-10-12") is None


def test_short_horizons_can_be_pruned_earlier(tmp_path):
    ar = Archive(tmp_path / "a.db")
    ar.add_forecasts([fc(0, 3600, 1.0, horizon="1h"), fc(0, 3600, 1.0, horizon="day_ahead")])
    ar.add_actuals("a", np.arange(0, 3600, 900), np.ones(4))
    ar.prune(0, "2000-01-01", short_before=10**6)
    assert [r.forecast.horizon for r in ar.evaluated("a", None, 0, 10**9)] == ["day_ahead"]
    ar.close()


def test_evaluating_a_year_of_rows_is_fast(tmp_path):
    import time

    ar = Archive(tmp_path / "a.db")
    n = 35040  # one year of quarter-hour issuances
    ar.add_forecasts([fc(900 * i, 900 * i + 3600, 1.0, issued=900 * i) for i in range(n)])
    ar.add_actuals("a", np.arange(0, 900 * (n + 4), 900), np.ones(n + 4))
    t0 = time.perf_counter()
    rows = ar.evaluated("a", None, 0, 10**9)
    assert len(rows) == n
    # Guards against query blow-ups (tens of seconds), not machine speed: CI runners are slower.
    assert time.perf_counter() - t0 < 5.0
    ar.close()


def test_actuals_between(tmp_path):
    ar = Archive(tmp_path / "a.db")
    ar.add_actuals("a", np.array([900, 0, 1800]), np.array([2.0, 1.0, 3.0]))
    times, wh = ar.actuals("a", 0, 1800)
    assert times.tolist() == [0, 900] and wh.tolist() == [1.0, 2.0]
    ar.close()
    assert ar.actuals("a", 0, 1800)[0].tolist() == []


def test_day_so_far_is_stored_with_the_forecast(tmp_path):
    from dataclasses import replace

    ar = Archive(tmp_path / "a.db")
    ar.add_forecasts([replace(fc(0, 3600, 5.0), day_ratio=1.2, day_share=0.4)])
    ar.add_actuals("a", np.arange(0, 3600, 900), np.ones(4))
    row = ar.evaluated("a", None, 0, 10**9)[0].forecast
    assert (row.day_ratio, row.day_share) == (1.2, 0.4)
    ar.close()


def test_archive_from_before_the_day_factor_is_migrated_not_replaced(tmp_path):
    import sqlite3

    path = tmp_path / "a.db"
    old = sqlite3.connect(path)
    old.executescript(
        """CREATE TABLE forecasts (
            plant TEXT NOT NULL, horizon TEXT NOT NULL, issued_at INTEGER NOT NULL,
            start INTEGER NOT NULL, end INTEGER NOT NULL,
            p10 REAL NOT NULL, p50 REAL NOT NULL, p90 REAL NOT NULL,
            raw_p50 REAL NOT NULL, model_p50 REAL NOT NULL, ratio REAL,
            cloud REAL NOT NULL, capacity REAL NOT NULL,
            PRIMARY KEY (plant, horizon, issued_at));
        INSERT INTO forecasts VALUES ('a','1h',0,0,3600,4,5,6,5,5,NULL,20,1000);"""
    )
    old.close()
    ar = Archive(path)
    assert not ar.recovered
    ar.add_actuals("a", np.arange(0, 3600, 900), np.ones(4))
    rows = ar.evaluated("a", None, 0, 10**9)
    assert len(rows) == 1 and rows[0].forecast.day_ratio is None
    ar.close()


def test_curves_keep_the_band_scale_they_were_issued_with(tmp_path):
    ar = Archive(tmp_path / "a.db")
    ar.save_curve("a", "2026-10-11", [[0, 1, 2, 3]], (1.2, 1.1))
    ar.save_curve("a", "2026-10-12", [[0, 1, 2, 3]])
    ar.save_curve("b", "2026-10-12", [[0, 1, 2, 3]])
    assert ar.curves_since("a", "2026-10-01") == [
        ("2026-10-11", [[0, 1, 2, 3]], (1.2, 1.1)),
        ("2026-10-12", [[0, 1, 2, 3]], (1.0, 1.0)),
    ]
    assert ar.curves_since("a", "2026-10-12") == [("2026-10-12", [[0, 1, 2, 3]], (1.0, 1.0))]
    ar.close()


def test_curves_from_before_the_band_scale_are_migrated(tmp_path):
    import sqlite3

    path = tmp_path / "a.db"
    old = sqlite3.connect(path)
    old.executescript(
        """CREATE TABLE curves (plant TEXT NOT NULL, day TEXT NOT NULL, data TEXT NOT NULL,
            PRIMARY KEY (plant, day));
        INSERT INTO curves VALUES ('a', '2026-10-11', '[[0, 1, 2, 3]]');"""
    )
    old.close()
    ar = Archive(path)
    assert not ar.recovered
    assert ar.curves_since("a", "2026-10-01") == [("2026-10-11", [[0, 1, 2, 3]], (1.0, 1.0))]
    assert ar.load_curve("a", "2026-10-11") == [[0, 1, 2, 3]]
    ar.close()


def test_hour_forecasts_issued_just_before_each_hour(tmp_path):
    ar = Archive(tmp_path / "a.db")
    ar.add_forecasts(
        [
            fc(3600, 7200, 500.0, issued=3000),  # issued 10 minutes before the hour: used
            fc(4500, 8100, 400.0, issued=3900),  # starts at a quarter, not an hour: skipped
            fc(7200, 10800, 300.0, issued=6900),
            fc(7200, 9000, 9.0, horizon="30m", issued=6900),  # other horizon
        ]
    )
    rows = ar.hour_forecasts("a", 0, 7201)
    assert rows == [(3600, 400.0, 500.0, 600.0), (7200, 240.0, 300.0, 360.0)]
    assert ar.hour_forecasts("a", 3601, 7200) == []
    ar.close()
