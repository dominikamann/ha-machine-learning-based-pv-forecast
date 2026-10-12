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
