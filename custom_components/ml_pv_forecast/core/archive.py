"""SQLite archive of issued forecasts, 15-minute actuals and morning curves."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import astuple, dataclass
from pathlib import Path

import numpy as np

QUARTER = 900

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    plant TEXT NOT NULL, horizon TEXT NOT NULL, issued_at INTEGER NOT NULL,
    start INTEGER NOT NULL, end INTEGER NOT NULL,
    p10 REAL NOT NULL, p50 REAL NOT NULL, p90 REAL NOT NULL,
    raw_p50 REAL NOT NULL, ratio REAL, cloud REAL NOT NULL, capacity REAL NOT NULL,
    PRIMARY KEY (plant, horizon, issued_at)
);
CREATE TABLE IF NOT EXISTS actuals (
    plant TEXT NOT NULL, start INTEGER NOT NULL, wh REAL NOT NULL, PRIMARY KEY (plant, start)
);
CREATE TABLE IF NOT EXISTS curves (
    plant TEXT NOT NULL, day TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY (plant, day)
);
"""


@dataclass(frozen=True)
class ArchivedForecast:
    plant: str
    horizon: str
    issued_at: int
    start: int
    end: int
    p10: float
    p50: float
    p90: float
    raw_p50: float
    ratio: float | None
    cloud: float
    capacity: float


@dataclass(frozen=True)
class EvalRow:
    forecast: ArchivedForecast
    actual: float
    persistence: float | None


def _finite(row: ArchivedForecast) -> bool:
    """Rows with NaN would violate NOT NULL (sqlite stores NaN as NULL)."""
    values = [v for k, v in vars(row).items() if isinstance(v, float) and k != "ratio"]
    return all(np.isfinite(values)) and (row.ratio is None or np.isfinite(row.ratio))


class Archive:
    """Blocking sqlite store; call from the executor."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self.recovered = False
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._db = self._open()
        except sqlite3.DatabaseError:
            path.rename(path.with_name(f"{path.name}.corrupt-{int(time.time())}"))
            self.recovered = True
            self._db = self._open()

    def _open(self) -> sqlite3.Connection:
        db = sqlite3.connect(self._path, check_same_thread=False)
        try:
            db.executescript(SCHEMA)
            db.execute("SELECT count(*) FROM forecasts").fetchone()
        except sqlite3.DatabaseError:
            db.close()
            raise
        return db

    def add_forecasts(self, rows: list[ArchivedForecast]) -> None:
        with self._lock, self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO forecasts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [astuple(r) for r in rows if _finite(r)],
            )

    def add_actuals(self, plant: str, times: np.ndarray, wh: np.ndarray) -> None:
        with self._lock, self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO actuals VALUES (?,?,?)",
                [
                    (plant, int(t), float(v))
                    for t, v in zip(times, wh, strict=True)
                    if np.isfinite(v)
                ],
            )

    def _actual_sum(self, plant: str, start: int, end: int) -> float | None:
        count, total = self._db.execute(
            "SELECT count(*), sum(wh) FROM actuals WHERE plant=? AND start>=? AND start<?",
            (plant, start, end),
        ).fetchone()
        return float(total) if count and count == (end - start) // QUARTER else None

    def evaluated(self, plant: str, horizon: str | None, since: int, until: int) -> list[EvalRow]:
        """Archived forecasts in [since, until] that have complete actuals."""
        query = "SELECT * FROM forecasts WHERE plant=? AND start>=? AND end<=?"
        args: tuple = (plant, since, until)
        if horizon is not None:
            query += " AND horizon=?"
            args = (*args, horizon)
        with self._lock:
            out = []
            for raw in self._db.execute(query + " ORDER BY start", args).fetchall():
                f = ArchivedForecast(*raw)
                actual = self._actual_sum(plant, f.start, f.end)
                if actual is None:
                    continue
                length = f.end - f.start
                out.append(EvalRow(f, actual, self._actual_sum(plant, f.start - length, f.start)))
            return out

    def save_curve(self, plant: str, day: str, data: list) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO curves VALUES (?,?,?)", (plant, day, json.dumps(data))
            )

    def load_curve(self, plant: str, day: str) -> list | None:
        with self._lock:
            row = self._db.execute(
                "SELECT data FROM curves WHERE plant=? AND day=?", (plant, day)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def prune(self, before: int, before_day: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM forecasts WHERE end < ?", (before,))
            self._db.execute("DELETE FROM actuals WHERE start < ?", (before,))
            self._db.execute("DELETE FROM curves WHERE day < ?", (before_day,))

    def close(self) -> None:
        with self._lock:
            self._db.close()
