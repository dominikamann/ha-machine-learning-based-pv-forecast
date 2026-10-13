"""SQLite archive of issued forecasts, 15-minute actuals and morning curves."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import astuple, dataclass, fields
from pathlib import Path

import numpy as np

QUARTER = 900

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    plant TEXT NOT NULL, horizon TEXT NOT NULL, issued_at INTEGER NOT NULL,
    start INTEGER NOT NULL, end INTEGER NOT NULL,
    p10 REAL NOT NULL, p50 REAL NOT NULL, p90 REAL NOT NULL,
    raw_p50 REAL NOT NULL, model_p50 REAL NOT NULL, ratio REAL,
    cloud REAL NOT NULL, capacity REAL NOT NULL, day_ratio REAL, day_share REAL,
    PRIMARY KEY (plant, horizon, issued_at)
);
CREATE TABLE IF NOT EXISTS actuals (
    plant TEXT NOT NULL, start INTEGER NOT NULL, wh REAL NOT NULL, PRIMARY KEY (plant, start)
);
CREATE INDEX IF NOT EXISTS forecasts_plant_start ON forecasts (plant, start);
CREATE TABLE IF NOT EXISTS curves (
    plant TEXT NOT NULL, day TEXT NOT NULL, data TEXT NOT NULL, band_lo REAL, band_hi REAL,
    PRIMARY KEY (plant, day)
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
    raw_p50: float  # after correction, before nowcast
    model_p50: float  # pure model output, before correction and nowcast
    ratio: float | None
    cloud: float
    capacity: float
    day_ratio: float | None = None  # actual/forecast since midnight when it was issued
    day_share: float | None = None  # share of the day's forecast energy already past


FORECAST_COLUMNS = [f.name for f in fields(ArchivedForecast)]
INSERT_FORECAST = (
    f"INSERT OR REPLACE INTO forecasts VALUES ({','.join('?' * len(FORECAST_COLUMNS))})"
)
NULLABLE = {"ratio", "day_ratio", "day_share"}
# Columns added later, with their type; older archives get them instead of being replaced.
ADDED_COLUMNS = {"day_ratio": "REAL", "day_share": "REAL"}


@dataclass(frozen=True)
class EvalRow:
    forecast: ArchivedForecast
    actual: float
    persistence: float | None


def _finite(row: ArchivedForecast) -> bool:
    """Rows with NaN would violate NOT NULL (sqlite stores NaN as NULL)."""
    values = [v for k, v in vars(row).items() if isinstance(v, float) and k not in NULLABLE]
    optional = [getattr(row, k) for k in NULLABLE if getattr(row, k) is not None]
    return all(np.isfinite(values)) and all(np.isfinite(optional))


class Archive:
    """Blocking sqlite store; call from the executor."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self.recovered = False
        self._closed = False
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
            columns = [row[1] for row in db.execute("PRAGMA table_info(forecasts)")]
            missing = FORECAST_COLUMNS[len(columns) :]
            if columns == FORECAST_COLUMNS[: len(columns)] and set(missing) <= set(ADDED_COLUMNS):
                with db:
                    for name in missing:
                        db.execute(f"ALTER TABLE forecasts ADD COLUMN {name} {ADDED_COLUMNS[name]}")
                columns = FORECAST_COLUMNS
            if columns != FORECAST_COLUMNS:
                raise sqlite3.DatabaseError("outdated archive schema")
            curve_columns = [row[1] for row in db.execute("PRAGMA table_info(curves)")]
            with db:
                for name in ("band_lo", "band_hi"):
                    if name not in curve_columns:
                        db.execute(f"ALTER TABLE curves ADD COLUMN {name} REAL")
        except sqlite3.DatabaseError:
            db.close()
            raise
        return db

    def add_forecasts(self, rows: list[ArchivedForecast]) -> None:
        with self._lock:
            if self._closed:
                return
            with self._db:
                self._db.executemany(
                    INSERT_FORECAST,
                    [astuple(r) for r in rows if _finite(r)],
                )

    def add_actuals(self, plant: str, times: np.ndarray, wh: np.ndarray) -> None:
        with self._lock:
            if self._closed:
                return
            with self._db:
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
            if self._closed:
                return []
            out = []
            for raw in self._db.execute(query + " ORDER BY start", args).fetchall():
                f = ArchivedForecast(*raw)
                actual = self._actual_sum(plant, f.start, f.end)
                if actual is None:
                    continue
                length = f.end - f.start
                out.append(EvalRow(f, actual, self._actual_sum(plant, f.start - length, f.start)))
            return out

    def actuals(self, plant: str, since: int, until: int) -> tuple[np.ndarray, np.ndarray]:
        """15-minute actuals (Wh) with start in [since, until)."""
        with self._lock:
            if self._closed:
                return np.array([], dtype=np.int64), np.array([])
            rows = self._db.execute(
                "SELECT start, wh FROM actuals"
                " WHERE plant=? AND start>=? AND start<? ORDER BY start",
                (plant, since, until),
            ).fetchall()
        return np.array([r[0] for r in rows], dtype=np.int64), np.array([r[1] for r in rows])

    def save_curve(
        self, plant: str, day: str, data: list, band: tuple[float, float] = (1.0, 1.0)
    ) -> None:
        """Store a morning curve with the band scale (lower, upper) that was applied to it."""
        with self._lock:
            if self._closed:
                return
            with self._db:
                self._db.execute(
                    "INSERT OR REPLACE INTO curves (plant, day, data, band_lo, band_hi)"
                    " VALUES (?,?,?,?,?)",
                    (plant, day, json.dumps(data), float(band[0]), float(band[1])),
                )

    def curves_since(self, plant: str, first_day: str) -> list[tuple[str, list, tuple]]:
        """Morning curves from ``first_day`` on, oldest first, with their band scale."""
        with self._lock:
            if self._closed:
                return []
            rows = self._db.execute(
                "SELECT day, data, band_lo, band_hi FROM curves"
                " WHERE plant=? AND day>=? ORDER BY day",
                (plant, first_day),
            ).fetchall()
        return [
            (day, json.loads(data), (1.0 if lo is None else lo, 1.0 if hi is None else hi))
            for day, data, lo, hi in rows
        ]

    def load_curve(self, plant: str, day: str) -> list | None:
        with self._lock:
            if self._closed:
                return None
            row = self._db.execute(
                "SELECT data FROM curves WHERE plant=? AND day=?", (plant, day)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def prune(self, before: int, before_day: str, short_before: int | None = None) -> None:
        """Drop old rows; short-horizon forecasts (the bulk) can be dropped earlier."""
        with self._lock:
            if self._closed:
                return
            with self._db:
                self._db.execute("DELETE FROM forecasts WHERE end < ?", (before,))
                if short_before is not None:
                    self._db.execute(
                        "DELETE FROM forecasts WHERE horizon IN ('30m', '1h', '6h') AND end < ?",
                        (short_before,),
                    )
                self._db.execute("DELETE FROM actuals WHERE start < ?", (before,))
                self._db.execute("DELETE FROM curves WHERE day < ?", (before_day,))

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._db.close()
