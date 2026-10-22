# -*- coding: utf-8 -*-
"""
telemetry.py
~~~~~~~~~~~~~
Race-replay data layer: OpenF1 ingest into SQLite plus replay frame assembly.

A replay needs channels that only make sense together -- 4 Hz car telemetry,
lap boundaries, tyre stints, gaps and race-control flags -- so one ingest
writes a whole session and the player reads it back from a local SQLite file
instead of re-hitting the API on every interaction.

Circuit geometry is not on OpenF1. It comes from the MultiViewer API that
FastF1 also uses, keyed by the ``circuit_key`` OpenF1 already reports, so no
hand-maintained circuit table is needed.

Track *positions* are an estimate and :func:`progress_at` documents exactly how
far that estimate can be trusted. Lap boundaries are measured; where a car sits
between two boundaries is not.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import json
import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from config import FASTF1_CONFIG, TEAM_COLORS
except ImportError:
    from .config import FASTF1_CONFIG, TEAM_COLORS

try:
    # The HTTP client is shared with the season-results layer: one rate
    # limiter for the whole process, so two modules can never spend the same
    # request budget twice.
    from openf1_client import (  # noqa: F401  (RateLimiter is re-exported)
        MULTIVIEWER_BASE, OPENF1_BASE, RateLimiter, cached_openf1,
        fetch_json as _shared_fetch_json,
    )
except ImportError:  # pragma: no cover - package-relative import
    from .openf1_client import (  # noqa: F401
        MULTIVIEWER_BASE, OPENF1_BASE, RateLimiter, cached_openf1,
        fetch_json as _shared_fetch_json,
    )

TELEMETRY_DB_NAME = "telemetry.sqlite"

# OpenF1's free tier allows 3 requests per second and 30 per minute, and
# returns HTTP 422 once a response grows too large -- which is why a full
# session of car_data must be requested one driver at a time. The published
# caps, the rate limiter and the retry policy live in ``openf1_client``.
SESSION_ENDPOINTS = ("laps", "stints", "intervals", "position", "pit", "race_control")

REQUEST_TIMEOUT_S = 60
REQUEST_ATTEMPTS = 4
TRACK_ATTEMPTS = 5
INGEST_WORKERS = 8

# Bumped whenever SCHEMA changes shape. The cache is rebuilt on a mismatch
# rather than migrated: it is derived data, and a stale column name silently
# reads back as None is far worse than one re-download.
SCHEMA_VERSION = 2

# Seconds of grid kept before the first timed lap, so the field start and the
# run to the first corner are on the replay.
FORMATION_MARGIN_S = 120

# A write-ahead log larger than this is folded back into the database when the
# cache is opened, so a reader does not replay tens of megabytes of log first.
WAL_CHECKPOINT_BYTES = 8 * 1024 * 1024

# One timeline drives every panel, so the trace and the track map can never
# disagree about what time it is. 2 Hz with client-side interpolation is smooth
# enough to read as motion at racing speed.
DEFAULT_FRAME_HZ = 2.0

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LapWindow:
    """A measured lap: when it started, how long it took, which lap it was."""

    number: int
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


def telemetry_db_path() -> Path:
    """Location of the replay SQLite cache, alongside the FastF1 disk cache."""
    return Path(FASTF1_CONFIG.cache_dir) / TELEMETRY_DB_NAME


def default_db() -> Path:
    """Alias kept for call sites that read better as ``default_db()``."""
    return telemetry_db_path()


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
#
# The rate limiter, the retry policy and the request headers live in
# ``openf1_client`` so the season-results layer and the replay cannot spend
# the same request budget twice. What follows is a thin seam over it.


def fetch_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = REQUEST_TIMEOUT_S,
    attempts: int = REQUEST_ATTEMPTS,
) -> Any:
    """GET a JSON document under the shared rate limit.

    Delegates to :func:`openf1_client.fetch_json`, which owns the limiter and
    the retry policy for the whole process. The wrapper keeps a single seam
    for HTTP access in this module and lets the replay ask for a longer
    timeout than the season-results layer needs.

    HTTP 429 is honoured rather than treated as an error: the service tells us
    how long to wait, so a burst that overshoots costs one pause instead of a
    failed ingest. Raises the last error when every attempt fails, so a caller
    never sees a half-built payload.
    """
    return _shared_fetch_json(url, params=params, timeout=timeout,
                              attempts=attempts)


def openf1(endpoint: str, **params: Any) -> List[Dict[str, Any]]:
    """Fetch one OpenF1 collection endpoint as a list of records.

    OpenF1 answers HTTP 404 for a query that matches nothing -- a driver who
    never set a lap time, a session without a given channel. That is an empty
    result, not a failure, so it becomes ``[]`` and lets the ingest finish with
    the drivers that do have data.
    """
    try:
        payload = fetch_json(f"{OPENF1_BASE}/{endpoint}", params=params)
    except RuntimeError as exc:
        if "404" in str(exc):
            _logger.info("OpenF1 %s returned no rows for %s", endpoint, params)
            return []
        raise
    return payload if isinstance(payload, list) else []


def openf1_sessions(year: Optional[int] = None) -> List[Dict[str, Any]]:
    """Every session OpenF1 holds, optionally narrowed to one season.

    Served from the shared on-disk response cache: the replay page asks for
    the same season list on every visit, and the answer changes only when a
    session is added.
    """
    if year:
        return cached_openf1("sessions", year=year)
    return cached_openf1("sessions")


def race_sessions(year: int) -> List[Dict[str, Any]]:
    """Races for a season, oldest first, skipping cancelled events."""
    races = [s for s in openf1_sessions(year)
             if s.get("session_name") == "Race" and not s.get("is_cancelled")]
    return sorted(races, key=lambda s: s.get("date_start") or "")


def safe_race_sessions(year: int) -> List[Dict[str, Any]]:
    """Races for a season, or an empty list when the API is unreachable.

    A transport failure (DNS, timeout, 5xx) is not an empty season, but it
    must not take the page down: the caller shows its empty state instead.
    """
    try:
        return race_sessions(year)
    except RuntimeError as exc:
        _logger.warning("OpenF1 unreachable for %s: %s", year, exc)
        return []


def multiviewer_circuit(circuit_key: int, year: int) -> Optional[Dict[str, Any]]:
    """Circuit centreline and corner markers, or None when unavailable.

    This is the only non-OpenF1 dependency in the replay, and it is small, so
    it is retried harder than the telemetry: losing it costs the whole track
    map while costing nothing but one extra request to recover.
    """
    try:
        return fetch_json(f"{MULTIVIEWER_BASE}/circuits/{circuit_key}/{year}",
                          attempts=TRACK_ATTEMPTS)
    except RuntimeError as exc:
        _logger.warning("Circuit %s/%s unavailable: %s", circuit_key, year, exc)
        return None


# --------------------------------------------------------------------------
# SQLite store
# --------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_key INTEGER PRIMARY KEY, year INTEGER, session_name TEXT,
    session_type TEXT, meeting_key INTEGER, circuit_key INTEGER,
    circuit_short_name TEXT, country_name TEXT, date_start TEXT, date_end TEXT
);
CREATE TABLE IF NOT EXISTS drivers (
    session_key INTEGER, driver_number INTEGER, broadcast_name TEXT,
    full_name TEXT, name_acronym TEXT, team_name TEXT, team_colour TEXT,
    country_code TEXT, PRIMARY KEY (session_key, driver_number)
);
CREATE TABLE IF NOT EXISTS laps (
    session_key INTEGER, driver_number INTEGER, lap_number INTEGER,
    start_t REAL, lap_duration REAL, s1 REAL, s2 REAL, s3 REAL,
    st_speed INTEGER, i1_speed INTEGER, i2_speed INTEGER, is_pit_out INTEGER,
    PRIMARY KEY (session_key, driver_number, lap_number)
);
CREATE TABLE IF NOT EXISTS stints (
    session_key INTEGER, driver_number INTEGER, stint_number INTEGER,
    lap_start INTEGER, lap_end INTEGER, compound TEXT, tyre_age_at_start INTEGER,
    PRIMARY KEY (session_key, driver_number, stint_number)
);
CREATE TABLE IF NOT EXISTS intervals (
    session_key INTEGER, driver_number INTEGER, t REAL,
    gap_to_leader REAL, gap_ahead REAL, PRIMARY KEY (session_key, driver_number, t)
);
CREATE TABLE IF NOT EXISTS positions (
    session_key INTEGER, driver_number INTEGER, t REAL, position INTEGER,
    PRIMARY KEY (session_key, driver_number, t)
);
CREATE TABLE IF NOT EXISTS car_data (
    session_key INTEGER, driver_number INTEGER, t REAL, speed INTEGER,
    throttle INTEGER, brake INTEGER, drs INTEGER, rpm INTEGER, n_gear INTEGER
);
CREATE INDEX IF NOT EXISTS idx_car_data ON car_data (session_key, driver_number, t);
CREATE TABLE IF NOT EXISTS pits (
    session_key INTEGER, driver_number INTEGER, lap_number INTEGER,
    t REAL, pit_duration REAL, PRIMARY KEY (session_key, driver_number, lap_number)
);
CREATE TABLE IF NOT EXISTS race_control (
    session_key INTEGER, t REAL, lap_number INTEGER, category TEXT,
    flag TEXT, scope TEXT, message TEXT
);
CREATE TABLE IF NOT EXISTS tracks (
    circuit_key INTEGER, year INTEGER, rotation REAL, xs TEXT, ys TEXT,
    corners TEXT, pit_loss TEXT, PRIMARY KEY (circuit_key, year)
);
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS ingest_log (
    session_key INTEGER PRIMARY KEY, ingested_at REAL, rows INTEGER
);
"""

# Dropped wholesale on a schema-version mismatch, in dependency-free order.
CACHE_TABLES = (
    "schema_version", "ingest_log", "race_control", "pits", "car_data",
    "positions", "intervals", "stints", "laps", "drivers", "sessions", "tracks",
)

# Table and column names are interpolated into SQL, so they are whitelisted
# rather than trusted: a name that is not on the list is a programming error,
# not a query.
READABLE_TABLES = frozenset({
    "laps", "stints", "intervals", "positions", "car_data", "pits",
    "race_control", "drivers", "sessions",
})

ORDERABLE_COLUMNS = frozenset({
    "t", "lap_number", "stint_number", "driver_number", "position",
    "date_start", "start_t",
})


class TelemetryStore:
    """SQLite cache of ingested sessions and circuit geometry.

    The store is shared by every Streamlit session (``st.cache_resource``), so
    it hands out **one connection per thread** instead of one connection for
    the process. A single shared connection would put every session into the
    same transaction: one session's rollback would then discard another
    session's half-written rows, and the ``with conn:`` atomicity the ingest
    depends on would be gone.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else telemetry_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._connections: List[sqlite3.Connection] = []
        self._connections_lock = threading.Lock()
        self._schema_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """This thread's connection, created and prepared on first use."""
        if self._closed:
            # The store was closed; the next use opens a fresh connection
            # rather than handing out one that has already been shut.
            with self._connections_lock:
                self._closed = False
            self._local = threading.local()
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            with self._connections_lock:
                self._connections.append(conn)
            self._local.conn = conn
            self._prepare(conn)
            conn.commit()
        return conn

    def write_lock(self) -> threading.Lock:
        """Lock held while a session is being written.

        Two ingests of the same session key would interleave their
        delete-then-insert passes; serialising them keeps the table contents
        consistent without relying on SQLite's busy timeout.
        """
        return self._write_lock

    def _prepare(self, conn: sqlite3.Connection) -> None:
        """Create the schema, discarding a cache written by an older one.

        ``CREATE TABLE IF NOT EXISTS`` silently keeps the old column names when
        a table already exists, so a renamed column reads back as ``None``
        everywhere instead of raising. Comparing a stored version is the only
        way to notice.

        The rebuild runs under a lock and re-checks the version inside it, so
        two threads opening a cold cache at the same time cannot each decide
        to drop the tables.

        WAL and ``synchronous=OFF`` are safe here and nowhere else: this file is
        a derived cache that any ingest can rebuild from the upstream API.
        ``journal_size_limit`` keeps the write-ahead log from growing without
        bound -- an unbounded WAL left a 44 MB file next to a 63 MB cache.
        """
        with self._schema_lock:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("PRAGMA journal_size_limit = 16777216")
            self._truncate_wal(conn)
            current = self._stored_version(conn)
            if current == SCHEMA_VERSION:
                conn.executescript(SCHEMA)
                return
            if current is not None:
                _logger.info("Rebuilding replay cache: schema v%s -> v%s",
                             current, SCHEMA_VERSION)
            for table in CACHE_TABLES:
                conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.executescript(SCHEMA)
            conn.execute("INSERT OR REPLACE INTO schema_version VALUES (?)",
                         (SCHEMA_VERSION,))

    @staticmethod
    def _truncate_wal(conn: sqlite3.Connection) -> None:
        """Fold a large write-ahead log back into the database file.

        A cache left behind by an ingest that was killed leaves a WAL of tens
        of megabytes next to the database. Every later reader has to replay
        that log before it can read, which is seconds of delay for a file that
        is a pure cache. The checkpoint is best-effort: a database that is
        locked by another writer simply keeps its log.
        """
        try:
            wal = Path(str(conn.execute("PRAGMA database_list").fetchone()[2]) + "-wal")
            if wal.exists() and wal.stat().st_size > WAL_CHECKPOINT_BYTES:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except (sqlite3.Error, OSError, IndexError, TypeError) as exc:
            _logger.debug("WAL checkpoint on open skipped: %s", exc)

    @staticmethod
    def _stored_version(conn: sqlite3.Connection) -> Optional[int]:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        ).fetchone()
        if row is None:
            return None
        return conn.execute("SELECT version FROM schema_version").fetchone()[0]

    def checkpoint(self) -> None:
        """Fold the write-ahead log back into the database file."""
        try:
            self._conn().execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error as exc:  # pragma: no cover - best effort
            _logger.debug("WAL checkpoint skipped: %s", exc)

    def close(self) -> None:
        """Close every connection this store handed out.

        The store stays usable: the next call reopens a connection and
        re-checks the schema version.
        """
        with self._connections_lock:
            connections, self._connections = self._connections, []
            self._closed = True
        for conn in connections:
            try:
                conn.close()
            except sqlite3.Error:  # pragma: no cover - best effort
                pass
        self._local = threading.local()

    # -- reads -------------------------------------------------------------

    def has_session(self, session_key: int) -> bool:
        row = self._conn().execute(
            "SELECT 1 FROM ingest_log WHERE session_key = ?", (session_key,)
        ).fetchone()
        return row is not None

    def session(self, session_key: int) -> Optional[Dict[str, Any]]:
        row = self._conn().execute(
            "SELECT * FROM sessions WHERE session_key = ?", (session_key,)
        ).fetchone()
        return dict(row) if row else None

    def sessions(self) -> List[Dict[str, Any]]:
        rows = self._conn().execute("SELECT * FROM sessions ORDER BY date_start").fetchall()
        return [dict(r) for r in rows]

    def drivers(self, session_key: int) -> List[Dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT * FROM drivers WHERE session_key = ? ORDER BY driver_number",
            (session_key,),
        ).fetchall()
        return [dict(r) for r in rows]

    def table(self, name: str, session_key: int, driver_number: Optional[int] = None,
              order: str = "t") -> List[Dict[str, Any]]:
        """Rows from one ingest table, optionally for a single driver."""
        if name not in READABLE_TABLES:
            raise ValueError(f"unknown telemetry table: {name!r}")
        if order not in ORDERABLE_COLUMNS:
            raise ValueError(f"unknown order column: {order!r}")
        sql = f"SELECT * FROM {name} WHERE session_key = ?"
        params: List[Any] = [session_key]
        if driver_number is not None:
            sql += " AND driver_number = ?"
            params.append(driver_number)
        rows = self._conn().execute(f"{sql} ORDER BY {order}", params).fetchall()
        return [dict(r) for r in rows]

    def has_car_data(self, session_key: int, driver_number: int) -> bool:
        """Whether a driver has any car telemetry for a session.

        OpenF1's coverage is partial: a driver can be on the grid without a
        single telemetry sample. The player has to be able to say so rather
        than show a car that reads zero speed.
        """
        row = self._conn().execute(
            "SELECT 1 FROM car_data WHERE session_key = ? AND driver_number = ? LIMIT 1",
            (session_key, driver_number),
        ).fetchone()
        return row is not None

    def car_data_window(self, session_key: int, driver_number: int,
                        t_from: float, t_to: float) -> List[Dict[str, Any]]:
        """Telemetry rows inside the replay window, plus the row before it.

        Reading only the window is what keeps a payload rebuild cheap: a race
        holds ~770k samples and roughly half of them sit before the replay
        starts. The single preceding row is what lets the player hold the last
        reading across the start of the window instead of showing zeros.
        """
        conn = self._conn()
        prior = conn.execute(
            "SELECT * FROM car_data WHERE session_key = ? AND driver_number = ?"
            " AND t < ? ORDER BY t DESC LIMIT 1",
            (session_key, driver_number, t_from),
        ).fetchone()
        rows = conn.execute(
            "SELECT * FROM car_data WHERE session_key = ? AND driver_number = ?"
            " AND t >= ? AND t <= ? ORDER BY t",
            (session_key, driver_number, t_from, t_to),
        ).fetchall()
        out = [dict(r) for r in rows]
        if prior is not None:
            out.insert(0, dict(prior))
        return out

    def race_control(self, session_key: int) -> List[Dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT * FROM race_control WHERE session_key = ? ORDER BY t", (session_key,)
        ).fetchall()
        return [dict(r) for r in rows]

    def time_extent(self, session_key: int) -> Optional[Tuple[float, float]]:
        """Earliest and latest telemetry sample in a session, in seconds."""
        row = self._conn().execute(
            "SELECT MIN(t) AS lo, MAX(t) AS hi FROM car_data WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        if row is None or row["lo"] is None:
            return None
        return float(row["lo"]), float(row["hi"])

    def track(self, circuit_key: int, year: int) -> Optional[Dict[str, Any]]:
        row = self._conn().execute(
            "SELECT * FROM tracks WHERE circuit_key = ? AND year = ?",
            (circuit_key, year),
        ).fetchone()
        return _decode_track(dict(row)) if row else None


def _decode_track(row: Dict[str, Any]) -> Dict[str, Any]:
    """Rehydrate the JSON-encoded geometry columns of a ``tracks`` row."""
    return {
        "circuit_key": row["circuit_key"],
        "year": row["year"],
        "rotation": row["rotation"] or 0.0,
        "xs": json.loads(row["xs"]),
        "ys": json.loads(row["ys"]),
        "corners": json.loads(row["corners"]),
        "pit_loss": json.loads(row["pit_loss"]) if row["pit_loss"] else {},
    }


def ensure_track(store: TelemetryStore, circuit_key: int, year: int) -> bool:
    """Fill in missing circuit geometry without re-ingesting the session.

    Telemetry ingest costs minutes; circuit geometry costs one request. Keeping
    them apart means a transient geometry failure is cheap to recover from
    instead of throwing away a completed download.
    """
    if store.track(circuit_key, year):
        return True
    circuit = multiviewer_circuit(circuit_key, year)
    if not circuit:
        return False
    with store._conn() as conn:
        _write_track(conn, circuit, circuit_key, year)
    return True


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------

def _parse_iso_series(values: List[Any]) -> pd.Series:
    """Parse a column of ISO-8601 timestamps, whatever precision each row uses.

    ``pd.to_datetime`` infers a *single* format from the first element and
    coerces every other row to ``NaT`` when it does not fit, so a column that
    mixes ``...T13:00:00.123456Z`` with ``...T13:00:01Z`` would lose rows --
    silently, and channel by channel. ``format="ISO8601"`` (pandas 2) accepts
    every ISO variant at once; older pandas falls back to a per-row parse of
    whatever the vectorised pass could not read.
    """
    series = pd.Series(values, dtype="object")
    try:
        return pd.to_datetime(series, utc=True, errors="coerce",
                              format="ISO8601")
    except (ValueError, TypeError):  # pragma: no cover - pandas < 2
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
        missing = parsed.isna()
        if missing.any():
            retry = pd.to_datetime(series[missing], utc=True, errors="coerce")
            parsed.loc[missing] = retry
        return parsed


def _seconds_since(iso: Optional[str], origin: pd.Timestamp) -> Optional[float]:
    """Parse one OpenF1 ISO timestamp into seconds relative to the session origin."""
    if not iso:
        return None
    stamp = _parse_iso_series([iso]).iloc[0]
    if pd.isna(stamp):
        return None
    return round((stamp - origin).total_seconds(), 3)


def _seconds_series(values: List[Any], origin: pd.Timestamp) -> List[Optional[float]]:
    """Vectorised :func:`_seconds_since` over a column of timestamps.

    Parsing scalars one at a time is the single slowest thing in an ingest:
    ~700k telemetry rows at roughly 0.25 ms each is three minutes of pure
    timestamp parsing. One vectorised call per driver removes it entirely.

    Rows whose timestamp cannot be parsed are reported rather than dropped in
    silence -- a channel that quietly loses half its rows looks exactly like a
    driver who stopped driving.
    """
    if not values:
        return []
    parsed = _parse_iso_series(values)
    dropped = int(parsed.isna().sum())
    if dropped:
        _logger.warning("Dropped %d of %d unparseable timestamps",
                        dropped, len(values))
    if parsed.isna().all():
        return [None] * len(values)
    origin_ts = pd.Timestamp(origin)
    deltas = (parsed - origin_ts).dt.total_seconds()
    return [None if pd.isna(d) else round(float(d), 3) for d in deltas]


def _session_origin(session_row: Dict[str, Any]) -> pd.Timestamp:
    """Reference clock for a session; falls back to the Unix epoch if absent."""
    stamp = pd.to_datetime(session_row.get("date_start"), utc=True, errors="coerce")
    return pd.Timestamp(0, tz="UTC") if pd.isna(stamp) else stamp


def _replay_origin(laps: List[Dict[str, Any]],
                   fallback: pd.Timestamp) -> pd.Timestamp:
    """Clock the replay is timed against: the first lap, not the session.

    OpenF1's ``date_start`` is the scheduled lights-out and can sit an hour
    before telemetry begins. Using it would give the whole replay a large
    negative lead-in -- a grid full of dead frames before the race starts.
    """
    stamps = [pd.to_datetime(rec.get("date_start"), utc=True, errors="coerce")
              for rec in laps]
    starts = [s for s in stamps if not pd.isna(s)]
    if not starts:
        return fallback
    return min(starts) - pd.Timedelta(seconds=FORMATION_MARGIN_S)


def _lap_rows(records: List[Dict[str, Any]], origin: pd.Timestamp) -> List[tuple]:
    """Map ``/laps`` records to ingest tuples, dropping rows without a clock.

    ``start_t`` is stored as seconds from the session origin rather than an ISO
    string: every other channel uses that clock, and a replay that has to parse
    timestamps on every read pays for it on every scrub.
    """
    starts = _seconds_series([r.get("date_start") for r in records], origin)
    rows = []
    for record, start in zip(records, starts):
        duration = record.get("lap_duration")
        if start is None or duration is None:
            continue
        rows.append((
            int(record["driver_number"]), int(record.get("lap_number") or 0), start,
            float(duration),
            record.get("duration_sector_1"), record.get("duration_sector_2"),
            record.get("duration_sector_3"), record.get("st_speed"),
            record.get("i1_speed"), record.get("i2_speed"),
            1 if record.get("is_pit_out_lap") else 0,
        ))
    return rows


def _sample_rows(records: List[Dict[str, Any]], origin: pd.Timestamp,
                 fields: Sequence[str]) -> List[tuple]:
    """Map timestamped telemetry records to ingest tuples."""
    times = _seconds_series([r.get("date") for r in records], origin)
    rows = []
    for record, t in zip(records, times):
        if t is None:
            continue
        rows.append((int(record["driver_number"]), t,
                     *[int(record.get(f) or 0) for f in fields]))
    return rows


def _pit_rows(records: List[Dict[str, Any]], origin: pd.Timestamp) -> List[tuple]:
    """Map ``/pit`` records to tuples of lap number, timestamp and stop time.

    A record whose date cannot be parsed is dropped. Pinning it to the session
    origin instead would put a stop at the start of the race and pin the car
    to the line for a whole lap.
    """
    times = _seconds_series([r.get("date") for r in records], origin)
    rows = []
    for record, t in zip(records, times):
        if t is None:
            continue
        rows.append((int(record["driver_number"]),
                     int(record.get("lap_number") or 0),
                     t, record.get("pit_duration")))
    return rows


def _interval_rows(records: List[Dict[str, Any]], origin: pd.Timestamp) -> List[tuple]:
    """Map ``/intervals`` records to tuples of gap-to-leader and gap-ahead."""
    times = _seconds_series([r.get("date") for r in records], origin)
    rows = []
    for record, t in zip(records, times):
        if t is None:
            continue
        rows.append((int(record["driver_number"]), t,
                     record.get("gap_to_leader"), record.get("interval")))
    return rows


def _control_rows(records: List[Dict[str, Any]], session_key: int,
                  origin: pd.Timestamp) -> List[tuple]:
    """Map race-control records to ingest tuples, in the order they happened."""
    times = _seconds_series([r.get("date") for r in records], origin)
    rows = []
    for record, t in zip(records, times):
        if t is None:
            continue
        rows.append((session_key, t, record.get("lap_number"),
                     record.get("category"), record.get("flag"),
                     record.get("scope"), record.get("message")))
    rows.sort(key=lambda row: row[1])
    return rows


def _split_by_driver(records: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Group a whole-session response by driver number."""
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["driver_number"]), []).append(record)
    return grouped


def _fetch_session_bundle(session_key: int) -> Dict[str, List[Dict[str, Any]]]:
    """Whole-session endpoints, each small enough to fetch in one call."""
    return {name: openf1(name, session_key=session_key) for name in SESSION_ENDPOINTS}


def _fetch_driver_bundle(session_key: int, driver_number: int) -> List[Dict[str, Any]]:
    """Per-driver telemetry, which the service rejects above a certain size."""
    return openf1("car_data", session_key=session_key, driver_number=driver_number)


def latest_ingested(store: TelemetryStore, year: Optional[int] = None) -> Optional[int]:
    """Session key of the most recently ingested race, if any.

    Lets the replay page open on something already playable instead of starting
    a multi-minute download the first time somebody visits it.
    """
    row = store._conn().execute(
        "SELECT s.session_key FROM sessions s"
        " JOIN ingest_log i ON i.session_key = s.session_key"
        + (" WHERE s.year = ?" if year else "")
        + " ORDER BY i.ingested_at DESC LIMIT 1",
        (year,) if year else (),
    ).fetchone()
    return int(row["session_key"]) if row else None


def ingest_session(store: TelemetryStore, session_key: int,
                   force: bool = False) -> Dict[str, Any]:
    """Download one session into SQLite. No-op when it is already cached.

    Costs 29 requests: one session lookup, one driver list, six whole-session
    endpoints, one car_data per driver, one circuit outline. That shape is what
    the rate limit forces -- pulling all six endpoints per driver instead would
    need 120 requests and blow past the free tier's 30 per minute.

    The per-driver telemetry is streamed to disk as it arrives, so the ingest
    never holds a whole session in memory: a race is ~770k samples, which is
    ~190 MB as Python dictionaries.

    Returns ``{'session_key', 'cached', 'row_count', 'session'}``.
    """
    if store.has_session(session_key) and not force:
        return {"session_key": session_key, "cached": True,
                "row_count": 0, "session": store.session(session_key)}

    session_rows = openf1("sessions", session_key=session_key)
    if not session_rows:
        raise RuntimeError(f"OpenF1 has no session {session_key}")
    meta = session_rows[0]
    fallback_origin = _session_origin(meta)

    driver_rows = openf1("drivers", session_key=session_key)
    numbers = sorted(int(d["driver_number"]) for d in driver_rows)

    bundle = _fetch_session_bundle(session_key)
    origin = _replay_origin(bundle["laps"], fallback_origin)
    track = multiviewer_circuit(int(meta["circuit_key"]), int(meta["year"]))

    row_count = _write_session(store, meta, driver_rows, bundle, numbers,
                               origin, track)
    return {
        "session_key": session_key,
        "cached": False,
        "row_count": row_count,
        "session": store.session(session_key),
    }


def _write_session(store: TelemetryStore, meta: Dict[str, Any],
                   driver_rows: List[Dict[str, Any]],
                   bundle: Dict[str, List[Dict[str, Any]]],
                   numbers: Sequence[int],
                   origin: pd.Timestamp,
                   track: Optional[Dict[str, Any]]) -> int:
    """Persist one ingested session as a single transaction.

    Everything is written inside one transaction, so a failure leaves the
    cache exactly as it was; the per-driver telemetry is streamed in as it is
    fetched rather than collected first.
    """
    conn = store._conn()
    key = int(meta["session_key"])
    laps_by_driver = _split_by_driver(bundle["laps"])

    with store.write_lock():
        with conn:
            _write_session_header(conn, key, meta, driver_rows, bundle,
                                  laps_by_driver, origin, track)
            _write_driver_telemetry(conn, key, numbers, origin)
            total = _session_row_count(conn, key)
            conn.execute("INSERT OR REPLACE INTO ingest_log VALUES (?,?,?)",
                         (key, time.time(), total))
    store.checkpoint()
    return total


def _write_session_header(conn: sqlite3.Connection, key: int,
                          meta: Dict[str, Any],
                          driver_rows: List[Dict[str, Any]],
                          bundle: Dict[str, List[Dict[str, Any]]],
                          laps_by_driver: Dict[int, List[Dict[str, Any]]],
                          origin: pd.Timestamp,
                          track: Optional[Dict[str, Any]]) -> None:
    """Write everything except the per-driver telemetry."""
    conn.execute("DELETE FROM ingest_log WHERE session_key = ?", (key,))
    for table in ("sessions", "drivers", "laps", "stints", "intervals",
                  "positions", "car_data", "pits", "race_control"):
        conn.execute(f"DELETE FROM {table} WHERE session_key = ?", (key,))

    conn.execute(
        "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?)",
        (key, meta.get("year"), meta.get("session_name"), meta.get("session_type"),
         meta.get("meeting_key"), meta.get("circuit_key"),
         meta.get("circuit_short_name"), meta.get("country_name"),
         meta.get("date_start"), meta.get("date_end")),
    )
    conn.executemany(
        "INSERT INTO drivers VALUES (?,?,?,?,?,?,?,?)",
        [(key, int(d["driver_number"]), d.get("broadcast_name"), d.get("full_name"),
          d.get("name_acronym"), d.get("team_name"), d.get("team_colour"),
          d.get("country_code")) for d in driver_rows],
    )

    conn.executemany(
        "INSERT OR REPLACE INTO laps VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [(key, *row) for rows in laps_by_driver.values()
         for row in _lap_rows(rows, origin)],
    )
    conn.executemany(
        "INSERT OR REPLACE INTO stints VALUES (?,?,?,?,?,?,?)",
        [(key, int(s["driver_number"]), int(s.get("stint_number") or 0),
          s.get("lap_start"), s.get("lap_end"), s.get("compound"),
          s.get("tyre_age_at_start"))
         for s in bundle["stints"] if s.get("compound")],
    )
    conn.executemany(
        "INSERT OR REPLACE INTO positions VALUES (?,?,?,?)",
        [(key, *row) for row in
         _sample_rows(bundle["position"], origin, ["position"])],
    )
    conn.executemany(
        "INSERT OR REPLACE INTO pits VALUES (?,?,?,?,?)",
        [(key, *row) for row in _pit_rows(
            [p for p in bundle["pit"] if p.get("pit_duration")], origin)],
    )
    conn.executemany(
        "INSERT OR REPLACE INTO intervals VALUES (?,?,?,?,?)",
        [(key, *row) for row in _interval_rows(bundle["intervals"], origin)],
    )
    conn.executemany(
        "INSERT INTO race_control VALUES (?,?,?,?,?,?,?)",
        _control_rows(bundle["race_control"], key, origin),
    )
    if track:
        _write_track(conn, track, int(meta["circuit_key"]), int(meta["year"]))


def _write_driver_telemetry(conn: sqlite3.Connection, key: int,
                            numbers: Sequence[int],
                            origin: pd.Timestamp) -> None:
    """Fetch, convert and write one driver's telemetry at a time.

    The fetches run concurrently (the rate limiter paces them), but each
    driver's records are converted and written as soon as they arrive and then
    dropped, so the peak memory is one driver rather than one session.
    """
    fields = ("speed", "throttle", "brake", "drs", "rpm", "n_gear")
    with ThreadPoolExecutor(max_workers=INGEST_WORKERS) as pool:
        futures = {pool.submit(_fetch_driver_bundle, key, n): n
                   for n in numbers}
        for future in as_completed(futures):
            # Popping releases the future, and with it the payload it holds,
            # so only the drivers still in flight are in memory.
            futures.pop(future)
            records = future.result()
            rows = [(key, *row) for row in _sample_rows(records, origin, fields)]
            if rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO car_data VALUES (?,?,?,?,?,?,?,?,?)",
                    rows,
                )
            del records, rows


def _write_track(conn: sqlite3.Connection, track: Dict[str, Any],
                 circuit_key: int, year: int) -> None:
    """Store circuit geometry, keeping only the fields the player draws."""
    corners = [
        {"x": c.get("trackPosition", {}).get("x"), "y": c.get("trackPosition", {}).get("y"),
         "number": c.get("number")}
        for c in track.get("corners", [])
        if c.get("trackPosition")
    ]
    conn.execute(
        "INSERT OR REPLACE INTO tracks VALUES (?,?,?,?,?,?,?)",
        (circuit_key, year, track.get("rotation"), json.dumps(track.get("x") or []),
         json.dumps(track.get("y") or []), json.dumps(corners),
         json.dumps(track.get("pitLoss") or {})),
    )


def _session_row_count(conn: sqlite3.Connection, session_key: int) -> int:
    """Total cached sample rows for a session, for the ingest log."""
    total = 0
    for table in ("laps", "car_data", "intervals", "positions", "stints"):
        row = conn.execute(
            f"SELECT COUNT(*) AS n FROM {table} WHERE session_key = ?", (session_key,)
        ).fetchone()
        total += row["n"]
    return total


# --------------------------------------------------------------------------
# Geometry and the position estimate
# --------------------------------------------------------------------------

def cumulative_lengths(xs: Sequence[float], ys: Sequence[float]) -> np.ndarray:
    """Cumulative arc length along a polyline; ``out[0] == 0``."""
    if len(xs) < 2:
        return np.zeros(len(xs))
    steps = np.hypot(np.diff(xs), np.diff(ys))
    return np.concatenate([[0.0], np.cumsum(steps)])


def point_at_fraction(xs: np.ndarray, ys: np.ndarray,
                      lengths: np.ndarray, fraction: float) -> Tuple[float, float]:
    """Point at a fraction of the way along the polyline, by arc length.

    Uniform-by-index sampling would bunch cars into the corners, where the
    MultiViewer centreline is densest; arc length keeps them evenly spread.
    """
    if len(xs) < 2:
        return (float(xs[0]), float(ys[0])) if len(xs) else (0.0, 0.0)

    total = float(lengths[-1])
    if total <= 0:
        return (float(xs[0]), float(ys[0]))

    target = min(max(fraction, 0.0), 1.0) * total
    upper = int(np.searchsorted(lengths, target, side="left"))
    upper = min(max(upper, 1), len(lengths) - 1)
    lower = upper - 1
    span = float(lengths[upper] - lengths[lower])
    ratio = 0.0 if span <= 0 else (target - float(lengths[lower])) / span
    return (
        float(xs[lower] + (xs[upper] - xs[lower]) * ratio),
        float(ys[lower] + (ys[upper] - ys[lower]) * ratio),
    )


def lap_windows(laps: List[Dict[str, Any]]) -> List[LapWindow]:
    """Measured laps of one driver, ordered and stripped of unusable rows.

    ``start`` is measured. A lap with no duration is interpolated from its
    neighbours rather than dropped, because a gap there would freeze the car
    on the start line for a whole lap.
    """
    usable = [lap for lap in laps if lap.get("start") is not None]
    usable.sort(key=lambda lap: lap["start"])
    if not usable:
        return []

    windows: List[LapWindow] = []
    for index, lap in enumerate(usable):
        duration = lap.get("duration")
        if not duration or duration <= 0:
            duration = _interpolated_duration(usable, index)
        if not duration or duration <= 0:
            continue
        windows.append(LapWindow(int(lap.get("number") or index + 1),
                                 float(lap["start"]), float(duration)))
    return windows


def _interpolated_duration(laps: List[Dict[str, Any]], index: int) -> Optional[float]:
    """Duration to assume for a lap that is missing one, from its neighbours."""
    before = next((lap.get("duration") for lap in reversed(laps[:index])
                   if lap.get("duration")), None)
    after = next((lap.get("duration") for lap in laps[index + 1:]
                  if lap.get("duration")), None)
    if before and after:
        return (float(before) + float(after)) / 2
    return float(before or after) if (before or after) else None


def progress_at(t: float, windows: List[LapWindow],
                pit_windows: Sequence[Tuple[float, float]] = ()) -> Tuple[int, float]:
    """Estimated (lap number, progress around the lap) at time ``t``.

    This is the one genuinely derived value in the player, and it is an
    approximation, not a measurement: lap start and lap duration come straight
    from OpenF1, but the circuit's real x/y telemetry is unavailable here, so a
    car is placed at constant progress around the lap. It reads correctly under
    green-flag racing and drifts wherever lap-average speed is unrepresentative
    -- through slow corners, during safety cars, and across pit stops. Pit
    windows are pinned to the start line because that is where the pit lane is.
    """
    if not windows:
        return (1, 0.0)

    starts = np.array([w.start for w in windows])
    index = int(np.searchsorted(starts, t, side="right")) - 1
    if index < 0:
        return (windows[0].number, 0.0)
    if index >= len(windows):
        return (windows[-1].number, 1.0)

    window = windows[index]
    progress = (t - window.start) / window.duration if window.duration else 0.0
    progress = min(max(progress, 0.0), 1.0)
    for pit_start, pit_end in pit_windows:
        if pit_start <= t < pit_end:
            # The pit lane starts at the line, so the *fraction* is pinned to
            # zero -- but the car is still on the lap it is running, which is
            # what :func:`replay_payload.progress_series` reports.
            return (window.number, 0.0)
    return (window.number, progress)


def step_hold(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Sample a telemetry channel onto ``grid`` by holding the last reading.

    The 4 Hz channels are state, not signals: interpolation between two pedal
    readings would invent throttle nobody applied. Samples before the first
    reading stay at zero.
    """
    if times.size == 0:
        return np.zeros(grid.size, dtype=np.int16)
    index = np.searchsorted(times, grid, side="right") - 1
    index = np.clip(index, 0, times.size - 1)
    return values[index]


def team_colour(team_name: Optional[str], api_colour: Optional[str]) -> str:
    """OpenF1's own team colour, falling back to the app's palette."""
    if api_colour and re_hex(api_colour):
        return f"#{api_colour.lstrip('#')}"
    for name, colour in TEAM_COLORS.items():
        if name.lower() == (team_name or "").lower():
            return colour
    return "#888888"


def re_hex(value: str) -> bool:
    """True when ``value`` is a 3- or 6-digit hex colour."""
    cleaned = value.lstrip("#")
    return len(cleaned) in (3, 6) and all(c in "0123456789abcdefABCDEF" for c in cleaned)