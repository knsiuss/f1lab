# -*- coding: utf-8 -*-
"""Tests for the replay data layer in src/telemetry.py and src/replay_payload.py.

Everything here runs on synthetic data: no network, no OpenF1, no Streamlit.
"""

import re
import sqlite3
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.telemetry import (
    RateLimiter, TelemetryStore, _control_rows, _interval_rows, _pit_rows,
    _seconds_since, _seconds_series, cumulative_lengths, ensure_track,
    ingest_session, lap_windows, latest_ingested, openf1, point_at_fraction,
    progress_at, step_hold, team_colour,
)
from src.replay_payload import (
    FRACTION_SCALE, _pit_spans, _position_by_lap, build_payload,
    player_html, progress_series,
)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def test_cumulative_lengths_of_a_square():
    lengths = cumulative_lengths([0, 3, 3, 0], [0, 0, 4, 4])
    assert lengths[0] == 0.0
    assert lengths[-1] == pytest.approx(10.0)  # 3 across, 4 down, 3 back


def test_point_at_fraction_walks_the_polyline():
    xs, ys = [0, 10, 10, 0], [0, 0, 10, 10]
    lengths = cumulative_lengths(xs, ys)
    assert point_at_fraction(xs, ys, lengths, 0.0) == (0.0, 0.0)
    assert point_at_fraction(xs, ys, lengths, 1.0) == (0.0, 10.0)
    assert point_at_fraction(xs, ys, lengths, 0.5) == pytest.approx((10.0, 5.0))


def test_point_at_fraction_clamps_out_of_range():
    xs, ys = [0, 10], [0, 0]
    lengths = cumulative_lengths(xs, ys)
    assert point_at_fraction(xs, ys, lengths, -5.0) == (0.0, 0.0)
    assert point_at_fraction(xs, ys, lengths, 5.0) == (10.0, 0.0)


def test_point_at_fraction_survives_degenerate_track():
    assert point_at_fraction([1, 1], [1, 1], cumulative_lengths([1, 1], [1, 1]), 0.5) == (1.0, 1.0)
    assert point_at_fraction([], [], np.zeros(0), 0.5) == (0.0, 0.0)


# --------------------------------------------------------------------------
# lap windows and the position estimate
# --------------------------------------------------------------------------

def _lap_rows(durations=(90.0, 92.0, 91.0), first_start=10.0):
    rows, start = [], first_start
    for index, duration in enumerate(durations):
        rows.append({"number": index + 1, "start": start, "duration": duration})
        start += duration
    return rows


def test_lap_windows_sorted_and_numbered():
    windows = lap_windows(_lap_rows())
    assert [w.number for w in windows] == [1, 2, 3]
    assert windows[0].start == 10.0
    assert windows[0].end == pytest.approx(100.0)


def test_lap_windows_interpolates_a_missing_duration():
    """A dropped lap must not freeze the car on the line for its whole lap."""
    rows = [{"number": 1, "start": 0.0, "duration": 90.0},
            {"number": 2, "start": 90.0, "duration": None},
            {"number": 3, "start": 180.0, "duration": 100.0}]
    windows = lap_windows(rows)
    assert len(windows) == 3
    assert windows[1].duration == pytest.approx(95.0)  # mean of its neighbours


def test_lap_windows_drops_laps_with_no_clock():
    assert lap_windows([{"number": 1, "start": None, "duration": 90.0}]) == []
    assert lap_windows([]) == []


def test_progress_at_mid_lap_is_half_way_around():
    windows = lap_windows(_lap_rows())
    number, progress = progress_at(10.0 + 45.0, windows)
    assert number == 1
    assert progress == pytest.approx(0.5)


def test_progress_at_resets_on_each_lap():
    windows = lap_windows(_lap_rows())
    assert progress_at(10.0 + 90.0, windows) == (2, pytest.approx(0.0))
    assert progress_at(10.0 + 90.0 + 92.0 + 91.0, windows) == (3, pytest.approx(1.0))


def test_progress_at_before_first_lap_and_after_last():
    windows = lap_windows(_lap_rows())
    assert progress_at(0.0, windows) == (1, 0.0)          # on the grid
    assert progress_at(10.0 + 1000.0, windows) == (3, 1.0)  # chequered flag


def test_progress_at_pins_a_car_in_the_pit_lane_to_the_line():
    windows = lap_windows(_lap_rows())
    number, progress = progress_at(40.0, windows, pit_windows=[(30.0, 50.0)])
    assert progress == 0.0


def test_progress_at_without_windows_is_stable():
    assert progress_at(99.0, []) == (1, 0.0)


def test_progress_series_matches_the_scalar_reference():
    windows = lap_windows(_lap_rows())
    grid = np.arange(0.0, 400.0, 0.5)
    series = progress_series(grid, windows, [])

    assert len(series) == len(grid)
    for t in (5.0, 10.0, 55.0, 100.0, 190.0, 300.0):
        _, expected = progress_at(t, windows)
        assert series[int(t * 2)] == pytest.approx(expected)


def test_progress_series_is_not_flat_over_a_lap():
    """Guards a real regression: quantising before the 0..1 -> uint16 scaling
    truncated every fraction to zero, parking the whole field on the line."""
    windows = lap_windows(_lap_rows())
    series = progress_series(np.arange(0.0, 105.0, 0.5), windows, [])
    assert series.min() == 0.0
    # Lap 1 spans t=10..100, so the peak fraction is just under a full lap and
    # resets to zero the moment lap 2 begins.
    assert series.max() == pytest.approx(0.99, abs=0.01)
    assert len(set(series.round(3).tolist())) > 50


def test_progress_series_empty_inputs_do_not_crash():
    grid = np.arange(0.0, 10.0, 1.0)
    assert list(progress_series(grid, [], [])) == [0.0] * 10
    assert list(progress_series(np.zeros(0), lap_windows(_lap_rows()), [])) == []


# --------------------------------------------------------------------------
# channel resampling
# --------------------------------------------------------------------------

def test_step_hold_keeps_the_last_reading_rather_than_interpolating():
    times = np.array([0.0, 10.0, 20.0])
    values = np.array([1, 2, 3])
    out = step_hold(times, values, np.array([0.0, 5.0, 9.9, 10.0, 25.0]))
    assert list(out) == [1, 1, 1, 2, 3]


def test_step_hold_before_first_sample_and_empty_input():
    out = step_hold(np.array([5.0]), np.array([9]), np.array([0.0]))
    assert list(out) == [9]
    assert list(step_hold(np.array([]), np.array([]), np.array([1.0, 2.0]))) == [0, 0]


# --------------------------------------------------------------------------
# rate limiting
# --------------------------------------------------------------------------

def test_rate_limiter_sleeps_once_the_burst_cap_is_hit():
    limiter = RateLimiter(burst=2, per_minute=1000)
    limiter.acquire()
    limiter.acquire()

    import time as clock
    start = clock.monotonic()
    limiter.acquire()  # must wait for the one-second window to roll
    assert clock.monotonic() - start >= 0.4


def test_openf1_treats_404_as_an_empty_result(monkeypatch):
    """A driver with no telemetry must not fail the whole ingest."""

    def not_found(url, params=None, **kwargs):
        raise RuntimeError(f"GET {url} failed after 4 attempts: 404 Client Error: "
                           "Not Found")

    monkeypatch.setattr("src.telemetry.fetch_json", not_found)
    assert openf1("car_data", session_key=1, driver_number=2) == []


def test_openf1_still_raises_on_other_failures(monkeypatch):
    def broken(url, params=None, **kwargs):
        raise RuntimeError("GET failed after 4 attempts: 500 Server Error")

    monkeypatch.setattr("src.telemetry.fetch_json", broken)
    with pytest.raises(RuntimeError):
        openf1("car_data", session_key=1, driver_number=2)


# --------------------------------------------------------------------------
# timestamp conversion
# --------------------------------------------------------------------------

ORIGIN = pd.Timestamp("2025-12-07T13:00:00Z")


def test_seconds_since_parses_one_timestamp():
    assert _seconds_since("2025-12-07T13:01:00Z", ORIGIN) == pytest.approx(60.0)
    assert _seconds_since(None, ORIGIN) is None
    assert _seconds_since("not-a-time", ORIGIN) is None


def test_seconds_series_matches_the_scalar_helper():
    """The vectorised path must agree with the readable one it replaces."""
    stamps = ["2025-12-07T13:00:00Z", "2025-12-07T13:01:00Z",
              "2025-12-07T13:00:30Z", None, "bad"]
    series = _seconds_series(stamps, ORIGIN)
    assert series == [_seconds_since(s, ORIGIN) for s in stamps]


def test_seconds_series_handles_empty_and_all_invalid():
    assert _seconds_series([], ORIGIN) == []
    assert _seconds_series([None, "nope"], ORIGIN) == [None, None]


def test_seconds_series_is_fast_enough_for_a_whole_session():
    """A scalar parse per row was ~3 minutes of a single ingest."""
    stamps = [f"2025-12-07T13:00:{i % 60:02d}Z" for i in range(50_000)]
    start = time.monotonic()
    _seconds_series(stamps, ORIGIN)
    assert time.monotonic() - start < 5.0


# --------------------------------------------------------------------------
# schema versioning
# --------------------------------------------------------------------------

def test_store_rebuilds_a_cache_written_by_an_older_schema(tmp_path):
    """A renamed column must not survive as a silent ``None``."""
    path = tmp_path / "old.sqlite"
    legacy = sqlite3.connect(path)
    legacy.execute("CREATE TABLE laps (session_key INTEGER, driver_number INTEGER,"
                   " lap_number INTEGER, date_start TEXT)")
    legacy.execute("CREATE TABLE ingest_log (session_key INTEGER PRIMARY KEY,"
                   " ingested_at REAL, rows INTEGER)")
    legacy.execute("INSERT INTO ingest_log VALUES (1, 0.0, 0)")
    legacy.commit()
    legacy.close()

    store = TelemetryStore(path)
    columns = {row["name"] for row in store._conn().execute(
        "SELECT name FROM pragma_table_info('laps')")}
    assert "start_t" in columns
    assert "date_start" not in columns
    # The stale ingest log went with it, so the session is re-downloaded.
    assert store.has_session(1) is False
    store.close()


# --------------------------------------------------------------------------
# colours
# --------------------------------------------------------------------------

def test_team_colour_prefers_the_api_value():
    assert team_colour("Some Team", "00D2BE") == "#00D2BE"


def test_team_colour_falls_back_to_the_palette():
    assert team_colour("Ferrari", None) == "#DC0000"
    assert team_colour("Unknown Team", "nonsense") == "#888888"


# --------------------------------------------------------------------------
# store round-trip
# --------------------------------------------------------------------------

@pytest.fixture
def seeded_store(tmp_path):
    """A store holding one synthetic session, written through the public API."""
    store = TelemetryStore(tmp_path / "telemetry.sqlite")
    origin = pd.Timestamp("2025-01-01T00:00:00Z")
    del origin
    with store._conn() as conn:
        conn.execute(
            "INSERT INTO sessions VALUES (1,2025,'Race','Race',9,63,'Sakhir',"
            "'Bahrain','2025-01-01T00:00:00Z','2025-01-01T02:00:00Z')")
        conn.execute("INSERT INTO drivers VALUES (1,1,'M VERSTAPPEN','Max VERSTAPPEN',"
                     "'VER','Red Bull Racing','4781D7','NED')")
        conn.executemany(
            "INSERT INTO laps VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            # lap 2 is a measured pit-out lap, so the player has a flag to show
            [(1, 1, n, float(n) * 90.0, 90.0 + n, 28.0, 30.0, 32.0, 330, 300, 310,
              1 if n == 2 else 0)
             for n in (1, 2, 3)])
        conn.execute("INSERT INTO stints VALUES (1,1,1,1,3,'SOFT',0)")
        conn.executemany("INSERT INTO intervals VALUES (?,?,?,?,?)",
                         [(1, 1, float(t), t, 1.5) for t in (0, 60, 120)])
        conn.executemany("INSERT INTO positions VALUES (?,?,?,?)",
                         [(1, 1, float(t), 1) for t in (0, 120)])
        conn.executemany(
            "INSERT INTO car_data VALUES (?,?,?,?,?,?,?,?,?)",
            [(1, 1, float(t), 200 + t, 80, 10, 0, 9000, 5)
             for t in range(0, 200, 5)])
        conn.execute("INSERT INTO pits VALUES (1,1,2,150.0,22.5)")
        conn.execute("INSERT INTO race_control VALUES (1,30.0,1,'Flag','GREEN',"
                     "'Track','GREEN LIGHT - PIT EXIT OPEN')")
        conn.execute(
            "INSERT INTO tracks VALUES (63,2025,92.0,'[0,10,10,0]','[0,0,10,10]',"
            "'[{\"x\":0,\"y\":0,\"number\":1}]','{\"normal\":\"24.0\"}')")
        conn.execute("INSERT INTO ingest_log VALUES (1,0.0,0)")
    return store


@pytest.fixture
def negative_origin_store(tmp_path):
    """A session whose telemetry starts *before* the replay origin.

    This is the real shape of an OpenF1 race: the session opens an hour
    before lights-out, and the replay grid is clamped to the formation
    window, so the first frame sits at a negative timestamp.
    """
    store = TelemetryStore(tmp_path / "negative.sqlite")
    with store._conn() as conn:
        conn.execute(
            "INSERT INTO sessions VALUES (1,2025,'Race','Race',9,63,'Sakhir',"
            "'Bahrain','2025-01-01T00:00:00Z','2025-01-01T02:00:00Z')")
        conn.execute("INSERT INTO drivers VALUES (1,1,'M VERSTAPPEN','Max VERSTAPPEN',"
                     "'VER','Red Bull Racing','4781D7','NED')")
        conn.executemany(
            "INSERT INTO laps VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [(1, 1, n, 120.0 + (n - 1) * 90.0, 90.0, 28.0, 30.0, 32.0, 330, 300, 310, 0)
             for n in (1, 2, 3)])
        conn.executemany(
            "INSERT INTO car_data VALUES (?,?,?,?,?,?,?,?,?)",
            [(1, 1, float(t), 200, 80, 10, 0, 9000, 5)
             for t in range(-120, 200, 5)])
        conn.execute("INSERT INTO ingest_log VALUES (1,0.0,0)")
    return store


@pytest.fixture
def partly_covered_store(tmp_path):
    """Two drivers on the grid, only one of them with telemetry.

    OpenF1's coverage is partial: a driver can be in the session without a
    single car-data sample.
    """
    store = TelemetryStore(tmp_path / "partial.sqlite")
    with store._conn() as conn:
        conn.execute(
            "INSERT INTO sessions VALUES (1,2025,'Race','Race',9,63,'Sakhir',"
            "'Bahrain','2025-01-01T00:00:00Z','2025-01-01T02:00:00Z')")
        conn.executemany(
            "INSERT INTO drivers VALUES (?,?,?,?,?,?,?,?)",
            [(1, 1, 'M VERSTAPPEN', 'Max VERSTAPPEN', 'VER', 'Red Bull Racing',
              '4781D7', 'NED'),
             (1, 4, 'L NORRIS', 'Lando NORRIS', 'NOR', 'McLaren', 'F47600', 'GBR')])
        conn.executemany(
            "INSERT INTO laps VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [(1, number, n, 120.0 + (n - 1) * 90.0, 90.0, 28.0, 30.0, 32.0,
              330, 300, 310, 0)
             for number in (1, 4) for n in (1, 2, 3)])
        conn.executemany(
            "INSERT INTO car_data VALUES (?,?,?,?,?,?,?,?,?)",
            [(1, 1, float(t), 200, 80, 10, 0, 9000, 5)
             for t in range(0, 200, 5)])
        conn.execute("INSERT INTO ingest_log VALUES (1,0.0,0)")
    return store


def test_store_round_trips_a_session(seeded_store):
    assert seeded_store.has_session(1)
    assert seeded_store.session(1)["circuit_short_name"] == "Sakhir"
    assert seeded_store.time_extent(1) == (0.0, 195.0)
    assert len(seeded_store.table("car_data", 1, 1)) == 40
    assert seeded_store.race_control(1)[0]["flag"] == "GREEN"


def test_store_rehydrates_track_geometry(seeded_store):
    track = seeded_store.track(63, 2025)
    assert track["xs"] == [0, 10, 10, 0]
    assert track["corners"][0]["number"] == 1
    assert track["pit_loss"]["normal"] == "24.0"


def test_store_missing_session_is_absent_not_broken(tmp_path):
    store = TelemetryStore(tmp_path / "empty.sqlite")
    assert store.has_session(999) is False
    assert store.session(999) is None
    assert store.track(1, 2020) is None
    assert store.time_extent(999) is None
    assert latest_ingested(store, 2025) is None
    store.close()


def test_latest_ingested_picks_the_most_recent_session_for_a_year(seeded_store):
    """The replay page opens on this, so it must not start a fresh download."""
    assert latest_ingested(seeded_store, 2025) == 1
    assert latest_ingested(seeded_store, 2024) is None
    assert latest_ingested(seeded_store) == 1


def test_ensure_track_fetches_geometry_without_reingesting(monkeypatch, tmp_path):
    """Circuit geometry costs one request, so a miss must not force a re-ingest."""
    store = TelemetryStore(tmp_path / "t.sqlite")
    assert store.track(70, 2025) is None

    calls = []

    def fake_fetch(url, params=None, **kwargs):
        calls.append(url)
        return {"x": [0, 1], "y": [0, 1], "corners": [],
                "rotation": 0, "pitLoss": {"normal": "24"}}

    monkeypatch.setattr("src.telemetry.fetch_json", fake_fetch)
    assert ensure_track(store, 70, 2025) is True
    assert len(calls) == 1

    # Second call is served from cache.
    assert ensure_track(store, 70, 2025) is True
    assert len(calls) == 1
    assert store.track(70, 2025)["xs"] == [0, 1]


def test_ensure_track_reports_failure_without_raising(monkeypatch, tmp_path):
    store = TelemetryStore(tmp_path / "t.sqlite")

    def boom(url, params=None, **kwargs):
        raise RuntimeError("GET failed after 5 attempts: 500")

    monkeypatch.setattr("src.telemetry.fetch_json", boom)
    assert ensure_track(store, 70, 2025) is False


def test_build_payload_is_empty_for_an_unknown_session(seeded_store):
    assert build_payload(seeded_store, 404) == {}


def test_build_payload_shapes_and_channels(seeded_store):
    payload = build_payload(seeded_store, 1, frame_hz=2.0)
    meta = payload["meta"]
    assert meta["circuit"] == "Sakhir"
    assert meta["totalLaps"] == 3

    n = payload["grid"]["n"]
    assert n == 390  # telemetry spans 0..195s, sampled at 2 Hz
    assert len(payload["drivers"]) == 1

    frac = np.frombuffer(__import__("base64").b64decode(payload["channels"]["frac"]),
                         dtype=np.uint16)
    speed = np.frombuffer(__import__("base64").b64decode(payload["channels"]["speed"]),
                          dtype=np.uint8)
    assert len(frac) == n
    assert len(speed) == n
    assert frac.max() <= FRACTION_SCALE


def test_build_payload_quantises_speed_to_two_kmh_steps(seeded_store):
    """Channels are held at 2 km/h resolution, so a reading loses at most 1 km/h."""
    payload = build_payload(seeded_store, 1, frame_hz=2.0)
    speed = np.frombuffer(__import__("base64").b64decode(payload["channels"]["speed"]),
                          dtype=np.uint8)
    # Samples sit every 5s with speed 200 + 5t, and each reading is held until
    # the next one, so the grid repeats in blocks of ten frames.
    truth = 200 + 5 * (np.arange(payload["grid"]["n"]) // 10)
    # Cast before scaling: uint8 arithmetic wraps at 256 in numpy.
    decoded = speed.astype(np.int16) * 2
    assert (decoded - truth).max() <= 1
    assert (decoded - truth).min() >= -1
    assert speed.max() <= 200  # never exceeds the uint8 range the decoder assumes


def test_build_payload_encodes_lap_progress_across_the_whole_grid(seeded_store):
    """Regression: the fraction channel once shipped all zeros, which parked
    every car on the start-finish line for the entire race."""
    payload = build_payload(seeded_store, 1, frame_hz=2.0)
    frac = np.frombuffer(__import__("base64").b64decode(payload["channels"]["frac"]),
                         dtype=np.uint16) / FRACTION_SCALE
    hz = payload["grid"]["hz"]

    assert len(np.unique(frac)) > 20
    assert frac.max() > 0.5

    # Read the lap window the payload itself reports, rather than assuming one.
    lap = payload["drivers"][0]["laps"][0]
    midpoint = (lap["start"] + lap["duration"] / 2) - payload["grid"]["t0"]
    assert frac[int(midpoint * hz)] == pytest.approx(0.5, abs=0.02)


def test_driver_block_reports_fastest_lap_and_top_speed(seeded_store):
    driver = build_payload(seeded_store, 1)["drivers"][0]
    assert driver["fastestLap"]["number"] == 1
    assert driver["topSpeed"] == 330
    assert driver["stints"][0]["compound"] == "SOFT"
    assert driver["stints"][0]["colour"] == "#FF3333"


def test_position_by_lap_holds_the_last_measured_position():
    """A lap starting before the next reading keeps the older position."""
    laps = [{"start": 0.0}, {"start": 60.0}, {"start": 200.0}]
    positions = [{"t": 0.0, "position": 3}, {"t": 120.0, "position": 1}]
    assert _position_by_lap(laps, positions) == [3, 3, 1]


def test_position_by_lap_without_data_is_all_none():
    assert _position_by_lap([{"start": 0.0}], []) == [None]


def test_pit_spans_falls_back_to_the_lap_number():
    laps = [{"start": 0.0}, {"start": 90.0}, {"start": 180.0}]
    spans = _pit_spans([{"lap_number": 2, "t": None, "pit_duration": 22.0}], laps, 90.0)
    assert spans == [(90.0, 112.0)]


def test_pit_spans_uses_a_measured_timestamp_when_present():
    spans = _pit_spans([{"lap_number": 2, "t": 150.0, "pit_duration": 22.0}], [], 90.0)
    assert spans == [(150.0, 172.0)]


def test_pit_spans_uses_the_lap_number_not_the_list_index():
    """A driver whose first lap is not lap 1 must get its own pit window.

    Indexing the lap list by ``lap_number - 1`` gave this driver lap 5's
    start when it pitted on lap 4.
    """
    laps = [{"number": 3, "start": 0.0}, {"number": 4, "start": 90.0},
            {"number": 5, "start": 180.0}]
    spans = _pit_spans([{"lap_number": 4, "t": None, "pit_duration": 22.0}],
                        laps, 90.0)
    assert spans == [(90.0, 112.0)]


# --------------------------------------------------------------------------
# timeline: the payload's duration is a span, not an absolute timestamp
# --------------------------------------------------------------------------

def test_payload_duration_is_the_grid_span(negative_origin_store):
    """The player counts the clock from the first frame.

    The grid can start before the session origin (the formation window), so
    an absolute end timestamp here left the last seconds of every session
    unreachable.
    """
    payload = build_payload(negative_origin_store, 1, frame_hz=2.0)
    grid = payload["grid"]
    span = (grid["n"] - 1) / grid["hz"]
    assert grid["t0"] == pytest.approx(-120.0)
    assert payload["meta"]["duration"] == pytest.approx(span)
    assert payload["meta"]["endTime"] == pytest.approx(grid["t0"] + span)
    # The end of the timeline is the last frame, not an absolute timestamp.
    assert payload["meta"]["duration"] + grid["t0"] == pytest.approx(
        payload["meta"]["endTime"])


def test_payload_duration_is_the_span_for_a_session_starting_at_zero(seeded_store):
    payload = build_payload(seeded_store, 1, frame_hz=2.0)
    grid = payload["grid"]
    assert payload["meta"]["duration"] == pytest.approx((grid["n"] - 1) / grid["hz"])


# --------------------------------------------------------------------------
# measured pit-out flag and telemetry coverage
# --------------------------------------------------------------------------

def test_measured_pit_out_flag_reaches_the_payload(seeded_store):
    """The player marks an out lap from the measured flag, not a guess."""
    laps = build_payload(seeded_store, 1)["drivers"][0]["laps"]
    assert laps[1]["out"] is True
    assert laps[0]["out"] is False


def test_driver_without_telemetry_is_flagged(partly_covered_store):
    """A driver with no telemetry must not read as a car parked in the pits."""
    payload = build_payload(partly_covered_store, 1)
    blocks = {d["number"]: d for d in payload["drivers"]}
    assert blocks[1]["hasTelemetry"] is True
    assert blocks[4]["hasTelemetry"] is False
    assert payload["meta"]["coverage"] == {"drivers": 2, "withTelemetry": 1}


def test_payload_exposes_the_position_feed_size_and_age(seeded_store):
    """The position feed is sampled rarely; the player has to be able to say so."""
    driver = build_payload(seeded_store, 1)["drivers"][0]
    assert driver["positionCount"] == len(driver["positions"])
    assert driver["positionUpdated"] == driver["positions"][-1][0]


# --------------------------------------------------------------------------
# timestamps: mixed ISO formats must all parse
# --------------------------------------------------------------------------

def test_seconds_series_parses_mixed_iso_formats():
    """One row with a different precision must not empty the whole channel.

    ``pd.to_datetime`` infers a single format from the first element, so a
    column mixing microseconds and whole seconds lost every row that did not
    match.
    """
    stamps = ["2025-12-07T13:00:00.123456Z", "2025-12-07T13:00:01Z",
              "2025-12-07T13:00:02+00:00"]
    series = _seconds_series(stamps, ORIGIN)
    assert series == [0.123, 1.0, 2.0]
    assert series == [_seconds_since(s, ORIGIN) for s in stamps]


def test_rows_with_an_unreadable_date_are_dropped_not_pinned_to_zero():
    """A row whose date cannot be read is dropped, not stamped at the origin."""
    origin = pd.Timestamp("2025-01-01T00:00:00Z")
    assert _pit_rows([{"driver_number": 1, "lap_number": 2, "date": "not-a-date",
                        "pit_duration": 22.0}], origin) == []
    assert _interval_rows([{"driver_number": 1, "date": "not-a-date"}], origin) == []
    assert _control_rows([{"date": "not-a-date", "message": "x"}], 1, origin) == []


# --------------------------------------------------------------------------
# the store: one connection per thread, whitelisted identifiers
# --------------------------------------------------------------------------

def test_store_hands_out_one_connection_per_thread(tmp_path):
    """Two sessions must not end up sharing one SQLite transaction."""
    store = TelemetryStore(tmp_path / "t.sqlite")
    seen = {}
    errors = []

    def grab(name):
        try:
            seen[name] = store._conn()
        except Exception as exc:  # pragma: no cover - failure detail
            errors.append(exc)

    threads = [threading.Thread(target=grab, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert seen[0] is not seen[1]
    store.close()


def test_store_survives_concurrent_readers_and_a_writer(tmp_path):
    store = TelemetryStore(tmp_path / "t.sqlite")
    errors = []

    def writer(number):
        try:
            with store._conn() as conn:
                conn.execute(
                    "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (number, 2025, 'Race', 'Race', 9, 63, 'Sakhir', 'Bahrain',
                     '2025-01-01T00:00:00Z', '2025-01-01T02:00:00Z'))
        except Exception as exc:  # pragma: no cover - failure detail
            errors.append(exc)

    def reader():
        try:
            for _ in range(25):
                store.sessions()
        except Exception as exc:  # pragma: no cover - failure detail
            errors.append(exc)

    threads = ([threading.Thread(target=writer, args=(i,)) for i in (1, 2)]
               + [threading.Thread(target=reader) for _ in range(2)])
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(store.sessions()) == 2
    store.close()


def test_store_reopens_after_being_closed(tmp_path):
    """Closing the store must not leave a closed connection behind."""
    store = TelemetryStore(tmp_path / "t.sqlite")
    store.sessions()
    store.close()
    assert store.sessions() == []
    store.close()


def test_store_rejects_an_unknown_table_or_order_column(seeded_store):
    """Identifiers are whitelisted, not interpolated on trust."""
    with pytest.raises(ValueError):
        seeded_store.table("sqlite_master", 1)
    with pytest.raises(ValueError):
        seeded_store.table("laps", 1, order="lap_number; DROP TABLE laps")


def test_car_data_window_returns_the_row_before_it(seeded_store):
    """The windowed read must keep the last reading before the window.

    Without it the player would show zeros at the start of the replay instead
    of holding the driver's last known values.
    """
    rows = seeded_store.car_data_window(1, 1, 100.0, 150.0)
    assert rows[0]["t"] < 100.0
    assert all(100.0 <= row["t"] <= 150.0 for row in rows[1:])
    assert len(rows) < len(seeded_store.table("car_data", 1, 1))


def test_telemetry_shares_the_openf1_rate_limiter():
    """One limiter for the process: two would double the request budget."""
    import sys
    from src import telemetry
    # The module used to carry its own limiter instance; the shared client
    # owns it now.
    assert not hasattr(telemetry, "LIMITER")
    assert telemetry.RateLimiter.__module__.endswith("openf1_client")
    client = sys.modules[telemetry.RateLimiter.__module__]
    assert hasattr(client, "LIMITER")


# --------------------------------------------------------------------------
# the ingest
# --------------------------------------------------------------------------

FAKE_SESSION = [{
    "session_key": 1, "year": 2025, "session_name": "Race", "session_type": "Race",
    "meeting_key": 9, "circuit_key": 63, "circuit_short_name": "Sakhir",
    "country_name": "Bahrain", "date_start": "2025-01-01T00:00:00Z",
    "date_end": "2025-01-01T02:00:00Z",
}]
FAKE_DRIVERS = [{
    "driver_number": 1, "broadcast_name": "M VERSTAPPEN",
    "full_name": "Max VERSTAPPEN", "name_acronym": "VER",
    "team_name": "Red Bull Racing", "team_colour": "4781D7", "country_code": "NED",
}]
FAKE_LAPS = [{
    "driver_number": 1, "lap_number": 1, "date_start": "2025-01-01T00:02:00Z",
    "lap_duration": 90.0, "duration_sector_1": 28.0, "duration_sector_2": 30.0,
    "duration_sector_3": 32.0, "st_speed": 330, "i1_speed": 300, "i2_speed": 310,
    "is_pit_out_lap": False,
}]
FAKE_CAR_DATA = [{
    "driver_number": 1, "date": f"2025-01-01T00:02:{i:02d}Z", "speed": 200,
    "throttle": 80, "brake": 10, "drs": False, "rpm": 9000, "n_gear": 5,
} for i in range(10)]


def _fake_openf1(endpoint, **params):
    return {
        "sessions": FAKE_SESSION, "drivers": FAKE_DRIVERS, "laps": FAKE_LAPS,
        "car_data": FAKE_CAR_DATA, "stints": [], "intervals": [], "position": [],
        "pit": [], "race_control": [],
    }.get(endpoint, [])


def test_ingest_records_the_row_count_it_wrote(tmp_path, monkeypatch):
    """The ingest log must say how many rows were cached, not zero."""
    monkeypatch.setattr("src.telemetry.openf1", _fake_openf1)
    monkeypatch.setattr("src.telemetry.multiviewer_circuit", lambda *a, **k: None)
    store = TelemetryStore(tmp_path / "t.sqlite")

    result = ingest_session(store, 1)
    assert result["row_count"] > 0
    logged = store._conn().execute(
        "SELECT rows FROM ingest_log WHERE session_key = 1").fetchone()
    assert logged["rows"] == result["row_count"]
    assert store.has_session(1)
    assert len(store.table("car_data", 1, 1)) == len(FAKE_CAR_DATA)


def test_ingest_is_atomic_when_a_driver_fetch_fails(tmp_path, monkeypatch):
    """A failed ingest must not leave a half-written session behind."""
    def failing_openf1(endpoint, **params):
        if endpoint == "car_data":
            raise RuntimeError("GET failed after 4 attempts: 500 Server Error")
        return _fake_openf1(endpoint, **params)

    monkeypatch.setattr("src.telemetry.openf1", failing_openf1)
    monkeypatch.setattr("src.telemetry.multiviewer_circuit", lambda *a, **k: None)
    store = TelemetryStore(tmp_path / "t.sqlite")

    with pytest.raises(RuntimeError):
        ingest_session(store, 1)
    assert store.has_session(1) is False
    assert store.table("car_data", 1, 1) == []


def test_race_sessions_survives_an_unreachable_api(monkeypatch):
    """A transport failure must not take the page down with it."""
    from src.telemetry import safe_race_sessions

    def boom(*args, **kwargs):
        raise RuntimeError("GET failed after 4 attempts: 500 Server Error")

    monkeypatch.setattr("src.telemetry.race_sessions", boom)
    assert safe_race_sessions(2025) == []


# --------------------------------------------------------------------------
# the player: escaping and the measured flags
# --------------------------------------------------------------------------

PLAYER_PATH = Path(__file__).parent.parent / "src" / "assets" / "replay_player.html"


def test_player_html_escapes_untrusted_values():
    """A driver name must not be able to close the script block."""
    payload = {"drivers": [{"name": "<img src=x onerror=alert(1)>"}]}
    markup = player_html("<script>__PAYLOAD__</script>", payload)
    assert "<img" not in markup
    assert "\\u003cimg" in markup


def test_player_escapes_every_interpolated_value():
    """The player must escape API strings before they reach innerHTML."""
    text = PLAYER_PATH.read_text(encoding="utf-8")
    assert "function esc(" in text
    assert "function html(" in text
    for match in re.finditer(r"\.innerHTML\s*=\s*(.*?);", text, re.S):
        target = match.group(1)
        if "${" in target:
            assert "html`" in target or "raw(" in target, target[:160]


def test_player_reads_the_measured_pit_out_flag():
    """The OUT marker must come from the measured flag, not a missing duration."""
    text = PLAYER_PATH.read_text(encoding="utf-8")
    assert "d.out = laps.map(l => !!l.out);" in text
    assert "const outLap = li >= 0 && !!d.out[li];" in text


def test_player_compares_lap_numbers_for_the_fastest_lap_marker():
    """The fastest-lap marker compared an array index with a lap number."""
    text = PLAYER_PATH.read_text(encoding="utf-8")
    assert "s.bestNo === bestOverall.lap" in text
    assert "d.bestLapNo.push(bestIdx >= 0 ? d.lapNo[bestIdx] : null);" in text
