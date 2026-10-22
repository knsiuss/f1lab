# -*- coding: utf-8 -*-
"""Tests for the OpenF1 season-results layer (src/openf1_results.py).

The tests never touch the network: the endpoint helpers are monkeypatched with
fixtures that mirror the real OpenF1 payloads (including the way OpenF1
reports a sprint session as ``session_type='Race'`` with
``session_name='Sprint'``).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import openf1_results as ofr


# ---------------------------------------------------------------------------
# Fixtures that mirror the real API payloads
# ---------------------------------------------------------------------------

SESSIONS = [
    # A normal Grand Prix weekend.
    {'session_key': 100, 'session_type': 'Practice', 'session_name': 'Practice 1',
     'meeting_key': 10, 'date_start': '2025-03-14T01:30:00+00:00', 'year': 2025,
     'is_cancelled': False},
    {'session_key': 101, 'session_type': 'Qualifying', 'session_name': 'Qualifying',
     'meeting_key': 10, 'date_start': '2025-03-15T05:00:00+00:00', 'year': 2025,
     'is_cancelled': False},
    {'session_key': 102, 'session_type': 'Race', 'session_name': 'Race',
     'meeting_key': 10, 'date_start': '2025-03-16T05:00:00+00:00', 'year': 2025,
     'is_cancelled': False},
    # A sprint weekend: OpenF1 types the sprint as a Race session.
    {'session_key': 200, 'session_type': 'Race', 'session_name': 'Sprint',
     'meeting_key': 20, 'date_start': '2025-03-22T03:00:00+00:00', 'year': 2025,
     'is_cancelled': False},
    {'session_key': 201, 'session_type': 'Race', 'session_name': 'Race',
     'meeting_key': 20, 'date_start': '2025-03-23T07:00:00+00:00', 'year': 2025,
     'is_cancelled': False},
    # A cancelled event must never reach the frame.
    {'session_key': 300, 'session_type': 'Race', 'session_name': 'Race',
     'meeting_key': 30, 'date_start': '2025-04-01T07:00:00+00:00', 'year': 2025,
     'is_cancelled': True},
]

DRIVERS = {
    1: {'driver_number': 1, 'first_name': 'Max', 'last_name': 'Verstappen',
        'full_name': 'Max VERSTAPPEN', 'team_name': 'Red Bull Racing'},
    4: {'driver_number': 4, 'first_name': 'Lando', 'last_name': 'Norris',
        'full_name': 'Lando NORRIS', 'team_name': 'McLaren'},
    44: {'driver_number': 44, 'first_name': 'Lewis', 'last_name': 'Hamilton',
         'full_name': 'Lewis HAMILTON', 'team_name': 'Ferrari'},
}

RESULTS = {
    102: [
        {'position': 1, 'driver_number': 4, 'number_of_laps': 57, 'points': 25.0,
         'dnf': False, 'dns': False, 'dsq': False, 'duration': 5167.469,
         'gap_to_leader': 0, 'meeting_key': 10, 'session_key': 102},
        {'position': 2, 'driver_number': 1, 'number_of_laps': 57, 'points': 18.0,
         'dnf': False, 'dns': False, 'dsq': False, 'duration': 5180.063,
         'gap_to_leader': 12.594, 'meeting_key': 10, 'session_key': 102},
        {'position': 3, 'driver_number': 44, 'number_of_laps': 56, 'points': 0.0,
         'dnf': False, 'dns': False, 'dsq': False, 'duration': None,
         'gap_to_leader': '+1 LAP', 'meeting_key': 10, 'session_key': 102},
    ],
    200: [
        {'position': 1, 'driver_number': 44, 'number_of_laps': 19, 'points': 8.0,
         'dnf': False, 'dns': False, 'dsq': False, 'duration': 1839.965,
         'gap_to_leader': 0, 'meeting_key': 20, 'session_key': 200},
    ],
}

GRIDS = {
    10: {1: 2, 4: 1, 44: 3},
    20: {1: 1, 4: 2, 44: 5},
}

EVENTS = [
    {'round': 1, 'name': 'Australian Grand Prix', 'date': pd.Timestamp('2025-03-16').to_pydatetime()},
    {'round': 2, 'name': 'Chinese Grand Prix', 'date': pd.Timestamp('2025-03-23').to_pydatetime()},
]


@pytest.fixture
def offline(monkeypatch):
    """Replace every endpoint call with the fixtures above."""
    monkeypatch.setattr(ofr, 'season_sessions', lambda year: list(SESSIONS))
    monkeypatch.setattr(ofr, 'session_result',
                        lambda key, ttl=None: list(RESULTS.get(key, [])))
    monkeypatch.setattr(ofr, 'starting_grid',
                        lambda meeting, session_key=None, ttl=None: dict(
                            GRIDS.get(meeting, {})))
    monkeypatch.setattr(ofr, 'drivers_for_session',
                        lambda key, ttl=None: dict(DRIVERS))
    monkeypatch.setattr(ofr, 'drivers_for_meeting',
                        lambda meeting, ttl=None: dict(DRIVERS))
    monkeypatch.setattr(ofr, '_calendar_events', lambda year: list(EVENTS))
    return None


# ---------------------------------------------------------------------------
# Session classification
# ---------------------------------------------------------------------------

def test_sprint_sessions_are_detected_by_name():
    # OpenF1 types a sprint as session_type='Race'; only session_name tells
    # a sprint apart from a Grand Prix.
    assert ofr._is_sprint({'session_type': 'Race', 'session_name': 'Sprint'})
    assert not ofr._is_sprint({'session_type': 'Race', 'session_name': 'Race'})
    assert not ofr._is_sprint({'session_type': 'Qualifying',
                               'session_name': 'Sprint Qualifying'})


def test_race_and_sprint_sessions_are_separated(offline):
    races = ofr.race_sessions(2025)
    sprints = ofr.sprint_sessions(2025)
    assert [s['session_key'] for s in races] == [102, 201]
    assert [s['session_key'] for s in sprints] == [200]


def test_cancelled_sessions_are_dropped(offline):
    assert all(s['session_key'] != 300 for s in ofr.race_sessions(2025))


# ---------------------------------------------------------------------------
# Row building
# ---------------------------------------------------------------------------

def test_format_duration():
    assert ofr.format_duration(5167.469) == '1:26:07.469'
    assert ofr.format_duration(91.5) == '1:31.500'
    assert ofr.format_duration(None) == ''
    assert ofr.format_duration(0) == ''


def test_time_retired_reconstruction():
    assert ofr._time_retired({'dnf': True}) == 'DNF'
    assert ofr._time_retired({'dns': True}) == 'DNS'
    assert ofr._time_retired({'dsq': True}) == 'DSQ'
    assert ofr._time_retired({'gap_to_leader': 0, 'duration': 5167.469}) == '1:26:07.469'
    assert ofr._time_retired({'gap_to_leader': 12.594}) == '+12.594'
    assert ofr._time_retired({'gap_to_leader': '+1 LAP'}) == '+1 LAP'


def test_driver_name_is_first_last():
    assert ofr._driver_name(DRIVERS[4]) == 'Lando Norris'
    assert ofr._driver_name({'full_name': 'Lando NORRIS'}) == 'Lando Norris'
    assert ofr._driver_name({}) == ''


def test_build_session_rows_schema(offline):
    session = next(s for s in SESSIONS if s['session_key'] == 102)
    rows = ofr.build_session_rows(session, EVENTS[0], 'Race')
    assert len(rows) == 3
    first = rows[0]
    assert first['Driver'] == 'Lando Norris'
    assert first['Team'] == 'McLaren'
    assert first['Starting Grid'] == 1
    assert first['Position'] == 1
    assert first['Points'] == 25.0
    assert first['Track'] == 'Australian Grand Prix'
    assert first['Round'] == 1
    assert first['SessionType'] == 'Race'
    assert first['Status'] == 'Finished'
    assert first['Time/Retired'] == '1:26:07.469'


# ---------------------------------------------------------------------------
# Season frame
# ---------------------------------------------------------------------------

def test_build_season_results_labels_sprints(offline):
    frame = ofr.build_season_results(2025)
    assert frame is not None
    counts = frame['SessionType'].value_counts().to_dict()
    assert counts == {'Race': 3, 'Sprint': 1}


def test_build_season_results_has_no_duplicate_rows(offline):
    frame = ofr.build_season_results(2025)
    key = ['SessionKey', 'Driver']
    assert not frame.duplicated(subset=key).any()


def test_build_season_results_is_in_calendar_order(offline):
    frame = ofr.build_season_results(2025)
    rounds = frame['Round'].tolist()
    assert rounds == sorted(rounds)


def test_build_season_results_returns_none_without_sessions(monkeypatch):
    monkeypatch.setattr(ofr, 'season_sessions', lambda year: [])
    assert ofr.build_season_results(2025) is None


def test_build_season_results_survives_a_failing_session(monkeypatch, offline):
    def boom(key):
        raise RuntimeError('service unavailable')

    monkeypatch.setattr(ofr, 'session_result', boom)
    frame = ofr.build_season_results(2025)
    # Every session failed, but the frame must not raise.
    assert frame is None or frame.empty


# ---------------------------------------------------------------------------
# Event matching
# ---------------------------------------------------------------------------

def test_match_event_finds_the_race_weekend(offline):
    race = next(s for s in SESSIONS if s['session_key'] == 102)
    sprint = next(s for s in SESSIONS if s['session_key'] == 200)
    assert ofr.match_event(EVENTS, race)['name'] == 'Australian Grand Prix'
    # A sprint on the Saturday belongs to the same Grand Prix weekend.
    assert ofr.match_event(EVENTS, sprint)['name'] == 'Chinese Grand Prix'


def test_match_event_without_calendar_returns_none():
    assert ofr.match_event([], {'date_start': '2025-03-16T05:00:00+00:00'}) is None


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def test_grid_for_event_uses_driver_names(monkeypatch, offline):
    monkeypatch.setattr(ofr, 'season_sessions', lambda year: list(SESSIONS))
    grid = ofr.grid_for_event(2025, 'Australian Grand Prix')
    assert grid == {'Lando Norris': 1, 'Max Verstappen': 2, 'Lewis Hamilton': 3}
    assert not [k for k in grid if k.startswith('#')]


def test_grid_for_event_unknown_event_is_empty(monkeypatch, offline):
    monkeypatch.setattr(ofr, 'season_sessions', lambda year: list(SESSIONS))
    assert ofr.grid_for_event(2025, 'Nowhere Grand Prix') == {}
    assert ofr.grid_for_event(2025, '') == {}


# ---------------------------------------------------------------------------
# Latency policy: what is cached for how long, and how many requests a
# season costs
# ---------------------------------------------------------------------------

def _session_ago(hours: float) -> dict:
    import datetime as dt
    when = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)
    return {'date_start': when.isoformat()}


def test_finished_and_future_sessions_are_cached_for_a_day():
    """Only a session that is running right now needs a short cache."""
    finished = ofr._session_ttl(_session_ago(48))
    future = ofr._session_ttl({'date_start': '2099-01-01T00:00:00+00:00'})
    live = ofr._session_ttl(_session_ago(1))
    assert finished == ofr.FINAL_SESSION_TTL_S
    assert future == ofr.FINAL_SESSION_TTL_S, (
        'a session that has not started yet will not change for hours'
    )
    assert live == ofr.LIVE_SESSION_TTL_S


def test_the_driver_map_is_fetched_once_per_season(monkeypatch, offline):
    """One request for the drivers of a season, not one per meeting.

    The driver metadata only changes when a driver is replaced, so a meeting
    is only fetched when it holds a number the map does not know yet.
    """
    calls = []

    def counting(meeting_key, ttl=None):
        calls.append(meeting_key)
        return dict(DRIVERS)

    monkeypatch.setattr(ofr, 'drivers_for_meeting', counting)

    drivers = {}
    for session in ofr._sessions_of_type(SESSIONS, 'Race'):
        ofr.build_session_rows(session, EVENTS[0], 'Race', drivers=drivers)
    assert len(calls) == 1, 'the driver map must be fetched once, not per meeting'

    # A driver the map does not know (a debut) triggers one more fetch.
    def with_a_newcomer(meeting_key, ttl=None):
        calls.append(meeting_key)
        merged = dict(DRIVERS)
        merged[99] = {'driver_number': 99, 'first_name': 'New',
                      'last_name': 'Comer', 'team_name': 'Test'}
        return merged

    monkeypatch.setattr(ofr, 'drivers_for_meeting', with_a_newcomer)
    monkeypatch.setattr(ofr, 'session_result', lambda key, ttl=None: [
        {'position': 1, 'driver_number': 99, 'number_of_laps': 10,
         'points': 10.0, 'dnf': False, 'dns': False, 'dsq': False,
         'duration': 900.0, 'gap_to_leader': 0}])
    ofr.build_session_rows(
        {'session_key': 999, 'meeting_key': 123, 'date_start': '2025-03-16T05:00:00+00:00'},
        EVENTS[0], 'Race', drivers={})
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Starting grid: a sprint weekend publishes two grids
# ---------------------------------------------------------------------------

def _grid_rows():
    # Shanghai 2025: rows for the sprint qualifying and for the Grand Prix
    # qualifying, both filed against the same meeting key.
    return [
        {'meeting_key': 1255, 'session_key': 9989, 'driver_number': 44, 'position': 1},
        {'meeting_key': 1255, 'session_key': 9989, 'driver_number': 1, 'position': 2},
        {'meeting_key': 1255, 'session_key': 9994, 'driver_number': 81, 'position': 1},
        {'meeting_key': 1255, 'session_key': 9994, 'driver_number': 63, 'position': 2},
    ]


def test_starting_grid_is_the_grid_before_that_session(monkeypatch):
    monkeypatch.setattr(ofr, 'cached_openf1',
                        lambda endpoint, ttl=None, **kw: _grid_rows())

    # The sprint (9993) starts from the sprint-qualifying grid (9989) ...
    assert ofr.starting_grid(1255, session_key=9993) == {44: 1, 1: 2}
    # ... and the Grand Prix (9998) from the qualifying grid (9994).
    assert ofr.starting_grid(1255, session_key=9998) == {81: 1, 63: 2}


def test_starting_grid_without_a_session_key_is_the_latest_grid(monkeypatch):
    monkeypatch.setattr(ofr, 'cached_openf1',
                        lambda endpoint, ttl=None, **kw: _grid_rows())
    assert ofr.starting_grid(1255) == {81: 1, 63: 2}


def test_starting_grid_falls_back_to_the_first_grid(monkeypatch):
    # A session earlier than every published grid takes the oldest one rather
    # than coming up empty.
    monkeypatch.setattr(ofr, 'cached_openf1',
                        lambda endpoint, ttl=None, **kw: _grid_rows())
    assert ofr.starting_grid(1255, session_key=9000) == {44: 1, 1: 2}


# ---------------------------------------------------------------------------
# Track label
# ---------------------------------------------------------------------------

def test_meeting_name_falls_back_to_the_meetings_feed(monkeypatch):
    seen = []

    def meetings(endpoint, ttl=None, **kw):
        seen.append((endpoint, kw))
        return [{'meeting_key': 1255, 'meeting_name': 'Chinese Grand Prix'}]

    monkeypatch.setattr(ofr, 'cached_openf1', meetings)

    # /sessions carries no meeting_name, so the Track label used to come out
    # blank whenever the FastF1 calendar was unreachable.
    assert ofr._meeting_name({'meeting_key': 1255, 'year': 2025}) == \
        'Chinese Grand Prix'
    assert seen and seen[0][0] == 'meetings'


def test_meeting_name_prefers_the_payload(monkeypatch):
    monkeypatch.setattr(ofr, 'cached_openf1',
                        lambda *a, **kw: pytest.fail('no request expected'))
    assert ofr._meeting_name({'meeting_key': 1,
                              'meeting_name': 'From the payload'}) == \
        'From the payload'


def test_meeting_name_survives_a_failing_feed(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError('offline')

    monkeypatch.setattr(ofr, 'cached_openf1', boom)
    assert ofr._meeting_name({'meeting_key': 1255, 'year': 2025}) == ''
    assert ofr._meeting_name({'meeting_key': 1255}) == ''
