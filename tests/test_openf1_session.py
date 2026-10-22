# -*- coding: utf-8 -*-
"""Tests for the OpenF1 session fallback (src/openf1_session.py).

FastF1 cannot read the official timing archive in this environment (the
static timing files answer ``401 Unauthorized``), so the analysis pages fall
back to the OpenF1 feed.  This module rebuilds a FastF1-shaped session from
that feed, and the rules it follows are worth pinning down:

* the lap table has the columns the app reads (``Driver``, ``LapNumber``,
  ``LapTime``, ``Position``, ``Compound``, ``Stint``, ``TyreLife``,
  ``IsInLap``, ``IsOutLap``, ``PitInTime``, ``PitOutTime``);
* lap and pit times are **elapsed** times from the start of the session, the
  way FastF1 stores them, so a pit stop can be measured;
* the track status is not published by the feed, so the column is left out
  instead of being faked;
* the running order is an estimate unless the official lap-by-lap order
  covers most of the laps.

The tests never touch the network: the OpenF1 client is monkeypatched with
fixtures that mirror the real payloads.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src import openf1_session as ofs
import openf1_results

SESSION_KEY = 9001

# --- Fixtures that mirror the real OpenF1 payloads -------------------------

SESSIONS = [
    {'session_key': 9001, 'session_type': 'Race', 'session_name': 'Race',
     'meeting_key': 900, 'date_start': '2026-10-11T13:00:00+00:00',
     'year': 2026, 'is_cancelled': False},
    {'session_key': 9002, 'session_type': 'Qualifying',
     'session_name': 'Qualifying', 'meeting_key': 900,
     'date_start': '2026-10-10T14:00:00+00:00', 'year': 2026,
     'is_cancelled': False},
    {'session_key': 9003, 'session_type': 'Race', 'session_name': 'Sprint',
     'meeting_key': 901, 'date_start': '2026-10-04T13:00:00+00:00',
     'year': 2026, 'is_cancelled': False},
]

DRIVERS = [
    {'driver_number': 1, 'full_name': 'Max VERSTAPPEN', 'broadcast_name': 'VER',
     'name_acronym': 'VER', 'team_name': 'Red Bull Racing'},
    {'driver_number': 4, 'full_name': 'Lando NORRIS', 'broadcast_name': 'NOR',
     'name_acronym': 'NOR', 'team_name': 'McLaren'},
]


def _lap(number, lap_number, offset_seconds, driver, pit_out=False):
    return {
        'driver_number': number,
        'lap_number': lap_number,
        'date_start': f'2026-10-11T13:{offset_seconds // 60:02d}:'
                      f'{offset_seconds % 60:02d}.000000+00:00',
        'lap_duration': 90.0 + lap_number,
        'duration_sector_1': 30.0,
        'duration_sector_2': 30.0,
        'duration_sector_3': 30.0 + lap_number,
        'is_pit_out_lap': pit_out,
        'st_speed': 300.0,
    }


LAPS = []
for _lap_no in range(1, 4):
    LAPS.append(_lap(1, _lap_no, _lap_no * 2, 1,
                     pit_out=(_lap_no == 3)))          # Verstappen pits on lap 2
    LAPS.append(_lap(4, _lap_no, _lap_no * 2 + 1, 4))  # Norris, no stop

STINTS = [
    {'driver_number': 1, 'stint_number': 1, 'compound': 'soft', 'lap_start': 1,
     'lap_end': 2, 'tyre_age_at_start': 0},
    {'driver_number': 1, 'stint_number': 2, 'compound': 'medium', 'lap_start': 3,
     'lap_end': 3, 'tyre_age_at_start': 0},
    {'driver_number': 4, 'stint_number': 1, 'compound': 'medium', 'lap_start': 1,
     'lap_end': 3, 'tyre_age_at_start': 0},
]

PIT = [
    {'driver_number': 1, 'lap_number': 2, 'date': '2026-10-11T13:04:10.000000+00:00',
     'pit_duration': 24.0, 'stop_duration': 24.0, 'lane_duration': 26.0},
]

RESULTS = [
    {'position': 1, 'driver_number': 1, 'number_of_laps': 3, 'points': 25.0,
     'dnf': False, 'dns': False, 'dsq': False, 'gap_to_leader': 0,
     'session_key': SESSION_KEY},
    {'position': 2, 'driver_number': 4, 'number_of_laps': 3, 'points': 18.0,
     'dnf': False, 'dns': False, 'dsq': False, 'gap_to_leader': 3.2,
     'session_key': SESSION_KEY},
]

CALENDAR = [
    {'round': 17, 'name': 'Singapore Grand Prix',
     'date': pd.Timestamp('2026-10-11', tz='UTC').to_pydatetime().replace(
         tzinfo=None)},
    {'round': 16, 'name': 'United States Grand Prix',
     'date': pd.Timestamp('2026-10-04', tz='UTC').to_pydatetime().replace(
         tzinfo=None)},
]


def _fake_cached_openf1(table, ttl=None, **params):
    """Stand-in for ``cached_openf1`` with the real payload shapes."""
    if table == 'sessions':
        return SESSIONS
    if table == 'drivers':
        return DRIVERS
    if table == 'laps':
        return LAPS
    if table == 'stints':
        return STINTS
    if table == 'pit':
        return PIT
    if table == 'session_result':
        return RESULTS
    return []


@pytest.fixture(autouse=True)
def _patched(monkeypatch):
    monkeypatch.setattr(ofs, 'cached_openf1', _fake_cached_openf1)
    monkeypatch.setattr(openf1_results, '_calendar_events',
                        lambda year: list(CALENDAR))
    monkeypatch.setattr(ofs, '_jolpica_positions', lambda year, round_number: ({}, set()))


# --- Tests ------------------------------------------------------------------

def test_find_session_maps_the_session_type():
    """'race' finds the Race session, 'sprint' the Sprint session."""
    race = ofs.find_session(2026, 'Singapore Grand Prix', 'race')
    assert race is not None
    assert race['session_key'] == SESSION_KEY

    sprint = ofs.find_session(2026, 'Singapore Grand Prix', 'sprint')
    assert sprint is None  # no sprint at this Grand Prix

    other = ofs.find_session(2026, 'United States Grand Prix', 'sprint')
    assert other is not None and other['session_key'] == 9003


def test_find_session_tolerates_a_naming_difference():
    """'Singapore GP' still finds the 'Singapore Grand Prix' session."""
    session = ofs.find_session(2026, 'Singapore GP', 'race')
    assert session is not None and session['session_key'] == SESSION_KEY


def test_build_laps_has_the_columns_the_app_reads():
    frame = ofs.build_laps(SESSION_KEY, year=2026, round_number=17)
    assert not frame.empty
    for column in ('Driver', 'DriverNumber', 'Team', 'LapNumber', 'LapTime',
                   'Position', 'Compound', 'Stint', 'TyreLife', 'IsInLap',
                   'IsOutLap', 'PitInTime', 'PitOutTime'):
        assert column in frame.columns, column


def test_build_laps_reads_names_teams_and_lap_times():
    frame = ofs.build_laps(SESSION_KEY, year=2026, round_number=17)
    verstappen = frame[frame['Driver'] == 'Max Verstappen']
    assert not verstappen.empty
    assert verstappen['Team'].iloc[0] == 'Red Bull Racing'
    assert verstappen['LapTime'].iloc[0] == pd.Timedelta(seconds=91.0)
    assert verstappen['Sector1Time'].iloc[0] == pd.Timedelta(seconds=30.0)
    assert frame[frame['Driver'] == 'Lando Norris']['Team'].iloc[0] == 'McLaren'


def test_build_laps_maps_tyres_stints_and_tyre_age():
    frame = ofs.build_laps(SESSION_KEY, year=2026, round_number=17)
    verstappen = frame[frame['Driver'] == 'Max Verstappen'].sort_values('LapNumber')
    assert list(verstappen['Compound']) == ['SOFT', 'SOFT', 'MEDIUM']
    assert list(verstappen['Stint']) == [1, 1, 2]
    # The tyre age counts the laps driven on the tyre, starting at one.
    assert list(verstappen['TyreLife']) == [1, 2, 1]


def test_build_laps_marks_the_pit_laps():
    frame = ofs.build_laps(SESSION_KEY, year=2026, round_number=17)
    verstappen = frame[frame['Driver'] == 'Max Verstappen'].sort_values('LapNumber')
    assert list(verstappen['IsInLap']) == [False, True, False]
    assert list(verstappen['IsOutLap']) == [False, False, True]
    norris = frame[frame['Driver'] == 'Lando Norris']
    assert not norris['IsInLap'].any()
    assert not norris['IsOutLap'].any()


def test_pit_times_are_elapsed_times_not_timestamps():
    """FastF1 stores these as elapsed times; the app measures stops with them."""
    frame = ofs.build_laps(SESSION_KEY, year=2026, round_number=17)
    pit_in = frame['PitInTime'].dropna()
    assert len(pit_in) == 1
    value = pit_in.iloc[0]
    assert isinstance(value, pd.Timedelta)
    # The stop happens about four minutes into the session.
    assert 200 <= value.total_seconds() <= 300


def test_a_pit_stop_can_be_measured_from_the_fallback_laps():
    """The stop time is the lane time minus the driving time, and it is sane."""
    from fastf1_extended import get_pit_stops

    frame = ofs.build_laps(SESSION_KEY, year=2026, round_number=17)
    stops = get_pit_stops(type('S', (), {'laps': frame})())
    assert len(stops) == 1
    stop = stops.iloc[0]
    assert stop['Driver'] == 'Max Verstappen'
    assert int(stop['Lap']) == 2
    assert bool(stop['Measured']) is True
    assert 15 <= float(stop['PitTime']) <= 60


def test_the_running_order_is_labelled_as_an_estimate():
    """Without an official lap-by-lap order the order is derived, and says so."""
    frame = ofs.build_laps(SESSION_KEY, year=2026, round_number=17)
    assert frame.attrs['position_source'] == 'estimated'
    assert frame['Position'].notna().all()


def test_the_official_order_is_only_used_for_the_race():
    """The public lap feed holds the race laps, so a sprint must not use it."""
    calls = []

    def _positions(year, round_number):
        calls.append((year, round_number))
        return {('max_verstappen', 1): 1, ('lando_norris', 1): 2}, {
            'max_verstappen', 'lando_norris'}

    ofs._jolpica_positions = _positions
    try:
        race = ofs.build_session(2026, 'Singapore Grand Prix', 'race')
        assert race is not None
        assert calls, 'the race may ask for the official order'
        sprint = ofs.build_session(2026, 'United States Grand Prix', 'sprint')
    finally:
        ofs._jolpica_positions = lambda year, round_number: ({}, set())
    assert sprint is not None
    assert sprint.laps.attrs['position_source'] == 'estimated'


def test_the_track_status_is_not_faked():
    """The feed publishes no safety-car flags, so the column is left out."""
    frame = ofs.build_laps(SESSION_KEY, year=2026, round_number=17)
    assert 'TrackStatus' not in frame.columns
    assert frame.attrs['track_status'] == 'unavailable'


def test_build_results_maps_the_classification():
    results = ofs.build_results(SESSION_KEY)
    assert list(results['Position']) == [1, 2]
    assert list(results['Driver']) == ['Max Verstappen', 'Lando Norris']
    assert list(results['Abbreviation']) == ['VER', 'NOR']
    assert list(results['Team']) == ['Red Bull Racing', 'McLaren']
    assert list(results['Laps']) == [3, 3]
    assert list(results['Points']) == [25.0, 18.0]
    assert list(results['Status']) == ['Finished', 'Finished']


def test_build_results_marks_a_retirement():
    rows = [dict(RESULTS[0]), dict(RESULTS[1], dnf=True)]
    original = _fake_cached_openf1

    def _table(table, ttl=None, **params):
        if table == 'session_result':
            return rows
        return original(table, ttl=ttl, **params)

    ofs.cached_openf1 = _table
    try:
        results = ofs.build_results(SESSION_KEY)
    finally:
        ofs.cached_openf1 = original
    assert results['Status'].iloc[1] == 'DNF'


def test_build_session_returns_a_usable_session():
    session = ofs.build_session(2026, 'Singapore Grand Prix', 'race')
    assert session is not None
    assert bool(session) is True
    assert session.source == 'OpenF1'
    assert session.session_key == SESSION_KEY
    assert session.track_status_available is False
    assert session.position_source == 'estimated'
    assert not session.laps.empty
    assert not session.results.empty


def test_build_session_is_none_when_the_weekend_has_not_run():
    """A future race has no laps yet: the app must show an empty state."""
    original = _fake_cached_openf1

    def _table(table, ttl=None, **params):
        if table == 'laps':
            return []
        return original(table, ttl=ttl, **params)

    ofs.cached_openf1 = _table
    try:
        session = ofs.build_session(2026, 'Singapore Grand Prix', 'race')
    finally:
        ofs.cached_openf1 = original
    assert session is None
