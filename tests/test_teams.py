# -*- coding: utf-8 -*-
"""Behaviour tests for teams.py - pit stops, reliability and team totals.

All tests are offline: the OpenF1 reads are replaced with small synthetic
frames so the arithmetic (median, reliability rate, confidence labels) is what
is actually being tested.
"""
import pandas as pd

import teams


# ---------------------------------------------------------------------------
# pit stops
# ---------------------------------------------------------------------------

def test_clean_duration_rejects_impossible_values():
    assert teams._clean_duration(2.4) == 2.4
    assert teams._clean_duration('3.1') == 3.1
    assert teams._clean_duration(0.2) is None      # a data artefact
    assert teams._clean_duration(-5) is None
    assert teams._clean_duration(999) is None      # not a stop
    assert teams._clean_duration(None) is None
    assert teams._clean_duration('n/a') is None


def _fake_stops():
    return pd.DataFrame([
        {'team': 'McLaren', 'duration_s': 2.4},
        {'team': 'McLaren', 'duration_s': 2.6},
        {'team': 'McLaren', 'duration_s': 3.0},
        {'team': 'Ferrari', 'duration_s': 2.2},
        {'team': 'Ferrari', 'duration_s': 2.9},
    ])


def test_pit_stop_league_medians_and_order(monkeypatch):
    monkeypatch.setattr(teams, 'pit_stops', lambda year, max_races=None: _fake_stops())
    league = teams.pit_stop_league(2026)
    assert list(league['team']) == ['Ferrari', 'McLaren']   # 2.55 < 2.6
    ferrari = league.iloc[0]
    assert ferrari['stops'] == 2
    assert ferrari['median_s'] == 2.55
    assert ferrari['best_s'] == 2.2
    assert ferrari['worst_s'] == 2.9


def test_pit_stop_league_carries_a_confidence_label(monkeypatch):
    monkeypatch.setattr(teams, 'pit_stops', lambda year, max_races=None: _fake_stops())
    league = teams.pit_stop_league(2026)
    # Two and three stops clear the minimum sample but stay at the 'low' level.
    assert set(league['level']) == {'low'}
    assert set(league['confidence']) == {'low'}


def test_a_single_stop_is_reported_as_insufficient(monkeypatch):
    stops = pd.DataFrame([{'team': 'Alpine', 'duration_s': 2.5}])
    monkeypatch.setattr(teams, 'pit_stops', lambda year, max_races=None: stops)
    league = teams.pit_stop_league(2026)
    assert league.iloc[0]['level'] == 'insufficient'


def test_pit_stop_league_without_data_returns_typed_empty_frame(monkeypatch):
    monkeypatch.setattr(teams, 'pit_stops',
                        lambda year, max_races=None: pd.DataFrame())
    league = teams.pit_stop_league(2026)
    assert league.empty
    assert 'median_s' in league.columns


def test_pit_stop_league_skips_rows_without_a_duration(monkeypatch):
    stops = _fake_stops()
    stops.loc[len(stops)] = {'team': 'McLaren', 'duration_s': float('nan')}
    monkeypatch.setattr(teams, 'pit_stops', lambda year, max_races=None: stops)
    league = teams.pit_stop_league(2026)
    mclaren = league[league['team'] == 'McLaren'].iloc[0]
    assert mclaren['stops'] == 3      # the NaN row is not counted


def test_pit_stops_aggregates_across_sessions(monkeypatch):
    sessions = [{'session_key': 1, 'meeting_key': 10, 'round_number': 1,
                 'location': 'Melbourne'},
                {'session_key': 2, 'meeting_key': 20, 'round_number': 2,
                 'location': 'Shanghai'}]
    monkeypatch.setattr(teams.openf1_results, 'race_sessions', lambda year: sessions)
    monkeypatch.setattr(teams, '_driver_teams',
                        lambda meeting_key: {1: {'driver': 'A', 'team': 'Alpine'}})
    monkeypatch.setattr(teams, '_rows',
                        lambda endpoint, **params: [
                            {'driver_number': 1, 'lap_number': 12,
                             'pit_duration': 22.5 if params['session_key'] == 1 else 0.5},
                        ])
    frame = teams.pit_stops(2026)
    # The 0.5 s row is an artefact and must not appear as a stop.
    assert len(frame) == 1
    assert frame.iloc[0]['duration_s'] == 22.5
    assert frame.iloc[0]['team'] == 'Alpine'


# ---------------------------------------------------------------------------
# reliability
# ---------------------------------------------------------------------------

def test_reliability_counts_starts_and_retirements(monkeypatch):
    sessions = [{'session_key': 1, 'meeting_key': 10},
                {'session_key': 2, 'meeting_key': 20}]
    monkeypatch.setattr(teams.openf1_results, 'race_sessions', lambda year: sessions)
    monkeypatch.setattr(teams, '_driver_teams',
                        lambda meeting_key: {7: {'driver': 'D', 'team': 'Sauber'}})
    results = {1: [{'driver_number': 7, 'position': 5, 'dnf': False}],
               2: [{'driver_number': 7, 'position': None, 'dnf': True}]}
    monkeypatch.setattr(teams.openf1_results, 'session_result',
                        lambda session_key: results[session_key])

    frame = teams.reliability(2026)
    row = frame.iloc[0]
    assert row['team'] == 'Sauber'
    assert row['starts'] == 2
    assert row['finishes'] == 1
    assert row['retirements'] == 1
    assert row['finish_rate'] == 50.0


def test_reliability_marks_a_dsq_as_a_retirement(monkeypatch):
    sessions = [{'session_key': 1, 'meeting_key': 10}]
    monkeypatch.setattr(teams.openf1_results, 'race_sessions', lambda year: sessions)
    monkeypatch.setattr(teams, '_driver_teams',
                        lambda meeting_key: {3: {'driver': 'C', 'team': 'Haas'}})
    monkeypatch.setattr(teams.openf1_results, 'session_result',
                        lambda session_key: [{'driver_number': 3, 'position': 12,
                                              'dnf': False, 'dsq': True}])
    row = teams.reliability(2026).iloc[0]
    assert row['retirements'] == 1


def test_reliability_without_data_is_empty(monkeypatch):
    monkeypatch.setattr(teams.openf1_results, 'race_sessions', lambda year: [])
    frame = teams.reliability(2026)
    assert frame.empty and 'finish_rate' in frame.columns


# ---------------------------------------------------------------------------
# season summary
# ---------------------------------------------------------------------------

def test_team_season_summary_counts_sprints_in_points_only(monkeypatch):
    frame = pd.DataFrame([
        {'Team': 'McLaren', 'Points': 25, 'Position': 1, 'SessionType': 'Race'},
        {'Team': 'McLaren', 'Points': 8, 'Position': 1, 'SessionType': 'Sprint'},
        {'Team': 'McLaren', 'Points': 18, 'Position': 2, 'SessionType': 'Race'},
        {'Team': 'Ferrari', 'Points': 15, 'Position': 3, 'SessionType': 'Race'},
    ])
    monkeypatch.setattr(teams.openf1_results, 'build_season_results', lambda year: frame)
    summary = teams.team_season_summary(2026)
    mclaren = summary[summary['team'] == 'McLaren'].iloc[0]
    assert mclaren['points'] == 51          # sprint counted
    assert mclaren['wins'] == 1             # only the Grand Prix win
    assert mclaren['podiums'] == 2
    assert mclaren['best_finish'] == 1


def test_team_season_summary_without_results(monkeypatch):
    monkeypatch.setattr(teams.openf1_results, 'build_season_results', lambda year: None)
    assert teams.team_season_summary(2026).empty
