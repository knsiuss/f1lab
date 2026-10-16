# -*- coding: utf-8 -*-
"""
test_h2h.py
~~~~~~~~~~~~~~~~~
Unit tests for the FastF1-backed head-to-head module.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import h2h


def _raw_results(rounds):
    """Build a FastF1-shaped results frame: (driver, team, color, country,
    position, grid, points, gap_seconds). One call = one session."""
    rows = []
    for round_no, entries in rounds.items():
        for driver, team, color, country, pos, grid, points, gap in entries:
            rows.append({
                'RoundNumber': round_no, 'FullName': driver, 'TeamName': team,
                'TeamColor': color, 'CountryCode': country,
                'DriverNumber': str(abs(hash(driver)) % 90 + 1),
                'Position': pos, 'GridPosition': grid, 'Points': points,
                'Time': pd.to_timedelta(gap, unit='s') if gap is not None else pd.NaT,
                'Status': 'Finished' if pos is not None else 'Accident',
                'Laps': 57,
            })
    return pd.DataFrame(rows)


def _normalise(rounds, session_type='Race'):
    """Normalise a synthetic session the way the cached loader does."""
    rows = h2h._normalise_results(_raw_results(rounds), session_type)
    round_no = next(iter(rounds))
    rows['Round'] = round_no
    rows['EventName'] = f'Race {round_no}'
    rows['Track'] = rows['EventName']
    rows['Code'] = f'R{round_no}'
    return rows


@pytest.fixture
def season_frame():
    """Three rounds, NOR vs PIA teammates plus VER as an outsider."""
    return pd.concat([
        _normalise({1: [
            ('Lando Norris', 'McLaren', 'FF8000', 'GBR', 1.0, 1.0, 25.0, 0.0),
            ('Oscar Piastri', 'McLaren', 'FF8000', 'AUS', 3.0, 3.0, 15.0, 2.5),
            ('Max Verstappen', 'Red Bull Racing', '3671C6', 'NED', 2.0, 2.0, 18.0, 0.895),
        ]}),
        # R2: Piastri wins the race, but from grid 3 - so the grid-based
        # quali record stays a clean 3-0 sweep for Norris below.
        _normalise({2: [
            ('Lando Norris', 'McLaren', 'FF8000', 'GBR', 2.0, 2.0, 18.0, 1.5),
            ('Oscar Piastri', 'McLaren', 'FF8000', 'AUS', 1.0, 3.0, 25.0, 0.0),
            ('Max Verstappen', 'Red Bull Racing', '3671C6', 'NED', None, 4.0, 0.0, None),
        ]}),
        _normalise({3: [
            ('Lando Norris', 'McLaren', 'FF8000', 'GBR', 1.0, 1.0, 25.0, 0.0),
            ('Oscar Piastri', 'McLaren', 'FF8000', 'AUS', 2.0, 2.0, 18.0, 0.612),
            ('Max Verstappen', 'Red Bull Racing', '3671C6', 'NED', 3.0, 3.0, 15.0, 8.481),
        ]}),
    ], ignore_index=True)


class TestPresentation:
    """Flag, colour and short-code helpers."""

    def test_flag_for_iso_and_ioc_codes(self):
        assert h2h.country_flag('GBR') == '\U0001F1EC\U0001F1E7'
        assert h2h.country_flag('NED') == '\U0001F1F3\U0001F1F1'
        assert h2h.country_flag('AUS') == '\U0001F1E6\U0001F1FA'

    def test_flag_empty_for_bad_input(self):
        assert h2h.country_flag(None) == ''
        assert h2h.country_flag('XX') == '\U0001F1FD\U0001F1FD'
        assert h2h.country_flag('') == ''
        assert h2h.country_flag('GBR1') == ''

    def test_team_color_normalises_hash(self):
        assert h2h.team_color('FF8000') == '#FF8000'
        assert h2h.team_color('#3671C6') == '#3671C6'
        assert h2h.team_color('ff8000') == '#FF8000'

    def test_team_color_fallback_for_bad_input(self):
        assert h2h.team_color(None) == '#888888'
        assert h2h.team_color('nope') == '#888888'

    def test_teammate_gets_distinct_tint(self):
        base = '#FF8000'
        assert h2h.driver_color(base, is_teammate_of_first=False) == base
        assert h2h.driver_color(base, is_teammate_of_first=True) != base

    def test_race_code_from_location(self):
        assert h2h.race_code('Australian Grand Prix', 'Melbourne') == 'AUS'
        assert h2h.race_code('Monaco Grand Prix', 'Monte Carlo') == 'MON'
        assert h2h.race_code('Unknown GP', 'Nowhere') == 'UNK'


class TestNormaliseResults:
    """Mapping FastF1 results frames onto the shared schema."""

    def test_columns_and_winner_gap(self):
        rows = _normalise({1: [
            ('Lando Norris', 'McLaren', 'FF8000', 'GBR', 1.0, 1.0, 25.0, 0.0),
            ('Max Verstappen', 'Red Bull Racing', '3671C6', 'NED', 2.0, 2.0, 18.0, 0.895),
        ]})
        assert list(rows['Driver']) == ['Lando Norris', 'Max Verstappen']
        assert list(rows['Team']) == ['McLaren', 'Red Bull Racing']
        assert list(rows['TeamColor']) == ['#FF8000', '#3671C6']
        assert rows.loc[0, 'TimeSec'] == 0.0
        assert rows.loc[1, 'TimeSec'] == pytest.approx(0.895)

    def test_dnf_has_no_position_or_time(self):
        rows = _normalise({1: [
            ('Max Verstappen', 'Red Bull Racing', '3671C6', 'NED', None, 4.0, 0.0, None),
        ]})
        assert pd.isna(rows.loc[0, 'Position'])
        assert pd.isna(rows.loc[0, 'TimeSec'])
        assert rows.loc[0, 'Points'] == 0.0

    def test_missing_column_raises(self):
        broken = pd.DataFrame({'FullName': [], 'TeamName': []})
        with pytest.raises(KeyError):
            h2h._normalise_results(broken, 'Race')


class TestDriverSummary:
    """Season aggregation per driver."""

    def test_totals(self, season_frame):
        summary = h2h.driver_summary(season_frame, 'Lando Norris')
        assert summary['points'] == 68
        assert summary['wins'] == 2
        assert summary['podiums'] == 3
        assert summary['races'] == 3
        assert summary['best'] == 1
        assert summary['dnfs'] == 0

    def test_poles_only_counted_when_quali_loaded(self, season_frame):
        assert h2h.driver_summary(season_frame, 'Lando Norris')['poles'] == 0
        quali = pd.concat([
            _normalise({1: [('Lando Norris', 'McLaren', 'FF8000', 'GBR', 1.0, 1.0, 0.0, None)]}, 'Qualifying'),
            _normalise({2: [('Lando Norris', 'McLaren', 'FF8000', 'GBR', 2.0, 2.0, 0.0, None)]}, 'Qualifying'),
        ], ignore_index=True)
        assert h2h.driver_summary(season_frame, 'Lando Norris', quali)['poles'] == 1

    def test_dnf_counts_and_average_excludes_it(self, season_frame):
        summary = h2h.driver_summary(season_frame, 'Max Verstappen')
        assert summary['dnfs'] == 1
        assert summary['avg_finish'] == pytest.approx(2.5)
        assert summary['races'] == 3

    def test_unknown_driver_returns_zeroed_summary(self, season_frame):
        summary = h2h.driver_summary(season_frame, 'Nobody')
        assert summary['points'] == 0
        assert summary['races'] == 0


class TestHeadToHead:
    """The comparison core."""

    def test_returns_none_for_invalid_pairs(self, season_frame):
        assert h2h.head_to_head(season_frame, None, 'Lando Norris', 'Lando Norris') is None
        assert h2h.head_to_head(season_frame, None, 'Lando Norris', 'Ghost') is None
        assert h2h.head_to_head(pd.DataFrame(), None, 'A', 'B') is None

    def test_records_and_points(self, season_frame):
        result = h2h.head_to_head(season_frame, None, 'Lando Norris', 'Oscar Piastri')
        assert result['races_shared'] == 3
        assert result['race_record'] == {'wins1': 2, 'wins2': 1, 'ties': 0}
        assert result['points'] == {'1': 68, '2': 58}

    def test_ties_counted_when_both_fail_to_classify(self, season_frame):
        rows = _normalise({1: [
            ('Lando Norris', 'McLaren', 'FF8000', 'GBR', None, 1.0, 0.0, None),
            ('Oscar Piastri', 'McLaren', 'FF8000', 'AUS', None, 2.0, 0.0, None),
        ]})
        result = h2h.head_to_head(rows, None, 'Lando Norris', 'Oscar Piastri')
        assert result['race_record'] == {'wins1': 0, 'wins2': 0, 'ties': 1}

    def test_race_table_gap_and_cumulative(self, season_frame):
        result = h2h.head_to_head(season_frame, None, 'Lando Norris', 'Oscar Piastri')
        table = result['race_table']
        assert len(table) == 3
        # Gap is the absolute finish-time difference.
        assert table.loc[0, 'Gap'] == pytest.approx(2.5)
        assert table.loc[1, 'Gap'] == pytest.approx(1.5)
        assert table.loc[2, 'Gap'] == pytest.approx(0.612)
        # Cumulative points accumulate in round order.
        assert list(table['Cum1']) == [25, 43, 68]
        assert list(table['Cum2']) == [15, 40, 58]

    def test_track_delta_and_wins_by_race(self, season_frame):
        result = h2h.head_to_head(season_frame, None, 'Lando Norris', 'Oscar Piastri')
        delta = result['track_delta']
        assert list(delta['Delta']) == [10, -7, 7]
        wins = result['wins_by_race']
        assert list(wins['Wins1']) == [1, 0, 1]
        assert list(wins['Wins2']) == [0, 1, 0]

    def test_grid_based_quali_record_without_quali_frame(self, season_frame):
        result = h2h.head_to_head(season_frame, None, 'Lando Norris', 'Oscar Piastri')
        assert result['quali_record'] == {'wins1': 3, 'wins2': 0, 'ties': 0,
                                          'source': 'grid', 'compared': 3}

    def test_quali_record_prefers_real_sessions(self, season_frame):
        quali = pd.concat([
            _normalise({1: [('Lando Norris', 'McLaren', 'FF8000', 'GBR', 2.0, 2.0, 0.0, None),
                            ('Oscar Piastri', 'McLaren', 'FF8000', 'AUS', 1.0, 1.0, 0.0, None)]}, 'Qualifying'),
        ], ignore_index=True)
        result = h2h.head_to_head(season_frame, quali, 'Lando Norris', 'Oscar Piastri')
        assert result['quali_record']['source'] == 'quali'
        assert result['quali_record']['wins2'] == 1

    def test_teammate_colours_are_distinct(self, season_frame):
        result = h2h.head_to_head(season_frame, None, 'Lando Norris', 'Oscar Piastri')
        assert result['driver1']['color'] == '#FF8000'
        assert result['driver2']['color'] != result['driver1']['color']

    def test_driver_side_metadata(self, season_frame):
        result = h2h.head_to_head(season_frame, None, 'Lando Norris', 'Oscar Piastri')
        assert result['driver1']['team'] == 'McLaren'
        assert result['driver1']['flag'] == '\U0001F1EC\U0001F1E7'
        assert result['driver2']['flag'] == '\U0001F1E6\U0001F1FA'
        assert result['driver1']['podiums'] == 3
        assert result['driver1']['last'] == 'NORRIS'

    def test_summary_rows_match_records(self, season_frame):
        result = h2h.head_to_head(season_frame, None, 'Lando Norris', 'Oscar Piastri')
        rows = {r['label']: (r['display1'], r['display2']) for r in result['summary_rows']}
        assert rows['Points'] == ('68', '58')
        assert rows['Wins'] == ('2', '1')
        assert rows['Race H2H'] == ('2', '1')
        assert rows['Quali H2H'] == ('3', '0')


class TestLoadSeasonResults:
    """Season assembly, with the network layer stubbed out."""

    @pytest.fixture(autouse=True)
    def _stub_calendar(self, monkeypatch):
        monkeypatch.setattr(h2h, 'get_completed_races',
                            lambda year: ['Race A', 'Race B'])

    def test_assembles_sessions_in_round_order(self, monkeypatch):
        captured = []

        def fake_session(year, event_name, session_type):
            captured.append((year, event_name, session_type))
            round_no = 1 if event_name == 'Race A' else 2
            return _normalise(
                {round_no: [('Lando Norris', 'McLaren', 'FF8000', 'GBR', 1.0, 1.0, 25.0, 0.0)]},
                session_type)

        monkeypatch.setattr(h2h, '_cached_session_results', fake_session)
        seen = []
        races, quali = h2h.load_season_results(
            2025, progress=lambda name, i, n: seen.append((name, i, n)))

        assert len(races) == 2
        assert list(races['Round']) == [1, 2]
        assert quali.empty
        assert len(seen) == 2
        assert seen[0][1] == 0

    def test_quali_loaded_on_demand(self, monkeypatch):
        monkeypatch.setattr(h2h, '_cached_session_results',
                            lambda year, name, t: _normalise(
                                {1: [('Lando Norris', 'McLaren', 'FF8000', 'GBR', 1.0, 1.0, 25.0, 0.0)]}, t))
        races, quali = h2h.load_season_results(2025, include_quali=True)
        assert not races.empty and not quali.empty
        assert set(quali['SessionType']) == {'Qualifying'}

    def test_no_events_yields_empty_frames(self, monkeypatch):
        monkeypatch.setattr(h2h, 'get_completed_races', lambda year: [])
        races, quali = h2h.load_season_results(2030)
        assert races.empty and quali.empty


class TestTeamRosters:
    """Team grouping with points-ordered drivers."""

    def test_roster_orders_drivers_by_points(self, season_frame):
        points = h2h.season_points(season_frame)
        rosters = h2h.team_rosters(season_frame, points)
        mclaren = rosters[rosters['Team'] == 'McLaren'].iloc[0]
        assert mclaren['Drivers'][0] == 'Lando Norris'
        assert mclaren['TeamColor'] == '#FF8000'
        assert rosters.iloc[0]['Points'] >= rosters.iloc[1]['Points']

    def test_empty_frame_returns_empty_rosters(self):
        assert h2h.team_rosters(pd.DataFrame()).empty
