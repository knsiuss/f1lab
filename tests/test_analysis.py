# -*- coding: utf-8 -*-
"""
test_analysis.py
~~~~~~~~~~~~~~~~
Unit tests for the analysis module.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import (
    calculate_driver_stats,
    calculate_teammate_comparison,
    compare_driver_pace,
    count_lead_changes,
    driver_performance_scores,
    gap_to_leader,
    latest_race_rows,
    points_progression,
    top3_overlap,
)


@pytest.fixture
def sample_race_data():
    """Create sample race data for testing."""
    return pd.DataFrame({
        'Driver': ['Driver A', 'Driver A', 'Driver B', 'Driver B'],
        'Team': ['Team X', 'Team X', 'Team Y', 'Team Y'],
        'Track': ['Race 1', 'Race 2', 'Race 1', 'Race 2'],
        'Position': [1, 2, 2, 1],
        'Points': [25, 18, 18, 25],
        'Starting Grid': [1, 3, 2, 1],
        'Finished': [True, True, True, True],
        'Set Fastest Lap': ['Yes', 'No', 'No', 'Yes']
    })


class TestDriverStats:
    """Tests for calculate_driver_stats function."""
    
    def test_calculates_total_points(self, sample_race_data):
        """Test total points calculation."""
        stats = calculate_driver_stats(sample_race_data)
        
        assert 'Total_Points' in stats.columns
        assert stats.loc['Driver A', 'Total_Points'] == 43
        assert stats.loc['Driver B', 'Total_Points'] == 43
    
    def test_calculates_wins(self, sample_race_data):
        """Test wins calculation."""
        stats = calculate_driver_stats(sample_race_data)
        
        assert 'Wins' in stats.columns
        assert stats.loc['Driver A', 'Wins'] == 1
        assert stats.loc['Driver B', 'Wins'] == 1
    
    def test_calculates_podiums(self, sample_race_data):
        """Test podium count calculation."""
        stats = calculate_driver_stats(sample_race_data)
        
        assert 'Podium' in stats.columns
        # Both positions are podiums (1 and 2)
        assert stats.loc['Driver A', 'Podium'] == 2
        assert stats.loc['Driver B', 'Podium'] == 2




class TestTeammateComparison:
    """Tests for calculate_teammate_comparison function."""

    @pytest.fixture
    def teammate_data(self):
        return pd.DataFrame({
            'Driver': ['NOR', 'PIA', 'VER', 'PER'],
            'Team': ['McLaren', 'McLaren', 'Red Bull', 'Red Bull'],
            'Track': ['Race 1', 'Race 1', 'Race 1', 'Race 1'],
            'Position': [1, 3, 2, 4],
            'Starting Grid': [2, 4, 1, 3],
            'Points': [25, 15, 18, 12],
            'Finished': [True, True, True, True]
        })

    def test_returns_comparison_dataframe(self, teammate_data):
        """Test that teammate comparison returns DataFrame."""
        result = calculate_teammate_comparison(teammate_data)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_compares_correct_drivers(self, teammate_data):
        """Test that correct teammate pairs are compared."""
        result = calculate_teammate_comparison(teammate_data)

        teams = result['Team'].unique()
        assert 'McLaren' in teams
        assert 'Red Bull' in teams

    def test_race_head_to_head(self, teammate_data):
        """Test race H2H scoring."""
        result = calculate_teammate_comparison(teammate_data)

        mcl_row = result[result['Team'] == 'McLaren'].iloc[0]
        # NOR P1, PIA P3 -> NOR wins race H2H
        assert '1 - 0' in mcl_row['Race H2H'] or '0 - 1' in mcl_row['Race H2H']

    def test_quali_head_to_head(self, teammate_data):
        """Test qualifying H2H scoring."""
        result = calculate_teammate_comparison(teammate_data)

        rb_row = result[result['Team'] == 'Red Bull'].iloc[0]
        # VER grid 1, PER grid 3 -> VER wins quali H2H
        assert '1 - 0' in rb_row['Quali H2H'] or '0 - 1' in rb_row['Quali H2H']

    def test_returns_empty_for_no_teammates(self):
        """Test that single-driver data returns empty DataFrame."""
        data = pd.DataFrame({
            'Driver': ['NOR'],
            'Team': ['McLaren'],
            'Track': ['Race 1'],
            'Position': [1],
            'Starting Grid': [1],
            'Points': [25],
            'Finished': [True]
        })
        result = calculate_teammate_comparison(data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_ignores_sprint_rows(self, teammate_data):
        """A sprint result is not a race result: it must not be compared."""
        frame = teammate_data.copy()
        frame['SessionType'] = 'Race'
        sprint = frame.copy()
        sprint['SessionType'] = 'Sprint'
        sprint['Position'] = [4, 2, 3, 1]  # a different finishing order
        combined = pd.concat([frame, sprint], ignore_index=True)

        result = calculate_teammate_comparison(combined)

        mcl_row = result[result['Team'] == 'McLaren'].iloc[0]
        # Race only: NOR P1, PIA P3 -> NOR wins 1-0. With the sprint counted
        # as well the record would be 1-1.
        assert mcl_row['Race H2H'] == '1 - 0'
        assert mcl_row['Races Together'] == 1


class TestDriverStatsPositions:
    """A driver without a classified position has no position statistics."""

    @pytest.fixture
    def retired_driver_data(self):
        return pd.DataFrame({
            'Driver': ['A', 'A', 'B'],
            'Team': ['X', 'X', 'Y'],
            'Track': ['R1', 'R2', 'R1'],
            'Position': [1, 2, 5],
            'Points': [25, 18, 10],
            'Starting Grid': [1, 2, 5],
            'Finished': [True, True, True],
        })

    def test_missing_positions_are_not_filled_with_zero(self, retired_driver_data):
        """A driver with one missing position still has a real average."""
        frame = retired_driver_data.copy()
        frame.loc[1, 'Position'] = None

        stats = calculate_driver_stats(frame)

        row = stats.loc['A']
        assert row['Avg_Position'] == pytest.approx(1.0)
        # The count-like columns are still filled.
        assert row['Races'] == 2
        assert row['Total_Points'] == 43

    def test_all_positions_missing_gives_nan_not_zero(self):
        """A driver who never classified gets NaN positions, not 0.0."""
        frame = pd.DataFrame({
            'Driver': ['A', 'B'],
            'Team': ['X', 'Y'],
            'Track': ['R1', 'R1'],
            'Position': [None, 5.0],
            'Points': [0, 10],
            'Starting Grid': [1, 5],
            'Finished': [False, True],
        })

        stats = calculate_driver_stats(frame)

        row = stats.loc['A']
        assert pd.isna(row['Avg_Position'])
        assert pd.isna(row['Best_Position'])
        assert pd.isna(row['Worst_Position'])
        assert row['Races'] == 1


class TestFastestLapsAvailability:
    """An empty fastest-lap column means "not loaded", not "nobody set one"."""

    @staticmethod
    def _frame(flags):
        return pd.DataFrame({
            'Driver': ['A', 'B'],
            'Team': ['X', 'Y'],
            'Track': ['R1', 'R1'],
            'Position': [1.0, 2.0],
            'Points': [25, 18],
            'Starting Grid': [1, 2],
            'Finished': [True, True],
            'Set Fastest Lap': flags,
        })

    def test_unavailable_when_the_column_is_empty(self):
        stats = calculate_driver_stats(self._frame(['', '']))
        assert stats.iloc[0]['Fastest_Laps'] is None
        assert stats.iloc[1]['Fastest_Laps'] is None

    def test_counted_when_the_data_is_loaded(self):
        stats = calculate_driver_stats(self._frame(['Yes', 'No']))
        assert stats.loc['A', 'Fastest_Laps'] == 1
        assert stats.loc['B', 'Fastest_Laps'] == 0

    def test_unavailable_when_the_column_is_absent(self):
        frame = self._frame(['Yes', 'No']).drop(columns=['Set Fastest Lap'])
        stats = calculate_driver_stats(frame)
        assert stats.iloc[0]['Fastest_Laps'] is None


class TestTop3Overlap:
    """'Shared in top 3' must compare the predicted and the actual podium."""

    def test_counts_the_shared_drivers(self):
        overlap = top3_overlap(['A', 'B', 'C', 'D'], {'A': 1, 'B': 5, 'C': 2, 'D': 3})
        assert overlap == 2   # A and C

    def test_a_predicted_driver_without_a_result_is_not_counted(self):
        """Regression: dropna() before head(3) pulled the 4th driver in."""
        # A was predicted first but has no recorded finish; D was predicted
        # fourth and actually finished second.
        overlap = top3_overlap(['A', 'B', 'C', 'D'], {'B': 4, 'C': 5, 'D': 2})
        assert overlap == 0

    def test_no_actual_result(self):
        assert top3_overlap(['A', 'B', 'C'], {}) == 0


# ---------------------------------------------------------------------------
# Season frame helpers
# ---------------------------------------------------------------------------

def _season_frame():
    """Two rounds, each with a race and a sprint for the same event."""
    rows = [
        # Round 1 -- Australian Grand Prix
        ('Australian Grand Prix', 1, 'Race', 1, 'NOR', 1, 25),
        ('Australian Grand Prix', 2, 'Race', 1, 'VER', 2, 18),
        ('Australian Grand Prix', 1, 'Sprint', 1, 'VER', 1, 8),
        ('Australian Grand Prix', 2, 'Sprint', 1, 'NOR', 2, 7),
        # Round 2 -- Chinese Grand Prix
        ('Chinese Grand Prix', 1, 'Race', 2, 'VER', 1, 25),
        ('Chinese Grand Prix', 2, 'Race', 2, 'NOR', 2, 18),
        ('Chinese Grand Prix', 1, 'Sprint', 2, 'NOR', 1, 8),
    ]
    return pd.DataFrame(
        rows,
        columns=['Track', 'Position', 'SessionType', 'Round', 'Driver',
                 'Starting Grid', 'Points'],
    )


class TestLatestRaceRows:
    """The 'recent results' panel must show one round, not the whole season."""

    def test_returns_only_the_last_round(self):
        frame = _season_frame()
        latest = latest_race_rows(frame)
        assert set(latest['Track']) == {'Chinese Grand Prix'}
        assert set(latest['SessionType']) == {'Race'}
        assert len(latest) == 2

    def test_sprint_of_the_same_round_is_excluded(self):
        frame = _season_frame()
        latest = latest_race_rows(frame)
        assert 'Sprint' not in set(latest['SessionType'])

    def test_empty_frame(self):
        assert latest_race_rows(pd.DataFrame()).empty


class TestPointsProgression:
    """Championship progression: one step per round, sprint points included."""

    def test_sprint_points_count_towards_the_round_total(self):
        frame = _season_frame()
        prog = points_progression(frame, drivers=['NOR'])

        nor = prog[prog['Driver'] == 'NOR']
        assert len(nor) == 2
        # Round 1: 25 (race) + 7 (sprint) = 32
        assert nor.iloc[0]['Points'] == 32
        # Round 2: 32 + 18 (race) + 8 (sprint) = 58
        assert nor.iloc[1]['Points'] == 58
        assert list(nor['RaceNum']) == [1, 2]

    def test_one_row_per_round_per_driver(self):
        frame = _season_frame()
        prog = points_progression(frame)
        assert not prog.duplicated(subset=['Driver', 'Track']).any()


class TestLeadChanges:
    """Lead changes must be counted in lap order, not driver order."""

    @staticmethod
    def _laps(order):
        """Lap table in driver-grouped order (as FastF1 returns it)."""
        rows = []
        for lap in range(1, len(order) + 1):
            driver = order[lap - 1]
            rows.append({'Driver': driver, 'LapNumber': lap,
                         'Position': 1, 'LapTime': pd.Timedelta(seconds=90)})
        frame = pd.DataFrame(rows)
        # FastF1 groups the rows by driver.
        return frame.sort_values(['Driver', 'LapNumber']).reset_index(drop=True)

    def test_leader_taking_the_lead_back_counts_twice(self):
        laps = self._laps(['VER', 'VER', 'NOR', 'VER'])
        assert count_lead_changes(laps) == 2

    def test_single_leader_has_no_changes(self):
        laps = self._laps(['VER', 'VER', 'VER'])
        assert count_lead_changes(laps) == 0

    def test_one_handover(self):
        laps = self._laps(['VER', 'VER', 'NOR', 'NOR'])
        assert count_lead_changes(laps) == 1

    def test_no_position_data(self):
        assert count_lead_changes(pd.DataFrame()) == 0


class TestGapToLeader:
    """The reference driver must be one who completed the laps."""

    @staticmethod
    def _laps():
        """NOR completes 4 laps, VER retires after 2 having been quickest."""
        rows = []
        for lap in range(1, 5):
            rows.append({'Driver': 'NOR', 'LapNumber': lap,
                         'LapTime': pd.Timedelta(seconds=90),
                         'Position': 1})
        for lap in range(1, 3):
            rows.append({'Driver': 'VER', 'LapNumber': lap,
                         'LapTime': pd.Timedelta(seconds=80),
                         'Position': 2})
        return pd.DataFrame(rows)

    def test_leader_is_not_the_fastest_cumulative_time(self):
        """VER's 2 laps are quicker in total, but VER is not the leader."""
        leader, frame = gap_to_leader(self._laps(), ['NOR', 'VER'])
        assert leader == 'NOR'

    def test_gap_is_not_reset_to_zero_after_the_reference_stops(self):
        leader, frame = gap_to_leader(self._laps(), ['NOR', 'VER'])
        ver = frame[frame['Driver'] == 'VER'].sort_values('LapNumber')
        assert len(ver) == 2
        # VER is 10s per lap quicker than NOR: the gap grows, it does not
        # collapse to zero when the reference driver is out of the picture.
        assert ver.iloc[0]['Gap'] == pytest.approx(-10.0, abs=0.01)
        assert ver.iloc[1]['Gap'] == pytest.approx(-20.0, abs=0.01)


class TestDriverPerformanceScores:
    """Scores are estimates built on clean laps, with a confidence level."""

    @staticmethod
    def _laps(n=20, base=90.0):
        return pd.DataFrame({
            'Driver': ['VER'] * n,
            'LapNumber': list(range(1, n + 1)),
            'LapTime': [pd.Timedelta(seconds=base + 0.05 * i) for i in range(n)],
            'Position': [1] * n,
            'TyreLife': list(range(1, n + 1)),
            'Stint': [1] * n,
            'Compound': ['SOFT'] * n,
            'IsOutLap': [False] * n,
            'IsInLap': [False] * n,
            'IsAccurate': [True] * n,
            'TrackStatus': [1] * n,
        })

    def test_scores_carry_estimated_and_confidence(self):
        scores = driver_performance_scores(self._laps(n=20))
        assert len(scores) == 1
        row = scores.iloc[0]
        assert row['Estimated'] == 'estimated'
        assert row['Level'] == 'good'
        assert row['Confidence'] == 'high'
        assert row['N'] == 20

    def test_out_laps_do_not_change_the_score(self):
        """A pit lap must not drag the consistency score down."""
        clean_laps = self._laps(n=20)
        dirty = clean_laps.copy()
        dirty.loc[5, 'LapTime'] = pd.Timedelta(seconds=200)
        dirty.loc[5, 'IsOutLap'] = True

        a = driver_performance_scores(clean_laps).iloc[0]
        b = driver_performance_scores(dirty).iloc[0]
        assert a['Overall'] == pytest.approx(b['Overall'])

    def test_insufficient_sample_gets_no_score(self):
        scores = driver_performance_scores(self._laps(n=1))
        assert scores.empty


class TestCompareDriverPace:
    """Head-to-head pace must use clean laps only."""

    @staticmethod
    def _laps():
        rows = []
        for driver, base in (('VER', 90.0), ('NOR', 91.0)):
            for lap in range(1, 6):
                rows.append({'Driver': driver, 'LapNumber': lap,
                             'LapTime': pd.Timedelta(seconds=base + lap),
                             'Position': 1, 'TyreLife': lap, 'Stint': 1,
                             'Compound': 'SOFT', 'IsOutLap': False,
                             'IsInLap': False, 'IsAccurate': True,
                             'TrackStatus': 1})
        return pd.DataFrame(rows)

    def test_uses_clean_laps(self):
        laps = self._laps()
        laps.loc[0, 'LapTime'] = pd.Timedelta(seconds=10)  # a partial pit lap
        laps.loc[0, 'IsInLap'] = True

        result = compare_driver_pace(laps, 'VER', 'NOR')
        assert result['n_a'] == 4
        assert result['n_b'] == 5
        # Clean laps only: VER 92-95 (mean 93.5), NOR 92-96 (mean 94.0).
        assert result['diff'] == pytest.approx(-0.5, abs=0.01)

    def test_confidence_follows_the_weaker_sample(self):
        result = compare_driver_pace(self._laps(), 'VER', 'NOR')
        assert result['level'] == 'low'
        assert result['confidence'] == 'low'
        assert result['estimated'] == 'estimated'

    def test_missing_driver_is_insufficient(self):
        result = compare_driver_pace(self._laps(), 'VER', 'HAM')
        assert result['level'] == 'insufficient'
        assert result['avg_b'] is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
