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
    calculate_driver_stats, calculate_team_stats,
    calculate_combined_constructor_standings,
    calculate_teammate_comparison
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


class TestTeamStats:
    """Tests for calculate_team_stats function."""
    
    def test_calculates_team_points(self, sample_race_data):
        """Test team total points calculation."""
        stats = calculate_team_stats(sample_race_data)
        
        assert 'Total_Points' in stats.columns
        assert stats.loc['Team X', 'Total_Points'] == 43
        assert stats.loc['Team Y', 'Total_Points'] == 43



class TestCombinedConstructorStandings:
    """Tests for calculate_combined_constructor_standings function."""

    @pytest.fixture
    def race_df(self):
        return pd.DataFrame({
            'Driver': ['NOR', 'VER', 'PIA', 'PER'],
            'Team': ['McLaren', 'Red Bull', 'McLaren', 'Red Bull'],
            'Track': ['Race 1', 'Race 1', 'Race 2', 'Race 2'],
            'Position': [1, 2, 1, 2],
            'Points': [25, 18, 25, 18],
            'Finished': [True, True, True, True]
        })

    @pytest.fixture
    def sprint_df(self):
        return pd.DataFrame({
            'Driver': ['NOR', 'VER'],
            'Team': ['McLaren', 'Red Bull'],
            'Track': ['Race 1', 'Race 1'],
            'Position': [1, 2],
            'Points': [8, 4],
            'Finished': [True, True]
        })

    def test_combines_team_points(self, race_df, sprint_df):
        """Test that constructor standings sum team points."""
        standings = calculate_combined_constructor_standings(race_df, sprint_df)

        assert 'Total_Points' in standings.columns
        # McLaren: 25+25 race + 8 sprint = 58
        # Red Bull: 18+18 race + 4 sprint = 40
        assert standings.loc['McLaren', 'Total_Points'] == 58
        assert standings.loc['Red Bull', 'Total_Points'] == 40

    def test_includes_position_ranking(self, race_df, sprint_df):
        """Test position ranking is assigned."""
        standings = calculate_combined_constructor_standings(race_df, sprint_df)

        assert standings.loc['McLaren', 'Position'] == 1
        assert standings.loc['Red Bull', 'Position'] == 2


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
