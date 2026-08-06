# -*- coding: utf-8 -*-
"""Tests for season-wide aggregation helpers in src/analysis.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import (
    calculate_form_trend,
    calculate_matchup,
    calculate_points_trajectory,
)


@pytest.fixture
def season_df():
    """Minimal race-results DataFrame in the shape produced by loader.clean_data."""
    rows = [
        # Track, Position, Driver, Team, Starting Grid, Points
        ('Australia', 1, 'NOR', 'McLaren', 1, 25),
        ('Australia', 2, 'VER', 'Red Bull', 3, 18),
        ('China', 2, 'NOR', 'McLaren', 2, 18),
        ('China', 1, 'VER', 'Red Bull', 1, 25),
        ('Japan', 1, 'NOR', 'McLaren', 1, 25),
        ('Japan', 3, 'VER', 'Red Bull', 4, 15),
        ('Bahrain', 2, 'NOR', 'McLaren', 2, 18),
        ('Bahrain', 1, 'VER', 'Red Bull', 1, 25),
    ]
    df = pd.DataFrame(
        rows, columns=['Track', 'Position', 'Driver', 'Team', 'Starting Grid', 'Points']
    )
    df['Finished'] = True
    return df


def test_matchup_race_record(season_df):
    result = calculate_matchup(season_df, 'NOR', 'VER')
    assert result is not None
    # positions: NOR 1,2,1,2 vs VER 2,1,3,1 -> split 2-2
    assert result['summary']['race_record'] == {'driver1': 2, 'driver2': 2}


def test_matchup_quali_record(season_df):
    result = calculate_matchup(season_df, 'NOR', 'VER')
    # grids: NOR [1,2,1,2] vs VER [3,1,4,1] -> split 2-2
    assert result['summary']['quali_record'] == {'driver1': 2, 'driver2': 2}


def test_matchup_points(season_df):
    result = calculate_matchup(season_df, 'NOR', 'VER')
    assert result['summary']['points'] == {'driver1': 86, 'driver2': 83}


def test_matchup_no_shared_races(season_df):
    solo = season_df[season_df['Driver'] == 'NOR']
    assert calculate_matchup(solo, 'NOR', 'VER') is None


def test_matchup_empty_input():
    assert calculate_matchup(pd.DataFrame(), 'NOR', 'VER') is None


def test_form_trend_shape(season_df):
    result = calculate_form_trend(season_df, ['NOR', 'VER'], window=3)
    assert 'form' in result and 'positions' in result
    form = result['form']
    assert list(form.columns) == ['NOR', 'VER']
    # Rolling window of 3 over NOR finishes [1, 2, 1, 2]
    assert form['NOR'].iloc[0] == pytest.approx(1.0)
    assert form['NOR'].iloc[-1] == pytest.approx((2 + 1 + 2) / 3, abs=0.01)


def test_points_trajectory_cumulative(season_df):
    traj = calculate_points_trajectory(season_df, ['NOR', 'VER'])
    # NOR 25, +18=43, +25=68, +18=86
    assert traj['NOR'].iloc[-1] == pytest.approx(86)
    assert traj['VER'].iloc[-1] == pytest.approx(83)
    assert traj['NOR'].iloc[1] == pytest.approx(43)