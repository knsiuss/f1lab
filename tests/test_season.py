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
    championship_trajectory,
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


@pytest.fixture
def combined_df(season_df):
    """The same season with a sprint row for every race."""
    sprint = season_df.copy()
    sprint['SessionType'] = 'Sprint'
    sprint['Position'] = [5, 6, 5, 6, 5, 6, 5, 6]
    sprint['Points'] = [8, 7, 8, 7, 8, 7, 8, 7]
    race = season_df.copy()
    race['SessionType'] = 'Race'
    return pd.concat([race, sprint], ignore_index=True)


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


# ---------------------------------------------------------------------------
# Sprint rows must not stand in for race rows
# ---------------------------------------------------------------------------

def test_matchup_ignores_sprint_rows(combined_df):
    """A sprint result is not a race result: it must not be compared."""
    result = calculate_matchup(combined_df, 'NOR', 'VER')
    assert result is not None
    # Race rows only: NOR 1,2,1,2 vs VER 2,1,3,1 -> split 2-2
    assert result['summary']['race_record'] == {'driver1': 2, 'driver2': 2}
    assert result['summary']['races_together'] == 4


def test_form_trend_ignores_sprint_rows(combined_df):
    result = calculate_form_trend(combined_df, ['NOR', 'VER'], window=3)
    form = result['form']
    # One column per driver, one row per race -- no duplicate race labels.
    assert list(form.columns) == ['NOR', 'VER']
    assert len(form.index) == 4
    assert form['NOR'].iloc[-1] == pytest.approx((2 + 1 + 2) / 3, abs=0.01)


def test_points_trajectory_ignores_sprint_rows(combined_df):
    traj = calculate_points_trajectory(combined_df, ['NOR'])
    assert traj['NOR'].iloc[-1] == pytest.approx(86)
    assert len(traj.index) == 4


def test_championship_trajectory_counts_sprint_points(combined_df):
    """The championship curve counts the sprint of the same weekend."""
    traj = championship_trajectory(combined_df, ['NOR', 'VER'])
    assert len(traj.index) == 4          # one step per round
    assert traj['NOR'].iloc[0] == pytest.approx(25 + 8)   # race + sprint
    assert traj['NOR'].iloc[-1] == pytest.approx(86 + 32)
    assert traj['VER'].iloc[-1] == pytest.approx(83 + 28)


# ---------------------------------------------------------------------------
# The forecast must not see the race it predicts
# ---------------------------------------------------------------------------

def _two_round_season():
    """Round 1: NOR wins. Round 2 (the target): VER wins, NOR is fifth."""
    rows = [
        ('Australian Grand Prix', 1, 'Race', 1, 'NOR', 1, 25),
        ('Australian Grand Prix', 2, 'Race', 1, 'VER', 2, 18),
        ('Chinese Grand Prix', 1, 'Race', 2, 'VER', 1, 25),
        ('Chinese Grand Prix', 5, 'Race', 2, 'NOR', 4, 10),
    ]
    df = pd.DataFrame(
        rows, columns=['Track', 'Position', 'SessionType', 'Round', 'Driver',
                       'Starting Grid', 'Points']
    )
    df['Team'] = df['Driver'].map({'NOR': 'McLaren', 'VER': 'Red Bull'})
    return df


def test_forecast_built_from_prior_rounds_has_no_look_ahead():
    """The Predictor and Report pages forecast from the rounds before the race.

    With the whole season the model's team-pace term (35% of the score) and
    the driver's form already contain the result of the race being predicted.
    """
    from src.prediction import predict_race, rows_before_race

    df = _two_round_season()
    grid = {'VER': 1, 'NOR': 2}

    history = rows_before_race(df, 'Chinese Grand Prix')
    assert set(history['Track']) == {'Australian Grand Prix'}

    honest = predict_race(history, grid)
    leaky = predict_race(df, grid)

    # Before the race NOR is the better driver on form and team pace; the
    # leaky forecast has already seen NOR's bad day.
    assert list(honest['Driver']) == ['NOR', 'VER']
    assert list(leaky['Driver']) == ['VER', 'NOR']


def test_rows_before_race_excludes_the_sprint_of_the_same_round():
    from src.prediction import rows_before_race

    rows = [
        ('Australian Grand Prix', 1, 'Race', 1, 'NOR', 25),
        ('Australian Grand Prix', 1, 'Sprint', 1, 'NOR', 8),
        ('Chinese Grand Prix', 1, 'Race', 2, 'NOR', 25),
        ('Chinese Grand Prix', 1, 'Sprint', 2, 'NOR', 8),
    ]
    df = pd.DataFrame(
        rows, columns=['Track', 'Position', 'SessionType', 'Round', 'Driver', 'Points']
    )

    history = rows_before_race(df, 'Chinese Grand Prix')

    assert set(history['Track']) == {'Australian Grand Prix'}
    assert set(history['SessionType']) == {'Race', 'Sprint'}


def test_rows_before_race_unknown_track_is_empty():
    from src.prediction import rows_before_race

    df = _two_round_season()
    assert rows_before_race(df, 'Monaco Grand Prix').empty