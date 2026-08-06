# -*- coding: utf-8 -*-
"""Tests for the prediction and fantasy model in src/prediction.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prediction import (
    add_predicted_points,
    add_price,
    compute_driver_form,
    compute_driver_prices,
    compute_team_pace,
    fantasy_points,
    forecast_vs_actual,
    optimal_lineup,
    predict_race,
    PRICE_MIN,
    PRICE_MAX,
)


@pytest.fixture
def season_df():
    """Two-team season so team pace is clearly separated."""
    rows = [
        ('R1', 1, 'NOR', 'McLaren', 1, 25),
        ('R1', 2, 'PIA', 'McLaren', 2, 18),
        ('R1', 3, 'VER', 'Red Bull', 3, 15),
        ('R1', 4, 'LEC', 'Red Bull', 4, 12),
        ('R2', 1, 'NOR', 'McLaren', 1, 25),
        ('R2', 3, 'PIA', 'McLaren', 3, 15),
        ('R2', 2, 'VER', 'Red Bull', 2, 18),
        ('R2', 5, 'LEC', 'Red Bull', 5, 10),
    ]
    df = pd.DataFrame(
        rows, columns=['Track', 'Position', 'Driver', 'Team', 'Starting Grid', 'Points']
    )
    return df


def test_team_pace_best_team_is_one(season_df):
    pace = compute_team_pace(season_df)
    assert pace['McLaren'] == pytest.approx(1.0)
    assert pace['Red Bull'] == pytest.approx(0.0)


def test_driver_form_bounds(season_df):
    form = compute_driver_form(season_df)
    assert form['NOR'] == pytest.approx(1.0)
    assert 0.0 <= form['LEC'] <= 1.0


def test_predict_race_order(season_df):
    grid = {'LEC': 1, 'VER': 2, 'NOR': 3, 'PIA': 4}
    pdf = predict_race(season_df, grid)
    assert list(pdf['Predicted']) == [1, 2, 3, 4]
    assert pdf.iloc[0]['Driver'] == 'NOR'  # strong team + form overcomes the grid
    assert pdf['WinProb'].sum() == pytest.approx(100.0, abs=0.5)


def test_predict_race_empty_grid(season_df):
    assert predict_race(season_df, {}).empty


def test_fantasy_points_scoring():
    assert fantasy_points(1) == 25
    assert fantasy_points(10) == 1
    assert fantasy_points(11) == 0
    assert fantasy_points(20) == 0


def test_driver_prices_bounds(season_df):
    prices = compute_driver_prices(season_df)
    for d, p in prices.items():
        assert PRICE_MIN <= p <= PRICE_MAX


def test_optimal_lineup_respects_budget(season_df):
    prices = compute_driver_prices(season_df)
    grid = {'LEC': 1, 'VER': 2, 'NOR': 3, 'PIA': 4}
    pdf = add_price(add_predicted_points(predict_race(season_df, grid)), prices)
    chosen = optimal_lineup(pdf, budget=50, count=2)
    assert len(chosen) <= 2
    # budget must hold for the selected 2 drivers
    cost = sum(pdf.loc[pdf['Driver'] == d, 'Price'].iloc[0] for d in chosen)
    assert cost <= 50


def test_optimal_lineup_empty():
    assert optimal_lineup(pd.DataFrame()) == []


def test_forecast_vs_actual_columns(season_df):
    grid = {'NOR': 1, 'VER': 2, 'PIA': 3, 'LEC': 4}
    pdf = predict_race(season_df, grid)
    joined = forecast_vs_actual(pdf, {'NOR': 1, 'VER': 2, 'PIA': 3, 'LEC': 4})
    for col in ['Actual', 'PredictedPts', 'ActualPts']:
        assert col in joined.columns