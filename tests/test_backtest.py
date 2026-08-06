# -*- coding: utf-8 -*-
"""Tests for walk-forward prediction backtesting in src/backtest.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest import (
    ESTIMATED,
    METRICS,
    evaluate_race,
    exact_match_rate,
    explain_prediction,
    grid_baseline,
    mean_abs_error,
    podium_overlap,
    points_mae,
    race_actual,
    race_grid,
    spearman_rank,
    walk_forward_backtest,
)
from src.prediction import predict_race

# Fixed grid order reused across every synthetic race.
GRID = ['A1', 'B1', 'A2', 'C1', 'B2']


def _season(races=4, grid=GRID):
    """Synthetic season where the finish order equals the grid order."""
    rows = []
    for r in range(1, races + 1):
        for pos, d in enumerate(grid, start=1):
            rows.append({
                'Track': f'R{r}', 'Position': pos, 'Driver': d,
                'Team': d[0], 'Starting Grid': pos,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Race helpers
# ---------------------------------------------------------------------------

def test_race_grid_maps_starting_position():
    race = _season(races=1)
    assert race_grid(race) == {d: i + 1 for i, d in enumerate(GRID)}


def test_race_actual_maps_finish_position():
    race = _season(races=1)
    assert race_actual(race) == {d: i + 1 for i, d in enumerate(GRID)}


def test_grid_baseline_is_pole_order():
    grid = {'VER': 3, 'NOR': 1, 'LEC': 2}
    assert grid_baseline(grid) == ['NOR', 'LEC', 'VER']


def test_race_grid_empty_on_missing_column():
    assert race_grid(pd.DataFrame({'Driver': ['A']})) == {}
    assert race_actual(pd.DataFrame({'Driver': ['A']})) == {}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_exact_match_rate():
    pred = {'A': 1, 'B': 2, 'C': 3}
    actual = {'A': 1, 'B': 3, 'C': 2}
    assert exact_match_rate(pred, actual) == pytest.approx(1 / 3, abs=0.001)


def test_mean_abs_error():
    pred = {'A': 1, 'B': 2, 'C': 3}
    actual = {'A': 1, 'B': 3, 'C': 2}
    assert mean_abs_error(pred, actual) == pytest.approx((0 + 1 + 1) / 3, abs=0.001)


def test_spearman_perfect_and_inverted():
    actual = {'A': 1, 'B': 2, 'C': 3}
    assert spearman_rank(actual, actual) == pytest.approx(1.0)
    inv = {'A': 3, 'B': 2, 'C': 1}
    assert spearman_rank(inv, actual) == pytest.approx(-1.0)


def test_podium_overlap():
    pred = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    actual = {'A': 2, 'B': 3, 'C': 1, 'D': 4}
    # predicted top-3 {A,B,C} == actual top-3 {A,B,C} -> full overlap
    assert podium_overlap(pred, actual) == pytest.approx(1.0)
    actual2 = {'A': 1, 'B': 2, 'C': 4, 'D': 3}
    # actual top-3 {A,B,D}; only A,B match the predicted top-3
    assert podium_overlap(pred, actual2) == pytest.approx(2 / 3, abs=0.001)


def test_points_mae():
    pred = {'A': 1, 'B': 2, 'C': 3}
    actual = {'A': 1, 'B': 3, 'C': 2}
    # pts: pred 25/18/15 vs actual 25/15/18 -> errs 0, 3, 3
    assert points_mae(pred, actual) == pytest.approx(2.0)


def test_metrics_insufficient_drivers():
    pred = {'A': 1}
    actual = {'A': 1}
    assert exact_match_rate(pred, actual) is None
    assert mean_abs_error(pred, actual) is None
    assert spearman_rank(pred, actual) is None


# ---------------------------------------------------------------------------
# evaluate_race
# ---------------------------------------------------------------------------

def test_evaluate_race_structure():
    race = _season(races=1)
    grid = race_grid(race)
    pred = predict_race(None, grid)
    actual = race_actual(race)
    model_m, base_m, n = evaluate_race(pred, actual)
    assert set(model_m) == set(METRICS)
    assert set(base_m) == set(METRICS)
    assert n == len(GRID)
    assert all(v is not None for v in model_m.values())


def test_evaluate_race_baseline_perfect_on_grid_equals_finish():
    race = _season(races=1)
    pred = predict_race(None, race_grid(race))
    actual = race_actual(race)
    _, base_m, _ = evaluate_race(pred, actual)
    # The grid baseline predicts the grid, which equals the finish here.
    assert base_m['exact_match_rate'] == 1.0
    assert base_m['mae'] == 0.0
    assert base_m['spearman'] == 1.0
    assert base_m['points_mae'] == 0.0


def test_evaluate_race_none_when_no_actual():
    race = _season(races=1)
    pred = predict_race(None, race_grid(race))
    assert evaluate_race(pred, {}) is None


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def test_walk_forward_backtests_later_races_only():
    per_race, summary = walk_forward_backtest(_season(races=4), min_train_races=2)
    # R1 has no history, R2 has only one prior race (below min) -> R3, R4 scored.
    assert list(per_race['Race']) == ['R3', 'R4']
    assert summary['n_races'] == 2
    assert summary['n_races_skipped'] == 0
    assert (per_race['Estimated'] == ESTIMATED).all()


def test_walk_forward_insufficient_history_returns_empty():
    per_race, summary = walk_forward_backtest(_season(races=2), min_train_races=5)
    assert per_race.empty
    assert summary['n_races'] == 0
    assert summary['n_races_skipped'] == 0


def test_walk_forward_skips_race_without_grid():
    season = _season(races=4)
    season.loc[season['Track'] == 'R2', 'Starting Grid'] = float('nan')
    per_race, summary = walk_forward_backtest(season, min_train_races=1)
    # R1 has no history; R2 has no grid (skipped); R3, R4 scored.
    assert list(per_race['Race']) == ['R3', 'R4']
    assert summary['n_races'] == 2
    assert summary['n_races_skipped'] == 1


def test_walk_forward_baseline_perfect_in_no_overtaking_season():
    per_race, summary = walk_forward_backtest(_season(races=4), min_train_races=1)
    assert (per_race['mae_baseline'] == 0.0).all()
    assert (per_race['exact_match_rate_baseline'] == 1.0).all()
    assert summary['mae_baseline'] == 0.0
    assert summary['exact_match_rate_baseline'] == 1.0


def test_walk_forward_empty_inputs():
    per_race, summary = walk_forward_backtest(pd.DataFrame())
    assert per_race.empty
    assert summary['n_races'] == 0


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

def test_explain_prediction_breaks_down_score():
    grid = {'A1': 1, 'B1': 2, 'A2': 3, 'C1': 4, 'B2': 5}
    pred = predict_race(None, grid)
    lines = explain_prediction(pred.iloc[0], field=len(grid))
    joined = '\n'.join(lines)
    assert 'predicted P1' in joined
    assert 'x 40%' in joined  # grid weight
    assert 'x 35%' in joined  # team weight
    assert 'x 25%' in joined  # form weight
    assert 'Score' in joined
