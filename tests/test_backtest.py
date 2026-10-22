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
    grid_baseline,
    mean_abs_error,
    podium_overlap,
    points_mae,
    race_actual,
    race_grid,
    spearman_rank,
    summarise,
    walk_forward_backtest,
)
from src.prediction import model_caveat, predict_race

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
# Session types: a sprint row must never override the race row
# ---------------------------------------------------------------------------

def _race_plus_sprint_frame():
    """One round with a race row and a sprint row per driver."""
    rows = []
    for session, positions in (('Race', {'A1': 1, 'B1': 2}),
                               ('Sprint', {'A1': 2, 'B1': 1})):
        for driver, pos in positions.items():
            rows.append({
                'Track': 'R1', 'Round': 1, 'SessionType': session,
                'Driver': driver, 'Team': driver[0], 'Position': pos,
                'Starting Grid': pos,
            })
    return pd.DataFrame(rows)


def test_race_grid_ignores_sprint_rows():
    """The combined frame must not let the sprint override the grid."""
    frame = _race_plus_sprint_frame()
    grid = race_grid(frame)
    assert grid == {'A1': 1, 'B1': 2}


def test_race_actual_ignores_sprint_rows():
    """The combined frame must not let the sprint override the result."""
    frame = _race_plus_sprint_frame()
    actual = race_actual(frame)
    assert actual == {'A1': 1, 'B1': 2}


def test_race_helpers_can_read_the_sprint_session_explicitly():
    """A caller that wants the sprint asks for it."""
    frame = _race_plus_sprint_frame()
    assert race_grid(frame, session_type='Sprint') == {'A1': 2, 'B1': 1}
    assert race_actual(frame, session_type='Sprint') == {'A1': 2, 'B1': 1}


def test_walk_forward_uses_race_rows_of_the_target_round():
    """A sprint row in the target round must not become the grid."""
    frame = _race_plus_sprint_frame()
    frame.loc[frame['SessionType'] == 'Sprint', 'Starting Grid'] = 9
    per_race, summary = walk_forward_backtest(frame, min_train_races=0)
    # The race grid (1, 2) is used, so the race is scored, not skipped.
    assert summary['n_races'] == 1
    assert summary['n_races_skipped'] == 0


# ---------------------------------------------------------------------------
# Podium overlap: the denominator is the number of drivers considered
# ---------------------------------------------------------------------------

def test_podium_overlap_with_two_drivers_is_full_overlap():
    """Two drivers (the module's minimum) must not report 2/3 as the score."""
    pred = {'A': 1, 'B': 2}
    actual = {'A': 1, 'B': 2}
    assert podium_overlap(pred, actual) == pytest.approx(1.0)


def test_podium_overlap_with_two_drivers_partial():
    """With two drivers both are inside the top-2, so the overlap is full."""
    pred = {'A': 1, 'B': 2}
    actual = {'A': 2, 'B': 1}
    # The order swaps but the *set* of top-2 drivers is the same, so this is
    # a full overlap -- and it is scored out of 2, not out of 3.
    assert podium_overlap(pred, actual) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Calendar order and the honest verdict
# ---------------------------------------------------------------------------

def _rounds_frame(order=None):
    """Four rounds with a 'Round' column; ``order`` shuffles the rows."""
    rows = []
    for rnd in range(1, 5):
        for pos, driver in enumerate(['A1', 'B1', 'A2', 'C1', 'B2'], start=1):
            rows.append({
                'Track': f'R{rnd}', 'Round': rnd, 'SessionType': 'Race',
                'Driver': driver, 'Team': driver[0], 'Position': pos,
                'Starting Grid': pos,
            })
    df = pd.DataFrame(rows)
    if order == 'reversed':
        df = df.iloc[::-1].reset_index(drop=True)
    return df


def test_walk_forward_follows_the_round_column():
    """The walk-forward order is the calendar order, not the row order."""
    per_race, _ = walk_forward_backtest(_rounds_frame(), min_train_races=2)
    assert list(per_race['Race']) == ['R3', 'R4']
    reversed_per_race, _ = walk_forward_backtest(
        _rounds_frame(order='reversed'), min_train_races=2)
    assert list(reversed_per_race['Race']) == ['R3', 'R4']


def test_summary_says_whether_the_model_beats_the_baseline():
    """The summary reports the comparison instead of leaving it to be found."""
    per_race, summary = walk_forward_backtest(_rounds_frame(), min_train_races=1)
    assert 'beats_baseline' in summary
    assert 'verdict' in summary
    assert isinstance(summary['beats_baseline'], bool)
    assert 'grid baseline' in summary['verdict']


def test_summary_verdict_reports_a_model_that_does_not_win():
    """A model that does not beat the grid baseline must say so."""
    per_race, summary = walk_forward_backtest(_rounds_frame(), min_train_races=1)
    if summary['mae_model'] >= summary['mae_baseline']:
        assert summary['beats_baseline'] is False
        assert ('does NOT beat' in summary['verdict']
                or 'only matches' in summary['verdict'])
    else:
        assert summary['beats_baseline'] is True
        assert 'beats the grid baseline' in summary['verdict']


# ---------------------------------------------------------------------------
# Ridge engine and the honest model-vs-baseline comparison
# ---------------------------------------------------------------------------

DRIVERS = ['A1', 'A2', 'A3', 'A4', 'A5']


def _inverted_season(rounds=8):
    """A season where the finish order is the grid order reversed.

    The grid never changes, so "the result is the grid" is badly wrong (2.4
    places out on average) and there is a stable pattern for a real model to
    learn: the car that starts further back finishes further forward.
    """
    rows = []
    n = len(DRIVERS)
    for rnd in range(1, rounds + 1):
        for i, driver in enumerate(DRIVERS):
            rows.append({
                'Round': rnd, 'Track': f'R{rnd}', 'SessionType': 'Race',
                'Driver': driver,
                'Team': 'Alpha' if i < 3 else 'Beta',
                'Starting Grid': i + 1,
                'Position': n - i,
                'Points': 25 - 5 * i,
                'Finished': 1,
            })
    return pd.DataFrame(rows)


def test_backtest_reports_which_engine_scored_each_race():
    """Every row says whether a real fit or the fallback produced it."""
    per_race, summary = walk_forward_backtest(_inverted_season(),
                                              min_train_races=1)
    assert 'Engine' in per_race.columns
    assert set(per_race['Engine']) <= {'ridge', 'heuristic'}
    assert (summary['n_engine_ridge'] + summary['n_engine_fallback']
            == summary['n_races'])
    # The earliest scored races have too little history; the regression takes
    # over once there is enough.
    assert summary['n_engine_fallback'] >= 1
    assert summary['n_engine_ridge'] >= 1
    assert list(per_race['Engine'])[-1] == 'ridge'


def test_walk_forward_model_beats_the_grid_on_a_learnable_pattern():
    """A pattern the model can learn from history must beat copying the grid."""
    per_race, summary = walk_forward_backtest(_inverted_season(),
                                              min_train_races=1)
    assert summary['beats_baseline'] is True
    assert summary['mae_model'] < summary['mae_baseline']
    assert summary['mae_model_beats_baseline'] >= 4
    assert 'beats the grid baseline' in summary['verdict']


def test_public_summarise_matches_the_walk_forward_summary():
    """``summarise`` is the same aggregation the walk-forward returns."""
    per_race, summary = walk_forward_backtest(_inverted_season(),
                                              min_train_races=1)
    assert summarise(per_race, summary['n_races_skipped']) == summary


def test_caveat_generated_from_the_measured_backtest():
    """The page shows the caveat: it must quote what the backtest measured."""
    _, summary = walk_forward_backtest(_inverted_season(), min_train_races=1)
    text = model_caveat(summary)
    assert f"{summary['mae_model']:.3f}" in text
    assert f"{summary['mae_baseline']:.3f}" in text
    assert 'does beat the baseline' in text
