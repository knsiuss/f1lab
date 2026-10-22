# -*- coding: utf-8 -*-
"""Tests for the prediction and fantasy model in src/prediction.py."""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import prediction
from src.prediction import (
    ESTIMATED,
    MODEL_CAVEAT,
    add_predicted_points,
    add_price,
    compute_driver_form,
    compute_driver_prices,
    compute_team_pace,
    fantasy_points,
    forecast_vs_actual,
    optimal_lineup,
    predict_race,
    rows_before_race,
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


def test_team_pace_all_nan_team_falls_back_to_neutral():
    # A team with only non-numeric positions averages to NaN: it must be
    # dropped (neutral 0.5 fallback) rather than poisoning the softmax.
    df = pd.DataFrame([
        {'Track': 'R1', 'Position': 1, 'Driver': 'NOR', 'Team': 'McLaren',
         'Starting Grid': 1, 'Points': 25},
        {'Track': 'R1', 'Position': 'DNF', 'Driver': 'VER', 'Team': 'Red Bull',
         'Starting Grid': 2, 'Points': 0},
    ])
    pace = compute_team_pace(df)
    assert 'Red Bull' not in pace
    pdf = predict_race(df, {'NOR': 1, 'VER': 2})
    assert pdf['Score'].notna().all()
    assert pdf['WinProb'].notna().all()


# ---------------------------------------------------------------------------
# The new season frame: rounds, sprints and mid-season team changes
# ---------------------------------------------------------------------------

def _openf1_frame():
    """Three rounds, each with a race and a sprint row per driver."""
    rows = []
    for rnd, track in enumerate(['Australian Grand Prix',
                                 'Chinese Grand Prix',
                                 'Japanese Grand Prix'], start=1):
        for session in ('Race', 'Sprint'):
            for pos, driver in enumerate(['VER', 'NOR'], start=1):
                rows.append({
                    'Track': track, 'Round': rnd, 'SessionType': session,
                    'Driver': driver, 'Team': 'Red Bull' if driver == 'VER' else 'McLaren',
                    'Position': pos, 'Starting Grid': pos, 'Points': 25 if pos == 1 else 18,
                })
    return pd.DataFrame(rows)


def test_rows_before_race_excludes_the_target_and_everything_after():
    """The cut is on the round: the target race and its sprint both go."""
    df = _openf1_frame()
    before = rows_before_race(df, 'Chinese Grand Prix')
    assert set(before['Track']) == {'Australian Grand Prix'}
    assert 'Chinese Grand Prix' not in set(before['Track'])
    assert 'Japanese Grand Prix' not in set(before['Track'])


def test_rows_before_race_keeps_sprint_rows_by_default():
    """All session types of the earlier rounds are history, not look-ahead."""
    df = _openf1_frame()
    before = rows_before_race(df, 'Chinese Grand Prix')
    assert set(before['SessionType']) == {'Race', 'Sprint'}


def test_rows_before_race_can_filter_to_races():
    df = _openf1_frame()
    before = rows_before_race(df, 'Chinese Grand Prix', session_type='Race')
    assert set(before['SessionType']) == {'Race'}


def test_rows_before_race_unknown_track_returns_empty_frame():
    """An unknown track must not fall back to the whole season."""
    df = _openf1_frame()
    before = rows_before_race(df, 'Monaco Grand Prix')
    assert before.empty
    assert len(before.columns) == len(df.columns)


def test_rows_before_race_empty_input_is_safe():
    assert rows_before_race(pd.DataFrame(), 'R1').empty
    assert rows_before_race(None, 'R1').empty


def test_driver_form_ignores_sprint_rows():
    """A sprint result must never stand in for a race result."""
    rows = []
    for rnd, track in [(1, 'R1'), (2, 'R2')]:
        rows.append({'Track': track, 'Round': rnd, 'SessionType': 'Race',
                     'Driver': 'VER', 'Team': 'RB', 'Position': 1,
                     'Starting Grid': 1, 'Points': 25})
        rows.append({'Track': track, 'Round': rnd, 'SessionType': 'Sprint',
                     'Driver': 'VER', 'Team': 'RB', 'Position': 20,
                     'Starting Grid': 20, 'Points': 0})
        rows.append({'Track': track, 'Round': rnd, 'SessionType': 'Race',
                     'Driver': 'NOR', 'Team': 'McL', 'Position': 2,
                     'Starting Grid': 2, 'Points': 18})
        rows.append({'Track': track, 'Round': rnd, 'SessionType': 'Sprint',
                     'Driver': 'NOR', 'Team': 'McL', 'Position': 1,
                     'Starting Grid': 1, 'Points': 25})
    df = pd.DataFrame(rows)
    form = compute_driver_form(df)
    # Race rows only: VER always P1, NOR always P2 -> VER is the better form.
    # With the sprint rows counted, VER's sprint P20s would drag it below NOR.
    assert form['VER'] == pytest.approx(1.0)
    assert form['NOR'] == pytest.approx(0.0)


def test_driver_form_follows_the_round_order():
    """Rows out of calendar order are still rolled over in round order."""
    rows = []
    for rnd, track in [(1, 'R1'), (2, 'R2'), (3, 'R3'), (4, 'R4')]:
        rows.append({'Track': track, 'Round': rnd, 'SessionType': 'Race',
                     'Driver': 'VER', 'Team': 'RB', 'Position': 1 + 4 * (rnd - 1),
                     'Starting Grid': 1, 'Points': 25})
        rows.append({'Track': track, 'Round': rnd, 'SessionType': 'Race',
                     'Driver': 'NOR', 'Team': 'McL', 'Position': 7,
                     'Starting Grid': 7, 'Points': 15})
    # Shuffle the frame: the calendar order must come from 'Round'.
    df = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)
    form = compute_driver_form(df)
    # In round order VER's last three races are P5, P9, P13 (mean 9) against
    # NOR's flat P7 -> VER is the worse form.
    assert form['VER'] == pytest.approx(0.0)
    assert form['NOR'] == pytest.approx(1.0)


def test_team_change_uses_the_latest_round():
    """A mid-season team change must not be attributed to the old team."""
    rows = [
        {'Track': 'R1', 'Round': 1, 'SessionType': 'Race', 'Driver': 'VER',
         'Team': 'Red Bull', 'Position': 1, 'Starting Grid': 1, 'Points': 25},
        {'Track': 'R2', 'Round': 2, 'SessionType': 'Race', 'Driver': 'VER',
         'Team': 'Ferrari', 'Position': 2, 'Starting Grid': 2, 'Points': 18},
    ]
    df = pd.DataFrame(rows)
    pdf = predict_race(df, {'VER': 1})
    assert pdf.iloc[0]['Team'] == 'Ferrari'


def test_predict_race_carries_honesty_labels():
    """The forecast is labelled estimated and carries a confidence level."""
    pdf = predict_race(_openf1_frame(), {'VER': 1, 'NOR': 2})
    assert (pdf['Estimated'] == ESTIMATED).all()
    assert 'Level' in pdf.columns and 'Confidence' in pdf.columns
    assert pdf.iloc[0]['Level'] == 'low'      # 3 races of history
    assert pdf.iloc[0]['Confidence'] == 'low'


def test_predict_race_publishes_the_model_caveat():
    """The model's own weakness against the grid baseline is visible."""
    pdf = predict_race(_openf1_frame(), {'VER': 1, 'NOR': 2})
    assert 'grid' in pdf.attrs['caveat'].lower()
    assert 'not' in pdf.attrs['caveat'].lower()
    assert MODEL_CAVEAT == pdf.attrs['caveat']


def test_predict_race_confidence_grows_with_history():
    """More race history behind the forecast -> a better confidence label."""
    small = predict_race(_openf1_frame(), {'VER': 1, 'NOR': 2})
    big_rows = []
    for rnd in range(1, 17):
        big_rows.append({'Track': f'R{rnd}', 'Round': rnd, 'SessionType': 'Race',
                         'Driver': 'VER', 'Team': 'RB', 'Position': 1,
                         'Starting Grid': 1, 'Points': 25})
    big = predict_race(pd.DataFrame(big_rows), {'VER': 1})
    assert small.iloc[0]['Level'] == 'low'
    assert big.iloc[0]['Level'] == 'good'
    assert big.iloc[0]['Confidence'] == 'high'


# ---------------------------------------------------------------------------
# Fantasy prices: no fabricated price for a driver with no points
# ---------------------------------------------------------------------------

def test_zero_point_driver_has_no_market_price():
    """A driver without points must not be priced at the floor."""
    df = pd.DataFrame([
        {'Driver': 'A', 'Points': 0},
        {'Driver': 'B', 'Points': 100},
    ])
    prices = compute_driver_prices(df)
    assert 'A' not in prices
    assert prices['B'] == PRICE_MAX


def test_prices_empty_when_nobody_scored():
    """No points at all -> no price table, not a table of invented $5M."""
    df = pd.DataFrame([{'Driver': 'A', 'Points': 0}])
    assert compute_driver_prices(df) == {}


def test_unpriced_driver_is_not_picked_by_the_lineup():
    """The greedy builder must not pick a driver with no market price."""
    df = pd.DataFrame([
        {'Driver': 'A', 'Points': 0},
        {'Driver': 'B', 'Points': 100},
    ])
    prices = compute_driver_prices(df)
    pdf = add_predicted_points(predict_race(df, {'A': 1, 'B': 2}))
    pdf = add_price(pdf, prices)
    assert not bool(pdf.loc[pdf['Driver'] == 'A', 'HasPrice'].iloc[0])
    chosen = optimal_lineup(pdf, budget=50, count=2)
    assert 'A' not in chosen

# ---------------------------------------------------------------------------
# Ridge regression (numpy only)
# ---------------------------------------------------------------------------

def test_ridge_fit_recovers_a_linear_relation():
    """With no penalty the fit must reproduce an exactly linear target.

    The coefficients are reported in the z-scored space the design matrix is
    built in, so the check de-standardises them before comparing with the
    relation the data was generated from.
    """
    rng = np.random.default_rng(7)
    x = rng.normal(size=(60, 2))
    y = 3.0 + 2.0 * x[:, 0] - 1.0 * x[:, 1]
    model = prediction.ridge_fit(x, y, lam=0.0)
    assert model is not None
    assert prediction.ridge_predict(model, x) == pytest.approx(y, abs=1e-8)
    raw_beta = np.asarray(model['beta']) / np.asarray(model['std'])
    assert raw_beta == pytest.approx([2.0, -1.0], abs=1e-8)
    raw_intercept = (float(model['intercept'])
                     - float(np.dot(np.asarray(model['beta']),
                                    np.asarray(model['mean'])
                                    / np.asarray(model['std']))))
    assert raw_intercept == pytest.approx(3.0, abs=1e-8)


def test_ridge_fit_shrinks_the_coefficients_with_the_penalty():
    """A larger lambda must pull the coefficients towards zero."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=(40, 3))
    y = 1.0 + x[:, 0] + 0.5 * x[:, 1] - 2.0 * x[:, 2] + rng.normal(size=40)
    light = prediction.ridge_fit(x, y, lam=0.1)
    heavy = prediction.ridge_fit(x, y, lam=1000.0)
    assert np.linalg.norm(heavy['beta']) < np.linalg.norm(light['beta'])


def test_ridge_fit_imputes_missing_cells_with_the_training_median():
    """A missing feature is filled with the column median and counted."""
    x = np.array([[1.0, 10.0], [2.0, np.nan], [3.0, 30.0], [4.0, 40.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    model = prediction.ridge_fit(x, y)
    assert model['n_imputed'] == 1
    assert model['medians'][1] == pytest.approx(30.0)  # median of 10, 30, 40
    predicted = prediction.ridge_predict(model, np.array([[2.5, np.nan]]))
    assert np.isfinite(predicted).all()


def test_ridge_fit_drops_rows_without_a_target():
    """A row with no finishing position cannot teach the model anything."""
    model = prediction.ridge_fit(np.arange(6.0).reshape(3, 2),
                                 np.array([1.0, np.nan, 3.0]))
    assert model['n'] == 2


def test_ridge_fit_returns_none_when_it_cannot_fit_honestly():
    assert prediction.ridge_fit([], []) is None
    assert prediction.ridge_fit(np.zeros((2, 2)), np.zeros(3)) is None
    assert prediction.ridge_predict(None, [[1.0, 2.0]]).size == 0


# ---------------------------------------------------------------------------
# Walk-forward features: no look-ahead
# ---------------------------------------------------------------------------

def _ridge_season(rounds=6):
    """A season with a Round column and a rotating finishing order."""
    drivers = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
    rows = []
    for rnd in range(1, rounds + 1):
        for i, driver in enumerate(drivers):
            rows.append({
                'Round': rnd, 'Track': f'Round {rnd}', 'SessionType': 'Race',
                'Driver': driver,
                'Team': 'Alpha' if i < 3 else 'Beta',
                'Starting Grid': i + 1,
                'Position': (i + rnd) % len(drivers) + 1,
                'Points': 25 - 5 * i,
                'Finished': 1,
            })
    return pd.DataFrame(rows)


def _grid(drivers=('A1', 'A2', 'A3', 'A4', 'A5', 'A6')):
    return {driver: i + 1 for i, driver in enumerate(drivers)}


def test_predict_race_switches_to_ridge_once_there_is_history():
    """Enough earlier rounds means a real fit, not the documented fallback."""
    season = _ridge_season(6)
    out = predict_race(season, _grid())
    assert out.attrs['engine'] == 'ridge'
    assert out.attrs['n_train'] == 30  # five earlier rounds of six drivers
    assert set(out.attrs['coefficients']) == set(prediction.FEATURES)
    assert set(out['Engine']) == {'ridge'}


def test_predict_race_falls_back_without_history():
    """With no history the documented heuristic scores the grid order."""
    out = predict_race(pd.DataFrame(), _grid(('A1', 'A2', 'A3')))
    assert out.attrs['engine'] == 'heuristic'
    assert out.attrs['coefficients'] == {}
    assert out.attrs['n_train'] == 0
    assert list(out['Grid']) == [1, 2, 3]


def test_training_rows_only_see_the_rounds_before_them():
    """A race's own result never enters the features of that same race."""
    season = _ridge_season(5)
    tampered = season.copy()
    mask = tampered['Track'] == 'Round 3'
    # Reverse the finishing order of round 3: the backmarker wins.
    tampered.loc[mask, 'Position'] = (tampered.loc[mask, 'Position']
                                      .map(lambda pos: 7 - pos))
    before = prediction._training_table(season)
    after = prediction._training_table(tampered)
    columns = list(prediction.FEATURES)
    # The rows of rounds 2 and 3 are built from rounds 1-2 only, so the wild
    # round 3 result cannot change them.
    for rnd in (2, 3):
        lhs = before[before['_round'] == rnd][columns].reset_index(drop=True)
        rhs = after[after['_round'] == rnd][columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(lhs, rhs)
    # Round 4 IS built on round 3, so its features do move.
    lhs = before[before['_round'] == 4][columns].reset_index(drop=True)
    rhs = after[after['_round'] == 4][columns].reset_index(drop=True)
    assert not lhs.equals(rhs)


def test_driver_gain_and_dnf_rate():
    """Positions gained and the finish-reliability rate read the same rows."""
    frame = pd.DataFrame([
        {'Track': 'R1', 'Driver': 'A', 'Starting Grid': 5, 'Position': 3,
         'Finished': 1},
        {'Track': 'R1', 'Driver': 'B', 'Starting Grid': 1, 'Position': 2,
         'Finished': 0},
        {'Track': 'R2', 'Driver': 'A', 'Starting Grid': 2, 'Position': 1,
         'Finished': 1},
    ])
    gain = prediction.compute_driver_gain(frame)
    assert gain['A'] == pytest.approx(1.5)   # (5-3 + 2-1) / 2
    assert gain['B'] == pytest.approx(-1.0)
    dnf = prediction.compute_dnf_rate(frame)
    assert dnf['A'] == pytest.approx(0.0)
    assert dnf['B'] == pytest.approx(1.0)


def test_feature_report_is_honest_about_what_is_missing():
    """The published inventory says which features the data does not carry."""
    report = prediction.feature_report()
    assert len(report) == 8
    assert len([entry for entry in report if entry['used']]) == 5
    missing = [str(entry['feature']) for entry in report
               if not entry['used']]
    assert 'Qualifying gap' in missing
    assert 'Circuit history' in missing
    assert 'Wet or dry' in missing
    for entry in report:
        assert entry['feature'] and entry['source'] and entry['note']


# ---------------------------------------------------------------------------
# The caveat quotes the measurement, never a guess
# ---------------------------------------------------------------------------

def test_model_caveat_quotes_the_numbers_it_was_given():
    summary = {'n_races': 10, 'mae_model': 2.7648, 'mae_baseline': 2.8693,
               'mae_model_beats_baseline': 6}
    text = prediction.model_caveat(summary)
    assert '2.765' in text
    assert '2.869' in text
    assert '6 of 10' in text
    assert 'does beat the baseline' in text
    assert 'Ridge regression' in text


def test_model_caveat_says_plainly_when_the_model_loses():
    summary = {'n_races': 20, 'mae_model': 3.1, 'mae_baseline': 2.8,
               'mae_model_beats_baseline': 4}
    text = prediction.model_caveat(summary)
    assert 'does NOT beat the baseline' in text
    assert '4 of 20' in text


def test_model_caveat_without_a_backtest_invents_no_number():
    """The published constant must not claim a measurement it has none of."""
    text = prediction.model_caveat()
    assert 'Ridge regression' in text
    assert not re.search(r'\d+\.\d+', text)
