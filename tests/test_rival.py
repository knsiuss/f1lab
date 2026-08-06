# -*- coding: utf-8 -*-
"""Tests for rival/competitor intelligence in src/rival.py."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rival import (
    ESTIMATED,
    build_watchlist,
    circuit_insights,
    head_to_head,
    rival_attribution_md,
)

_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10}


def _season():
    """Three races; A1 dominant, B1/C1 chasing, D1 near the back."""
    rows = {}
    plans = {
        # track: grid order, finish order
        'R1': (['A1', 'B1', 'A2', 'C1', 'D1'],
               ['A1', 'B1', 'A2', 'C1', 'D1']),
        'R2': (['B1', 'A1', 'A2', 'C1', 'D1'],
               ['A1', 'B1', 'A2', 'C1', 'D1']),
        'R3': (['A1', 'B1', 'A2', 'C1', 'D1'],
               ['A1', 'A2', 'B1', 'C1', 'D1']),
    }
    for track, (grid_order, finish_order) in plans.items():
        grid_pos = {d: i + 1 for i, d in enumerate(grid_order)}
        for pos, d in enumerate(finish_order, start=1):
            rows.setdefault(track, []).append({
                'Track': track, 'Position': pos, 'Driver': d,
                'Team': d[0], 'Starting Grid': grid_pos[d],
                'Points': _POINTS[pos], 'Set Fastest Lap': 'Yes' if d == 'A1' else 'No',
            })
    return pd.concat([pd.DataFrame(rows[t]) for t in rows], ignore_index=True)


# ---------------------------------------------------------------------------
# Head-to-head
# ---------------------------------------------------------------------------

def test_head_to_head_counts_wins():
    # A1 finishes ahead of B1 in all 3 races (B1 second in R1/R2, third in R3).
    h = head_to_head(_season(), 'A1', 'B1')
    assert h['races'] == 3
    assert h['a_wins'] == 3
    assert h['b_wins'] == 0
    assert h['level'] != 'insufficient'
    assert h['estimated'] == ESTIMATED
    assert h['points_diff'] > 0


def test_head_to_head_insufficient_data():
    one_race = _season()[_season()['Track'] == 'R1']
    h = head_to_head(one_race, 'A1', 'B1')
    assert h['level'] == 'insufficient'


def test_head_to_head_net_position():
    h = head_to_head(_season(), 'B1', 'A2')
    # B1 (driver_a) ahead in R1/R2, A2 ahead in R3 -> B1 better on average.
    assert h['a_wins'] == 2
    assert h['b_wins'] == 1
    assert h['net_pos'] < 0


def test_rival_attribution_says_insufficient():
    season = pd.DataFrame({'Track': [], 'Position': [], 'Driver': [],
                           'Team': [], 'Starting Grid': [], 'Points': []})
    md = rival_attribution_md(head_to_head(season, 'A1', 'B1'))
    assert 'insufficient' in md


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

def test_watchlist_leaders_without_focus():
    wl = build_watchlist(_season())
    assert list(wl['Driver']) == ['A1', 'B1', 'A2', 'C1', 'D1']
    assert wl.iloc[0]['Points'] == wl['Points'].max()
    assert wl.iloc[0]['GapToFocus'] == 0


def test_watchlist_around_focus_includes_nearest_rivals():
    wl = build_watchlist(_season(), focus='A1', top=3)
    assert 'A1' in set(wl['Driver'])
    assert len(wl) == 3
    # nearest rivals by points are B1 (2nd) and A2/C1 (tied around 3rd/4th)
    assert 'B1' in set(wl['Driver'])
    assert (wl['Estimated'] == ESTIMATED).all()


def test_watchlist_empty_on_no_data():
    assert build_watchlist(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# Circuit insights
# ---------------------------------------------------------------------------

def test_circuit_insights_reports_winner_and_pole_to_win():
    ins = circuit_insights(_season(), 'R1')
    assert ins['winner'] == 'A1'
    assert ins['pole'] == 'A1'
    assert ins['pole_to_win'] is True
    assert ins['fastest_lap'] == 'A1'
    assert ins['estimated'] == ESTIMATED
    assert ins['assumptions']


def test_circuit_insights_insufficient_missing_track():
    ins = circuit_insights(_season(), 'Monaco')
    assert ins['level'] == 'insufficient'