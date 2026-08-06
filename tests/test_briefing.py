# -*- coding: utf-8 -*-
"""Tests for the data-driven race briefing/debrief in src/briefing.py."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.briefing import (
    championship_leader_md,
    form_guide_md,
    post_race_debrief,
    pre_race_briefing,
)

_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10}


def _season():
    """Five-car, four-race season. R3 has real movement vs the grid."""
    races = {
        'R1': (['A1', 'A2', 'B1', 'B2', 'C1'], ['A1', 'A2', 'B1', 'B2', 'C1']),
        'R2': (['A1', 'A2', 'B1', 'B2', 'C1'], ['A1', 'A2', 'B1', 'B2', 'C1']),
        # B1 starts P4 and finishes P2 (gainer); C1 starts P2, finishes P5.
        'R3': (['A1', 'C1', 'A2', 'B1', 'B2'], ['A1', 'B1', 'A2', 'B2', 'C1']),
        'R4': (['A1', 'C1', 'A2', 'B1', 'B2'], ['A1', 'B1', 'A2', 'B2', 'C1']),
    }
    rows = []
    for track, (grid_order, finish_order) in races.items():
        grid_pos = {d: i + 1 for i, d in enumerate(grid_order)}
        for pos, d in enumerate(finish_order, start=1):
            rows.append({
                'Track': track, 'Position': pos, 'Driver': d,
                'Team': d[0], 'Starting Grid': grid_pos[d],
                'Points': _POINTS[pos],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Championship context
# ---------------------------------------------------------------------------

def test_championship_leader_md_full_season():
    md = championship_leader_md(_season())
    # A1 wins all four races -> 100 pts; A2 and B1 tie on 66.
    assert 'A1' in md and '100' in md and '34' in md


def test_championship_leader_md_up_to_race_excludes_future():
    md = championship_leader_md(_season(), up_to_race='R3')
    # Standings after R2: A1 50, A2 36 -> gap 14.
    assert '50' in md and '14' in md


def test_championship_leader_md_empty_data():
    assert championship_leader_md(pd.DataFrame()) == ""


# ---------------------------------------------------------------------------
# Pre-race briefing
# ---------------------------------------------------------------------------

def test_form_guide_md_lists_leaders():
    md = form_guide_md(_season())
    assert 'Current form leaders' in md
    assert 'A1' in md
    assert 'estimated' in md


def test_pre_race_briefing_contains_all_sections():
    grid = {'A1': 1, 'C1': 2, 'A2': 3, 'B1': 4, 'B2': 5}
    md = pre_race_briefing(_season(), grid, race='R5')
    assert 'Pre-race briefing — R5' in md
    assert 'leads the championship' in md
    assert 'Current form leaders' in md
    assert 'Model favourites' in md
    assert 'estimated' in md


def test_pre_race_briefing_empty_grid():
    # Without a grid there is no forecast, but season context still shows.
    md = pre_race_briefing(_season(), {})
    assert 'leads the championship' in md
    assert 'Model favourites' not in md


# ---------------------------------------------------------------------------
# Post-race debrief
# ---------------------------------------------------------------------------

def test_debrief_winner_podium_and_movement():
    md = post_race_debrief(_season(), 'R3')
    assert 'wins at' in md
    assert 'P1 **A1**' in md and 'P2 **B1**' in md and 'P3 **A2**' in md
    assert 'gained **2** places' in md and '(P4 → P2)' in md
    assert 'lost **3** places' in md and '(P2 → P5)' in md
    assert 'Championship leader unchanged: **A1**' in md


def test_debrief_forecast_podium_match():
    forecast = pd.DataFrame({'Driver': ['A1', 'B1', 'A2', 'B2', 'C1']})
    md = post_race_debrief(_season(), 'R3', forecast=forecast)
    assert 'matched **3/3** of the podium' in md


def test_debrief_forecast_miss_reported_honestly():
    # A wrong forecast still reports the match count rather than hiding it.
    forecast = pd.DataFrame({'Driver': ['B2', 'C1', 'B1', 'A2', 'A1']})
    md = post_race_debrief(_season(), 'R3', forecast=forecast)
    assert 'matched' in md and 'of the podium' in md
    assert '/3' in md


def test_debrief_championship_lead_change():
    rows = [
        # R1: A1 wins, B1 second.
        {'Track': 'R1', 'Position': 1, 'Driver': 'A1', 'Team': 'A',
         'Starting Grid': 1, 'Points': 25},
        {'Track': 'R1', 'Position': 2, 'Driver': 'B1', 'Team': 'B',
         'Starting Grid': 2, 'Points': 18},
        # R2: B1 wins, A1 only third -> B1 overtakes.
        {'Track': 'R2', 'Position': 1, 'Driver': 'B1', 'Team': 'B',
         'Starting Grid': 2, 'Points': 25},
        {'Track': 'R2', 'Position': 2, 'Driver': 'C1', 'Team': 'C',
         'Starting Grid': 3, 'Points': 18},
        {'Track': 'R2', 'Position': 3, 'Driver': 'A1', 'Team': 'A',
         'Starting Grid': 1, 'Points': 15},
    ]
    md = post_race_debrief(pd.DataFrame(rows), 'R2')
    assert 'Championship lead changes hands: **A1** → **B1**' in md


def test_debrief_no_result():
    assert 'No recorded result' in post_race_debrief(_season(), 'Monaco')
