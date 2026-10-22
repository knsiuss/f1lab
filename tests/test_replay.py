# -*- coding: utf-8 -*-
"""Tests for the race-replay pivot logic in src/replay.py."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.replay import build_position_replay, positions_by_lap


def test_positions_pivot_basic():
    laps = pd.DataFrame({
        'Driver': ['a', 'b', 'a', 'b', 'a', 'b'],
        'LapNumber': [1, 1, 2, 2, 3, 3],
        'Position': [1, 2, 1, 2, 1, 2],
    })
    pivot, order = positions_by_lap(laps)
    assert order == [1, 2, 3]
    assert list(pivot.columns) == ['a', 'b']
    assert pivot.loc[1, 'a'] == 1
    assert pivot.loc[1, 'b'] == 2


def test_positions_forward_fill():
    laps = pd.DataFrame({
        'Driver': ['a', 'b', 'a'],           # b misses lap 2
        'LapNumber': [1, 1, 2],
        'Position': [1, 2, 1],
    })
    pivot, order = positions_by_lap(laps)
    # b should carry position 2 into lap 2
    assert pivot.loc[2, 'b'] == 2


def test_positions_driver_filter():
    laps = pd.DataFrame({
        'Driver': ['a', 'b'],
        'LapNumber': [1, 1],
        'Position': [1, 2],
    })
    pivot, _ = positions_by_lap(laps, drivers=['a'])
    assert list(pivot.columns) == ['a']


def test_positions_empty():
    (pivot, order) = positions_by_lap(pd.DataFrame())
    assert pivot.empty and order == []


def test_positions_missing_columns():
    laps = pd.DataFrame({'Driver': ['a'], 'LapNumber': [1]})  # no Position col
    pivot, _ = positions_by_lap(laps)
    assert pivot.empty


# --------------------------------------------------------------------------
# the animated figure
# --------------------------------------------------------------------------

def _pivot():
    return pd.DataFrame({'a': [1, 2], 'b': [2, 1]}, index=[1, 2])


def test_position_replay_builds_one_frame_per_lap():
    fig = build_position_replay(_pivot(), {'a': 'Ferrari', 'b': 'McLaren'})
    assert len(fig.frames) == 2
    assert [frame.name for frame in fig.frames] == ['1', '2']


def test_position_replay_lap_counter_uses_its_own_axis():
    """The lap counter must not add a category to the driver axis.

    The counter used to be a trace with a numeric x on the categorical
    driver axis, which put a stray "1" tick on it.
    """
    fig = build_position_replay(_pivot(), {'a': 'Ferrari', 'b': 'McLaren'})
    counters = [trace for frame in fig.frames for trace in frame.data
                if trace.xaxis == 'x2']
    assert counters, "the lap counter should live on the overlaid axis"
    for frame in fig.frames:
        for trace in frame.data:
            if trace.xaxis == 'x2':
                continue
            assert all(isinstance(x, str) for x in trace.x), trace.x
    assert fig.layout.xaxis2.overlaying == 'x'


def test_position_replay_of_an_empty_pivot_is_an_empty_figure():
    fig = build_position_replay(pd.DataFrame(), {})
    assert not fig.frames
