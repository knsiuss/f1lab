# -*- coding: utf-8 -*-
"""Tests for session-to-session pace comparison in src/compare.py."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.compare import ESTIMATED, compare_sessions


def _laps(driver='VER', base=90.0, n=12, stint=1):
    """Synthetic clean laps with a linear tyre-age drift."""
    return pd.DataFrame({
        'Driver': [driver] * n,
        'Stint': [stint] * n,
        'LapTime': [base + 0.05 * age for age in range(1, n + 1)],
        'TyreLife': list(range(1, n + 1)),
        'Compound': ['SOFT'] * n,
    })


def test_compare_sessions_slower_session_positive_delta():
    a = _laps(base=90.0)
    b = _laps(base=91.0)  # session B is 1s/lap slower
    out = compare_sessions(a, b)
    assert len(out) == 1
    row = out.iloc[0]
    assert row['Driver'] == 'VER'
    # BasePace_A (90) - BasePace_B (91) = -1 -> session A faster.
    assert row['Delta'] == -1.0
    assert row['Estimated'] == ESTIMATED


def test_compare_sessions_driver_filter():
    a = pd.concat([_laps(driver='VER', base=90.0), _laps(driver='LEC', base=92.0)])
    b = pd.concat([_laps(driver='VER', base=91.0), _laps(driver='LEC', base=93.0)])
    out = compare_sessions(a, b, driver='VER')
    assert set(out['Driver']) == {'VER'}


def test_compare_sessions_multiple_stints():
    a = pd.concat([_laps(stint=1, base=90.0), _laps(stint=2, base=90.5)])
    b = pd.concat([_laps(stint=1, base=91.0), _laps(stint=2, base=91.0)])
    out = compare_sessions(a, b)
    assert len(out) == 2
    assert set(out['Stint']) == {1, 2}
    # Stint 1: 90 - 91 = -1.0; stint 2: 90.5 - 91.0 = -0.5.
    assert (out['Delta'] == [-1.0, -0.5]).all()


def test_compare_sessions_empty_on_insufficient_fit():
    # One lap per stint cannot produce a base-pace fit -> excluded.
    a = _laps(n=1)
    b = _laps(n=1)
    assert compare_sessions(a, b).empty


def test_compare_sessions_empty_inputs():
    assert compare_sessions(pd.DataFrame(), pd.DataFrame()).empty
