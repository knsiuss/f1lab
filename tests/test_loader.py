# -*- coding: utf-8 -*-
"""
test_loader.py
~~~~~~~~~~~~~~
Unit tests for the schema normalisation in src/loader.py.

The season data itself comes from the OpenF1 API (src/openf1_results.py);
this module only owns the shape every consumer relies on.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import pandas as pd
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loader import clean_data, race_tracks, REQUIRED_COLUMNS


def _frame(rows):
    return pd.DataFrame(rows, columns=['Track', 'Round', 'SessionType', 'Position',
                                        'Driver', 'Team', 'Starting Grid', 'Laps',
                                        'Time/Retired', 'Status', 'Points'])


class TestCleanData:
    """Tests for clean_data."""

    def test_clean_data_creates_finished_column(self):
        sample = pd.DataFrame({
            'Driver': ['Driver A', 'Driver B'],
            'Points': ['25', '18'],
            'Position': ['1', '2'],
            'Starting Grid': ['1', '3'],
            'Laps': ['57', '57'],
            'Time/Retired': ['+0.000', '+5.123'],
        })

        cleaned = clean_data(sample)

        assert 'Finished' in cleaned.columns
        assert cleaned['Points'].dtype in ['int64', 'float64']
        assert cleaned['Position'].dtype in ['int64', 'float64']

    def test_clean_data_handles_dnf(self):
        sample = pd.DataFrame({
            'Driver': ['Driver A', 'Driver B'],
            'Points': ['0', '0'],
            'Position': [None, '20'],
            'Starting Grid': ['1', '3'],
            'Laps': ['10', '57'],
            'Time/Retired': ['DNS', '+1 Lap'],
        })

        cleaned = clean_data(sample)

        assert bool(cleaned.loc[0, 'Finished']) is False
        assert bool(cleaned.loc[1, 'Finished']) is True

    def test_status_column_drives_the_finished_flag(self):
        # The API reports the retirement status directly; it wins over the
        # historic text column.
        frame = _frame([
            ('Australian Grand Prix', 1, 'Race', 1, 'A', 'T', 1, 57, '1:26:07.469', 'Finished', 25.0),
            ('Australian Grand Prix', 1, 'Race', 12, 'B', 'T', 2, 30, 'DNF', 'DNF', 0.0),
            ('Australian Grand Prix', 1, 'Race', None, 'C', 'T', 3, 0, 'DNS', 'DNS', 0.0),
        ])
        cleaned = clean_data(frame)
        assert list(cleaned['Finished']) == [True, False, False]

    def test_clean_data_does_not_raise_on_missing_columns(self):
        # A partial API response must degrade, not crash the page.
        frame = pd.DataFrame({'Driver': ['A'], 'Points': [10]})
        cleaned = clean_data(frame)
        assert 'Finished' in cleaned.columns

    def test_clean_data_strict_mode_raises_on_missing_columns(self):
        frame = pd.DataFrame({'Driver': ['A']})
        with pytest.raises(ValueError):
            clean_data(frame, strict=True)

    def test_clean_data_on_empty_frame(self):
        cleaned = clean_data(pd.DataFrame())
        assert cleaned.empty


class TestRaceTracks:
    """Tests for the calendar-ordered track list."""

    def test_race_tracks_follow_the_round_order(self):
        frame = _frame([
            ('Chinese Grand Prix', 2, 'Sprint', 1, 'A', 'T', 1, 19, '30:39.965', 'Finished', 8.0),
            ('Australian Grand Prix', 1, 'Race', 1, 'A', 'T', 1, 57, '1:26:07.469', 'Finished', 25.0),
            ('Australian Grand Prix', 1, 'Race', 2, 'B', 'T', 2, 57, '+12.594', 'Finished', 18.0),
        ])
        assert race_tracks(frame) == ['Australian Grand Prix', 'Chinese Grand Prix']

    def test_race_tracks_on_empty_frame(self):
        assert race_tracks(pd.DataFrame()) == []


def test_required_columns_are_the_documented_ones():
    assert REQUIRED_COLUMNS == ['Points', 'Position', 'Starting Grid', 'Laps',
                                 'Time/Retired']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
