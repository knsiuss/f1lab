# -*- coding: utf-8 -*-
"""
test_fastf1_extended.py
~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the extended FastF1 module.

Uses mocking to avoid actual API calls during testing.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import pytest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip all tests if fastf1 is not installed
pytest.importorskip("fastf1", reason="fastf1 not installed")

from src.fastf1_extended import (
    get_tyre_stints,
    get_pit_stops,
    get_race_control_messages,
)


class TestGetTyreStints:
    """Tests for get_tyre_stints function."""

    def test_returns_empty_df_for_none(self):
        """Test None session returns empty DataFrame."""
        result = get_tyre_stints(None)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_identifies_tyre_stints(self):
        """Test that tyre stints are correctly identified."""
        mock_session = MagicMock()
        mock_session.laps = pd.DataFrame({
            'Driver': ['VER', 'VER', 'VER', 'VER', 'VER', 'VER'],
            'LapNumber': [1, 2, 3, 4, 5, 6],
            'Compound': ['SOFT', 'SOFT', 'SOFT', 'HARD', 'HARD', 'HARD'],
        })

        stints = get_tyre_stints(mock_session)

        assert isinstance(stints, pd.DataFrame)
        assert len(stints) == 2  # Two stints
        ver_stints = stints[stints['Driver'] == 'VER']
        assert len(ver_stints) == 2

    def test_filters_by_driver(self):
        """Test filtering by specific driver."""
        mock_session = MagicMock()
        mock_session.laps = pd.DataFrame({
            'Driver': ['VER', 'VER', 'NOR', 'NOR'],
            'LapNumber': [1, 2, 1, 2],
            'Compound': ['SOFT', 'SOFT', 'MEDIUM', 'MEDIUM'],
        })

        nor_stints = get_tyre_stints(mock_session, driver='NOR')

        assert len(nor_stints) == 1
        assert nor_stints['Driver'].iloc[0] == 'NOR'


class TestGetPitStops:
    """Tests for get_pit_stops (cross-lap measurement + honest estimates)."""

    @staticmethod
    def _session(laps):
        mock_session = MagicMock()
        mock_session.laps = laps
        return mock_session

    def test_returns_empty_df_for_none(self):
        result = get_pit_stops(None)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_measures_duration_across_lap_boundary(self):
        """PitIn on lap N pairs with PitOut on lap N+1 -> positive duration."""
        laps = pd.DataFrame({
            'Driver': ['VER'] * 4,
            'LapNumber': [10, 11, 12, 13],
            'Compound': ['SOFT', 'SOFT', 'HARD', 'HARD'],
            'PitInTime': [pd.NaT, pd.Timedelta(seconds=1000.0), pd.NaT, pd.NaT],
            'PitOutTime': [pd.NaT, pd.NaT, pd.Timedelta(seconds=1022.5), pd.NaT],
        })
        result = get_pit_stops(self._session(laps))

        assert len(result) == 1
        row = result.iloc[0]
        assert row['Lap'] == 11          # pit-entry lap
        assert row['PitTime'] == 22.5    # measured across the boundary
        assert bool(row['Measured']) is True
        assert row['Compound'] == 'HARD'  # compound fitted for the next stint

    def test_unmeasurable_stop_is_labelled_estimate(self):
        """No following PitOutTime -> model default, flagged not measured."""
        from src.config import MODEL_CONFIG

        laps = pd.DataFrame({
            'Driver': ['VER'] * 3,
            'LapNumber': [1, 2, 3],
            'Compound': ['SOFT', 'SOFT', 'SOFT'],
            'PitInTime': [pd.NaT, pd.Timedelta(seconds=500.0), pd.NaT],
            'PitOutTime': [pd.NaT, pd.NaT, pd.NaT],
        })
        result = get_pit_stops(self._session(laps))

        assert len(result) == 1
        row = result.iloc[0]
        assert bool(row['Measured']) is False
        assert row['PitTime'] == round(float(MODEL_CONFIG['pit_loss_sec']), 1)


class TestGetRaceControlMessages:
    """Tests for get_race_control_messages function."""

    def test_returns_empty_df_for_none(self):
        """Test None session returns empty DataFrame."""
        result = get_race_control_messages(None)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_returns_formatted_messages(self):
        """Test that messages are formatted."""
        mock_session = MagicMock()
        mock_session.race_control_messages = pd.DataFrame({
            'Time': [pd.Timedelta(seconds=3661)],
            'Category': ['Flag'],
            'Message': ['YELLOW FLAG'],
            'Flag': ['YELLOW'],
            'Scope': ['Track'],
            'Sector': [1],
            'Lap': [5],
        })

        result = get_race_control_messages(mock_session)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert 'Message' in result.columns

    def test_handles_empty_messages(self):
        """Test empty messages return empty DataFrame."""
        mock_session = MagicMock()
        mock_session.race_control_messages = pd.DataFrame()

        result = get_race_control_messages(mock_session)
        assert len(result) == 0

    def test_handles_missing_attribute(self):
        """Test session without race_control_messages."""
        mock_session = MagicMock()
        del mock_session.race_control_messages

        result = get_race_control_messages(mock_session)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
