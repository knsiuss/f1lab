# -*- coding: utf-8 -*-
"""
test_fastf1_extended.py
~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the extended FastF1 module.

Uses mocking to avoid actual API calls during testing.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip all tests if fastf1 is not installed
pytest.importorskip("fastf1", reason="fastf1 not installed")

from src.fastf1_extended import (
    get_tyre_degradation,
    get_best_sectors,
    get_detailed_pit_analysis,
    get_race_control_messages,
    get_session_info,
    get_tyre_stints,
    format_lap_time,
    format_f1_time,
    get_compound_color,
)


class TestFormatUtilities:
    """Tests for formatting utility functions."""

    def test_format_lap_time_valid(self):
        """Test valid lap time formatting."""
        assert format_lap_time(90.123) == "1:30.123"

    def test_format_lap_time_nan(self):
        """Test NaN returns dash."""
        assert format_lap_time(float('nan')) == "-"

    def test_format_f1_time_valid(self):
        """Test valid F1 time formatting."""
        td = pd.Timedelta(seconds=90.123)
        result = format_f1_time(td)
        assert isinstance(result, str)
        assert ":" in result

    def test_format_f1_time_long(self):
        """Test F1 time over 1 hour."""
        td = pd.Timedelta(seconds=3661.500)
        result = format_f1_time(td)
        assert result.startswith("1:")

    def test_format_f1_time_nan(self):
        """Test NaN returns N/A."""
        assert format_f1_time(pd.NaT) == "N/A"

    def test_get_compound_color_known(self):
        """Test known compound colors."""
        assert get_compound_color('SOFT') == '#FF3333'
        assert get_compound_color('MEDIUM') == '#FFF200'
        assert get_compound_color('HARD') == '#EBEBEB'

    def test_get_compound_color_unknown(self):
        """Test unknown compound returns gray."""
        assert get_compound_color('UNKNOWN') == '#808080'
        assert get_compound_color('C5') == '#808080'


class TestGetSessionInfo:
    """Tests for get_session_info function."""

    def test_returns_empty_dict_for_none(self):
        """Test that None session returns empty dict."""
        result = get_session_info(None)
        assert result == {}

    def test_extracts_event_info(self):
        """Test that event info is extracted."""
        mock_session = MagicMock()
        mock_session.event = {
            'EventName': 'Bahrain Grand Prix',
            'OfficialEventName': 'Bahrain Grand Prix',
            'Country': 'Bahrain',
            'Location': 'Sakhir',
            'CircuitShortName': 'Bahrain',
            'RoundNumber': 1,
            'EventDate': pd.Timestamp('2025-03-15'),
        }
        mock_session.name = 'Race'
        mock_session.date = pd.Timestamp('2025-03-15')

        info = get_session_info(mock_session)

        assert info['event_name'] == 'Bahrain Grand Prix'
        assert info['country'] == 'Bahrain'
        assert info['round'] == 1


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


class TestGetTyreDegradation:
    """Tests for get_tyre_degradation function."""

    def test_returns_empty_dict_for_none(self):
        """Test None session returns empty dict."""
        result = get_tyre_degradation(None, 'VER')
        assert result == {}

    def test_calculates_degradation(self):
        """Test degradation calculation."""
        mock_session = MagicMock()
        mock_laps = pd.DataFrame({
            'Driver': ['VER'] * 10,
            'LapNumber': list(range(1, 11)),
            'LapTime': [pd.Timedelta(seconds=90 + i * 0.2) for i in range(10)],
            'Compound': ['SOFT'] * 5 + ['HARD'] * 5,
        })
        mock_laps['IsAccurate'] = True

        mock_session.laps = MagicMock()
        mock_session.laps.pick_driver.return_value = mock_laps
        mock_session.laps.copy.return_value = mock_laps

        # Mock get_tyre_stints to return known stints
        with patch('src.fastf1_extended.get_tyre_stints') as mock_stints:
            mock_stints.return_value = pd.DataFrame({
                'Driver': ['VER', 'VER'],
                'Stint': [0, 1],
                'Compound': ['SOFT', 'HARD'],
                'StartLap': [1, 6],
                'EndLap': [5, 10],
                'Laps': [5, 5],
            })
            result = get_tyre_degradation(mock_session, 'VER')

        assert isinstance(result, dict)
        assert 'Stint 1' in result or 'Stint 2' in result


class TestGetBestSectors:
    """Tests for get_best_sectors function."""

    def test_returns_empty_df_for_none(self):
        """Test None session returns empty DataFrame."""
        result = get_best_sectors(None)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_finds_best_sectors_per_driver(self):
        """Test best sector identification."""
        mock_session = MagicMock()
        mock_laps = pd.DataFrame({
            'Driver': ['NOR', 'NOR', 'VER', 'VER'],
            'Team': ['McLaren', 'McLaren', 'Red Bull', 'Red Bull'],
            'LapNumber': [1, 2, 1, 2],
            'Sector1Time': [pd.Timedelta(seconds=25.0), pd.Timedelta(seconds=24.5),
                            pd.Timedelta(seconds=24.8), pd.Timedelta(seconds=25.1)],
            'Sector2Time': [pd.Timedelta(seconds=30.0), pd.Timedelta(seconds=30.5),
                            pd.Timedelta(seconds=29.8), pd.Timedelta(seconds=30.1)],
            'Sector3Time': [pd.Timedelta(seconds=30.0), pd.Timedelta(seconds=29.5),
                            pd.Timedelta(seconds=29.8), pd.Timedelta(seconds=30.1)],
            'LapTime': [pd.Timedelta(seconds=85.0), pd.Timedelta(seconds=84.5),
                        pd.Timedelta(seconds=84.4), pd.Timedelta(seconds=85.3)],
        })
        mock_session.laps = mock_laps

        result = get_best_sectors(mock_session)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # Two drivers
        assert 'Best_Sector1' in result.columns
        assert 'Best_Sector2' in result.columns
        assert 'Best_Sector3' in result.columns

    def test_calculates_theoretical_best(self):
        """Test theoretical best lap calculation."""
        mock_session = MagicMock()
        mock_laps = pd.DataFrame({
            'Driver': ['NOR'],
            'Team': ['McLaren'],
            'LapNumber': [1],
            'Sector1Time': [pd.Timedelta(seconds=25.0)],
            'Sector2Time': [pd.Timedelta(seconds=30.0)],
            'Sector3Time': [pd.Timedelta(seconds=30.0)],
            'LapTime': [pd.Timedelta(seconds=85.0)],
        })
        mock_session.laps = mock_laps

        result = get_best_sectors(mock_session)

        assert 'TheoreticalBest' in result.columns


class TestGetDetailedPitAnalysis:
    """Tests for get_detailed_pit_analysis function."""

    def test_returns_empty_df_for_none(self):
        """Test None session returns empty DataFrame."""
        result = get_detailed_pit_analysis(None)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_uses_pit_stops_if_available(self):
        """Test that session.pit_stops is used when available."""
        mock_session = MagicMock()
        mock_session.pit_stops = pd.DataFrame({
            'Driver': ['VER', 'NOR'],
            'Lap': [12, 14],
            'Stop': [1, 1],
            'Duration': [22.5, 23.1],
        })
        mock_session.laps = None

        result = get_detailed_pit_analysis(mock_session)

        assert isinstance(result, pd.DataFrame)
        assert 'LapNumber' in result.columns

    def test_falls_back_to_lap_data(self):
        """Test fallback to lap-based pit detection."""
        mock_session = MagicMock()
        mock_session.pit_stops = None
        mock_laps = MagicMock()
        mock_laps.pick_pit_stops.return_value = pd.DataFrame({
            'Driver': ['VER'],
            'LapNumber': [12],
            'PitInTime': [pd.Timestamp('2025-03-15 15:30:00')],
            'PitOutTime': [pd.Timestamp('2025-03-15 15:30:23')],
        })
        mock_session.laps = mock_laps

        result = get_detailed_pit_analysis(mock_session)

        assert isinstance(result, pd.DataFrame)


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
