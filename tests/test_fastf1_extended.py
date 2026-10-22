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

from fastf1_extended import ordered_drivers

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip all tests if fastf1 is not installed
pytest.importorskip("fastf1", reason="fastf1 not installed")

from src.fastf1_extended import (
    get_tyre_stints,
    get_pit_stops,
    get_race_control_messages,
    pit_laps_by_driver,
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

    def test_uses_the_fastf1_stint_column(self):
        """A same-compound pit stop is a new stint, and FastF1 knows it."""
        mock_session = MagicMock()
        mock_session.laps = pd.DataFrame({
            'Driver': ['VER'] * 4,
            'LapNumber': [1, 2, 3, 4],
            'Compound': ['SOFT', 'SOFT', 'SOFT', 'SOFT'],
            'Stint': [1, 1, 2, 2],
        })

        stints = get_tyre_stints(mock_session)

        assert len(stints) == 2
        first = stints[stints['Stint'] == 1].iloc[0]
        second = stints[stints['Stint'] == 2].iloc[0]
        assert first['StartLap'] == 1 and first['EndLap'] == 2
        assert first['Laps'] == 2
        assert second['StartLap'] == 3 and second['EndLap'] == 4
        assert second['Laps'] == 2

    def test_same_compound_stop_is_visible(self):
        """Regression: a same-compound stop was invisible in the stint map."""
        mock_session = MagicMock()
        mock_session.laps = pd.DataFrame({
            'Driver': ['VER'] * 4,
            'LapNumber': [1, 2, 3, 4],
            'Compound': ['SOFT'] * 4,
            'Stint': [1, 1, 2, 2],
        })

        stints = get_tyre_stints(mock_session)
        assert len(stints) == 2


class TestPitLapsByDriver:
    """The pit-stop frame names its lap column 'Lap', not 'LapNumber'."""

    def test_reads_the_lap_column(self):
        pit_stops = pd.DataFrame({
            'Driver': ['VER', 'VER', 'NOR'],
            'Stop': [1, 2, 1],
            'Lap': [12, 30, 20],
            'PitTime': [22.0, 21.5, 22.5],
        })

        laps = pit_laps_by_driver(pit_stops)

        assert laps['VER'] == [12, 30]
        assert laps['NOR'] == [20]

    def test_empty_frame(self):
        assert pit_laps_by_driver(pd.DataFrame()) == {}


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

    def test_formats_a_timestamp_as_an_elapsed_race_time(self):
        """Regression: a Timestamp came out as an empty string."""
        mock_session = MagicMock()
        mock_session.session_start_time = pd.Timestamp('2025-03-16 05:00:00')
        mock_session.race_control_messages = pd.DataFrame({
            'Time': [pd.Timestamp('2025-03-16 05:12:34.500')],
            'Category': ['Flag'],
            'Message': ['YELLOW FLAG'],
            'Flag': ['YELLOW'],
        })

        result = get_race_control_messages(mock_session)

        assert result.iloc[0]['Time'] == '12:34.500'

    def test_formats_a_timestamp_without_a_session_start(self):
        """Without an origin the wall-clock time is shown, never an empty cell."""
        mock_session = MagicMock()
        mock_session.session_start_time = None
        mock_session.race_control_messages = pd.DataFrame({
            'Time': [pd.Timestamp('2025-03-16 05:12:34')],
            'Message': ['YELLOW FLAG'],
        })

        result = get_race_control_messages(mock_session)

        assert result.iloc[0]['Time'] == '05:12:34'

    def test_timestamp_format_helper(self):
        from src.fastf1_extended import format_f1_time

        assert format_f1_time(pd.Timestamp('2025-03-16 05:12:34')) == '05:12:34'
        assert format_f1_time(pd.Timedelta(seconds=90.5)) == '1:30.500'



class TestOrderedDrivers:
    """`ordered_drivers` keeps the frame's spelling and sorts by finish."""

    @staticmethod
    def _session(rows, columns=('Position', 'Abbreviation', 'Driver')):
        return MagicMock(results=pd.DataFrame(rows, columns=list(columns)))

    def test_fastf1_codes_and_full_names_both_resolve(self):
        """The tyre chart said "0 drivers" when a full-name frame met codes."""
        frame = pd.DataFrame({'Driver': ['Max Verstappen', 'Charles Leclerc']})
        session = self._session([
            {'Position': 2, 'Abbreviation': 'VER', 'Driver': 'Max Verstappen'},
            {'Position': 1, 'Abbreviation': 'LEC', 'Driver': 'Charles Leclerc'},
        ])

        assert ordered_drivers(session, frame) == ['Charles Leclerc', 'Max Verstappen']

    def test_abbreviated_frame_is_returned_unchanged_in_order(self):
        frame = pd.DataFrame({'Driver': ['LEC', 'VER', 'HAM']})
        session = self._session([
            {'Position': 1, 'Abbreviation': 'VER'},
            {'Position': 2, 'Abbreviation': 'HAM'},
            {'Position': 3, 'Abbreviation': 'LEC'},
        ])

        assert ordered_drivers(session, frame) == ['VER', 'HAM', 'LEC']

    def test_unknown_drivers_keep_frame_order_at_the_end(self):
        frame = pd.DataFrame({'Driver': ['UNKNOWN', 'VER']})
        session = self._session([{'Position': 1, 'Abbreviation': 'VER'}])

        assert ordered_drivers(session, frame) == ['VER', 'UNKNOWN']

    def test_a_missing_or_empty_frame_gives_no_drivers(self):
        session = self._session([{'Position': 1, 'Abbreviation': 'VER'}])

        assert ordered_drivers(session, None) == []
        assert ordered_drivers(session, pd.DataFrame()) == []
        assert ordered_drivers(session, pd.DataFrame({'LapNumber': [1]})) == []

    def test_a_session_without_results_still_returns_every_driver(self):
        """A practice session has results, but a session object may not."""
        frame = pd.DataFrame({'Driver': ['VER', 'LEC']})
        session = MagicMock(results=None)
        session.results = None

        assert ordered_drivers(session, frame) == ['VER', 'LEC']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
