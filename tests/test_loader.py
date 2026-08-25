# -*- coding: utf-8 -*-
"""
test_loader.py
~~~~~~~~~~~~~~
Unit tests for the data loader module.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loader import load_data, clean_data, load_season_data


class TestLoadData:
    """Tests for load_data function."""
    
    def test_load_data_valid_file(self):
        """Test loading a valid CSV file."""
        data_path = Path(__file__).parent.parent / 'data' / 'Formula1_2025Season_RaceResults.csv'
        if data_path.exists():
            df = load_data(str(data_path))
            assert df is not None
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
    
    def test_load_data_invalid_file(self):
        """Test loading a non-existent file returns None."""
        df = load_data('non_existent_file.csv')
        assert df is None


class TestLoadSeasonData:
    """Tests for load_season_data (year-aware season filenames)."""

    def _write(self, tmp_path, year, kind, drivers):
        csv = "Driver,Points,Position,Starting Grid,Laps,Time/Retired\n"
        for i, d in enumerate(drivers, start=1):
            csv += f"{d},{25 - (i - 1) * 8},{i},{i},57,+2.345\n"
        path = tmp_path / f'Formula1_{year}Season_{kind}Results.csv'
        path.write_text(csv, encoding='utf-8')

    def test_load_season_data_year_in_filenames(self, tmp_path):
        """Uses the passed year in the filenames, not a hardcoded season."""
        self._write(tmp_path, 2024, 'Race', ['VER', 'NOR'])
        self._write(tmp_path, 2024, 'Sprint', ['NOR', 'VER'])
        df = load_season_data(str(tmp_path), 2024)
        assert df is not None
        assert set(df['SessionType']) == {'Race', 'Sprint'}
        assert len(df) == 4
        assert 'Finished' in df.columns

    def test_load_season_data_missing_year_returns_none(self, tmp_path):
        """A season with no files returns None rather than falling back to 2025."""
        assert load_season_data(str(tmp_path), 2024) is None


class TestCleanData:
    """Tests for clean_data function."""
    
    def test_clean_data_creates_finished_column(self):
        """Test that clean_data creates a Finished column."""
        # Create sample data
        sample_data = pd.DataFrame({
            'Driver': ['Driver A', 'Driver B'],
            'Points': ['25', '18'],
            'Position': ['1', '2'],
            'Starting Grid': ['1', '3'],
            'Laps': ['57', '57'],
            'Time/Retired': ['+0.000', '+5.123']
        })
        
        cleaned = clean_data(sample_data)
        
        assert 'Finished' in cleaned.columns
        assert cleaned['Points'].dtype in ['int64', 'float64']
        assert cleaned['Position'].dtype in ['int64', 'float64']
    
    def test_clean_data_handles_dnf(self):
        """Test that clean_data handles DNF entries."""
        sample_data = pd.DataFrame({
            'Driver': ['Driver A', 'Driver B'],
            'Points': ['0', '0'],
            'Position': [None, '20'],
            'Starting Grid': ['1', '3'],
            'Laps': ['10', '57'],
            'Time/Retired': ['DNS', '+1 Lap']
        })
        
        cleaned = clean_data(sample_data)
        
        # DNS should not be marked as finished
        assert cleaned.loc[0, 'Finished'] == False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
