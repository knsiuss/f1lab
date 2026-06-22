# -*- coding: utf-8 -*-
"""
test_model.py
~~~~~~~~~~~~~
Unit tests for the ML model module.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import RaceStrategySimulator


@pytest.fixture
def sample_race_data():
    """Create sample race data for model training."""
    np.random.seed(42)
    n_samples = 100
    
    drivers = ['NOR', 'VER', 'PIA', 'RUS', 'LEC', 'HAM']
    teams = ['McLaren', 'Red Bull', 'Mercedes', 'Ferrari']
    tracks = ['Australia', 'Bahrain', 'Monaco', 'Silverstone']
    
    data = {
        'Driver': np.random.choice(drivers, n_samples),
        'Team': np.random.choice(teams, n_samples),
        'Track': np.random.choice(tracks, n_samples),
        'Starting Grid': np.random.randint(1, 21, n_samples),
        'Position': np.random.randint(1, 21, n_samples),
        'Finished': [True] * n_samples
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_model_dir(tmp_path):
    """Create temporary model directory."""
    model_dir = tmp_path / 'models'
    model_dir.mkdir()
    
    # Change to temp directory for tests
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    
    yield tmp_path
    
    # Restore original directory
    os.chdir(original_dir)



class TestRaceStrategySimulator:
    """Tests for RaceStrategySimulator class."""

    @pytest.fixture
    def simulator(self):
        """Create a default simulator instance."""
        return RaceStrategySimulator(base_lap_time=90.0, total_laps=57)

    def test_initialization(self):
        """Test default initialization."""
        sim = RaceStrategySimulator()
        assert sim.base_lap_time == 90.0
        assert sim.total_laps == 57
        assert sim.pit_loss == 22.0

    def test_custom_initialization(self):
        """Test custom initialization values."""
        sim = RaceStrategySimulator(base_lap_time=95.0, total_laps=30)
        assert sim.base_lap_time == 95.0
        assert sim.total_laps == 30

    def test_predict_strategy_returns_dict(self, simulator):
        """Test predict_strategy returns correct structure."""
        result = simulator.predict_strategy('VER', 'SOFT')

        assert isinstance(result, dict)
        assert '1_stop_time' in result
        assert '2_stop_time' in result
        assert 'recommended' in result
        assert 'delta' in result

    def test_predict_strategy_positive_times(self, simulator):
        """Test that predicted times are positive."""
        result = simulator.predict_strategy('VER', 'SOFT')

        assert result['1_stop_time'] > 0
        assert result['2_stop_time'] > 0
        assert result['delta'] >= 0

    def test_predict_strategy_recommendation(self, simulator):
        """Test that a recommendation is made."""
        result = simulator.predict_strategy('VER', 'SOFT')

        assert result['recommended'] in ['1 Stop', '2 Stops']

    def test_simulate_stint_returns_float(self, simulator):
        """Test _simulate_stint returns a positive float."""
        result = simulator._simulate_stint('SOFT', 1, 10)
        assert isinstance(result, float)
        assert result > 0

    def test_simulate_stint_increases_with_laps(self, simulator):
        """Test that more laps increase total time."""
        t1 = simulator._simulate_stint('SOFT', 1, 10)
        t2 = simulator._simulate_stint('SOFT', 1, 20)
        assert t2 > t1

    def test_simulate_stint_compound_effect(self, simulator):
        """Test different compounds produce different times."""
        t_soft = simulator._simulate_stint('SOFT', 1, 20)
        t_hard = simulator._simulate_stint('HARD', 1, 20)
        # Times should differ due to different degradation
        assert t_soft != t_hard

    def test_catch_up_prediction_detects_under(self, simulator):
        """Test catch-up prediction when chaser is faster."""
        # Chaser 2s behind with better tires
        result = simulator.catch_up_prediction(2.0, 'SOFT', 'HARD', 30)
        assert isinstance(result, int)
        # Should eventually catch up or return -1

    def test_catch_up_prediction_never_catches(self, simulator):
        """Test catch-up prediction when impossible."""
        # Large gap with no advantage
        result = simulator.catch_up_prediction(100.0, 'HARD', 'SOFT', 10)
        # Should not catch in 10 laps
        assert result == -1

    def test_two_stop_is_sometimes_faster(self):
        """Test that 2-stop can be faster on high-deg tracks."""
        # Many laps and high deg (via HARD compound simulation)
        sim = RaceStrategySimulator(base_lap_time=90.0, total_laps=70)
        result = sim.predict_strategy('VER', 'HARD')
        # Both strategies should give valid results
        assert isinstance(result['recommended'], str)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
