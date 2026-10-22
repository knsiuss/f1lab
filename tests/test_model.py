# -*- coding: utf-8 -*-
"""
test_model.py
~~~~~~~~~~~~~
Unit tests for the strategy simulator in src/model.py.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MODEL_CONFIG
from src.model import (
    ESTIMATED,
    RaceStrategySimulator,
    calculate_degradation_curve,
    calculate_fuel_correction,
)


@pytest.fixture
def simulator():
    """A default simulator instance."""
    return RaceStrategySimulator(base_lap_time=90.0, total_laps=57)


class TestRaceStrategySimulator:
    """Tests for RaceStrategySimulator class."""

    def test_initialization(self):
        """Test default initialization."""
        sim = RaceStrategySimulator()
        assert sim.base_lap_time == 90.0
        assert sim.total_laps == 57
        assert sim.pit_loss == 22.0
        assert sim.n_sample == 0

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

    # ------------------------------------------------------------------
    # Strategy projection: the strategy must be modelled, not just labelled
    # ------------------------------------------------------------------

    def test_one_stop_includes_pit_loss(self, simulator):
        """A 1-stop strategy must pay for its single pit stop."""
        result = simulator.predict_strategy('VER', 'SOFT')
        stop_lap = simulator.total_laps // 2
        expected = (
            simulator._simulate_stint('SOFT', 0, stop_lap)
            + simulator._simulate_stint('MEDIUM', stop_lap, simulator.total_laps)
            + simulator.pit_loss
        )
        assert result['1_stop_time'] == pytest.approx(expected)

    def test_two_stop_includes_two_pit_losses(self, simulator):
        """A 2-stop strategy must pay for both pit stops."""
        result = simulator.predict_strategy('VER', 'SOFT')
        total = simulator.total_laps
        stop_a, stop_b = total // 3, (2 * total) // 3
        expected = (
            simulator._simulate_stint('SOFT', 0, stop_a)
            + simulator._simulate_stint('MEDIUM', stop_a, stop_b)
            + simulator._simulate_stint('HARD', stop_b, total)
            + 2 * simulator.pit_loss
        )
        assert result['2_stop_time'] == pytest.approx(expected)

    def test_two_stop_uses_three_stints_not_two(self, simulator):
        """A 2-stop strategy must actually change tyres twice."""
        result = simulator.predict_strategy('VER', 'SOFT')
        assert result['compounds_2_stop'] == ['SOFT', 'MEDIUM', 'HARD']
        assert len(result['stop_laps_2_stop']) == 2
        # Two pit losses, not one.
        one_stop_penalties = simulator.pit_loss
        two_stop_penalties = 2 * simulator.pit_loss
        assert two_stop_penalties > one_stop_penalties

    def test_two_stop_is_sometimes_faster(self):
        """On a long, high-degradation race the 2-stop must win."""
        sim = RaceStrategySimulator(base_lap_time=90.0, total_laps=70)
        result = sim.predict_strategy('VER', 'HARD')
        # The tyre reset saves more than the extra pit stop costs.
        assert result['2_stop_time'] < result['1_stop_time']
        assert result['recommended'] == '2 Stops'
        assert result['delta'] == pytest.approx(
            abs(result['1_stop_time'] - result['2_stop_time']))

    def test_one_stop_wins_on_low_degradation(self, simulator):
        """On hard tyres the extra stop is not worth the pit loss."""
        result = simulator.predict_strategy('VER', 'HARD')
        assert result['1_stop_time'] < result['2_stop_time']
        assert result['recommended'] == '1 Stop'

    def test_two_stop_wins_on_high_degradation(self, simulator):
        """On soft tyres the tyre reset is worth the extra stop."""
        result = simulator.predict_strategy('VER', 'SOFT')
        assert result['2_stop_time'] < result['1_stop_time']
        assert result['recommended'] == '2 Stops'

    def test_second_compound_is_configurable(self, simulator):
        """The 2-stop tyre plan is the caller's, not hardcoded to MEDIUM."""
        result = simulator.predict_strategy('VER', 'SOFT', second_tire='HARD',
                                            third_tire='SOFT')
        assert result['compounds_2_stop'] == ['SOFT', 'HARD', 'SOFT']

    def test_driver_is_carried_but_is_only_a_label(self, simulator):
        """The driver name is reported; it does not fake per-driver data."""
        a = simulator.predict_strategy('VER', 'SOFT')
        b = simulator.predict_strategy('LEC', 'SOFT')
        assert a['driver'] == 'VER'
        assert b['driver'] == 'LEC'
        # No per-driver data exists, so the projection must not change.
        assert a['1_stop_time'] == b['1_stop_time']
        assert any('label' in s for s in a['assumptions'])

    def test_strategy_carries_honesty_labels(self, simulator):
        """Every derived figure is estimated and carries a confidence."""
        result = simulator.predict_strategy('VER', 'SOFT')
        assert result['estimated'] == ESTIMATED
        assert result['level'] == 'insufficient'
        assert result['confidence'] == 'low'
        assert result['assumptions']

    def test_confidence_follows_the_sample(self):
        """The confidence label comes from the sample behind the projection."""
        good = RaceStrategySimulator(90.0, 57, n_sample=15).predict_strategy('VER', 'SOFT')
        poor = RaceStrategySimulator(90.0, 57, n_sample=3).predict_strategy('VER', 'SOFT')
        assert good['level'] == 'good' and good['confidence'] == 'high'
        assert poor['level'] == 'low' and poor['confidence'] == 'low'

    def test_too_few_laps_returns_insufficient(self, simulator):
        """No strategy can be projected when there is no race left."""
        result = simulator.predict_strategy('VER', 'SOFT', current_lap=56)
        assert result['recommended'] == 'insufficient data'
        assert result['level'] == 'insufficient'
        assert result['reason']
        # The caller always gets numbers it can format.
        assert isinstance(result['1_stop_time'], float)
        assert isinstance(result['2_stop_time'], float)


class TestCatchUpPrediction:
    """The chase model: one-off advantage, 1-based lap numbers."""

    def test_catch_up_prediction_detects_under(self, simulator):
        """A chaser within reach catches on a 1-based lap number."""
        # 0.4s behind: the one-off fresh-tyre advantage (-0.5s) closes it.
        result = simulator.catch_up_prediction(0.4, 'SOFT', 'HARD', 30)
        assert result == 1

    def test_fresh_tyre_advantage_is_one_off_not_per_lap(self, simulator):
        """A 1.0s gap is NOT closed in one lap on identical tyres."""
        result = simulator.catch_up_prediction(1.0, 'MEDIUM', 'MEDIUM', 30)
        assert result == -1

    def test_catch_up_prediction_never_catches(self, simulator):
        """Test catch-up prediction when impossible."""
        # Large gap with no advantage
        result = simulator.catch_up_prediction(100.0, 'HARD', 'SOFT', 10)
        # Should not catch in 10 laps
        assert result == -1

    def test_chaser_not_behind_cannot_catch(self, simulator):
        """A chaser that is not behind has nothing to catch."""
        assert simulator.catch_up_prediction(0.0, 'SOFT', 'HARD', 10) == -1
        assert simulator.catch_up_prediction(-2.0, 'SOFT', 'HARD', 10) == -1

    def test_no_laps_remaining_returns_minus_one(self, simulator):
        assert simulator.catch_up_prediction(1.0, 'SOFT', 'HARD', 0) == -1

    def test_returns_lap_number_not_zero_based_index(self, simulator):
        """The returned lap number is 1-based, so a lap-1 catch is 1, not 0."""
        result = simulator.catch_up_prediction(0.1, 'SOFT', 'HARD', 5)
        assert result == 1
        assert result != 0


class TestFuelModel:
    """The fuel term must be meaningful and honestly documented."""

    def test_fuel_correction_depends_on_total_laps(self):
        """total_laps sets how much fuel is carried -- it is not ignored."""
        short = calculate_fuel_correction(0, 20)
        long = calculate_fuel_correction(0, 57)
        assert short != long
        assert long < short  # more fuel carried -> slower (negative gain)

    def test_fuel_correction_is_zero_at_the_end_of_the_distance(self):
        assert calculate_fuel_correction(57, 57) == 0.0

    def test_fuel_gain_is_negative(self):
        """Negative = faster as the fuel burns off."""
        assert calculate_fuel_correction(10, 57) < 0

    def test_fuel_term_cancels_between_strategies(self, simulator):
        """The fuel term is identical in both strategies (documented)."""
        result = simulator.predict_strategy('VER', 'SOFT')
        # Recompute both strategies with the fuel term removed: the gap
        # between them must not change, because fuel does not differentiate.
        total = simulator.total_laps
        stop_one = total // 2
        stop_a, stop_b = total // 3, (2 * total) // 3
        one_no_fuel = (simulator.base_lap_time * total
                       + calculate_degradation_curve('SOFT') * stop_one * (stop_one - 1) / 2
                       + calculate_degradation_curve('MEDIUM') * (total - stop_one) * (total - stop_one - 1) / 2
                       + simulator.pit_loss)
        two_no_fuel = (simulator.base_lap_time * total
                       + calculate_degradation_curve('SOFT') * stop_a * (stop_a - 1) / 2
                       + calculate_degradation_curve('MEDIUM') * (stop_b - stop_a) * (stop_b - stop_a - 1) / 2
                       + calculate_degradation_curve('HARD') * (total - stop_b) * (total - stop_b - 1) / 2
                       + 2 * simulator.pit_loss)
        modelled_gap = result['1_stop_time'] - result['2_stop_time']
        assert modelled_gap == pytest.approx(one_no_fuel - two_no_fuel, abs=0.05)


class TestDegradationCurve:
    def test_known_compounds(self):
        rates = MODEL_CONFIG['degradation_rates']
        assert calculate_degradation_curve('SOFT') == rates['SOFT']
        assert calculate_degradation_curve('HARD') == rates['HARD']
        assert calculate_degradation_curve('UNKNOWN') == rates['DEFAULT']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
