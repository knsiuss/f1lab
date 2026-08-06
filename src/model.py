# -*- coding: utf-8 -*-
"""
model.py
~~~~~~~~
Race strategy simulation tools.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import sys
import logging
from pathlib import Path

# Ensure src/ is importable no matter how this module is reached
# (as `model` from a page, or as `src.model` from the test suite).
sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL_CONFIG

logger = logging.getLogger(__name__)

# Physics knobs live in config.MODEL_CONFIG so they can be tuned in one place.
_DEGRADATION_RATES = MODEL_CONFIG['degradation_rates']


def calculate_degradation_curve(compound: str) -> float:
    """
    Calculate time loss per lap due to tire wear (seconds).
    Returns the degradation factor (s/lap).
    """
    c = str(compound).upper()
    for key, rate in _DEGRADATION_RATES.items():
        if key != 'DEFAULT' and key in c:
            return rate
    return _DEGRADATION_RATES['DEFAULT']


def calculate_fuel_correction(current_lap: int, total_laps: int) -> float:
    """
    Calculate time gained due to fuel burn (seconds).
    Returns negative time delta (time gained) relative to heavy start.
    Avg gain ~0.06s per lap driven (see config.MODEL_CONFIG).
    """
    return float(current_lap) * MODEL_CONFIG['fuel_gain_per_lap']


class RaceStrategySimulator:
    """Race strategy optimizer using tyre degradation and fuel models."""

    def __init__(self,
                 base_lap_time: float = MODEL_CONFIG['default_base_lap_time'],
                 total_laps: int = MODEL_CONFIG['default_total_laps']):
        self.base_lap_time = base_lap_time
        self.total_laps = total_laps
        self.pit_loss = MODEL_CONFIG['pit_loss_sec']  # Avg pit loss in seconds
        
    def predict_strategy(self, driver: str, start_tire: str, current_lap: int = 0) -> dict:
        """
        Predict optimal strategy and finish time.
        """
        # 1-stop: a single stint on the starting compound + one pit stop
        t_1stop = self._simulate_stint(start_tire, current_lap, self.total_laps) + self.pit_loss

        # 2-stop (simplified split): two stints + two pit stops
        stop_lap = current_lap + (self.total_laps - current_lap) // 2
        t_2stop_s1 = self._simulate_stint(start_tire, current_lap, stop_lap)
        t_2stop_s2 = self._simulate_stint('MEDIUM', stop_lap, self.total_laps)  # Assume switch to Med
        t_2stop = t_2stop_s1 + t_2stop_s2 + 2 * self.pit_loss

        return {
            '1_stop_time': t_1stop,
            '2_stop_time': t_2stop,
            'recommended': '1 Stop' if t_1stop < t_2stop else '2 Stops',
            'delta': abs(t_1stop - t_2stop)
        }

    def _simulate_stint(self, compound: str, start_lap: int, end_lap: int) -> float:
        """Simulate total time for a stint."""
        total_time = 0.0
        deg_factor = calculate_degradation_curve(compound)
        
        for lap in range(start_lap, end_lap):
            # Physics: Base + Deg - Fuel
            tyre_age = lap - start_lap
            deg_penalty = tyre_age * deg_factor
            fuel_gain = calculate_fuel_correction(lap, self.total_laps)
            
            lap_time = self.base_lap_time + deg_penalty + fuel_gain
            total_time += lap_time
            
        return total_time

    def catch_up_prediction(self, chaser_gap: float, chaser_tire: str, leader_tire: str, laps_remaining: int) -> int:
        """
        Predict lap when chaser catches leader.
        Returns lap number or -1 if never.
        """
        deg_chaser = calculate_degradation_curve(chaser_tire)
        deg_leader = calculate_degradation_curve(leader_tire)
        
        current_gap = chaser_gap
        
        for i in range(laps_remaining):
            # Relative pace (neg = chaser faster)
            # Simplified: assume chaser has fresher tyres (see config).
            pace_delta = MODEL_CONFIG['fresh_tyre_advantage'] + (deg_chaser * i) - (deg_leader * i)
            current_gap += pace_delta
            
            if current_gap <= 0:
                return i # Laps to catch
                
        return -1
