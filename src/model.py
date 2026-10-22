# -*- coding: utf-8 -*-
"""
model.py
~~~~~~~~
Race strategy simulation tools.

A transparent projection of race time for a given tyre plan, built on the
physics knobs in ``config.MODEL_CONFIG`` (degradation per compound, fuel
burn, pit loss) so every number can be traced back to a stated assumption.

Data Honesty contract
---------------------
Every number this module produces is an **estimate** and is labelled as such:
the returned dictionary carries ``estimated``, a ``level`` and a
``confidence`` taken from the sample behind the projection
(``pace.data_quality``: good >= 15 clean laps, moderate >= 8, low >= 2,
insufficient < 2), plus the list of assumptions the projection rests on.

The simulator has no per-driver data of its own: ``base_lap_time`` and the
degradation rates are either supplied by the caller (typically fitted from
the driver's clean laps) or taken from the configuration. The sample size
that supports the projection is therefore passed in by the caller
(``n_sample``); with the default of 0 the level is ``insufficient``, which is
the honest label for a projection that rests on constants alone.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import sys
import logging
from pathlib import Path
from typing import List

# Ensure src/ is importable no matter how this module is reached
# (as `model` from a page, or as `src.model` from the test suite).
sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL_CONFIG  # noqa: E402
from pace import data_quality  # noqa: E402

logger = logging.getLogger(__name__)

# Banner applied to every derived metric (project convention).
ESTIMATED = "estimated"

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
    Seconds gained (negative = faster) from the fuel burnt off by ``current_lap``.

    The car is loaded with fuel for ``total_laps`` laps and gets faster lap by
    lap as that fuel burns off, so the gain at any point is proportional to the
    fuel still in the tank, i.e. to the laps still to run. ``total_laps`` is
    therefore not decoration: it sets how much fuel is carried, and with it
    how much time the heavy car loses at the start of the race.

    Note what this term does and does not do: it is a function of the
    **absolute** lap, so it is identical in every strategy's projection and
    cancels in a strategy comparison. That is deliberate and physically right
    -- the car covers the same distance and burns the same fuel whatever the
    tyre plan -- so the 1-stop vs 2-stop decision is driven by the tyre reset
    and the pit loss, not by fuel.
    """
    laps_remaining = max(0, int(total_laps) - int(current_lap))
    return float(laps_remaining) * MODEL_CONFIG['fuel_gain_per_lap']


class RaceStrategySimulator:
    """Race strategy projection with an explicit fuel and pit-loss model."""

    def __init__(self,
                 base_lap_time: float = MODEL_CONFIG['default_base_lap_time'],
                 total_laps: int = MODEL_CONFIG['default_total_laps'],
                 n_sample: int = 0):
        """
        Args:
            base_lap_time: clean-air lap time on a fresh tyre (seconds).
            total_laps: race distance in laps.
            n_sample: number of clean laps behind ``base_lap_time`` (and the
                degradation rates). It drives the confidence label: 0 means
                "modelled from constants only", which is reported as
                ``insufficient`` rather than dressed up as a confident
                projection.
        """
        self.base_lap_time = base_lap_time
        self.total_laps = total_laps
        self.pit_loss = MODEL_CONFIG['pit_loss_sec']  # Avg pit loss in seconds
        self.n_sample = int(n_sample or 0)

    # ------------------------------------------------------------------
    # Strategy projection
    # ------------------------------------------------------------------

    def predict_strategy(self, driver: str, start_tire: str, current_lap: int = 0,
                         second_tire: str = 'MEDIUM',
                         third_tire: str = 'HARD') -> dict:
        """
        Project the race time of a one-stop and a two-stop strategy.

        A one-stop strategy is two stints (start tyre -> second tyre) and one
        pit stop; a two-stop strategy is three stints and two pit stops. The
        fuel burn is the same in both (same distance, same fuel), so the
        comparison is decided by the tyre reset against the pit loss.

        Args:
            driver: the driver's name. The simulator holds **no per-driver
                data**, so this is a label carried into the result (and into
                the assumptions) rather than a model input; a caller with the
                driver's own fitted pace should pass that as ``base_lap_time``.
            start_tire: compound for the opening stint.
            current_lap: lap the driver is on (0 = the start of the race).
            second_tire: compound for the second stint.
            third_tire: compound for the third stint (two-stop only).

        Returns:
            dict with ``1_stop_time``, ``2_stop_time``, ``recommended`` and
            ``delta`` (the existing keys, unchanged) plus the honesty labels:
            ``estimated``, ``level``, ``confidence``, ``n_sample``, the tyre
            plans used and the list of assumptions.
        """
        current_lap = max(0, int(current_lap))
        remaining = max(0, int(self.total_laps) - current_lap)
        too_short = remaining < 2
        if too_short:
            # Nothing left to race: the projection covers a minimal two-lap
            # window so the caller always gets numbers, and the result is
            # labelled "insufficient data" instead of a recommendation.
            remaining = 2
            current_lap = max(0, int(self.total_laps) - remaining)

        stop_one = current_lap + remaining // 2
        stop_two_a = current_lap + remaining // 3
        stop_two_b = current_lap + (2 * remaining) // 3

        t_one = (self._simulate_stint(start_tire, current_lap, stop_one)
                 + self._simulate_stint(second_tire, stop_one, self.total_laps)
                 + self.pit_loss)
        t_two = (self._simulate_stint(start_tire, current_lap, stop_two_a)
                 + self._simulate_stint(second_tire, stop_two_a, stop_two_b)
                 + self._simulate_stint(third_tire, stop_two_b, self.total_laps)
                 + 2 * self.pit_loss)

        quality = data_quality(self.n_sample)
        return {
            '1_stop_time': round(t_one, 3),
            '2_stop_time': round(t_two, 3),
            'recommended': ('insufficient data' if too_short
                            else ('1 Stop' if t_one < t_two else '2 Stops')),
            'delta': round(abs(t_one - t_two), 3),
            'driver': driver,
            'estimated': ESTIMATED,
            'level': 'insufficient' if too_short else quality['level'],
            'confidence': 'low' if too_short else quality['confidence'],
            'n_sample': quality['n'],
            'reason': ('not enough laps left to project a strategy'
                       if too_short else None),
            'compounds_1_stop': [start_tire, second_tire],
            'compounds_2_stop': [start_tire, second_tire, third_tire],
            'stop_laps_1_stop': [stop_one],
            'stop_laps_2_stop': [stop_two_a, stop_two_b],
            'assumptions': self._assumptions(driver, start_tire, second_tire,
                                             third_tire, current_lap),
        }

    def _simulate_stint(self, compound: str, start_lap: int, end_lap: int) -> float:
        """Simulate total time for a stint.

        Physics per lap:
            * base lap time (fresh tyre, clean air);
            * tyre ageing: ``deg * tyre_age`` counted from the stint start;
            * fuel: the car carries fuel for the whole race, so the penalty is
              a function of the **absolute** lap -- the car is heaviest at the
              start and gets lighter as the tank empties. The term is
              therefore identical in every strategy and cancels in a strategy
              comparison (the car burns the same fuel over the same
              distance); the strategy decision is made by the tyre reset
              against the pit loss.
        """
        laps = max(0, int(end_lap) - int(start_lap))
        if laps <= 0:
            return 0.0
        deg_factor = calculate_degradation_curve(compound)

        total_time = 0.0
        for lap in range(int(start_lap), int(end_lap)):
            tyre_age = lap - int(start_lap)
            # Seconds slower than an empty tank: the fuel still in the tank.
            fuel_penalty = -calculate_fuel_correction(lap, self.total_laps)
            lap_time = self.base_lap_time + deg_factor * tyre_age + fuel_penalty
            total_time += lap_time

        return total_time

    def _assumptions(self, driver: str, start_tire: str, second_tire: str,
                     third_tire: str, current_lap: int) -> List[str]:
        """Every assumption the projection rests on, as readable strings."""
        return [
            f"base lap time (fresh tyre, clean air): {self.base_lap_time}s",
            f"tyre degradation per lap (s): {_DEGRADATION_RATES}",
            f"pit loss per stop: {self.pit_loss}s",
            (f"fuel: a full load for {self.total_laps} laps, burnt off lap by "
             f"lap ({MODEL_CONFIG['fuel_gain_per_lap']}s per lap). The term is "
             f"the same in every strategy, so it cancels in the comparison"),
            (f"race distance: {self.total_laps} laps "
             f"(projection starts at lap {current_lap})"),
            (f"tyre plan: 1-stop {start_tire} -> {second_tire}, "
             f"2-stop {start_tire} -> {second_tire} -> {third_tire}"),
            "the compound plan is the caller's, not an optimisation",
            (f"driver '{driver}' is a label only: the simulator holds no "
             f"per-driver data, so it does not change the projection"),
            f"sample behind this projection: {self.n_sample} clean laps "
            f"(level {data_quality(self.n_sample)['level']})",
        ]

    # ------------------------------------------------------------------
    # Racecraft
    # ------------------------------------------------------------------

    def catch_up_prediction(self, chaser_gap: float, chaser_tire: str,
                            leader_tire: str, laps_remaining: int) -> int:
        """Lap number on which the chaser catches the leader, or -1.

        Physics:
            * the fresh-tyre advantage is a **one-off** gain on the first lap
              of the chase (fresh tyres cost the leader time once), not a
              per-lap bonus -- a per-lap bonus would close a 1s gap in one
              lap on identical tyres;
            * after that the gap changes only through tyre ageing: the chaser
              loses ``deg`` per lap of tyre age and the leader loses their
              own, so the per-lap difference is ``deg_chaser - deg_leader``.

        Args:
            chaser_gap: gap to the leader in seconds (positive = behind).
            chaser_tire / leader_tire: the compounds in use.
            laps_remaining: laps left in the race.

        Returns:
            The **1-based** lap number on which the gap is closed, or -1 when
            the chaser does not catch the leader within ``laps_remaining``
            laps -- including when the chaser is not behind to begin with.
        """
        if chaser_gap is None or laps_remaining is None:
            return -1
        laps_remaining = int(laps_remaining)
        if laps_remaining <= 0:
            return -1
        gap = float(chaser_gap)
        if gap <= 0:
            return -1  # already ahead: there is nothing to catch

        deg_chaser = calculate_degradation_curve(chaser_tire)
        deg_leader = calculate_degradation_curve(leader_tire)
        for lap in range(1, laps_remaining + 1):
            if lap == 1:
                # One-off fresh-tyre advantage, paid out on the first lap.
                gap += MODEL_CONFIG['fresh_tyre_advantage']
            else:
                gap += (deg_chaser - deg_leader) * (lap - 1)
            if gap <= 0:
                return lap
        return -1
