# -*- coding: utf-8 -*-
"""
test_setupfingerprint.py
~~~~~~~~~~~~~~~~~~~~~~~~
Behaviour tests for the setup-fingerprint module using synthetic telemetry.

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.setupfingerprint import (
    classify_corner,
    detect_corners,
    fingerprint,
    fingerprint_shift,
    normalise_fingerprint,
    _cornering_metrics,
)
from src.config import AERO_CONFIG


# ---------------------------------------------------------------------------
# Synthetic trace builders
# ---------------------------------------------------------------------------

def _trace_with_corners(apex_speeds, base=250.0, straight=60,
                        dip_half=6, bottom=5):
    """Speed trace: flat base, one braking dip per apex speed given.

    Dips have a short flat bottom (like a real corner's apex region) so the
    detected minimum sits exactly on the target apex speed.
    """
    speed = []
    for apex in apex_speeds:
        speed.extend([base] * straight)
        ramp = np.linspace(base, apex, dip_half)
        speed.extend(ramp)
        speed.extend([apex] * bottom)
        speed.extend(ramp[::-1])
    speed.extend([base] * straight)
    return np.asarray(speed, dtype=float)


def _car_data_frame(speed, dt=0.1):
    t0 = pd.Timestamp('2025-01-01')
    times = [t0 + pd.Timedelta(seconds=i * dt) for i in range(len(speed))]
    return pd.DataFrame({'Time': times, 'Speed': speed})


@pytest.fixture
def cfg():
    return {**AERO_CONFIG}


# ---------------------------------------------------------------------------
# Corner detection
# ---------------------------------------------------------------------------

class TestDetectCorners:
    def test_finds_one_corner_per_dip(self, cfg):
        speed = _trace_with_corners([100.0, 150.0])
        corners = detect_corners(speed, cfg)
        assert len(corners) == 2

    def test_ignores_noise_below_drop_threshold(self, cfg):
        # 5 km/h wobble is below min_speed_drop_kmh (15).
        speed = _trace_with_corners([245.0, 247.0])
        assert detect_corners(speed, cfg) == []

    def test_too_short_trace_returns_empty(self, cfg):
        assert detect_corners(np.array([200.0, 100.0]), cfg) == []

    def test_apex_speed_matches_dip_bottom(self, cfg):
        speed = _trace_with_corners([110.0])
        corners = detect_corners(speed, cfg)
        assert len(corners) == 1
        assert abs(corners[0]['apex_speed'] - 110.0) <= AERO_CONFIG['smooth_window']


# ---------------------------------------------------------------------------
# Classification and cornering metrics
# ---------------------------------------------------------------------------

class TestClassifyCorner:
    def test_buckets(self, cfg):
        assert classify_corner(80, cfg) == 'slow'
        assert classify_corner(AERO_CONFIG['slow_max_kmh'], cfg) == 'slow'
        assert classify_corner(150, cfg) == 'medium'
        assert classify_corner(240, cfg) == 'fast'


class TestCorneringMetrics:
    def test_bucket_means_and_braking(self, cfg):
        speed = _trace_with_corners([100.0, 150.0, 200.0, 230.0])
        metrics = _cornering_metrics(_car_data_frame(speed), cfg)

        assert metrics is not None
        assert metrics['corners'] == 4
        assert abs(metrics['SlowApex'] - 100.0) <= 10
        assert abs(metrics['MediumApex'] - 150.0) <= 10
        # Two fast corners average to their midpoint.
        assert abs(metrics['FastApex'] - 215.0) <= 10
        assert metrics['BrakeDecelG'] > 0

    def test_too_few_corners_reports_none_values(self, cfg):
        speed = _trace_with_corners([100.0])  # 1 corner < min_corners_total
        metrics = _cornering_metrics(_car_data_frame(speed), cfg)

        assert metrics is not None
        assert metrics['SlowApex'] is None
        assert metrics['BrakeDecelG'] is None

    def test_unusable_frame_returns_none(self, cfg):
        assert _cornering_metrics(None, cfg) is None
        assert _cornering_metrics(pd.DataFrame(), cfg) is None


# ---------------------------------------------------------------------------
# Fingerprint assembly and honesty gating
# ---------------------------------------------------------------------------

def _laps_frame(per_driver_laps):
    """Build a FastF1-like lap table: {driver: [(lap_time_s, SpeedST), ...]}."""
    rows = []
    for driver, entries in per_driver_laps.items():
        for i, (lap_s, st) in enumerate(entries, start=1):
            rows.append({
                'Driver': driver,
                'Team': f'Team_{driver}',
                'LapNumber': i,
                'LapTime': pd.Timedelta(seconds=lap_s),
                'SpeedST': st,
                'TrackStatus': 1,
                'IsAccurate': True,
                'IsOutLap': False,
            })
    return pd.DataFrame(rows)


class TestFingerprint:
    def test_top_speed_is_median_of_best_k(self):
        laps = _laps_frame({
            'AAA': [(90, 300), (90, 310), (90, 305), (90, 280), (90, 290)],
        })
        fp = fingerprint(laps)
        # top-3 ST values are 310/305/300 -> median 305.
        assert len(fp) == 1
        assert fp.iloc[0]['TopSpeed'] == 305.0

    def test_rows_exist_even_without_session_telemetry(self):
        laps = _laps_frame({'AAA': [(90, 300)]})
        fp = fingerprint(laps, session=None)
        row = fp.iloc[0]
        assert row['TopSpeed'] == 300.0
        assert row['SlowApex'] is None          # telemetry needed for cornering
        assert row['Estimated'] == 'estimated'
        assert row['Confidence'] in {'low', 'high', 'medium'}

    def test_empty_input_returns_empty_frame(self):
        assert fingerprint(pd.DataFrame()).empty
        assert fingerprint(None).empty

    def test_non_clean_laps_do_not_count(self):
        laps = _laps_frame({'AAA': [(90, 300), (90, 320), (90, 315)]})
        laps.loc[laps.index[-1], 'IsAccurate'] = False   # one dirty lap
        laps.loc[laps.index[-2], 'IsOutLap'] = True      # another dirty lap
        fp = fingerprint(laps)
        # Only one clean lap remains; median of its single trap value.
        assert fp.iloc[0]['TopSpeed'] == 300.0


class TestNormaliseFingerprint:
    def test_session_best_index_is_one(self):
        fp = pd.DataFrame({
            'Driver': ['A', 'B'],
            'TopSpeed': [320.0, 300.0],
            'SlowApex': [110.0, 105.0],
        })
        norm = normalise_fingerprint(fp)
        assert norm.iloc[0]['TopSpeedIdx'] == 1.000
        assert norm.iloc[1]['TopSpeedIdx'] == round(300.0 / 320.0, 3)
        assert norm.iloc[0]['SlowApexIdx'] == 1.000

    def test_missing_metric_stays_nan_not_zero(self):
        fp = pd.DataFrame({
            'Driver': ['A', 'B'],
            'TopSpeed': [320.0, np.nan],
        })
        norm = normalise_fingerprint(fp)
        assert np.isnan(norm.iloc[1]['TopSpeedIdx'])

    def test_empty_passthrough(self):
        assert normalise_fingerprint(pd.DataFrame()).empty


class TestFingerprintShift:
    def test_delta_positive_means_bigger_in_b(self):
        a = pd.DataFrame({'Driver': ['AAA'], 'TopSpeed': [310.0]})
        b = pd.DataFrame({'Driver': ['AAA'], 'TopSpeed': [318.0]})
        shift = fingerprint_shift(a, b)
        assert shift.iloc[0]['TopSpeed_delta'] == 8.0
        assert shift.iloc[0]['AbsShift'] == 8.0

    def test_drivers_needing_both_sides(self):
        a = pd.DataFrame({'Driver': ['AAA', 'BBB'], 'TopSpeed': [310.0, 305.0]})
        b = pd.DataFrame({'Driver': ['AAA'], 'TopSpeed': [312.0]})
        shift = fingerprint_shift(a, b)
        assert list(shift['Driver']) == ['AAA']

    def test_no_overlap_returns_empty(self):
        a = pd.DataFrame({'Driver': ['AAA'], 'TopSpeed': [310.0]})
        b = pd.DataFrame({'Driver': ['ZZZ'], 'TopSpeed': [310.0]})
        assert fingerprint_shift(a, b).empty


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
