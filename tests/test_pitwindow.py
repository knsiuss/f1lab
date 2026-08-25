# -*- coding: utf-8 -*-
"""Tests for the pit-window radar in src/pitwindow.py."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pitwindow import ESTIMATED, pit_window


def _laps(base=90.0, n=30, driver="VER", deg=0.0, stint=1):
    """Synthetic clean laps from one stint with a linear drift."""
    return pd.DataFrame({
        "Driver": [driver] * n,
        "Stint": [stint] * n,
        "LapTime": [base + (deg * age + 0.001) for age in range(1, n + 1)],
        "TyreLife": list(range(1, n + 1)),
        "LapNumber": list(range(1, n + 1)),
        "Compound": ["MEDIUM"] * n,
    })


def test_pit_window_returns_radar_rows():
    out = pit_window(_laps(deg=0.1), total_laps=50)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["Driver"] == "VER"
    assert row["Estimated"] == ESTIMATED
    assert 0 < row["OptimalLap"] <= 49
    assert row["WindowStart"] <= row["OptimalLap"] <= row["WindowEnd"]


def test_pit_window_symmetric_optimum_is_midpoint():
    # One stop, the same compound both stints => absorb the lap number of pits
    # balance the tyre-age penalty, so the optimum sits at the race midpoint.
    out = pit_window(_laps(deg=0.1), total_laps=50).iloc[0]
    assert abs(out["OptimalLap"] - 25) <= 2


def test_pit_window_pit_loss_does_not_move_optimum():
    # A constant pit-loss adds the same offset to every candidate stop lap, so
    # it must not change the argmin (the window shifts, the optimum doesn't).
    a = pit_window(_laps(deg=0.1), total_laps=50, pit_loss=10.0).iloc[0]["OptimalLap"]
    b = pit_window(_laps(deg=0.1), total_laps=50, pit_loss=40.0).iloc[0]["OptimalLap"]
    assert a == b


def test_pit_window_multiple_drivers():
    laps = pd.concat([_laps(driver="VER"), _laps(driver="LEC", base=91.0)])
    out = pit_window(laps, total_laps=50)
    assert set(out["Driver"]) == {"VER", "LEC"}


def test_pit_window_multiplicative_stops_widen_window():
    # A larger tolerance yields a wider (or equal) window.
    out_narrow = pit_window(_laps(deg=0.1), total_laps=50, tolerance=0.0)
    out_wide = pit_window(_laps(deg=0.1), total_laps=50, tolerance=5.0)
    w0 = out_narrow.iloc[0]
    w5 = out_wide.iloc[0]
    assert (w5["WindowEnd"] - w5["WindowStart"]) >= (w0["WindowEnd"] - w0["WindowStart"])


def test_pit_window_empty_inputs():
    assert pit_window(pd.DataFrame()).empty


def test_pit_window_impossible_race_length_empty():
    assert pit_window(_laps(), total_laps=1, n_stops=1).empty


def test_pit_window_no_meaningful_deg_blank_window():
    # Sub-threshold / negative degradation must not fabricate a pit-at-lap-1.
    for d in (-0.01, 0.0, 0.005):
        out = pit_window(_laps(deg=d), total_laps=50)
        row = out.iloc[0]
        assert pd.isna(row["OptimalLap"])
        assert pd.isna(row["WindowStart"]) and pd.isna(row["WindowEnd"])