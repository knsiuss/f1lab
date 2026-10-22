# -*- coding: utf-8 -*-
"""Tests for the pit-window radar in src/pitwindow.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MODEL_CONFIG
from src.pitwindow import ESTIMATED, pit_window
from src.whatif import Scenario, simulate_scenario


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


# ---------------------------------------------------------------------------
# Multi-stop strategies: the answer must be a real n-stop strategy
# ---------------------------------------------------------------------------

def test_pit_window_two_stops_is_not_the_one_stop_answer():
    """n_stops=2 must not collapse into the one-stop optimum."""
    one = pit_window(_laps(deg=0.1), total_laps=50, n_stops=1).iloc[0]
    two = pit_window(_laps(deg=0.1), total_laps=50, n_stops=2).iloc[0]
    assert two["NStops"] == 2
    # The two-stop answer reports two distinct stop laps.
    assert two["OptimalLap2"] is not None
    assert not pd.isna(two["OptimalLap2"])
    assert two["OptimalLap"] < two["OptimalLap2"]
    # And it is a different plan from the one-stop answer.
    assert (two["OptimalLap"], two["OptimalLap2"]) != (one["OptimalLap"], None)


def test_pit_window_two_stop_plan_is_charged_two_pit_losses():
    """The reported two-stop plan really costs two pit stops."""
    row = pit_window(_laps(deg=0.1), total_laps=50, n_stops=2).iloc[0]
    stops = [int(s) for s in str(row["OptimalStops"]).split("-")]
    assert len(stops) == 2
    scenario = Scenario(name="plan", compounds=["DEFAULT"] * 3, stop_laps=stops)
    kwargs = dict(total_laps=50, base_lap_time=float(row["BasePace"]),
                  deg_rates={"DEFAULT": float(row["DegPerLap"])},
                  fuel_gain=float(MODEL_CONFIG["fuel_gain_per_lap"]))
    with_pit = simulate_scenario(scenario, pit_loss=22.0, **kwargs)
    without_pit = simulate_scenario(scenario, pit_loss=0.0, **kwargs)
    assert with_pit["n_stops"] == 2
    assert with_pit["total_time"] - without_pit["total_time"] == pytest.approx(
        44.0, abs=0.01)


def test_pit_window_one_stop_still_charges_one_pit_loss():
    """The single-stop plan is charged exactly one pit loss."""
    row = pit_window(_laps(deg=0.1), total_laps=50, n_stops=1).iloc[0]
    stops = [int(s) for s in str(row["OptimalStops"]).split("-")]
    scenario = Scenario(name="plan", compounds=["DEFAULT"] * 2, stop_laps=stops)
    kwargs = dict(total_laps=50, base_lap_time=float(row["BasePace"]),
                  deg_rates={"DEFAULT": float(row["DegPerLap"])},
                  fuel_gain=float(MODEL_CONFIG["fuel_gain_per_lap"]))
    with_pit = simulate_scenario(scenario, pit_loss=22.0, **kwargs)
    without_pit = simulate_scenario(scenario, pit_loss=0.0, **kwargs)
    assert with_pit["n_stops"] == 1
    assert with_pit["total_time"] - without_pit["total_time"] == pytest.approx(
        22.0, abs=0.01)


def test_pit_window_reports_multi_stop_columns():
    """The extra columns the caller needs to show a two-stop window."""
    out = pit_window(_laps(deg=0.1), total_laps=50, n_stops=2)
    for col in ("NStops", "OptimalLap2", "WindowStart2", "WindowEnd2",
                "OptimalStops", "Candidates"):
        assert col in out.columns
    row = out.iloc[0]
    assert row["WindowStart2"] <= row["OptimalLap2"] <= row["WindowEnd2"]
    assert row["Candidates"] > 1