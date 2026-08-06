# -*- coding: utf-8 -*-
"""Tests for normalised pace / data-quality analysis in src/pace.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pace import (
    ESTIMATED,
    clean_laps_for_analysis,
    data_quality,
    degradation_summary,
    fit_degradation,
    long_run_quality,
    normalised_pace_by_stint,
)


def _laps(driver="VER", compound="SOFT", n=20, deg=0.08, base=90.0, stint=1):
    """Synthetic lap table: lap_time = base + deg * tyre_age."""
    ages = list(range(1, n + 1))
    return pd.DataFrame({
        "Driver": [driver] * n,
        "Compound": [compound] * n,
        "Stint": [stint] * n,
        "TyreLife": ages,
        "LapTime": [base + deg * a for a in ages],
    })


def test_clean_drops_out_lap_and_safety_car():
    df = _laps(n=6)
    df["IsOutLap"] = [True, False, False, False, False, False]
    df["IsInLap"] = [False, False, False, True, False, False]
    df["TrackStatus"] = [1, 1, 1, 1, 2, 1]  # 2 = not green
    clean = clean_laps_for_analysis(df)
    assert len(clean) == 3
    assert "LapTime_s" in clean.columns
    assert clean["LapTime_s"].notna().all()


def test_clean_keeps_all_when_markers_missing():
    df = _laps(n=5).drop(columns=["Compound"])
    clean = clean_laps_for_analysis(df)
    assert len(clean) == 5


def test_empty_laps_returns_empty():
    out = clean_laps_for_analysis(pd.DataFrame())
    assert out.empty


def test_data_quality_thresholds():
    assert data_quality(15)["level"] == "good"
    assert data_quality(15)["confidence"] == "high"
    assert data_quality(8)["level"] == "moderate"
    assert data_quality(2)["level"] == "low"
    assert data_quality(1)["level"] == "insufficient"
    for n in (2, 8, 15):
        assert data_quality(n)["estimated"] == ESTIMATED


def test_fit_degradation_recovers_slope_and_intercept():
    fit = fit_degradation(_laps(), driver="VER", compound="SOFT")
    assert fit["level"] == "good"
    assert fit["base_pace"] == pytest.approx(90.0, abs=0.01)
    assert fit["deg_per_lap"] == pytest.approx(0.08, abs=0.001)
    assert fit["n"] == 20


def test_fit_degradation_insufficient_data():
    fit = fit_degradation(_laps(n=1))
    assert fit["level"] == "insufficient"
    assert "base_pace" not in fit


def test_fit_filters_by_driver():
    df = pd.concat([_laps(driver="VER"), _laps(driver="LEC")])
    fit = fit_degradation(df, driver="VER")
    assert fit["n"] == 20


def test_normalised_pace_by_stint_structure():
    out = normalised_pace_by_stint(_laps())
    assert len(out) == 1
    row = out.iloc[0]
    assert row["BasePace"] == pytest.approx(90.0, abs=0.01)
    assert row["DegPerLap"] == pytest.approx(0.08, abs=0.001)
    assert row["Level"] == "good"
    assert row["Estimated"] == ESTIMATED


def test_normalised_pace_two_stints():
    df = pd.concat([_laps(n=12, stint=1), _laps(n=12, stint=2)])
    out = normalised_pace_by_stint(df)
    assert len(out) == 2
    assert set(out["Stint"]) == {1, 2}


def test_degradation_summary_per_compound():
    df = pd.concat([_laps(compound="SOFT"), _laps(compound="MEDIUM")])
    out = degradation_summary(df)
    assert len(out) == 2
    assert set(out["Compound"]) == {"SOFT", "MEDIUM"}


def test_long_run_quality_respects_min_stint():
    df = pd.concat([_laps(n=20, stint=1), _laps(n=5, stint=2)])
    out = long_run_quality(df, min_stint=10)
    assert len(out) == 1
    assert out.iloc[0]["Length"] == 20


def test_aggregations_empty_on_no_data():
    assert normalised_pace_by_stint(pd.DataFrame()).empty
    assert degradation_summary(pd.DataFrame()).empty
    assert long_run_quality(pd.DataFrame()).empty
