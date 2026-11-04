# -*- coding: utf-8 -*-
"""Tests for src/race_analysis.py on synthetic session lap tables."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import race_analysis as ra  # noqa: E402


def _laps():
    rows = []
    specs = [("VER", "Red Bull", [80.0, 79.5, 79.0, 78.5, 95.0]),
             ("NOR", "McLaren", [81.0, 80.5, 80.0, 79.8, 81.2])]
    for driver, team, times in specs:
        for i, seconds in enumerate(times, start=1):
            rows.append({
                "Driver": driver, "Team": team, "LapNumber": i,
                "LapTime": pd.Timedelta(seconds=seconds),
                "Position": 1 if driver == "VER" else 2,
                "Compound": "MEDIUM", "Stint": 1,
                "TrackStatus": "1", "IsAccurate": True,
            })
    return pd.DataFrame(rows)


def test_validate_laps_finds_duplicates_and_bad_rows():
    laps = _laps()
    laps.loc[0, "LapTime"] = pd.Timedelta(seconds=-5)
    dup = laps.iloc[[0]]
    laps = pd.concat([laps, dup], ignore_index=True)
    issues = ra.validate_laps(laps)
    kinds = set(issues["Issue"])
    assert "duplicate lap" in kinds
    assert "non-positive lap time" in kinds


def test_validate_clean_table_has_no_issues():
    assert ra.validate_laps(_laps()).empty
    assert ra.data_status(_laps()) == "Verified"
    assert ra.data_status(pd.DataFrame()) == "Unavailable"


def test_race_overview_fastest_and_average():
    overview = ra.race_overview(_laps())
    assert overview["fastest_driver"] == "VER"
    assert overview["fastest_lap_s"] == pytest.approx(78.5)
    assert overview["total_laps"] == 5


def test_results_table_sorted_by_position():
    table = ra.results_table(_laps())
    assert table.iloc[0]["Driver"] == "VER"
    assert table.iloc[0]["BestLap_s"] == pytest.approx(78.5)
    assert table.iloc[1]["Position"] == 2


def test_align_telemetry_resamples_by_distance():
    frames = {
        "VER": pd.DataFrame({"Distance": [0, 100, 200],
                             "Speed": [100, 200, 150], "Brake": [0, 1, 0]}),
        "NOR": pd.DataFrame({"Distance": [0, 50, 200],
                             "Speed": [90, 180, 140], "Brake": [0, 0, 1]}),
        "EMPTY": pd.DataFrame({"Speed": [1]}),
    }
    aligned = ra.align_telemetry(frames, grid_points=5)
    assert set(aligned) == {"VER", "NOR"}
    assert len(aligned["VER"]) == 5
    summary = ra.telemetry_summary(aligned["VER"])
    assert summary["max_speed"] == pytest.approx(200)
    assert summary["braking_points"] == 1


def test_stint_timeline_and_degradation_slope():
    laps = _laps()
    stints = pd.DataFrame([{"Driver": "VER", "Stint": 1, "Compound": "MEDIUM",
                            "StartLap": 1, "EndLap": 5}])
    timeline = ra.stint_timeline(stints)
    assert timeline.iloc[0]["StintLength"] == 5
    slopes = ra.degradation_slope(laps, min_laps=4)
    assert set(slopes["Driver"]) == {"VER", "NOR"}
    # NOR's laps trend faster over the stint (negative slope); the
    # fixture's VER pit-lap outlier would dominate a naive fit, which
    # is why the fit runs on clean laps only.
    nor = slopes[slopes["Driver"] == "NOR"].iloc[0]
    assert nor["Slope_s_per_lap"] < 0


def test_pit_summary_and_position_changes():
    pits = pd.DataFrame({"Driver": ["VER", "NOR"], "Lap": [5, 4],
                         "PitTime": [2.5, 3.0]})
    summary = ra.pit_summary(pits)
    assert summary["total"] == 2
    assert summary["fastest_s"] == pytest.approx(2.5)
    assert summary["window"] == (4, 5)
    changes = ra.position_changes(_laps())
    assert set(changes["Driver"]) == {"VER", "NOR"}
    assert (changes["Gain"] == 0).all()


def test_conservative_overtakes_exclude_pit_laps():
    laps = _laps()
    # NOR improves on lap 3 (P3 -> P1); lap 5 is their pit lap.
    laps.loc[laps["Driver"] == "NOR", "Position"] = [3, 3, 1, 1, 1]
    events = ra.conservative_overtakes(laps, {"NOR": [5]})
    assert len(events) == 1
    assert events.iloc[0]["Lap"] == 3


def test_pace_distribution_and_gap_stats():
    dist = ra.pace_distribution(_laps())
    assert not dist.empty
    assert dist.iloc[0]["Driver"] == "VER"  # fastest median
    gap = pd.DataFrame({"Driver": ["NOR", "NOR", "NOR"],
                        "LapNumber": [1, 2, 3],
                        "Gap": [5.0, 3.0, 1.0]})
    stats = ra.gap_stats(gap)
    assert stats.iloc[0]["FinalGap_s"] == pytest.approx(1.0)
    assert stats.iloc[0]["ClosingRate_s_per_lap"] == pytest.approx(2.0)


def test_automated_insights_are_derived():
    found = ra.automated_insights(_laps())
    assert found
    for item in found:
        assert item["text"] and item["evidence"]
    assert any("VER" in item["text"] for item in found)


def test_theoretical_best_sums_best_sectors():
    summary = pd.DataFrame([{"Driver": "VER", "Sector1Best": 20.0,
                             "Sector2Best": 30.0, "Sector3Best": 25.0}])
    best = ra.theoretical_best(summary)
    assert best.iloc[0]["TheoreticalBest_s"] == pytest.approx(75.0)
