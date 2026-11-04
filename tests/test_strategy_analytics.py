# -*- coding: utf-8 -*-
"""Unit tests for strategy_analytics (tyre strategy & pit-stop calculations).

Synthetic FastF1-style frames only: no network, no session downloads.
Each test pins one brief requirement -- measured vs estimated pit times
stay separate, compound usage is computed per lap (not per stint),
unknown compounds are reported rather than guessed, and small samples
return no number instead of a fabricated one.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy_analytics import (
    calculate_average_compound_pace,
    calculate_average_pit_duration,
    calculate_compound_usage,
    calculate_fastest_pit_stop,
    calculate_pit_stop_table,
    calculate_stint_length,
    calculate_total_pit_stops,
    calculate_tyre_degradation,
    compare_strategies,
    degradation_trend,
    generate_strategy_insights,
    optimize_pit_lap,
    simulate_race_strategy,
)


def _laps():
    """Three drivers, three stints, pit laps at 10 and 20."""
    rows = []
    for driver in ("VER", "NOR"):
        for lap in range(1, 31):
            compound = "SOFT" if lap < 10 else ("MEDIUM" if lap < 20 else "HARD")
            rows.append({
                "Driver": driver, "Team": "T1" if driver == "VER" else "T2",
                "LapNumber": lap, "LapTime": pd.Timedelta(seconds=90 + lap / 10),
                "Compound": compound, "Stint": 1 if lap < 10 else (2 if lap < 20 else 3),
                "TyreLife": (lap - 1) % 10 + 1,
                "IsOutLap": lap in (10, 20), "IsInLap": lap in (9, 19),
                "IsAccurate": True, "TrackStatus": "1",
            })
    return pd.DataFrame(rows)


def _pit_stops():
    return pd.DataFrame([
        {"Driver": "VER", "Stop": 1, "Lap": 10, "PitTime": 2.4,
         "Compound": "MEDIUM", "Measured": True},
        {"Driver": "VER", "Stop": 2, "Lap": 20, "PitTime": 2.1,
         "Compound": "HARD", "Measured": True},
        {"Driver": "NOR", "Stop": 1, "Lap": 10, "PitTime": 25.0,
         "Compound": "MEDIUM", "Measured": False},
    ])


def _stints():
    return pd.DataFrame([
        {"Driver": "VER", "Stint": 1, "Compound": "SOFT",
         "StartLap": 1, "EndLap": 9, "Laps": 9},
        {"Driver": "VER", "Stint": 2, "Compound": "MEDIUM",
         "StartLap": 10, "EndLap": 19, "Laps": 10},
        {"Driver": "VER", "Stint": 3, "Compound": "HARD",
         "StartLap": 20, "EndLap": 30, "Laps": 11},
    ])


def test_total_pit_stops_counts_records_and_measures_split():
    result = calculate_total_pit_stops(_pit_stops())
    assert result["total"] == 3
    assert result["per_driver"] == {"VER": 2, "NOR": 1}
    assert result["measured"] == 2
    assert result["estimated"] == 1
    assert result["status"] == "partial"


def test_average_pit_duration_uses_only_valid_times():
    result = calculate_average_pit_duration(_pit_stops())
    assert result["valid"] == 3
    assert result["average"] == pytest.approx((2.4 + 2.1 + 25.0) / 3, abs=0.01)
    assert result["fastest"] == 2.1
    assert result["slowest"] == 25.0


def test_fastest_pit_stop_reports_driver_team_and_lap():
    result = calculate_fastest_pit_stop(_pit_stops(), _laps())
    assert result["driver"] == "VER"
    assert result["team"] == "T1"
    assert result["lap"] == 20
    assert result["duration"] == 2.1
    assert result["measured"] is True


def test_pit_stop_table_one_row_per_driver():
    table = calculate_pit_stop_table(_pit_stops(), _laps())
    assert len(table) == 2
    ver = table[table["Driver"] == "VER"].iloc[0]
    assert ver["Pit Stops"] == 2
    assert ver["Pit Window"] == "L10–L20"
    assert ver["Avg Duration (s)"] == pytest.approx(2.25)


def test_compound_usage_is_per_lap_not_per_stint():
    result = calculate_compound_usage(_laps())
    assert result["total_laps"] == 60
    assert result["usage"]["SOFT"] == {"laps": 18, "pct": 30.0}
    assert result["usage"]["MEDIUM"]["laps"] == 20
    assert result["by_driver"]["VER"]["HARD"] == 11


def test_compound_usage_unknown_is_reported_not_guessed():
    laps = _laps()
    laps.loc[laps.index[:3], "Compound"] = "PZERO"  # not a real compound
    result = calculate_compound_usage(laps)
    assert result["usage"]["UNKNOWN"]["laps"] == 3


def test_stint_length_recomputes_and_drops_invalid():
    stints = _stints()
    stints["Laps"] = 999  # stale column must be recomputed
    stints.loc[2, "EndLap"] = 5  # invalid: ends before it starts
    out = calculate_stint_length(stints)
    assert len(out) == 2
    assert out.iloc[1]["Laps"] == 10


def test_average_compound_pace_filtered_excludes_pit_laps():
    pace = calculate_average_compound_pace(_laps(), mode="filtered")
    assert set(pace["Compound"]) == {"SOFT", "MEDIUM", "HARD"}
    soft = pace[pace["Compound"] == "SOFT"].iloc[0]
    # In lap 9 is dropped from each driver's 9 SOFT laps: 2 * 8 = 16.
    assert soft["Laps"] == 2 * 8
    assert soft["Drivers"] == 2


def test_degradation_and_trend_shapes():
    laps = _laps()
    deg = calculate_tyre_degradation(laps)
    assert not deg.empty
    assert "DegPerLap" in deg.columns
    trend = degradation_trend(laps)
    assert not trend.empty
    assert set(trend["Compound"]) == {"SOFT", "MEDIUM", "HARD"}
    assert "MedianLapTime_s" in trend.columns


def test_simulate_and_compare_strategies_ranked():
    one = simulate_race_strategy(["SOFT", "HARD"], [15], 30,
                                 base_lap_time=90.0, pit_loss=25.0)
    assert one["total_time"] > 0
    assert one["estimated"] == "estimated"
    table = compare_strategies(
        [{"name": "One-stop", "compounds": ["SOFT", "HARD"], "stop_laps": [15]},
         {"name": "Two-stop", "compounds": ["SOFT", "MEDIUM", "HARD"],
          "stop_laps": [10, 20]}],
        30, base_lap_time=90.0, pit_loss=25.0)
    assert len(table) == 2
    assert "GapToBest" in table.columns


def test_optimize_pit_lap_returns_windows():
    windows = optimize_pit_lap(_laps(), total_laps=30, n_stops=1)
    assert not windows.empty
    assert {"Driver", "OptimalLap", "WindowStart", "WindowEnd"} \
        <= set(windows.columns)


def test_insights_are_number_backed():
    insights = generate_strategy_insights(_laps(), _stints(), _pit_stops())
    titles = {item["title"] for item in insights}
    assert "Most used compound" in titles
    assert "Fastest pit stop" in titles
    assert "Longest stint" in titles
    for item in insights:
        assert item["value"] and item["detail"]


def test_empty_frames_return_unavailable_not_crash():
    assert calculate_total_pit_stops(pd.DataFrame())["total"] == 0
    assert calculate_average_pit_duration(pd.DataFrame())["average"] is None
    assert calculate_fastest_pit_stop(pd.DataFrame())["driver"] is None
    assert calculate_compound_usage(pd.DataFrame())["total_laps"] == 0
    assert calculate_stint_length(pd.DataFrame()).empty
    assert calculate_average_compound_pace(pd.DataFrame()).empty
    assert calculate_tyre_degradation(pd.DataFrame()).empty
    assert optimize_pit_lap(pd.DataFrame()).empty
