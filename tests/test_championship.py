# -*- coding: utf-8 -*-
"""Tests for the Championship Race data layer (src/championship.py)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from championship import (  # noqa: E402
    analytical_summary,
    championship_positions,
    consistency,
    count_lead_changes,
    cumulative,
    gap_vs_reference,
    insights,
    round_points,
    scenarios,
    short_track,
    standings_total,
)


def _frame():
    rows = []
    data = [
        # track, round, session, driver, team, no, position, points, status
        ("Australian Grand Prix", 1, "Race", "Max Verstappen", "Red Bull Racing", 1, 1, 25, "Finished"),
        ("Australian Grand Prix", 1, "Race", "Lando Norris", "McLaren", 4, 2, 18, "Finished"),
        ("Australian Grand Prix", 1, "Sprint", "Max Verstappen", "Red Bull Racing", 1, 1, 8, "Finished"),
        ("Australian Grand Prix", 1, "Sprint", "Lando Norris", "McLaren", 4, 2, 7, "Finished"),
        ("Chinese Grand Prix", 2, "Race", "Lando Norris", "McLaren", 4, 1, 25, "Finished"),
        ("Chinese Grand Prix", 2, "Race", "Max Verstappen", "Red Bull Racing", 1, 2, 18, "Finished"),
        ("Japanese Grand Prix", 3, "Race", "Lando Norris", "McLaren", 4, 1, 25, "Finished"),
        ("Japanese Grand Prix", 3, "Race", "Max Verstappen", "Red Bull Racing", 1, 5, 10, "Finished"),
        ("Japanese Grand Prix", 3, "Race", "Charles Leclerc", "Ferrari", 16, 3, 15, "Finished"),
    ]
    for track, rnd, session, driver, team, no, pos, pts, status in data:
        rows.append({
            "Track": track, "Round": rnd, "SessionType": session,
            "Driver": driver, "Team": team, "No": no,
            "Position": pos, "Points": pts, "Status": status,
            "Finished": True,
        })
    return pd.DataFrame(rows)


def test_short_track_strips_grand_prix_suffix():
    assert short_track("Mexico City Grand Prix") == "Mexico City"
    assert short_track("Australian Grand Prix") == "Australian"


def test_round_points_combined_sums_race_and_sprint():
    points = round_points(_frame(), "combined")
    assert points.loc["Max Verstappen", "Australian"] == 33
    assert points.loc["Lando Norris", "Australian"] == 25


def test_round_points_race_only_excludes_sprint():
    points = round_points(_frame(), "race")
    assert points.loc["Max Verstappen", "Australian"] == 25


def test_round_points_absent_driver_is_nan_not_zero():
    points = round_points(_frame(), "combined")
    # Leclerc only appears from round 3.
    assert pd.isna(points.loc["Charles Leclerc", "Australian"])
    assert points.loc["Charles Leclerc", "Japanese"] == 15


def test_cumulative_never_shrinks():
    cum = cumulative(_frame())
    assert cum.loc["Lando Norris"].tolist() == [25, 50, 75]
    assert cum.loc["Max Verstappen"].tolist() == [33, 51, 61]


def test_championship_positions_rank_all_drivers():
    positions = championship_positions(cumulative(_frame()))
    # Verstappen leads after round 1 (33 vs 25), Norris leads them after.
    assert positions.loc["Max Verstappen", "Australian"] == 1
    assert positions.loc["Lando Norris", "Chinese"] == 2
    assert positions.loc["Lando Norris", "Japanese"] == 1
    assert positions.loc["Charles Leclerc", "Japanese"] == 3


def test_count_lead_changes():
    assert count_lead_changes(cumulative(_frame())) == 1


def test_gap_vs_reference_sign():
    gaps = gap_vs_reference(cumulative(_frame()), "Lando Norris")
    assert gaps.loc["Max Verstappen", "Japanese"] == pytest.approx(-14)
    assert gaps.loc["Lando Norris", "Japanese"] == 0


def test_standings_total_combined():
    totals = standings_total(_frame())
    assert totals.index[0] == "Lando Norris"
    assert totals["Lando Norris"] == 75


def test_consistency_counts_wins_podiums_and_averages():
    table = consistency(_frame(), ["Lando Norris", "Charles Leclerc"])
    row = table[table["Driver"] == "Lando Norris"].iloc[0]
    assert row["Wins"] == 2
    assert row["Podiums"] == 3
    assert row["AvgPts"] == pytest.approx(25.0)
    late = table[table["Driver"] == "Charles Leclerc"].iloc[0]
    assert late["Races"] == 1


def test_scenarios_elimination_and_needed():
    table = scenarios(_frame(), remaining_races=1, remaining_sprints=0)
    assert table.iloc[0]["Status"] == "Leads the championship"
    # Leclerc needs 75 - 15 + 1 = 61 points but only 26 remain.
    late = table[table["Driver"] == "Charles Leclerc"].iloc[0]
    assert late["PointsNeeded"] == 61
    assert late["MaxAvailable"] == 26
    assert late["Status"] == "Mathematically eliminated"
    # Verstappen needs 15 and can still overtake.
    verstappen = table[table["Driver"] == "Max Verstappen"].iloc[0]
    assert verstappen["Status"] == "Can still overtake"


def test_insights_are_derived_and_traceable():
    found = insights(_frame(), ["Max Verstappen", "Lando Norris"])
    assert found, "expected at least one derived insight"
    for item in found:
        assert item["text"] and item["evidence"]
    assert any("Lando Norris" in item["text"] for item in found)


def test_insights_with_four_or_more_rounds():
    """The lead-extension insight reads the cumulative frame by driver."""
    frame = _frame()
    extra = frame[frame["Track"] == "Japanese Grand Prix"].copy()
    extra["Track"] = "Bahrain Grand Prix"
    extra["Round"] = 4
    frame = pd.concat([frame, extra], ignore_index=True)
    found = insights(frame, ["Max Verstappen", "Lando Norris"])
    assert found
    assert any("extended" in item["text"] or "Lando Norris" in item["text"]
               for item in found)


def test_analytical_summary_names_the_leader():
    assert "Lando Norris" in analytical_summary(_frame())
