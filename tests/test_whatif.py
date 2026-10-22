# -*- coding: utf-8 -*-
"""Tests for the what-if strategy simulator in src/whatif.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MODEL_CONFIG
from src.whatif import (
    ESTIMATED,
    Scenario,
    compare_scenarios,
    scenario_risk,
    simulate_scenario,
    stint_time,
    stop_lap_problem,
    undercut_delta,
)

# Baseline knobs (mirror config defaults for readable assertions).
BASE = 90.0
FUEL = MODEL_CONFIG["fuel_gain_per_lap"]  # -0.06
PIT = MODEL_CONFIG["pit_loss_sec"]        # 22.0
DEG = MODEL_CONFIG["degradation_rates"]


def test_stint_time_matches_formula():
    # SOFT, laps 0..10, lap_time = 90 + 0.12*age - 0.06*lap = 90 + 0.06*lap
    total = stint_time("SOFT", 0, 10, BASE, DEG, FUEL)
    assert total == pytest.approx(90 * 10 + 0.06 * sum(range(10)), abs=0.001)


def test_simulate_scenario_one_stop():
    sc = Scenario(name="S>H@10", compounds=["SOFT", "HARD"], stop_laps=[10])
    res = simulate_scenario(sc, total_laps=30)
    # stint1 [0,10) = 902.7 ; stint2 [10,30) = 1784.2 ; + pit 22
    assert res["total_time"] == pytest.approx(902.7 + 1784.2 + PIT, abs=0.01)
    assert res["n_stops"] == 1
    assert res["estimated"] == ESTIMATED
    assert res.get("assumptions")


def test_more_stops_consume_pit_loss():
    # With zero degradation the extra stop's tyre-reset saves nothing, so the
    # added stop must cost exactly pit_loss. (With degrading tyres the reset
    # can offset the pit loss, which is the modelling point under test.)
    flat = {"DEFAULT": 0.0}
    one = Scenario(name="1", compounds=["SOFT", "MEDIUM", "HARD"], stop_laps=[10, 20])
    two = Scenario(name="2", compounds=["SOFT", "HARD"], stop_laps=[20])
    r1 = simulate_scenario(one, total_laps=30, deg_rates=flat)
    r2 = simulate_scenario(two, total_laps=30, deg_rates=flat)
    assert r1["n_stops"] == 2
    assert r2["n_stops"] == 1
    assert r1["total_time"] == pytest.approx(r2["total_time"] + PIT, abs=0.01)


def test_compare_scenarios_ranks_best_first():
    early = Scenario(name="early", compounds=["SOFT", "HARD"], stop_laps=[10])
    late = Scenario(name="late", compounds=["SOFT", "HARD"], stop_laps=[15])
    df = compare_scenarios([early, late], total_laps=30)
    assert list(df["Rank"]) == [1, 2]
    assert df.iloc[0]["Name"] == "early"
    assert df.iloc[0]["GapToBest"] == pytest.approx(0.0)
    assert df.iloc[1]["GapToBest"] > 0
    assert df.iloc[0]["Estimated"] == ESTIMATED
    assert {"Name", "TotalTime", "GapToBest", "Stops", "Compounds",
            "Risk", "Confidence", "Estimated"} <= set(df.columns)


def test_undercut_delta_sign():
    # Degrading SOFT makes the earlier switch to HARD faster: delta negative.
    d = undercut_delta(total_laps=30, compound_new="HARD", compound_old="SOFT",
                       earlier_by=1)
    assert d["delta_sec"] < 0
    assert d["estimated"] == ESTIMATED
    assert d["assumptions"]


def test_undercut_delta_matches_compare():
    d = undercut_delta(total_laps=30, compound_new="HARD", compound_old="SOFT",
                       earlier_by=1)
    early = Scenario(name="early", compounds=["SOFT", "HARD"], stop_laps=[14])
    late = Scenario(name="late", compounds=["SOFT", "HARD"], stop_laps=[15])
    df = compare_scenarios([early, late], total_laps=30)
    expected = df.iloc[0]["TotalTime"] - df.iloc[1]["TotalTime"]
    assert d["delta_sec"] == pytest.approx(expected, abs=0.01)


def test_risk_labels():
    assert scenario_risk(0) == "very low"
    assert scenario_risk(1) == "low"
    assert scenario_risk(2) == "medium"
    assert scenario_risk(4) == "high"


def test_simulate_insufficient_laps():
    sc = Scenario(name="none", compounds=["SOFT"])
    res = simulate_scenario(sc, total_laps=0)
    assert res["level"] == "insufficient"


def test_weather_and_sc_delay_add_time():
    sc = Scenario(name="base", compounds=["SOFT", "HARD"], stop_laps=[15])
    clean = simulate_scenario(sc, total_laps=30)["total_time"]
    with_events = simulate_scenario(
        sc, total_laps=30, weather_penalty=0.5, sc_delay=8.0
    )["total_time"]
    assert with_events == pytest.approx(clean + 0.5 * 30 + 8.0, abs=0.01)


def test_traffic_and_rival_response_add_time():
    sc = Scenario(name="base", compounds=["SOFT", "HARD"], stop_laps=[15])
    clean = simulate_scenario(sc, total_laps=30)["total_time"]
    with_social = simulate_scenario(
        sc, total_laps=30, traffic_penalty=0.3, rival_response=5.0
    )
    assert with_social["total_time"] == pytest.approx(
        clean + 0.3 * 30 + 5.0, abs=0.01)
    assert any("traffic penalty" in a for a in with_social["assumptions"])
    assert any("rival response delay" in a for a in with_social["assumptions"])


def test_undercut_delta_with_rival_response():
    # A large rival reaction to the earlier stop can flip a beneficial undercut.
    base = undercut_delta(total_laps=30, compound_new="HARD",
                          compound_old="SOFT", earlier_by=1)
    reacted = undercut_delta(total_laps=30, compound_new="HARD",
                             compound_old="SOFT", earlier_by=1,
                             rival_response=10.0)
    # rival_response charges the earlier variant -> delta moves +10 towards
    # "later faster".
    assert reacted["delta_sec"] == pytest.approx(
        base["delta_sec"] + 10.0, abs=0.01)
    assert any("rival response to the earlier stop" in a
               for a in reacted["assumptions"])


# ---------------------------------------------------------------------------
# Stop-plan rules: every declared stop is paid for, invalid plans are refused
# ---------------------------------------------------------------------------

def test_stop_lap_zero_is_rejected_not_simulated_as_a_free_stop():
    """A stop on lap 0 is not a strategy -- it must not score a free time."""
    sc = Scenario(name="lap0", compounds=["SOFT", "HARD"], stop_laps=[0])
    res = simulate_scenario(sc, total_laps=30)
    assert res.get("level") == "insufficient"
    assert "total_time" not in res
    assert res.get("reason")


def test_rejected_lap_zero_stop_would_otherwise_look_fastest():
    """Regression: the old code turned stop_lap=0 into a 0-stop race."""
    real = simulate_scenario(
        Scenario(name="real", compounds=["SOFT", "HARD"], stop_laps=[15]),
        total_laps=30)
    bogus = simulate_scenario(
        Scenario(name="bogus", compounds=["SOFT", "HARD"], stop_laps=[0]),
        total_laps=30)
    # The bogus plan must not be scored at all (it used to be the fastest).
    assert "total_time" not in bogus
    assert real["total_time"] > 0


def test_stop_lap_beyond_race_distance_is_rejected():
    """A stop after the flag is not a strategy."""
    sc = Scenario(name="late", compounds=["SOFT", "HARD"], stop_laps=[30])
    res = simulate_scenario(sc, total_laps=30)
    assert res.get("level") == "insufficient"
    assert "total_time" not in res


def test_stop_lap_on_final_lap_is_the_boundary_and_is_allowed():
    """A stop on the last lap is allowed: the new tyre still runs one lap."""
    sc = Scenario(name="last", compounds=["SOFT", "HARD"], stop_laps=[29])
    res = simulate_scenario(sc, total_laps=30)
    assert res.get("total_time") is not None
    assert res["n_stops"] == 1
    # One lap beyond the flag is not a strategy.
    beyond = simulate_scenario(
        Scenario(name="beyond", compounds=["SOFT", "HARD"], stop_laps=[30]),
        total_laps=30)
    assert beyond.get("level") == "insufficient"


def test_more_stops_than_compounds_is_rejected_not_dropped():
    """Two stops with two compounds must not silently become one stop."""
    sc = Scenario(name="two", compounds=["SOFT", "HARD"], stop_laps=[10, 20])
    res = simulate_scenario(sc, total_laps=30)
    assert res.get("level") == "insufficient"
    assert "total_time" not in res
    assert res.get("n_stops") == 2


def test_stop_laps_must_increase():
    """A plan whose stops go backwards is refused."""
    sc = Scenario(name="backwards", compounds=["SOFT", "MEDIUM", "HARD"],
                  stop_laps=[20, 10])
    res = simulate_scenario(sc, total_laps=30)
    assert res.get("level") == "insufficient"
    assert "total_time" not in res


def test_n_stops_comes_from_the_scenario():
    """The reported stop count is the number of stops that were declared."""
    res = simulate_scenario(
        Scenario(name="two", compounds=["SOFT", "MEDIUM", "HARD"],
                 stop_laps=[10, 20]),
        total_laps=30)
    assert res["n_stops"] == 2
    assert res["stops_laps"] == [10, 20]


def test_every_declared_stop_is_charged_its_pit_loss():
    """Two stops must cost two pit losses, not one."""
    sc = Scenario(name="two", compounds=["SOFT", "MEDIUM", "HARD"],
                  stop_laps=[10, 20])
    with_pit = simulate_scenario(sc, total_laps=30, pit_loss=22.0)
    without_pit = simulate_scenario(sc, total_laps=30, pit_loss=0.0)
    assert with_pit["total_time"] - without_pit["total_time"] == pytest.approx(
        44.0, abs=0.01)


def test_stop_lap_problem_helper():
    """The shared rule reports why a plan cannot be simulated."""
    assert stop_lap_problem([10], ["SOFT", "HARD"], 30) is None
    assert stop_lap_problem([], ["SOFT"], 30) is None
    assert stop_lap_problem([0], ["SOFT", "HARD"], 30) is not None
    assert stop_lap_problem([30], ["SOFT", "HARD"], 30) is not None
    assert stop_lap_problem([10, 20], ["SOFT", "HARD"], 30) is not None
    assert stop_lap_problem([20, 10], ["SOFT", "MEDIUM", "HARD"], 30) is not None


def test_undercut_delta_rejects_a_race_too_short_for_a_stop():
    """A race with no room for a stop reports insufficient, not a number."""
    d = undercut_delta(total_laps=2, compound_new="HARD", compound_old="SOFT",
                       earlier_by=1)
    assert d["delta_sec"] is None
    assert d["level"] == "insufficient"
    assert d.get("reason")
