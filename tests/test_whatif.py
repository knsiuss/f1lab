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
