# -*- coding: utf-8 -*-
"""Tests for the Jolpica service and the source-priority config.

No network: Jolpica documents are synthetic fixtures shaped like the
live API, and the fallback path is exercised with a synthetic season
frame.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import jolpica  # noqa: E402
from sources import DATA_STATUSES, priority, provenance  # noqa: E402


DRIVER_STANDINGS = {
    "MRData": {"StandingsTable": {"StandingsLists": [{
        "DriverStandings": [
            {"position": "1", "points": "380", "wins": "9",
             "Driver": {"givenName": "Andrea Kimi",
                        "familyName": "Antonelli", "code": "ANT",
                        "permanentNumber": "12"},
             "Constructors": [{"name": "Mercedes"}]},
            {"position": "2", "points": "361", "wins": "3",
             "Driver": {"givenName": "Charles",
                        "familyName": "Leclerc", "code": "LEC",
                        "permanentNumber": "16"},
             "Constructors": [{"name": "Ferrari"}]},
        ]}]}},
}

CONSTRUCTOR_STANDINGS = {
    "MRData": {"StandingsTable": {"StandingsLists": [{
        "ConstructorStandings": [
            {"position": "1", "points": "630", "wins": "12",
             "Constructor": {"name": "Mercedes"}},
            {"position": "2", "points": "504", "wins": "2",
             "Constructor": {"name": "Ferrari"}},
        ]}]}},
}


def _patch(monkeypatch, payload):
    monkeypatch.setattr(jolpica, "_get", lambda path, ttl: payload)


def test_driver_standings_parsing(monkeypatch):
    _patch(monkeypatch, DRIVER_STANDINGS)
    table = jolpica.driver_standings(2026)
    assert table.iloc[0]["Driver"] == "Andrea Kimi Antonelli"
    assert table.iloc[0]["Points"] == 380
    assert table.iloc[0]["source_name"] == "Jolpica"
    assert table.iloc[0]["data_status"] == "Verified"


def test_constructor_standings_parsing(monkeypatch):
    _patch(monkeypatch, CONSTRUCTOR_STANDINGS)
    table = jolpica.constructor_standings(2026)
    assert table.iloc[0]["Constructor"] == "Mercedes"
    assert table.iloc[1]["Wins"] == 2


def test_standings_unavailable_when_api_fails(monkeypatch):
    monkeypatch.setattr(jolpica, "_get", lambda path, ttl: None)
    assert jolpica.driver_standings(2026) is None
    assert jolpica.constructor_standings(2026) is None


def test_constructor_fallback_computes_from_results(monkeypatch):
    monkeypatch.setattr(jolpica, "_get", lambda path, ttl: None)
    season = pd.DataFrame({
        "Team": ["Ferrari", "Ferrari", "Mercedes", "Mercedes"],
        "Points": [25, 18, 15, 12],
        "Position": [1, 2, 3, 4],
    })
    table, source = jolpica.constructor_standings_with_fallback(2026, season)
    assert "fallback" in source
    assert table.iloc[0]["Constructor"] == "Ferrari"
    assert table.iloc[0]["Points"] == 43
    assert table.iloc[0]["data_status"] == "Estimated"


RACE_RESULTS = {
    "MRData": {"RaceTable": {"Races": [{
        "Results": [
            {"position": "1", "grid": "1", "laps": "78", "points": "25",
             "status": {"status": "Finished"},
             "Time": {"time": "1:54:32.851"},
             "Driver": {"givenName": "Charles", "familyName": "Leclerc",
                        "code": "LEC"},
             "Constructors": [{"name": "Ferrari"}]},
            {"position": "2", "grid": "2", "laps": "78", "points": "18",
             "status": {"status": "Finished"},
             "Time": {"time": "+3.131"},
             "Driver": {"givenName": "Max", "familyName": "Verstappen",
                        "code": "VER"},
             "Constructors": [{"name": "Red Bull"}]},
        ]}]}},
}

PIT_STOPS = {
    "MRData": {"RaceTable": {"Races": [{
        "PitStops": [
            {"driverId": "leclerc", "lap": "25", "stop": "1",
             "duration": "2.412", "time": "14:22:10"},
        ]}]}},
}


def test_race_results_parsing(monkeypatch):
    _patch(monkeypatch, RACE_RESULTS)
    table = jolpica.race_results(2026, 8)
    assert table.iloc[0]["Driver"] == "Charles Leclerc"
    assert table.iloc[0]["Points"] == 25
    assert table.iloc[1]["Time"] == "+3.131"
    assert table.iloc[0]["source_name"] == "Jolpica"


def test_pit_stops_parsing(monkeypatch):
    _patch(monkeypatch, PIT_STOPS)
    table = jolpica.pit_stops(2026, 8)
    assert table.iloc[0]["DriverId"] == "leclerc"
    assert table.iloc[0]["Lap"] == 25
    assert table.iloc[0]["Duration"] == "2.412"


def test_results_unavailable_when_api_fails(monkeypatch):
    monkeypatch.setattr(jolpica, "_get", lambda path, ttl: None)
    assert jolpica.race_results(2026, 8) is None
    assert jolpica.sprint_results(2026, 8) is None
    assert jolpica.qualifying_results(2026, 8) is None
    assert jolpica.pit_stops(2026, 8) is None


def test_source_priority_table_matches_the_decision():
    assert priority("standings") == ("Jolpica", "computed_from_results")
    assert priority("pit_stops") == ("OpenF1", "Jolpica")
    assert priority("weather_forecast") == ("Open-Meteo", "Unavailable")
    assert priority("rain_radar") == ("RainViewer", "Unavailable")
    assert priority("unknown_dataset") == (None, None)


def test_provenance_rejects_unknown_status():
    record = provenance("standings", "Jolpica", "Verified")
    assert record["source_name"] == "Jolpica"
    with pytest.raises(ValueError):
        provenance("standings", "Jolpica", "made-up")
    assert set(DATA_STATUSES) == {"Verified", "Available", "Partial",
                                  "Estimated", "Unavailable"}
