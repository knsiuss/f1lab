# -*- coding: utf-8 -*-
"""Tests for the season-discovery logic in src/seasons.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.seasons import discover_available_years


def test_discovers_year_from_race_results(tmp_path):
    (tmp_path / "Formula1_2026Season_RaceResults.csv").write_text("", encoding="utf-8")
    assert discover_available_years(tmp_path) == [2026]


def test_years_sorted_ascending(tmp_path):
    for year in (2024, 2026, 2025):
        (tmp_path / f"Formula1_{year}Season_RaceResults.csv").write_text("", encoding="utf-8")
    assert discover_available_years(tmp_path) == [2024, 2025, 2026]


def test_ignores_qualifying_and_unrelated_files(tmp_path):
    (tmp_path / "Formula1_2025Season_RaceResults.csv").write_text("", encoding="utf-8")
    (tmp_path / "Formula1_2025Season_QualifyingResults.csv").write_text("", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("", encoding="utf-8")
    assert discover_available_years(tmp_path) == [2025]


def test_missing_dir_returns_empty(tmp_path):
    assert discover_available_years(tmp_path / "does-not-exist") == []


def test_empty_dir_returns_empty(tmp_path):
    assert discover_available_years(tmp_path) == []


def test_supported_years_includes_live_calendar_year(monkeypatch):
    """The in-progress season is streamable even before its CSVs exist."""
    import src.config as config
    monkeypatch.setattr(config, "_current_calendar_year", lambda: 2030)
    years = config.FASTF1_CONFIG.get_supported_years()
    assert years[-1] == 2030
    assert 2030 in years


def test_supported_years_respects_data_when_its_latest(monkeypatch):
    """A season year beyond the current calendar year is still honoured from data."""
    import src.config as config
    monkeypatch.setattr(config, "_current_calendar_year", lambda: 2024)
    # max_supported_year (from data discovery) is 2025, so 2025 remains included.
    years = config.FASTF1_CONFIG.get_supported_years()
    assert years[-1] == 2025
