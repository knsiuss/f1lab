# -*- coding: utf-8 -*-
"""Tests for resolve_csv_track in src/season_config.py (GP name -> CSV Track)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import season_config
from src.season_config import resolve_csv_track

_CALENDAR = [
    {'round': 1, 'name': 'Australian Grand Prix'},
    {'round': 2, 'name': 'Chinese Grand Prix'},
    {'round': 3, 'name': 'Japanese Grand Prix'},
]
_CSV_TRACKS = ['Australia', 'China', 'Japan']


def _patch_calendar(monkeypatch):
    monkeypatch.setattr(
        season_config, 'get_season_calendar', lambda year: _CALENDAR
    )


def test_exact_csv_name_passes_through():
    assert resolve_csv_track(2025, 'China', _CSV_TRACKS) == 'China'


def test_gp_name_resolves_positionally_real(monkeypatch):
    _patch_calendar(monkeypatch)
    assert resolve_csv_track(2025, 'Australian Grand Prix', _CSV_TRACKS) == 'Australia'
    assert resolve_csv_track(2025, 'Japanese Grand Prix', _CSV_TRACKS) == 'Japan'


def test_case_insensitive_match(monkeypatch):
    _patch_calendar(monkeypatch)
    assert resolve_csv_track(2025, 'china', _CSV_TRACKS) == 'China'


def test_unknown_and_future_races_return_none(monkeypatch):
    _patch_calendar(monkeypatch)
    assert resolve_csv_track(2025, 'Monaco Grand Prix', _CSV_TRACKS) is None
    assert resolve_csv_track(2025, '', _CSV_TRACKS) is None
    assert resolve_csv_track(2025, None, _CSV_TRACKS) is None


def test_gp_beyond_csv_range_returns_none(monkeypatch):
    # Calendar has 3 rounds but CSV only has the first one (season in
    # progress) — round 2 must not alias to the wrong track.
    monkeypatch.setattr(
        season_config, 'get_season_calendar', lambda year: _CALENDAR
    )
    assert resolve_csv_track(2025, 'Chinese Grand Prix', ['Australia']) is None
