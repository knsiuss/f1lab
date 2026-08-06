# -*- coding: utf-8 -*-
"""Tests for the weekend-report builder in src/report.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.report import (
    build_insights,
    build_report,
    race_result_md,
    season_standings_md,
)


@pytest.fixture
def report_df():
    rows = [
        ('Australia', 1, 'NOR', 'McLaren', 1, 58, 25),
        ('Australia', 2, 'VER', 'Red Bull', 3, 58, 18),
        ('Australia', 3, 'PIA', 'McLaren', 2, 58, 15),
        ('China', 1, 'VER', 'Red Bull', 1, 56, 25),
        ('China', 2, 'NOR', 'McLaren', 2, 56, 18),
        ('China', 3, 'PIA', 'McLaren', 4, 56, 15),
    ]
    df = pd.DataFrame(
        rows, columns=['Track', 'Position', 'Driver', 'Team', 'Starting Grid', 'Laps', 'Points']
    )
    df['SessionType'] = 'Race'
    return df


def test_season_standings_table(report_df):
    md = season_standings_md(report_df)
    assert 'NOR' in md and 'VER' in md
    assert 'Points' in md.splitlines()[0]


def test_race_result_table(report_df):
    md = race_result_md(report_df, 'Australia')
    assert 'Lando Norris' in md or 'NOR' in md
    assert 'Pts' in md


def test_build_insights_winner_and_pole(report_df):
    insights = build_insights(report_df, 'Australia')
    joined = " ".join(insights)
    assert 'NOR' in joined
    assert 'pole' in joined.lower()


def test_build_insights_gainer(report_df):
    insights = build_insights(report_df, 'Australia')
    # PIA started P2 finished P3 -> net -1; VER started P3 finished P2 -> gained 1
    assert any('gained' in i for i in insights)


def test_build_report_full(report_df):
    md = build_report(2025, 'Australia', report_df, extra={'pit_stops': 30})
    assert '# F1 2025 Weekend Report' in md
    assert '## Australia' in md
    assert '## Race Result' in md
    assert '**30**' in md


def test_build_report_empty_df():
    md = build_report(2025, 'Australia', pd.DataFrame())
    assert '# F1 2025 Weekend Report' in md
    assert '_No season data available._' in md


def test_build_report_extra_sections(report_df):
    md = build_report(2025, 'Australia', report_df, extra_sections=['## Tyre Strategy', '_MEDIUM 2x_'])
    assert '## Tyre Strategy' in md
    assert '_MEDIUM 2x_' in md