# -*- coding: utf-8 -*-
"""Tests for the app shell: the page menu, the chart scale, the season progress."""

import ast
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

PAGES_DIR = ROOT / 'src' / 'pages'


# ---------------------------------------------------------------------------
# The page list (one source for the menu and the navigation)
# ---------------------------------------------------------------------------

def _page_list():
    """The PAGES constant of src/f1.py, read from the source (no import)."""
    tree = ast.parse((ROOT / 'src' / 'f1.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'PAGES':
            return ast.literal_eval(node.value)
    return []


def test_every_page_is_in_the_navigation_list():
    pages = _page_list()
    assert pages, 'the app must define its pages in one list'
    listed = {path for _title, path, _icon in pages}
    on_disk = {f'pages/{p.name}' for p in PAGES_DIR.glob('*.py')}
    assert on_disk == listed, (
        'the navigation list and the pages folder disagree: '
        f'{on_disk ^ listed}'
    )


def test_page_titles_are_unique_and_ordered():
    pages = _page_list()
    titles = [title for title, _path, _icon in pages]
    assert len(titles) == len(set(titles)), 'two pages share a title'
    assert titles[0] == 'Dashboard', 'the dashboard is the entry point'


def test_the_hamburger_button_is_wired():
    source = (ROOT / 'src' / 'f1.py').read_text(encoding='utf-8')
    assert "key='nav_toggle'" in source, 'the hamburger button must have a key'
    assert 'nav_open' in source, 'the button must toggle the menu state'


def test_the_hamburger_button_is_styled_as_a_hamburger():
    css = (ROOT / 'src' / 'assets' / 'style.css').read_text(encoding='utf-8')
    assert '.st-key-nav_toggle' in css, (
        'the hamburger button needs a style (the key becomes a st-key- class)'
    )
    assert 'linear-gradient' in css, 'the bars are drawn with a gradient'


# ---------------------------------------------------------------------------
# Chart layout: one height scale
# ---------------------------------------------------------------------------

def test_chart_height_scale_is_used_everywhere():
    """No chart may hardcode its height any more."""
    for page in sorted(PAGES_DIR.glob('*.py')):
        source = page.read_text(encoding='utf-8')
        # Only the chart layouts matter; an iframe height is a different thing.
        layouts = re.findall(r'_fig_layout\((.*?)\n\s*\)', source, re.S)
        for layout in layouts:
            raw = re.findall(r'height=\s*(\d+)', layout)
            assert not raw, (
                f'{page.name} hardcodes chart heights {raw}; use '
                "styles.chart_height('row') / scaled_height(...) instead"
            )
            for expression in re.findall(r'height=([^\n,)]+)', layout):
                assert ('chart_height(' in expression
                        or 'scaled_height(' in expression), (
                    f'{page.name} sets a chart height outside the shared scale: '
                    f'height={expression.strip()}'
                )


def test_axis_titles_use_one_form():
    """Every axis title is set the same way, so the charts look alike."""
    for page in sorted(PAGES_DIR.glob('*.py')):
        source = page.read_text(encoding='utf-8')
        assert 'xaxis=dict(title=dict(' not in source, (
            f'{page.name} still nests the axis title inside the axis dict; '
            'use xaxis_title=dict(...)'
        )
        assert 'yaxis=dict(title=dict(' not in source, (
            f'{page.name} still nests the axis title inside the axis dict; '
            'use yaxis_title=dict(...)'
        )


def test_chart_geometry_is_not_overridden_per_chart():
    """Margins and grid colours come from the shared layout, not per chart."""
    for page in sorted(PAGES_DIR.glob('*.py')):
        source = page.read_text(encoding='utf-8')
        assert 'margin=dict(' not in source, (
            f'{page.name} overrides the chart margins; the shared layout owns them'
        )
        assert 'gridcolor=GRID' not in source, (
            f'{page.name} repeats the grid colour; the shared layout owns it'
        )


def test_the_height_scale_is_ordered():
    from src.styles import CHART_HEIGHT, chart_height, scaled_height
    assert CHART_HEIGHT['tile'] < CHART_HEIGHT['row'] < CHART_HEIGHT['block']
    assert CHART_HEIGHT['block'] < CHART_HEIGHT['feature'] < CHART_HEIGHT['tall']
    assert chart_height() == CHART_HEIGHT['block'], (
        'a chart without an explicit height uses the default of the scale'
    )
    assert scaled_height('bars', 3) == CHART_HEIGHT['bars'] + 3 * 26
    with pytest.raises(KeyError):
        chart_height('does-not-exist')


def test_fig_layout_uses_the_scale_by_default():
    from src.styles import CHART_HEIGHT, fig_layout
    layout = fig_layout()
    assert layout['height'] == CHART_HEIGHT['block']
    assert layout['margin'] == dict(l=56, r=22, t=26, b=42)
    # Axes grow with their labels, so a long race name is not cut off.
    assert layout['xaxis']['automargin'] is True
    assert layout['yaxis']['automargin'] is True


# ---------------------------------------------------------------------------
# Season progress: the sidebar and the dashboard must agree
# ---------------------------------------------------------------------------

def test_season_progress_counts_the_races_with_results(monkeypatch):
    import shared

    frame = pd.DataFrame({
        'Track': ['A', 'A', 'B', 'B'],
        'Round': [1, 1, 2, 2],
        'SessionType': ['Race', 'Sprint', 'Race', 'Sprint'],
    })
    monkeypatch.setattr(shared, 'load_race_data', lambda year: frame)
    monkeypatch.setattr('season_config.get_season_calendar',
                        lambda year: [{'name': f'R{i}'} for i in range(3)])

    completed, total = shared.season_progress(2025)
    assert completed == 2, 'sprint rows are not races'
    assert total == 3


def test_season_progress_without_data(monkeypatch):
    import shared

    monkeypatch.setattr(shared, 'load_race_data', lambda year: None)
    monkeypatch.setattr('season_config.get_season_calendar', lambda year: [])
    assert shared.season_progress(2025) == (0, 0)
