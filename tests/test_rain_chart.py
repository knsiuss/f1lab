# -*- coding: utf-8 -*-
"""The rain-timeline chart of the Report page, built without Streamlit.

The figure builder lives in the page module, and importing that module pulls
``shared``/``season_config`` (seconds, plus a calendar request). So this test
compiles the one function out of the page source and hands it the design
tokens: what is checked is the chart it builds, not the page around it.
"""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import plotly.graph_objects as go  # noqa: E402

from styles import (  # noqa: E402
    COMPARE_COLOURS,
    LABEL_FONT,
    chart_height,
    fig_layout,
)

PAGE = ROOT / 'src' / 'pages' / '4_Report.py'


def _builder():
    """`_rain_timeline_figure` from the page source, without the page."""
    tree = ast.parse(PAGE.read_text(encoding='utf-8'))
    node = next((item for item in tree.body
                 if isinstance(item, ast.FunctionDef)
                 and item.name == '_rain_timeline_figure'), None)
    assert node is not None, 'the page no longer builds a rain timeline chart'
    namespace = {
        'go': go,
        '_fig_layout': fig_layout,
        'chart_height': chart_height,
        'LABEL_FONT': LABEL_FONT,
        'COMPARE_COLOURS': COMPARE_COLOURS,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(PAGE), 'exec'),
         namespace)
    return namespace['_rain_timeline_figure']


def _timeline(chances, amounts=None):
    amounts = amounts if amounts is not None else [0.0] * len(chances)
    steps = []
    for index, (chance, amount) in enumerate(zip(chances, amounts)):
        steps.append({
            'time_utc': f'2026-10-23T{10 + index // 4:02d}:{15 * (index % 4):02d}:00Z',
            'minutes_ahead': 15 * index,
            'probability': chance,
            'precipitation_mm': amount,
        })
    return {'steps': steps, 'step_minutes': 15, 'window_minutes': 120}


def test_each_step_becomes_one_point_in_time_order():
    figure = _builder()(_timeline([10, 40, 90, 5]))
    trace = figure.data[0]
    assert trace.type == 'scatter'
    assert list(trace.y) == [10, 40, 90, 5]
    assert list(trace.x) == ['10:00', '10:15', '10:30', '10:45']


def test_the_line_uses_the_shared_accent_and_does_not_fill():
    figure = _builder()(_timeline([5, 19, 20, 80, None]))
    trace = figure.data[0]
    assert trace.line.color == COMPARE_COLOURS[0]
    assert trace.marker.color == COMPARE_COLOURS[0]
    assert trace.mode == 'lines+markers'


def test_a_missing_chance_draws_a_gap_not_a_zero():
    figure = _builder()(_timeline([10, None, 30]))
    assert list(figure.data[0].y) == [10, None, 30]


def test_the_hover_names_the_amount_or_says_it_is_missing():
    figure = _builder()(_timeline([10, 20], amounts=[0.4, None]))
    hover = list(figure.data[0].customdata)
    assert '0.4 mm' in hover[0]
    assert 'not published' in hover[1]


def test_the_chart_follows_the_shared_layout_rules():
    figure = _builder()(_timeline([10, 20]))
    layout = figure.layout
    assert layout.title.text is None                   # the section names it
    assert layout.height == chart_height('row')        # the one height scale
    assert layout.xaxis.title.text.startswith('Time')
    assert layout.yaxis.title.text.startswith('Chance')
    assert layout.yaxis.range == (0, 105)
    assert layout.showlegend is False


def test_an_empty_timeline_does_not_crash_the_chart():
    figure = _builder()({'steps': []})
    assert len(figure.data[0].y) == 0
