# -*- coding: utf-8 -*-
"""Tests for the shared UI components (src/ui.py).

The components are the one place where values from the data source (driver
names, team names, messages) are turned into markup, so the escaping is
tested here rather than assumed.
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src import ui


def test_esc_neutralises_markup():
    assert ui.esc('<img src=x onerror=alert(1)>') == \
        '&lt;img src=x onerror=alert(1)&gt;'
    assert ui.esc('"quoted"') == '&quot;quoted&quot;'
    assert ui.esc(None) == ''
    assert ui.esc(7) == '7'


def test_result_rows_escape_driver_names():
    """A driver name from the API must never become markup."""
    captured = {}

    class _Capture:
        def markdown(self, body, unsafe_allow_html=False):
            captured['body'] = body

    original = ui.st.markdown
    ui.st.markdown = _Capture().markdown
    try:
        ui.result_rows([{
            'position': 1,
            'name': '<script>alert(1)</script>',
            'team': 'McLaren',
            'value': '25 pts',
            'colour': '#F47600',
        }])
    finally:
        ui.st.markdown = original

    body = captured['body']
    assert '<script>' not in body
    assert '&lt;script&gt;' in body
    # The team colour is the one data-driven inline style that is allowed.
    assert '#F47600' in body


def test_result_rows_render_positions_without_emoji():
    captured = {}

    class _Capture:
        def markdown(self, body, unsafe_allow_html=False):
            captured['body'] = body

    original = ui.st.markdown
    ui.st.markdown = _Capture().markdown
    try:
        ui.result_rows([
            {'position': 1, 'name': 'A', 'team': 'T', 'value': '1'},
            {'position': 2, 'name': 'B', 'team': 'T', 'value': '2'},
        ])
    finally:
        ui.st.markdown = original

    body = captured['body']
    for emoji in ('\U0001F947', '\U0001F948', '\U0001F949'):
        assert emoji not in body
    assert 'f1-row-pos--p1' in body
    assert 'f1-row-pos--p2' in body


def test_kpis_render_the_expected_fields():
    captured = {}

    class _Capture:
        def markdown(self, body, unsafe_allow_html=False):
            captured['body'] = body

    original = ui.st.markdown
    ui.st.markdown = _Capture().markdown
    try:
        ui.kpis([{'label': 'Races', 'value': '12/24', 'sub': 'done',
                   'tone': 'accent'}])
    finally:
        ui.st.markdown = original

    body = captured['body']
    assert 'Races' in body and '12/24' in body and 'done' in body
    assert 'f1-kpi--accent' in body


def test_kpis_with_no_items_render_nothing():
    calls = []

    class _Capture:
        def markdown(self, body, unsafe_allow_html=False):
            calls.append(body)

    original = ui.st.markdown
    ui.st.markdown = _Capture().markdown
    try:
        ui.kpis([])
    finally:
        ui.st.markdown = original
    assert calls == []


def test_chips_render_pairs_and_plain_strings():
    captured = {}

    class _Capture:
        def markdown(self, body, unsafe_allow_html=False):
            captured['body'] = body

    original = ui.st.markdown
    ui.st.markdown = _Capture().markdown
    try:
        ui.chips([('season', '2026'), 'OpenF1 live'])
    finally:
        ui.st.markdown = original

    body = captured['body']
    assert '2026' in body and 'season' in body and 'OpenF1 live' in body


def test_section_escapes_its_title():
    captured = {}

    class _Capture:
        def markdown(self, body, unsafe_allow_html=False):
            captured['body'] = body

    original = ui.st.markdown
    ui.st.markdown = _Capture().markdown
    try:
        ui.section('<b>bad</b>', note_text='<i>note</i>')
    finally:
        ui.st.markdown = original

    body = captured['body']
    assert '<b>bad</b>' not in body
    assert '&lt;b&gt;bad&lt;/b&gt;' in body
    assert '&lt;i&gt;note&lt;/i&gt;' in body


def test_note_kinds_use_distinct_classes():
    bodies = []

    class _Capture:
        def markdown(self, body, unsafe_allow_html=False):
            bodies.append(body)

    original = ui.st.markdown
    ui.st.markdown = _Capture().markdown
    try:
        ui.note('plain')
        ui.note('careful', kind='warn')
        ui.estimate_note('estimated figure')
    finally:
        ui.st.markdown = original

    assert 'f1-note' in bodies[0]
    assert 'f1-note--warn' in bodies[1]
    assert 'f1-note--estimate' in bodies[2]


def test_empty_state_escapes_title_and_hint():
    captured = {}

    class _Capture:
        def markdown(self, body, unsafe_allow_html=False):
            captured['body'] = body

    original = ui.st.markdown
    ui.st.markdown = _Capture().markdown
    try:
        ui.empty('<script>x</script>', 'hint')
    finally:
        ui.st.markdown = original

    body = captured['body']
    assert '<script>' not in body
    assert 'hint' in body


# ---------------------------------------------------------------------------
# Regression: Streamlit renders a markdown block as HTML only until the first
# blank line, and a line holding nothing but spaces counts as blank. Markup
# with an empty line in the middle silently turns into a rendered code block
# (the "latest race" rows once shipped as raw HTML source).
# ---------------------------------------------------------------------------

def _capture_markdown(call):
    captured = {}

    class _Capture:
        def markdown(self, body, unsafe_allow_html=False):
            captured['body'] = body

    original = ui.st.markdown
    ui.st.markdown = _Capture().markdown
    try:
        call()
    finally:
        ui.st.markdown = original
    return captured.get('body', '')


def test_components_never_emit_a_blank_line():
    checks = [
        lambda: ui.page_header('Title', 'Sub', [('a', 'b')]),
        lambda: ui.section('Section', 'note'),
        lambda: ui.kpis([{'label': 'L', 'value': 'V', 'sub': 'S'}]),
        lambda: ui.kpis([{'label': 'L', 'value': 'V'}]),
        lambda: ui.result_rows([
            {'position': 1, 'name': 'A', 'team': 'T', 'value': '1',
             'colour': '#FFFFFF'},
            {'position': 4, 'name': 'B', 'value': '4'},
        ]),
        lambda: ui.note('text'),
        lambda: ui.estimate_note('text'),
        lambda: ui.chips([('a', 'b'), 'plain']),
        lambda: ui.legend([('#FFFFFF', 'Label')]),
        lambda: ui.caption('text'),
        lambda: ui.empty('Title', 'hint'),
        lambda: ui.card('<p>x</p>'),
    ]
    for check in checks:
        body = _capture_markdown(check)
        assert body.strip(), 'component rendered nothing'
        for line in body.splitlines():
            assert line.strip(), (
                'component emitted a blank line - Streamlit would render the '
                'rest of the block as code, not HTML'
            )
