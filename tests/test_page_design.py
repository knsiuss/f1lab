# -*- coding: utf-8 -*-
"""Design-system guard for the pages.

The pages were redesigned onto one shared system (src/ui.py + the classes in
assets/style.css). These tests keep them there: a page that starts writing
its own inline HTML, emoji or ``---`` separators again is caught here instead
of by a reader.

The checks are static (the page source) plus a rendered check of the markup
the shared components emit -- no browser, no network.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
PAGES = sorted((ROOT / 'src' / 'pages').glob('*.py'))

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

# The emoji that used to decorate the result rows.
EMOJI = re.compile('[\U0001F300-\U0001FAFF☀-➿]')


@pytest.mark.parametrize('page', PAGES, ids=lambda p: p.name)
def test_page_uses_the_shared_components(page):
    source = page.read_text(encoding='utf-8')
    assert 'from ui import' in source, (
        f'{page.name} does not import the shared UI components'
    )


@pytest.mark.parametrize('page', PAGES, ids=lambda p: p.name)
def test_page_has_no_markdown_separators(page):
    """``st.markdown("---")`` was the old divider; sections replace it."""
    source = page.read_text(encoding='utf-8')
    assert 'st.markdown("---")' not in source
    assert "st.markdown('---')" not in source
    assert 'st.divider()' not in source


@pytest.mark.parametrize('page', PAGES, ids=lambda p: p.name)
def test_page_has_no_emoji(page):
    source = page.read_text(encoding='utf-8')
    found = EMOJI.findall(source)
    assert not found, f'{page.name} still uses emoji: {found[:5]}'


@pytest.mark.parametrize('page', PAGES, ids=lambda p: p.name)
def test_page_does_not_hand_write_a_page_header(page):
    """The page title block belongs to ui.page_header."""
    source = page.read_text(encoding='utf-8')
    assert '<h1' not in source, (
        f'{page.name} hand-writes an <h1>; use ui.page_header()'
    )


def test_shared_components_emit_a_single_page_header():
    """ui.page_header is the only title block, and it renders one header."""
    from src import ui

    bodies = []

    class _Capture:
        def markdown(self, body, unsafe_allow_html=False):
            bodies.append(body)

    original = ui.st.markdown
    ui.st.markdown = _Capture().markdown
    try:
        ui.page_header('Title', 'Subtitle', [('season', '2026')])
    finally:
        ui.st.markdown = original

    assert len(bodies) == 1
    body = bodies[0]
    assert body.count('f1-page-header') == 1
    assert 'f1-page-title' in body
    assert 'f1-chip' in body


def test_no_page_downloads_the_telemetry_without_being_asked():
    """The car telemetry is the largest download in the app.

    It must sit behind an explicit user action, so that opening a page never
    waits for it.
    """
    for page in sorted(PAGES):
        source = page.read_text(encoding='utf-8')
        if 'load_telemetry=True' not in source:
            continue
        # The call must be guarded: a button (or an explicit "loaded" flag)
        # has to stand between the page and the download.
        assert ('st.button(' in source
                or 'setup_fingerprint_loaded' in source
                or 'ingest_session' in source), (
            f'{page.name} loads the telemetry without asking the user first'
        )


def test_no_page_uses_the_old_inline_team_palette():
    """Team colours come from the API (shared.team_colour), not the dict."""
    for page in PAGES:
        source = page.read_text(encoding='utf-8')
        assert 'TEAM_COLORS.get(' not in source, (
            f'{page.name} still uses the hardcoded team palette; '
            'use shared.team_colour(team, year)'
        )
